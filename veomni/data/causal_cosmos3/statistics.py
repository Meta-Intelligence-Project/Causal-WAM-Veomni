# SPDX-License-Identifier: OpenMDW-1.1
"""Fit one LeRobot dataset's state/action statistics with native block encoding.

Derived from cosmos-framework-ascend/tools/compute_causal_action_stats.py.
Only numeric rows are read; every complete single-block start is counted once.
"""

import json
import logging
import os
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .readers.lerobot import ActionColumns, episode_frame_bounds, plan_window, read_metadata, read_window
from .transforms.processing import ActionProcessor, Geometry


class QuantileAccumulator:
    """Per-channel population moments and exact or reservoir quantiles."""

    def __init__(self, *, method, capacity, seed):
        if method not in ("exact", "reservoir") or capacity < 1 or seed < 0:
            raise ValueError("Expected exact/reservoir method, positive capacity and nonnegative seed")
        self.method, self.capacity, self.seed = method, capacity, seed
        self.count = 0
        self.counts = None

    def update(self, values, masks):
        values = values.detach().cpu().float().numpy()
        masks = masks.detach().cpu().bool().numpy()
        if values.ndim != 2 or not len(values) or masks.shape != values.shape:
            raise ValueError("Expected nonempty [N, D] values and masks")
        if not np.isfinite(values[masks]).all():
            raise ValueError("Nonfinite value in a valid statistics channel")
        width = values.shape[1]
        if self.counts is None:
            self.counts = np.zeros(width, dtype=np.int64)
            self.mean = np.zeros(width, dtype=np.float64)
            self.m2 = np.zeros(width, dtype=np.float64)
            self.minimum = np.full(width, np.inf)
            self.maximum = np.full(width, -np.inf)
            self.rows = np.empty((self.capacity, width), dtype=np.float32) if self.method == "reservoir" else None
            self.chunks = [[] for _ in range(width)]
            self.rng = [np.random.default_rng(np.random.SeedSequence([self.seed, d])) for d in range(width)]
        elif width != len(self.counts):
            raise ValueError("Cannot merge different template widths")
        self.count += len(values)
        for d in range(width):
            # 不同子集的 mask 可以不同；无效补零不进入该通道的计数、矩统计或蓄水池。
            x = values[masks[:, d], d].astype(np.float64)
            if not len(x):
                continue
            old, n = int(self.counts[d]), len(x)
            total = old + n
            batch_mean = x.mean()
            delta = batch_mean - self.mean[d]
            # 合并每批原始有效值的均值与中心二阶矩，包含批间均值差；std 使用总体分母 N。
            self.m2[d] += np.square(x - batch_mean).sum() + delta * delta * old * n / total
            self.mean[d] += delta * n / total
            self.minimum[d] = min(self.minimum[d], x.min())
            self.maximum[d] = max(self.maximum[d], x.max())
            if self.method == "exact":
                self.chunks[d].append(x.astype(np.float32))
            else:
                fill = min(n, max(0, self.capacity - old))
                self.rows[old : old + fill, d] = x[:fill]
                remaining = x[fill:]
                if len(remaining):
                    # Algorithm R 对该通道跨所有子集的有效值连续采样，不在子集边界重置。
                    slots = self.rng[d].integers(0, np.arange(old + fill + 1, total + 1))
                    selected = np.flatnonzero(slots < self.capacity)
                    # 批内同一槽位可多次命中，仅保留最后一次写入，等价于逐条更新。
                    _, last = np.unique(slots[selected[::-1]], return_index=True)
                    selected = selected[::-1][last]
                    self.rows[slots[selected], d] = remaining[selected]
            self.counts[d] = total

    def finalize(self, rotation_groups=(), *, bounds="q01_q99"):
        if bounds not in ("q01_q99", "min_max"):
            raise ValueError("bounds must be q01_q99 or min_max")
        if not self.count:
            raise ValueError("Statistics population is empty")
        valid = self.counts > 0
        qs = np.zeros((2, len(valid)), dtype=np.float64)
        for d in np.flatnonzero(valid):
            values = (
                np.concatenate(self.chunks[d])
                if self.method == "exact"
                else self.rows[: min(self.counts[d], self.capacity), d]
            )
            qs[:, d] = np.quantile(values, [0.01, 0.99], method="linear")
        # 仅选择归一化边界，所有实测统计仍完整保存；min/max 来自全量有效值。
        if bounds == "min_max":
            low = np.where(valid, self.minimum, 0).astype(np.float32)
            high = np.where(valid, self.maximum, 0).astype(np.float32)
        else:
            low, high = qs[0].astype(np.float32), qs[1].astype(np.float32)
        # q01/q99 保留实测值；只有有效四元数通道的归一化边界固定为理论范围。
        for group in rotation_groups:
            if len(group) != 4:
                raise ValueError("Fixed quaternion bounds require four-component rotation groups")
            indexes = np.asarray(group)
            if valid[indexes].any() and not valid[indexes].all():
                raise ValueError("Incomplete quaternion statistics coverage")
            if valid[indexes].all():
                low[indexes], high[indexes] = -1, 1
        metrics = {
            "min": np.where(valid, self.minimum, 0).tolist(),
            "max": np.where(valid, self.maximum, 0).tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(0, self.m2 / np.maximum(self.counts, 1))).tolist(),
            "count": [self.count],
            "valid_counts": self.counts.tolist(),
            **{k: v.tolist() for k, v in zip(("q01", "q99"), qs)},
        }
        return dict(low=low.tolist(), high=high.tolist(), valid=valid.tolist(), **metrics)

    def summary(self):
        sampled = self.counts if self.method == "exact" else np.minimum(self.counts, self.capacity)
        return {
            "rows": self.count,
            "valid_counts": self.counts.tolist(),
            "sampled_counts": sampled.tolist(),
            "approximate": bool((sampled < self.counts).any()),
        }


def discover_dataset_roots(paths):
    """Discover roots below parent directories, without traversing data or videos."""
    roots = set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        found = False
        for directory, children, _ in os.walk(path):
            children.sort()
            root = Path(directory)
            if not (root / "meta/info.json").is_file():
                continue
            if not (root / "data").is_dir():
                raise ValueError(f"Missing data directory: {root}")
            roots.add(root)
            found = True
            children[:] = []
        if not found:
            raise ValueError(f"No LeRobot datasets found under {path}")
    if not roots:
        raise ValueError("No dataset paths supplied")
    return sorted(roots)


def _ordered_map(function, items, workers):
    """Bound in-flight reads and retain population/RNG order across worker counts."""
    if workers == 1:
        yield from map(function, items)
        return
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        try:
            for _ in range(workers):
                if (item := next(items, None)) is not None:
                    pending.append(pool.submit(function, item))
            while pending:
                yield pending.popleft().result()
                if (item := next(items, None)) is not None:
                    pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


def compute_statistics(
    root,
    *,
    columns=None,
    geometry=None,
    method="reservoir",
    reservoir_size=50_000,
    seed=42,
    bounds="q01_q99",
    limit=None,
    batch_size=512,
    num_workers=4,
    log_every=1000,
):
    """Scan all episodes of ONE dataset, without padding or dataset partitioning.

    States are the anchors at each legal block start. Actions use exactly the
    training ActionProcessor(statistics=None), including absolute grippers/hands
    and canonical relative quaternions. A source owns one state/action mask, as
    in the native reader. limit is a debug block cap, recorded as partial output.
    """
    if batch_size < 1 or num_workers < 1 or (limit is not None and limit < 1) or log_every < 0:
        raise ValueError("Batch size, workers and limit must be positive; log interval must be nonnegative")
    if bounds not in ("q01_q99", "min_max"):
        raise ValueError("bounds must be q01_q99 or min_max")
    root = Path(root).resolve()
    columns, geometry = columns or ActionColumns(), geometry or Geometry()
    processor = ActionProcessor(None)
    state = QuantileAccumulator(method=method, capacity=reservoir_size, seed=seed)
    action = QuantileAccumulator(method=method, capacity=reservoir_size, seed=seed + 1)
    info, metadata = read_metadata(root)
    block = geometry.actions_per_block
    episodes, skipped = [], []
    for index, episode in enumerate(metadata["episode_index"]):
        ep = {key: values[index].item() for key, values in metadata.items()}
        first, stop = episode_frame_bounds(ep["dataset_to_index"] - ep["dataset_from_index"], columns=columns)
        count = max(0, stop - first - block)
        if count:
            episodes.append((int(episode), ep, first, count))
        else:
            skipped.append(int(episode))
    available = sum(item[3] for item in episodes)
    total = min(limit, available) if limit is not None else available
    if not total:
        raise ValueError(f"No complete {block}-action blocks in {root}")

    def batches():
        remaining = total
        for episode, ep, first, count in episodes:
            count = min(count, remaining)
            for offset in range(0, count, batch_size):
                yield episode, ep, first + offset, min(batch_size, count - offset)
            remaining -= count
            if not remaining:
                break

    def encode(item):
        episode, ep, first, count = item
        window = plan_window(
            root,
            episode,
            first,
            count + block - 1,
            columns=columns,
            block_size=block,
            video_stride=geometry.video_stride,
            pad_to_block=False,
            info=info,
            ep=ep,
        )
        raw = read_window(window, video=False)
        indexes = torch.arange(count)[:, None] + torch.arange(block)[None, :]
        # Flatten overlapping windows into independent blocks, keeping each anchor.
        raw["state"] = torch.cat((raw["state"][indexes].flatten(0, 1), raw["state"][-1:]))
        raw["action"] = raw["action"][indexes].flatten(0, 1)
        raw["real_action_steps"] = count * block
        return count, processor(raw, geometry)

    processed = 0
    logging.info("%s: scanning %d/%d blocks", root, total, available)
    for count, encoded in _ordered_map(encode, batches(), num_workers):
        state.update(encoded["states"], encoded["state_mask"].expand_as(encoded["states"]))
        action.update(encoded["action"], encoded["action_mask"].expand_as(encoded["action"]))
        previous, processed = processed, processed + count
        if log_every and (processed // log_every > previous // log_every or processed == total):
            logging.info("%s: %d/%d blocks (%.1f%%)", root, processed, total, 100 * processed / total)
    rotations = processor.template.rotation_groups
    return dict(
        state=state.finalize(rotations, bounds=bounds),
        action=action.finalize(rotations, bounds=bounds),
        provenance=dict(
            kind="fitted",
            dataset_root=str(root),
            population="all_valid_single_block_starts",
            start_stride=1,
            template_id=processor.template.template_id,
            geometry=asdict(geometry),
            columns=asdict(columns),
            episodes=[int(episode) for episode in metadata["episode_index"]],
            skipped_episodes=skipped,
            blocks=processed,
            available_blocks=available,
            partial=processed < available,
            method=method,
            quantiles=[0.01, 0.99],
            interpolation="linear",
            std_ddof=0,
            seed=seed,
            reservoir_size=reservoir_size if method == "reservoir" else None,
            bounds=bounds,
            quaternion_bounds=[-1, 1],
            rotation_groups=[list(group) for group in rotations],
            state=state.summary(),
            action=action.summary(),
        ),
    )


def save_statistics(statistics, path):
    """Atomically replace a dataset's statistics only after successful computation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".causal_stats_", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(statistics, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
