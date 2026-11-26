"""Finite, deterministic validation iteration over existing video/action datasets."""

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

from ..readers.lerobot import EmptyTemporalSampleError
from .dataset import MODES, ActionDataset, LeRobotDataset
from .frame_dataset import ActionFrameDataset, LeRobotFrameDataset
from .pipeline import collate_action_pack


VALIDATION_VERSION = "causal-cosmos3-validation-v3"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validation_seed(seed, identity, stream):
    """Derive rank/order/step independent 63-bit seeds without Python hash()."""
    return int(fingerprint([VALIDATION_VERSION, seed, identity, stream])[:16], 16) % (2**63)


def dataset_subsets(dataset):
    """Visit the existing concat/weighted/uniform source wrappers without reading."""
    if isinstance(dataset, (LeRobotDataset, LeRobotFrameDataset)):
        return [dataset]
    if hasattr(dataset, "dataset"):
        return dataset_subsets(dataset.dataset)
    children = getattr(dataset, "datasets", getattr(dataset, "_datasets", ()))
    return [subset for child in children for subset in dataset_subsets(child)]


def iter_frame_samples(dataset, rank=0, world_size=1, *, raw=False):
    """Finite data inspection in root/index order, independent of source sampling weights.

    No duplicate candidates are added to equalize ranks. Errors retain their source
    context and propagate. This iterator is not a distributed model evaluator.
    """
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Expected world_size > 0 and 0 <= rank < world_size")
    subsets = sorted(dataset_subsets(dataset), key=lambda ds: (ds.source, ds.root))
    if not subsets or any(not isinstance(ds, LeRobotFrameDataset) for ds in subsets):
        raise ValueError("iter_frame_samples requires frame datasets")
    offset = 0
    for ds in subsets:
        for index in range((rank - offset) % world_size, len(ds), world_size):
            yield ds.read_sample(index) if raw else ds[index]
        offset += len(ds)


def build_validation_indices(dataset):
    """按固定顺序筛掉无法成块的 frame anchor；只保存原始索引，不解码视频。"""
    indices = []
    for ds in sorted(dataset_subsets(dataset), key=lambda ds: (ds.source, ds.root)):
        for index in range(len(ds)):
            if isinstance(ds, LeRobotFrameDataset):
                try:
                    ds.read_sample(index, video=False)
                except EmptyTemporalSampleError:
                    continue
            indices.append((ds, index))
    if not indices:
        raise ValueError("Validation requires at least one legal window")
    return indices


def iter_validation_batches(dataset, rank=0, world_size=1, *, seed=42, indices=None):
    """Yield singleton packs with fixed task/history/seeds and equal DP rounds.

    Screen frame anchors before sharding, or reuse the supplied screened indices.
    Only assigned valid candidates are prepared; dataset RNG and loader state are unused.
    Old datasets retain empty packs for failed reads. Frame datasets require full
    history and propagate read errors to the evaluator's cross-rank synchronization.
    """
    subsets = sorted(dataset_subsets(dataset), key=lambda ds: (ds.source, ds.root))
    if any(isinstance(ds, LeRobotFrameDataset) and not ds.full_history for ds in subsets):
        raise ValueError("Frame validation requires full_history=True")
    indices = build_validation_indices(dataset) if indices is None else indices
    if not indices:
        raise ValueError("Validation requires at least one legal window")
    window_count = len(indices)
    is_action = isinstance(subsets[0], (ActionDataset, ActionFrameDataset))
    if any(isinstance(ds, (ActionDataset, ActionFrameDataset)) != is_action for ds in subsets):
        raise ValueError("Validation requires a single dataset stage")
    if is_action:
        modes = MODES if subsets[0].mode == "joint" else (subsets[0].mode,)
    else:
        modes = ("video",)
    streams = ("time", "video", "action") if is_action else ("time", "video")
    for mode in modes:
        for turn in range(math.ceil(window_count / world_size)):
            offset = turn * world_size + rank
            candidate = offset % window_count
            ds, index = indices[candidate]
            # 判断是frame的训练还是episode训练
            is_frame_dataset = isinstance(ds, LeRobotFrameDataset)
            if is_frame_dataset:
                sample = ds.make_sample(index, mode=mode) if is_action else ds.make_sample(index)
                window, caption, fps = sample["window"], sample["caption"], sample["source_fps"]
            else:
                window, caption, fps = ds.plan_read_window(index)
            identity = dict(
                source=ds.source,
                root=ds.root,
                episode=window.episode,
                candidate_index=index,
                caption=caption,
            )
            if is_frame_dataset:
                identity.update(
                    anchor=window.anchor,
                    past=window.past,
                    future=window.future,
                    max_blocks=window.max_blocks,
                    video_stride=window.video_stride,
                )
                if is_action:
                    identity.update(real_actions=sample["real_action_steps"], actions=len(sample["action"]))
                else:
                    identity.update(
                        real_intervals=int(sample["observation_valid_steps"].sum()) - 1,
                        intervals=len(sample["timestamps"]) - 1,
                    )
                history = sample["history_blocks"]
            else:
                identity["start"] = window.start
                if is_action:
                    # Preserve existing Action identities and their derived seeds.
                    identity.update(real_actions=window.real_actions, actions=window.actions)
                else:
                    identity.update(real_intervals=window.real_intervals, intervals=window.intervals)
                history_seed = validation_seed(seed, identity, "history")
                history = ds.select_history(ds.num_video_latents(window), rng=np.random.default_rng(history_seed))
                options = dict(mode=mode, history=history) if is_action else dict(history=history)
                sample = ds.__getitem__(index, **options)
            if is_action:
                geometry = asdict(ds.geometry)
            else:
                geometry = dict(
                    latent_block_size=ds.latent_block_size,
                    temporal_compression=ds.temporal_compression,
                    spatial_compression=ds.spatial_compression,
                    patch_size=ds.patch_size,
                )
            record = dict(
                identity=identity,
                candidate_index=index,
                mode=mode,
                history_blocks=history,
                seeds={stream: validation_seed(seed, {**identity, "mode": mode}, stream) for stream in streams},
                window=asdict(window),
                caption=caption,
                source_fps=fps,
                geometry=geometry,
                resolution=ds.resolution,
                viewpoint=ds.viewpoint,
            )
            if not is_frame_dataset:
                record["history_seed"] = history_seed
            samples = [sample] if sample else []
            yield dict(
                batch=collate_action_pack(samples), record=record, weight=int(offset < window_count and bool(samples))
            )
