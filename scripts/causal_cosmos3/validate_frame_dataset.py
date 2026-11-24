"""Check frame samples against source Parquet and independent PyAV decoding (CPU).

Run with PYTHONPATH=. python scripts/causal_cosmos3/validate_frame_dataset.py
--root /path/to/lerobot --source robotwin --viewpoint concat_view --report /tmp/report.json.
The source is read-only. No model, checkpoint, accelerator or generated statistics required.
"""

import argparse
import gc
import json
import os
import resource
import statistics
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from veomni.data.causal_cosmos3 import (
    ActionColumns,
    ActionFrameDataset,
    ActionProcessor,
    FrameCandidateConfig,
    FrameLabelConfig,
)


def reference_video(path, queries, height, width):
    """Decode independently, then select the nearest PTS for each requested position."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(max(0, int((min(queries) - 0.5) / float(stream.time_base))), stream=stream, backward=True)
        best = [None] * len(queries)
        distances = [(float("inf"), float("inf"))] * len(queries)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * stream.time_base)
            selected = [i for i, query in enumerate(queries) if (abs(pts - query), pts) < distances[i]]
            if selected:
                array = frame.reformat(
                    width=width, height=height, format="rgb24", interpolation="BICUBIC"
                ).to_ndarray()
                for i in selected:
                    best[i], distances[i] = array, (abs(pts - queries[i]), pts)
            if pts > max(queries) + 0.05:
                break
        if any(frame is None for frame in best):
            raise AssertionError(f"No reference frames: {path}")
        return np.stack(best, axis=0).transpose(3, 0, 1, 2)


class ProbeDataset(Dataset):
    def __init__(self, dataset, indexes):
        self.dataset, self.indexes = dataset, indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        before = len(os.listdir("/proc/self/fd"))
        start = time.perf_counter()
        sample = self.dataset[self.indexes[index]]
        sample["probe"] = dict(
            pid=os.getpid(),
            seconds=time.perf_counter() - start,
            fd_before=before,
            fd_after=len(os.listdir("/proc/self/fd")),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        source_handles = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                source_handles += os.readlink(f"/proc/self/fd/{fd}").startswith(self.dataset.root + "/")
            except FileNotFoundError:  # DataLoader IPC can close handles concurrently.
                pass
        sample["probe"]["source_handles_after"] = source_handles
        return sample


def passthrough(sample):
    return sample


def inspection_tokenize(text):
    """Explicit inspection IDs; not a model vocabulary. Must support spawn workers."""
    return [1, *text.encode(), 2]


def benchmark(dataset, indexes, workers):
    loader = DataLoader(
        ProbeDataset(dataset, indexes),
        batch_size=None,
        num_workers=workers,
        collate_fn=passthrough,
        multiprocessing_context="spawn" if workers else None,
    )
    start, probes, outputs = time.perf_counter(), [], 0
    for sample in loader:
        probes.append(sample["probe"])
        outputs += sum(
            value.numel() * value.element_size() for value in sample.values() if isinstance(value, torch.Tensor)
        )
    elapsed = time.perf_counter() - start
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe["pid"], []).append(probe)
        assert probe["source_handles_after"] == 0, probe
    return dict(
        workers=workers,
        samples=len(indexes),
        seconds=elapsed,
        samples_per_second_including_startup=len(indexes) / elapsed,
        median_read_seconds=statistics.median(probe["seconds"] for probe in probes),
        tensor_bytes=outputs,
        processes=[
            dict(
                pid=pid,
                peak_rss_mib=max(p["peak_rss_mib"] for p in rows),
                fd_before=rows[0]["fd_before"],
                fd_after=rows[-1]["fd_after"],
                per_read_fd_delta_max=max(p["fd_after"] - p["fd_before"] for p in rows),
                source_handles_after_max=max(p["source_handles_after"] for p in rows),
            )
            for pid, rows in grouped.items()
        ],
    )


def reference_relative_actions(state, target, state_mask, action_mask):
    """Independent NumPy oracle for scalar differences, absolute hands and xyzw rotations."""

    def unit(q):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        largest = np.take_along_axis(q, np.abs(q).argmax(axis=-1)[..., None], axis=-1)
        pivot = np.where(q[..., 3:4] != 0, q[..., 3:4], largest)
        return np.where(pivot < 0, -q, q)

    state = np.where(state_mask, state, 0)
    expected = target - state
    expected[:, 28:42] = target[:, 28:42]  # Grippers and hands are absolute targets.
    for group in (slice(17, 21), slice(24, 28)):
        if state_mask[group].all():
            state[group] = unit(state[group])
        if action_mask[group].all():
            a, b = state[group].copy(), unit(target[:, group])
            a[:3] *= -1  # Inverse of the current orientation.
            xyz = a[3] * b[:, :3] + b[:, 3:4] * a[:3] + np.cross(a[:3], b[:, :3])
            w = a[3] * b[:, 3:4] - (a[:3] * b[:, :3]).sum(axis=-1, keepdims=True)
            expected[:, group] = unit(np.concatenate((xyz, w), axis=-1))
    return np.where(action_mask, expected, 0), state


def validate(args):
    root = Path(args.root).resolve()
    columns = ActionColumns(**args.columns)
    common = dict(
        source=args.source,
        viewpoint=args.viewpoint,
        columns=columns,
        candidates=FrameCandidateConfig(future_frames=args.future_frames, anchor_stride=args.anchor_stride),
        labels=FrameLabelConfig(**args.labels),
        decode_threads=1,
        tokenize=inspection_tokenize,
    )
    start = time.perf_counter()
    ds = ActionFrameDataset(root, processor=ActionProcessor(None), video=True, **common)
    report = dict(
        root=str(root),
        source=args.source,
        candidates=len(ds),
        episodes=len(ds.episode_rows),
        init_seconds=time.perf_counter() - start,
        index_bytes=ds.cumulative.nbytes + ds.episode_rows.nbytes + ds.anchor_starts.nbytes,
        settings=dict(
            past_frames=0, future_frames=args.future_frames, anchor_stride=args.anchor_stride, columns=vars(columns)
        ),
        checks=[],
    )
    slots = sorted({0, len(ds.episode_rows) // 2, len(ds.episode_rows) - 1})
    indexes = []
    maximum_error = 0.0
    for slot in slots:
        first, stop = int(ds.cumulative[slot]), int(ds.cumulative[slot + 1])
        for index in sorted({first, (first + stop - 1) // 2, stop - 1}):
            sample = ds[index]
            plan = sample["window"]
            names = list(
                dict.fromkeys(["timestamp", columns.state, columns.target, columns.state_mask, columns.target_mask])
            )
            # Independent source projection; use physical episode offsets, not Dataset payloads.
            table = pq.read_table(plan.data_path, columns=names).slice(plan.row_start, plan.length)
            source_states = np.asarray(table[columns.state].to_pylist(), dtype=np.float64)
            source_targets = np.asarray(table[columns.target].to_pylist(), dtype=np.float64)
            t, block = plan.anchor, ds.geometry.actions_per_block
            legal_stop = min(plan.length, plan.length + 1 - columns.target_offset)
            real = min(args.future_frames, legal_stop - t - 1)
            total = ((real + block - 1) // block) * block
            state_rows = np.minimum(t + np.arange(0, total, block), t + real)
            target_rows = t + np.minimum(np.arange(total), real - 1) + columns.target_offset
            sm = np.asarray(table[columns.state_mask][t].as_py(), dtype=bool)
            am = np.asarray(table[columns.target_mask][int(target_rows[0])].as_py(), dtype=bool)
            encoded, anchors = [], []
            for position, row in enumerate(state_rows):
                expected, anchor = reference_relative_actions(
                    source_states[row].copy(),
                    source_targets[target_rows[position * block : (position + 1) * block]].copy(),
                    sm,
                    am,
                )
                encoded.append(expected)
                anchors.append(anchor)
            expected = np.concatenate(encoded)
            np.testing.assert_allclose(sample["action"], expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(sample["anchors"], np.stack(anchors), atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(sample["action_valid_steps"], np.arange(total) < real)
            assert len(sample["states"]) == total // block
            maximum_error = max(maximum_error, float(np.abs(sample["action"].numpy() - expected).max()))
            source_times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            queries = source_times[t + np.minimum(np.arange(0, total + 1, ds.geometry.video_stride), real)]
            for camera, video_path, offset in zip(plan.cameras, plan.video_paths, plan.video_offsets, strict=True):
                expected_video = reference_video(video_path, queries + offset, camera.height, camera.width)
                actual_video = sample["video"][
                    :, :, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width
                ]
                np.testing.assert_array_equal(actual_video, expected_video)
            report["checks"].append(
                dict(
                    episode=plan.episode,
                    anchor=t,
                    real_actions=real,
                    padded_actions=total,
                    blocks=len(anchors),
                    video_frames=len(queries),
                    caption=sample["caption"],
                    exact_pixel_match=True,
                )
            )
            indexes.append(index)
    statistics_path = Path(args.statistics) if args.statistics else root / "meta/causal_action_stats.json"
    report["statistics"] = dict(path=str(statistics_path), available=statistics_path.exists())
    if statistics_path.exists():
        measured = ActionFrameDataset(root, processor=ActionProcessor(statistics_path), video=False, **common)
        for index in indexes:
            assert torch.isfinite(measured[index]["action"]).all()
        report["statistics"]["compatible"] = True
    report["processing"] = dict(
        reference="independent Parquet block states and NumPy relative actions", maximum_absolute_error=maximum_error
    )
    gc.collect()
    benchmark_indexes = [indexes[i % len(indexes)] for i in range(args.benchmark_samples)]
    report["benchmarks"] = [benchmark(ds, benchmark_indexes, workers) for workers in dict.fromkeys((0, args.workers))]
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--viewpoint", choices=("ego_view", "concat_view"), required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--benchmark-samples", type=int, default=18)
    parser.add_argument("--future-frames", type=int, default=128)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--columns", type=json.loads, default={}, help="ActionColumns overrides as a JSON object")
    parser.add_argument("--labels", type=json.loads, default={}, help="FrameLabelConfig overrides as a JSON object")
    parser.add_argument("--statistics", help="Optional measured block statistics path")
    args = parser.parse_args()
    if args.workers < 0 or args.benchmark_samples < 1:
        parser.error("workers must be nonnegative and benchmark-samples must be positive")
    torch.set_num_threads(1)
    report = validate(args)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
