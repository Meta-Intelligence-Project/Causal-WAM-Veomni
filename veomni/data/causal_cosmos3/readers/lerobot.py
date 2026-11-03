"""On-demand LeRobot v3 I/O. No persistent file handles or decoded-table cache."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .video import decode_into


@dataclass(frozen=True)
class ActionColumns:
    state: str = "state_unified"
    target: str = "action_unified"
    state_mask: str = "mask_state"
    target_mask: str = "mask_action"
    target_offset: int = 0


@dataclass(frozen=True)
class Camera:
    """Camera field and its output rectangle, in pixels (top, left, height, width)."""

    key: str
    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class Window:
    root: str
    episode: int
    start: int
    real_actions: int
    actions: int
    data_path: str
    row_start: int
    columns: ActionColumns
    fps: float
    video_stride: int
    cameras: tuple[Camera, ...]
    video_paths: tuple[str, ...]
    video_offsets: tuple[float, ...]
    height: int
    width: int
    action_width: int
    state_width: int

    @property
    def video_frames(self):
        return self.actions // self.video_stride + 1


def _numpy(column):
    """Copy only a requested window; outputs must not retain row-group buffers."""
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError("Null values in a required LeRobot column")
    if pa.types.is_fixed_size_list(array.type):
        values = array.flatten()
        if values.null_count:
            raise ValueError("Null feature values")
        return values.to_numpy(zero_copy_only=False).reshape(len(array), array.type.list_size).copy()
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy()
        widths = np.diff(offsets)
        if not len(widths) or np.any(widths != widths[0]):
            raise ValueError("Ragged action/state features are unsupported")
        values = array.flatten()
        if values.null_count:
            raise ValueError("Null feature values")
        return values.to_numpy(zero_copy_only=False).reshape(len(array), int(widths[0])).copy()
    return array.to_numpy(zero_copy_only=False).copy()


def read_rows(path, start, stop, columns):
    """Read file-local [start, stop); project intersecting row groups and release them."""
    parts = {c: [] for c in dict.fromkeys(columns)}
    with pq.ParquetFile(path) as parquet:
        if not 0 <= start < stop <= parquet.metadata.num_rows:
            raise IndexError(f"{path}: invalid row interval [{start}, {stop})")
        offset = 0
        for i in range(parquet.num_row_groups):
            end = offset + parquet.metadata.row_group(i).num_rows
            lo, hi = max(start, offset), min(stop, end)
            if lo < hi:
                table = parquet.read_row_group(i, columns=list(parts), use_threads=False).slice(lo - offset, hi - lo)
                for column in parts:
                    parts[column].append(_numpy(table[column]))
                del table
            offset = end
            if offset >= stop:
                break
    return {c: values[0] if len(values) == 1 else np.concatenate(values) for c, values in parts.items()}


def read_metadata(root, cameras=()):
    """Read numeric episode columns once, retaining no Arrow tables or file handles."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0: {root}")
    columns = ["episode_index", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
    for camera in cameras:
        key = camera.key if isinstance(camera, Camera) else camera
        columns += [f"videos/{key}/{field}" for field in ("chunk_index", "file_index", "from_timestamp")]
    columns = list(dict.fromkeys(columns))
    parts = {c: [] for c in columns + ["data/file_from_offset", "source_fps"]}
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        with pq.ParquetFile(path) as parquet:
            requested = columns + [
                c for c in ("data/file_from_offset", "source_fps") if c in parquet.schema_arrow.names
            ]
            for batch in parquet.iter_batches(columns=requested, batch_size=4096, use_threads=False):
                table = pa.Table.from_batches([batch])
                for column in parts:
                    if column not in requested:
                        values = np.full(len(batch), np.nan)
                    elif column in ("source_fps", "data/file_from_offset"):
                        values = np.asarray(table[column].to_numpy(), dtype=np.float64)
                    else:
                        values = _numpy(table[column])
                    parts[column].append(values)
    if not parts["episode_index"]:
        raise ValueError(f"No episodes in {root}")
    episodes = {c: np.concatenate(v) for c, v in parts.items()}
    order = np.argsort(episodes["episode_index"])
    episodes = {c: v[order] for c, v in episodes.items()}
    if len(np.unique(episodes["episode_index"])) != len(order):
        raise ValueError(f"Duplicate episode IDs in {root}")
    file_starts = {}
    for chunk, file, start in zip(
        episodes["data/chunk_index"], episodes["data/file_index"], episodes["dataset_from_index"]
    ):
        key = (int(chunk), int(file))
        file_starts[key] = min(file_starts.get(key, int(start)), int(start))
    for i in range(len(order)):
        if np.isnan(episodes["data/file_from_offset"][i]):
            key = (int(episodes["data/chunk_index"][i]), int(episodes["data/file_index"][i]))
            episodes["data/file_from_offset"][i] = episodes["dataset_from_index"][i] - file_starts[key]
    episodes["data/file_from_offset"] = episodes["data/file_from_offset"].astype(np.int64)
    return info, episodes


def plan_window(
    root,
    episode: int,
    start: int,
    max_actions: int,
    *,
    columns=None,
    cameras=(),
    block_size=32,
    video_stride=4,
    pad_to_block=True,
    info,
    ep,
):
    """Describe an episode-local window before opening any data/video file.

    Short windows repeat the tail to the nearest block. The sampler owns
    which starts are legal candidates (and subtask captions); this reader
    never resamples a failed request or silently changes its start.
    """
    root = Path(root).resolve()
    cameras = tuple(cameras)
    columns = columns or ActionColumns()
    if min(max_actions, block_size, video_stride) <= 0 or block_size % video_stride:
        raise ValueError("Window/block/video stride must be positive and aligned")
    length = ep["dataset_to_index"] - ep["dataset_from_index"]
    first, stop = max(0, -columns.target_offset), min(length, length + 1 - columns.target_offset)
    if not first <= start < stop - 1:
        raise IndexError(f"Episode {episode}: start {start} outside legal action interval [{first}, {stop - 1})")
    real = min(max_actions, stop - start - 1)
    actions = ((real + block_size - 1) // block_size) * block_size if pad_to_block else real
    height = width = 0
    video_paths, offsets = [], []
    for camera in cameras:
        if min(camera.top, camera.left) < 0 or min(camera.height, camera.width) <= 0:
            raise ValueError("Invalid camera rectangle")
        feature = info["features"][camera.key]
        if feature["dtype"] != "video":
            raise ValueError(f"{camera.key} is not a video feature")
        height, width = max(height, camera.top + camera.height), max(width, camera.left + camera.width)
        prefix = f"videos/{camera.key}"
        video_paths.append(
            str(
                root
                / info["video_path"].format(
                    video_key=camera.key,
                    chunk_index=ep[prefix + "/chunk_index"],
                    file_index=ep[prefix + "/file_index"],
                )
            )
        )
        offsets.append(float(ep[prefix + "/from_timestamp"]))
    action_width = int(info["features"][columns.target]["shape"][0])
    state_width = int(info["features"][columns.state]["shape"][0])
    return Window(
        str(root),
        episode,
        start,
        real,
        actions,
        str(root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])),
        int(ep["data/file_from_offset"]) + start,
        columns,
        float(info["fps"]),
        video_stride,
        cameras,
        tuple(video_paths),
        tuple(offsets),
        height,
        width,
        action_width,
        state_width,
    )


def read_window(window: Window, *, video=True, tolerance_s=1e-4, decode_threads=0):
    """Materialize raw absolute targets; source time offset is applied once."""
    c, n, total = window.columns, window.real_actions, window.actions
    start, target_start = window.row_start, window.row_start + c.target_offset
    lo, hi = min(start, target_start), max(start + n + 1, target_start + n)
    values = read_rows(
        window.data_path, lo, hi, [c.state, c.target, c.state_mask, c.target_mask, "timestamp", "task_index"]
    )
    state = values[c.state][start - lo : start - lo + n + 1]
    target = values[c.target][target_start - lo : target_start - lo + n]
    if state.shape != (n + 1, window.state_width) or target.shape != (n, window.action_width):
        raise ValueError("State/action dimensions differ from info.json")
    times = values["timestamp"][start - lo : start - lo + n + 1].astype(np.float64)
    if not np.allclose(np.diff(times), 1 / window.fps, atol=1e-5, rtol=1e-4):
        raise ValueError("Non-contiguous source timestamps")

    def padded(array, length):
        out = np.empty((length, array.shape[1]), dtype=np.float32)
        out[: len(array)] = array
        out[len(array) :] = array[-1]
        return torch.from_numpy(out)

    timestamps = np.empty(total + 1, dtype=np.float64)
    timestamps[: n + 1] = times
    timestamps[n + 1 :] = times[-1] + np.arange(1, total - n + 1) / window.fps
    result = dict(
        state=padded(state, total + 1),
        action=padded(target, total),
        state_mask=torch.from_numpy(values[c.state_mask][start - lo].copy()).bool(),
        action_mask=torch.from_numpy(values[c.target_mask][start - lo].copy()).bool(),
        timestamps=torch.from_numpy(timestamps),
        real_action_steps=n,
        task_index=int(values["task_index"][start - lo]),
        window=window,
    )
    if video and window.cameras:
        canvas = np.zeros((3, window.video_frames, window.height, window.width), dtype=np.uint8)
        queries = np.minimum(timestamps[:: window.video_stride], times[-1])
        for camera, path, offset in zip(window.cameras, window.video_paths, window.video_offsets, strict=True):
            region = canvas[:, :, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width]
            decode_into(path, queries + offset, region, tolerance_s=tolerance_s, threads=decode_threads)
        result["video"] = torch.from_numpy(canvas)
    return result
