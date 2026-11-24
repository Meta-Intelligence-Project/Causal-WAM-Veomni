"""On-demand LeRobot v3 I/O. No persistent file handles or decoded-table cache."""

import json
from dataclasses import dataclass, replace
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

    def __post_init__(self):
        if type(self.target_offset) is not int:
            raise ValueError("target_offset must be an integer")
        if self.target != self.state and self.target_offset != 0:
            raise ValueError("Nonzero target_offset requires target to be the state column")
        if self.target == self.state and self.target_mask != self.state_mask:
            raise ValueError("Actions derived from state must use state_mask as target_mask")


_DEFAULT_ACTION_COLUMNS = ActionColumns()


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
    """Shared frame-interval plan; legacy action count fields also serve video reads."""

    root: str
    episode: int
    start: int
    real_actions: int
    actions: int
    data_path: str
    row_start: int
    columns: ActionColumns | None
    fps: float
    video_stride: int
    cameras: tuple[Camera, ...]
    video_paths: tuple[str, ...]
    video_offsets: tuple[float, ...]
    height: int
    width: int
    action_width: int | None
    state_width: int | None

    @property
    def real_intervals(self):
        return self.real_actions

    @property
    def intervals(self):
        return self.actions

    @property
    def video_frames(self):
        return self.intervals // self.video_stride + 1


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


def read_info(root):
    """Read and validate the LeRobot v3 info once."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0: {root}")
    return info


def read_metadata(root, cameras=(), *, info=None):
    """Read numeric episode columns once, retaining no Arrow tables or file handles."""
    root = Path(root)
    if info is None:
        info = read_info(root)
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


def episode_frame_bounds(length, *, columns=None):
    """Return exclusive frame bounds, restricted by action targets when present."""
    if columns is None:
        return 0, length
    offset = columns.target_offset
    return max(0, -offset), min(length, length + 1 - offset)


def read_source_fps(root, info, episodes):
    """Read source rates without parsing subtask intervals or retaining JSON records."""
    fps = episodes["source_fps"].copy()
    rows = {int(ep): i for i, ep in enumerate(episodes["episode_index"])}
    path = Path(root) / "meta/episodes.jsonl"
    if path.exists():
        with path.open() as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    row = rows.get(record["episode_index"])
                    if row is not None and "source_fps" in record:
                        fps[row] = float(record["source_fps"])
    fps[np.isnan(fps)] = float(info["fps"])
    if (
        not np.isfinite(float(info["fps"]))
        or float(info["fps"]) <= 0
        or not np.isfinite(fps).all()
        or (fps <= 0).any()
    ):
        raise ValueError("Storage/source FPS must be positive and finite")
    return fps


def _storage_fields(root, ep, info, columns, cameras):
    """Physical paths/layout shared by episode windows and frame anchors."""
    root = Path(root).resolve()
    cameras = tuple(cameras)
    height = width = 0
    video_paths, offsets = [], []
    for camera in cameras:
        if min(camera.top, camera.left) < 0 or min(camera.height, camera.width) <= 0:
            raise ValueError("Invalid camera rectangle")
        if info["features"][camera.key]["dtype"] != "video":
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
    return dict(
        data_path=str(
            root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
        ),
        columns=columns,
        fps=float(info["fps"]),
        cameras=cameras,
        video_paths=tuple(video_paths),
        video_offsets=tuple(offsets),
        height=height,
        width=width,
    )


def plan_window(
    root,
    episode: int,
    start: int,
    max_actions: int,
    *,
    columns=_DEFAULT_ACTION_COLUMNS,
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
    Explicit columns=None reads video/task data only; omitted columns retain
    the default action schema. max_actions counts source-frame intervals.
    """
    root = Path(root).resolve()
    cameras = tuple(cameras)
    if min(max_actions, block_size, video_stride) <= 0 or block_size % video_stride:
        raise ValueError("Window/block/video stride must be positive and aligned")
    length = ep["dataset_to_index"] - ep["dataset_from_index"]
    first, stop = episode_frame_bounds(length, columns=columns)
    if not first <= start < stop - 1:
        raise IndexError(f"Episode {episode}: start {start} outside legal frame interval [{first}, {stop - 1})")
    real = min(max_actions, stop - start - 1)
    actions = ((real + block_size - 1) // block_size) * block_size if pad_to_block else real
    return Window(
        root=str(root),
        episode=episode,
        start=start,
        real_actions=real,
        actions=actions,
        row_start=int(ep["data/file_from_offset"]) + start,
        video_stride=video_stride,
        action_width=int(info["features"][columns.target]["shape"][0]) if columns is not None else None,
        state_width=int(info["features"][columns.state]["shape"][0]) if columns is not None else None,
        **_storage_fields(root, ep, info, columns, cameras),
    )


def read_window(window: Window, *, video=True, tolerance_s=1e-4, decode_threads=0):
    """Read one timeline and optional absolute targets; all modalities share tail padding."""
    c, n, total = window.columns, window.real_intervals, window.intervals
    start = window.row_start
    lo, hi = start, start + n + 1
    columns = ["timestamp", "task_index"]
    if c is not None:
        target_start = start + c.target_offset
        lo, hi = min(start, target_start), max(start + n + 1, target_start + n)
        columns = [c.state, c.target, c.state_mask, c.target_mask, *columns]
    values = read_rows(window.data_path, lo, hi, columns)
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
        timestamps=torch.from_numpy(timestamps),
        task_index=int(values["task_index"][start - lo]),
        window=window,
    )
    if c is not None:
        state = values[c.state][start - lo : start - lo + n + 1]
        target = values[c.target][target_start - lo : target_start - lo + n]
        if state.shape != (n + 1, window.state_width) or target.shape != (n, window.action_width):
            raise ValueError("State/action dimensions differ from info.json")
        result.update(
            state=padded(state, total + 1),
            action=padded(target, total),
            state_mask=torch.from_numpy(values[c.state_mask][start - lo].copy()).bool(),
            action_mask=torch.from_numpy(values[c.target_mask][start - lo].copy()).bool(),
            real_action_steps=n,
        )
    if video and window.cameras:
        queries = np.minimum(timestamps[:: window.video_stride], times[-1])
        result["video"] = decode_window_video(window, queries, tolerance_s=tolerance_s, decode_threads=decode_threads)
    return result


def decode_window_video(window, timestamps, *, tolerance_s=1e-4, decode_threads=0):
    """Decode explicit episode-local timestamps into the window's camera canvas."""
    queries = np.asarray(timestamps, dtype=np.float64)
    canvas = np.zeros((3, len(queries), window.height, window.width), dtype=np.uint8)
    for camera, path, offset in zip(window.cameras, window.video_paths, window.video_offsets, strict=True):
        region = canvas[:, :, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width]
        decode_into(path, queries + offset, region, tolerance_s=tolerance_s, threads=decode_threads)
    return torch.from_numpy(canvas)


@dataclass(frozen=True)
class FrameWindow:
    """An exact inclusive [anchor-past, anchor+future] request, before model padding."""

    episode: int
    anchor: int
    past: int
    future: int
    length: int
    data_path: str
    row_start: int  # File-local start of the episode, not the requested window.
    columns: ActionColumns | None
    fps: float
    video_stride: int
    cameras: tuple[Camera, ...]
    video_paths: tuple[str, ...]
    video_offsets: tuple[float, ...]
    height: int
    width: int


def plan_frame_window(root, episode, anchor, past, future, *, info, ep, columns=None, cameras=(), video_stride=1):
    """Plan without payload I/O, using a valid anchor and configuration from the dataset."""
    length = int(ep["dataset_to_index"] - ep["dataset_from_index"])
    return FrameWindow(
        episode=int(episode),
        anchor=anchor,
        past=past,
        future=future,
        length=length,
        row_start=int(ep["data/file_from_offset"]),
        video_stride=video_stride,
        **_storage_fields(root, ep, info, columns, cameras),
    )


def read_frame_window(
    window: FrameWindow, *, label_columns=(), video=True, tolerance_s=1e-4, decode_threads=0, block_size=None
):
    """Read one projected range; clamp placeholders within this episode and expose masks.

    Action j is the transition from observation j to j+1, read at j+target_offset.
    Its two observations and its target row must exist. Video offsets are multiples
    of video_stride relative to the anchor, so the anchor is always included.
    Labels are returned only at the anchor.

    For forward action training, block_size counts source-frame action intervals.
    Determine the real endpoint and final block length before reading payloads;
    repeat the final valid action/state and extrapolate padded timestamps once.
    """
    w, c = window, window.columns
    last_state = w.length - 1
    if block_size is not None:
        real = min(w.future, w.length - w.anchor - 1, w.length - w.anchor - c.target_offset)
        total = ((real + block_size - 1) // block_size) * block_size
        w = replace(w, future=total)
        last_state = w.anchor + real
    # 请求索引保留越界值；读取索引截到有效观测终点，直接补齐最终窗口。
    state_frame_indices = np.arange(w.anchor - w.past, w.anchor + w.future + 1, dtype=np.int64)
    state_read_indices = state_frame_indices.clip(0, last_state)
    state_valid_steps = (state_frame_indices >= 0) & (state_frame_indices <= last_state)

    # video 直接从 t-k 开始取帧；Dataset 保证 anchor 落在抽帧网格上。
    video_frame_indices = np.arange(w.anchor - w.past, w.anchor + w.future + 1, w.video_stride, dtype=np.int64)
    video_read_indices = video_frame_indices.clip(0, last_state)
    video_valid_steps = (video_frame_indices >= 0) & (video_frame_indices <= last_state)

    # 读取对应action
    read_start, read_stop = int(state_read_indices.min()), int(state_read_indices.max()) + 1
    columns = ["timestamp", *label_columns]
    if c is not None:
        # action 对应 [t-k, t+a)；只有来自 state 的动作目标才允许非零偏移。
        action_step_indices = np.arange(w.anchor - w.past, w.anchor + w.future, dtype=np.int64)
        action_frame_indices = action_step_indices + c.target_offset
        last_target = w.length - 1 if block_size is None else last_state - 1 + c.target_offset
        action_read_indices = action_frame_indices.clip(0, last_target)
        action_valid_steps = (
            (action_step_indices >= 0)
            & (action_step_indices + 1 <= last_state)
            & (action_frame_indices >= 0)
            & (action_frame_indices < w.length)
        )
        if len(action_read_indices):
            read_start = min(read_start, int(action_read_indices.min()))
            read_stop = max(read_stop, int(action_read_indices.max()) + 1)
        columns += [c.state, c.target, c.state_mask, c.target_mask]
    values = read_rows(w.data_path, w.row_start + read_start, w.row_start + read_stop, columns)
    times = values["timestamp"].astype(np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / w.fps, atol=1e-5, rtol=1e-4):
        raise ValueError("Non-contiguous source timestamps")
    if c is not None:
        # raw 按 episode 首尾补齐；block 读取按最后有效 action/观测补齐。
        # values 只读真实存在的行；*_read_indices 已指向对应端点，重复取数即可补齐。
        # 补齐位置的有效性 mask 仍为 False；减去 read_start 是把 episode 帧号换成 values 内的数组下标。
        state = values[c.state][state_read_indices - read_start]
        action = values[c.target][action_read_indices - read_start]
        masks = {}
        for name, column, read_indices in (
            ("state_mask", c.state_mask, state_read_indices),
            ("action_mask", c.target_mask, action_read_indices),
        ):
            selected = (
                values[column][read_indices - read_start]
                if len(read_indices)
                else values[column][w.anchor - read_start : w.anchor - read_start + 1]
            )
            if not np.isin(selected, (0, 1)).all() or not (selected == selected[0]).all():
                raise ValueError(f"{column}: expected a constant binary feature mask within the window")
            masks[name] = torch.from_numpy(selected[0].copy()).bool()
    timestamps = times[state_read_indices - read_start].copy()
    if block_size is not None:
        timestamps[real + 1 :] = timestamps[real] + np.arange(1, total - real + 1) / w.fps
    # 统一组织返回字段。
    result = dict(
        window=w,
        anchor_index=w.past,
        observation_valid_steps=torch.from_numpy(state_valid_steps),
        video_valid_steps=torch.from_numpy(video_valid_steps),
        timestamps=torch.from_numpy(timestamps),
        anchor_labels={name: values[name][w.anchor - read_start] for name in label_columns},
    )
    if c is not None:
        result.update(
            state=torch.from_numpy(state.astype(np.float32)),
            action=torch.from_numpy(action.astype(np.float32)),
            action_valid_steps=torch.from_numpy(action_valid_steps),
            **masks,
        )
        if block_size is not None:
            result["real_action_steps"] = real
    if video and w.cameras:
        result["video"] = decode_window_video(
            w, times[video_read_indices - read_start], tolerance_s=tolerance_s, decode_threads=decode_threads
        )
    return result
