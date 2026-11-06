"""Read one contiguous episode span for multiple independently anchored decisions."""

import numpy as np
import torch

from .lerobot import read_rows
from .video import decode_into


def read_episode_rows(window, *, history_steps=0):
    """Read the union once, retaining each tick's masks and task identity.

    ``window`` may include one decision of cheap row lookahead. Historical
    timestamps are checked in the same read; only the requested future states
    and targets are retained. No Arrow buffers or persistent handles survive.
    """
    c = window.columns
    if c is None:
        raise ValueError("Episode decision segments require action columns")
    n, total, start = window.real_intervals, window.intervals, window.row_start
    first = max(0, window.start - history_steps)
    target_start = start + c.target_offset
    lo = min(start - window.start + first, target_start)
    hi = max(start + n + 1, target_start + n)
    values = read_rows(
        window.data_path, lo, hi, ["timestamp", "task_index", c.state, c.target, c.state_mask, c.target_mask]
    )
    all_times = values["timestamp"].astype(np.float64)
    if not np.allclose(np.diff(all_times), 1 / window.fps, atol=1e-5, rtol=1e-4):
        raise ValueError("Non-contiguous episode segment timestamps")

    def pad(array, length):
        result = np.empty((length, *array.shape[1:]), dtype=array.dtype)
        result[: len(array)] = array
        result[len(array) :] = array[-1]
        return torch.from_numpy(result)

    state_rows = slice(start - lo, start - lo + n + 1)
    target_rows = slice(target_start - lo, target_start - lo + n)
    states, actions = values[c.state][state_rows], values[c.target][target_rows]
    if states.shape != (n + 1, window.state_width) or actions.shape != (n, window.action_width):
        raise ValueError("State/action dimensions differ from info.json")
    times = np.empty(total + 1, dtype=np.float64)
    times[: n + 1] = all_times[state_rows]
    times[n + 1 :] = times[n] + np.arange(1, total - n + 1) / window.fps
    return dict(
        window=window,
        timestamps=torch.from_numpy(times),
        state=pad(states.astype(np.float32), total + 1),
        action=pad(actions.astype(np.float32), total),
        state_masks=pad(values[c.state_mask][state_rows], total + 1),
        action_masks=pad(values[c.target_mask][state_rows], total + 1),
        task_indexes=pad(values["task_index"][state_rows], total + 1),
    )


def read_episode_video(window, *, history_steps, reset, anchor_timestamp, tolerance_s=1e-4, decode_threads=0):
    """Decode one actual segment span, including its boundary observation.

    A reset includes the fixed history prefix. Continuations contain their
    already-encoded first RGB boundary, which the streaming condition removes.
    Negative history repeats episode reset; a final padded tail repeats the
    final real observation and cannot become a later segment's history.
    """
    past = history_steps if reset else 0
    if past < 0 or past % window.video_stride:
        raise ValueError("Episode history must be nonnegative and video-stride aligned")
    ticks = window.start + np.arange(-past, window.intervals + 1, window.video_stride)
    ticks = np.clip(ticks, 0, window.start + window.real_intervals)
    canvas = np.zeros((3, len(ticks), window.height, window.width), dtype=np.uint8)
    queries = float(anchor_timestamp) + (ticks - window.start) / window.fps
    for camera, path, offset in zip(window.cameras, window.video_paths, window.video_offsets, strict=True):
        region = canvas[:, :, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width]
        decode_into(path, queries + offset, region, tolerance_s=tolerance_s, threads=decode_threads)
    return dict(video=torch.from_numpy(canvas), video_frame_ticks=ticks.tolist())
