# SPDX-License-Identifier: OpenMDW-1.1
"""Shared history sampling geometry; padding only maps to the episode's first image."""

from dataclasses import dataclass
from numbers import Integral


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class HistoryWindow:
    """Logical sample times and the real observation supplying each image.

    Ticks are relative to an actual episode reset, not an arbitrary training
    segment. Negative logical ticks may use tick zero; missing nonnegative
    observations must never be replaced with a different frame.
    """

    history_blocks: int
    logical_ticks: tuple[int, ...]
    observation_ticks: tuple[int, ...]

    @property
    def window_start_tick(self):
        return self.logical_ticks[0]

    @property
    def required_observation_ticks(self):
        return tuple(dict.fromkeys(self.observation_ticks))

    @property
    def history_steps(self):
        """历史覆盖的 action 时间间隔，独立于预测 horizon。"""
        return self.logical_ticks[-1] - self.logical_ticks[0]


def build_history_window(
    tick,
    *,
    history_blocks=None,
    history_steps=None,
    actions_per_block=32,
    video_stride=4,
    history_padding="available",
):
    """Sample a current-anchored window, optionally repeating the true reset image.

    ``available`` retains the original complete-block warmup. ``repeat_first``
    always returns the configured number of history blocks, including at tick 0.
    The caller must resolve ``observation_ticks`` against real measured images.
    """
    tick = _integer(tick, "tick")
    actions_per_block = _integer(actions_per_block, "actions_per_block", 1)
    video_stride = _integer(video_stride, "video_stride", 1)
    if actions_per_block % video_stride:
        raise ValueError("actions_per_block must be divisible by video_stride")
    if history_padding not in ("available", "repeat_first"):
        raise ValueError("history_padding must be available or repeat_first")
    if history_steps is not None:
        # P按真实时间步采样；P=48、H=32同样合法，不再按H取整。
        length = _integer(history_steps, "history_steps")
        if length % video_stride:
            raise ValueError("history_steps must be divisible by video_stride")
        if history_padding != "repeat_first":
            raise ValueError("Explicit history_steps requires repeat_first padding")
        count = 0  # 仅旧接口使用history_blocks；新窗口以history_steps为准。
    else:
        history_blocks = _integer(history_blocks, "history_blocks", 1)
        count = min(tick // actions_per_block, history_blocks) if history_padding == "available" else history_blocks
        length = count * actions_per_block
    logical_ticks = tuple(range(tick - length, tick + 1, video_stride))
    return HistoryWindow(count, logical_ticks, tuple(max(0, t) for t in logical_ticks))
