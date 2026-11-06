# SPDX-License-Identifier: OpenMDW-1.1
"""Causal online observations with persistent, bounded episode VAE state."""

import math

import torch

from ..cosmos3_core.episode_vae import BoundedEpisodeVAE
from .history_window import build_history_window


class EpisodeObservationEncoder:
    """Encode each new observation group once on a fixed stream time axis.

    A text/model-context reset must call ``clear`` and provide the measured raw
    history again. Anchor changes alone retain convolution history. Complete
    latent groups define the stream phase; an incompatible anchor fails before
    modifying the current stream instead of silently shifting its time axis.
    """

    def __init__(
        self,
        tokenizer,
        *,
        history_steps=None,
        history_blocks=None,
        action_horizon=32,
        history_padding="repeat_first",
        actions_per_block=32,
        video_stride=4,
        reuse_reset_prefix=True,
        max_latents=None,
    ):
        for name, value, minimum in (
            ("video_stride", video_stride, 1),
            ("action_horizon", action_horizon, 1),
            ("actions_per_block", actions_per_block, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a positive integer")
        if history_padding != "repeat_first":
            raise ValueError("Episode streaming requires repeat_first history padding")
        if history_steps is None:
            if type(history_blocks) is not int or history_blocks < 1:
                raise ValueError("Episode streaming requires history_steps or positive history_blocks")
            history_steps = history_blocks * actions_per_block
        if type(history_steps) is not int or history_steps < 0:
            raise ValueError("history_steps must be a nonnegative integer")
        self.tokenizer = tokenizer
        self.group = int(tokenizer.temporal_compression_factor)
        self.video_stride, self.action_horizon = video_stride, action_horizon
        self.actions_per_block, self.history_blocks = actions_per_block, history_blocks
        self.history_padding, self.history_steps = history_padding, history_steps
        self.latent_ticks = self.group * self.video_stride
        if self.group < 1 or history_steps % self.latent_ticks:
            raise ValueError("Episode history_steps must contain complete VAE temporal groups")
        self.history_latents = 1 + history_steps // self.latent_ticks
        self.reuse_reset_prefix = reuse_reset_prefix
        capacity = (
            max_latents
            if max_latents is not None
            else self.history_latents + math.ceil(action_horizon / self.latent_ticks)
        )
        if type(capacity) is not int or capacity < self.history_latents:
            raise ValueError("max_latents must hold the configured history")
        self.episode = BoundedEpisodeVAE(tokenizer, max_latents=capacity)
        self.clear()

    @property
    def state(self):
        return self.episode.state

    @property
    def root_origin(self):
        return self.stream_origin_tick

    @property
    def latent_count(self):
        return self.episode.latent_count

    @property
    def count(self):
        if self.window is None or self.encoded_tick is None:
            return 0
        return max(
            0,
            min(
                len(self.window.logical_ticks),
                1 + (self.encoded_tick - self.window.window_start_tick) // self.video_stride,
            ),
        )

    def clear(self):
        self.episode.clear()
        self.frames = {}
        self.first_image = self.image_shape = None
        self.latest_tick = -1
        self.anchor = self.window = self.stream_origin_tick = self.encoded_tick = None
        self.last_finished_id = None
        self.last_new_latents = None
        self.new_frame_ids = self.prefix_global_ids = ()
        self.reanchors = 0
        self.reused_reset_latents = 0

    def _image(self, image):
        image = torch.as_tensor(image).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError("Observed images must be CPU uint8 [3,H,W]")
        if self.image_shape is not None and tuple(image.shape) != self.image_shape:
            raise ValueError("Observed image geometry changed; reset is required")
        return image.clone()

    def add(self, tick, image, *, prune=True):
        if type(tick) is not int or tick < 0:
            raise ValueError("Observation tick must be a nonnegative integer")
        image = self._image(image)
        previous = self.first_image if tick == 0 else self.frames.get(tick)
        if previous is not None:
            if not torch.equal(previous, image):
                raise ValueError("A measured image cannot be overwritten at an existing tick")
            self._feed_available()
            return
        if tick < self.latest_tick:
            raise ValueError("New episode observations must arrive in nondecreasing tick order")
        self.image_shape = tuple(image.shape)
        if tick == 0:
            self.first_image = image
        self.frames[tick] = image
        self.latest_tick = tick
        if prune:
            self.prune()
        self._feed_available()

    def prune(self):
        oldest = self.latest_tick - self.history_steps - self.action_horizon
        self.frames = {tick: image for tick, image in self.frames.items() if tick >= oldest}

    def _frame(self, tick):
        tick = max(0, tick)
        return self.first_image if tick == 0 else self.frames.get(tick)

    def _normalized(self, frames):
        mean = getattr(self.tokenizer, "latent_mean", None)
        device = mean.device if mean is not None else torch.device("cpu")
        if device.type != "cpu":
            getattr(torch, device.type).set_device(device)
        return torch.stack(frames, dim=1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0

    def prepare(self, anchor):
        if type(anchor) is not int or anchor < 0:
            raise ValueError("Window anchor must be a nonnegative integer")
        if self.stream_origin_tick is not None and (anchor - self.stream_origin_tick) % self.latent_ticks:
            raise ValueError("Anchor changes the episode VAE phase; reset the stream before using this anchor")
        if self.anchor != anchor:
            self.reanchors += int(self.anchor is not None)
        self.anchor = anchor
        self.window = build_history_window(
            anchor,
            history_steps=self.history_steps,
            history_padding=self.history_padding,
            actions_per_block=self.actions_per_block,
            video_stride=self.video_stride,
        )
        if self.stream_origin_tick is None:
            self.stream_origin_tick = self.window.window_start_tick
        self._feed_available()

    @torch.no_grad()
    def _feed_available(self):
        if self.window is None:
            return
        try:
            if not self.episode.active:
                frame = self._frame(self.stream_origin_tick)
                if frame is None:
                    return
                self.episode.reset(self._normalized([frame]))
                self.encoded_tick = self.stream_origin_tick
            while self.encoded_tick + self.latent_ticks <= self.anchor:
                ticks = range(
                    self.encoded_tick + self.video_stride,
                    self.encoded_tick + self.latent_ticks + 1,
                    self.video_stride,
                )
                frames = [self._frame(tick) for tick in ticks]
                if any(frame is None for frame in frames):
                    break
                self.episode.append(self._normalized(frames))
                self.encoded_tick += self.latent_ticks
        except BaseException:
            self.clear()
            raise

    def _latents_for(self, ids):
        ring_ids = self.episode.latent_ids
        if not ids:
            return self.episode.latents[:, :, :0]
        if ids[0] < ring_ids[0] or ids[-1] > ring_ids[-1]:
            raise ValueError("Requested anchor is outside the retained episode latent history; reset is required")
        return self.episode.latents[:, :, ids[0] - ring_ids[0] : ids[-1] - ring_ids[0] + 1]

    @torch.no_grad()
    def finish(self, anchor):
        prepared = self.count if self.anchor == anchor else 0
        cold = self.last_finished_id is None
        self.prepare(anchor)
        if self.encoded_tick is None or self.encoded_tick < anchor:
            required = (
                (self.stream_origin_tick,)
                if self.encoded_tick is None
                else range(
                    self.encoded_tick + self.video_stride,
                    self.encoded_tick + self.latent_ticks + 1,
                    self.video_stride,
                )
            )
            missing = list(dict.fromkeys(max(0, tick) for tick in required if self._frame(tick) is None))
            raise ValueError(f"missing real historical video observations at ticks {missing}")
        target = (anchor - self.stream_origin_tick) // self.latent_ticks
        ids = tuple(range(target - self.history_latents + 1, target + 1))
        history = self._latents_for(ids).squeeze(0)
        new_start = ids[0] if cold else self.last_finished_id + 1
        new_ids = tuple(range(new_start, target + 1)) if new_start <= target else ()
        self.last_new_latents = self._latents_for(new_ids).squeeze(0).detach().clone()
        self.new_frame_ids, self.prefix_global_ids = new_ids, ids
        self.last_finished_id = target if cold else max(self.last_finished_id, target)
        return history, dict(
            sampled_frames_encoded_before_predict=prepared,
            streaming_window_rebuilt=cold,
            streaming_reanchors=self.reanchors,
            reused_reset_latents=0,
            history_encoding="episode_stream",
            history_origin_tick=self.stream_origin_tick,
            new_history_latents=len(new_ids),
        )
