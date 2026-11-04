# SPDX-License-Identifier: OpenMDW-1.1
"""Native measured decision windows, without historical action/state placeholders."""

import math
from dataclasses import dataclass
from numbers import Integral

import torch

from .history_window import build_history_window


@dataclass(frozen=True)
class RTCObservation:
    tick: int
    image: torch.Tensor


@dataclass
class DecisionWindowMetadata:
    block_size: int
    action_frame_ids: torch.Tensor
    action_mask: torch.Tensor
    anchors: torch.Tensor
    state_mask: torch.Tensor
    template: object
    statistics: object
    observed_latents: int
    layout_mode: str = "decision_window"
    current_state_index: int = 0

    def action_indexes_for_group(self, group):
        if group != self.observed_latents:
            raise ValueError("Requested group is not this decision's current horizon")
        return torch.arange(len(self.action_frame_ids), dtype=torch.long)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class RTCObservedBatchBuilder:
    """Use actual observation ticks, one fresh anchor and one future action horizon."""

    def __init__(
        self,
        *,
        template,
        statistics,
        tokenize,
        prompt,
        image_shape,
        controller_hz=30.0,
        history_steps=1024,
        action_horizon=32,
        video_stride=4,
        temporal_compression=4,
        vae=None,
    ):
        self.template, self.statistics, self.tokenize = template, statistics, tokenize
        self.vae = vae
        self.layout_mode = "decision_window"
        self.history_steps = _integer(history_steps, "history_steps")
        self.action_horizon = _integer(action_horizon, "action_horizon", 1)
        self.video_stride = _integer(video_stride, "video_stride", 1)
        self.temporal_compression = _integer(temporal_compression, "temporal_compression", 1)
        if (self.video_stride, self.temporal_compression) != (4, 4):
            raise ValueError("Online windows currently use the trained stride4/VAE4 geometry")
        self.actions_per_latent = video_stride * temporal_compression
        if self.history_steps % self.actions_per_latent or self.action_horizon % self.actions_per_latent:
            raise ValueError("History and horizon must contain complete VAE temporal groups")
        self.image_shape, self.controller_hz = tuple(image_shape), float(controller_hz)
        if isinstance(controller_hz, bool) or not math.isfinite(self.controller_hz) or self.controller_hz <= 0:
            raise ValueError("controller_hz must be finite and positive")
        if len(self.image_shape) != 3 or self.image_shape[0] != 3 or any(size < 1 for size in self.image_shape):
            raise ValueError("image_shape must be RGB [3,H,W]")
        self.text_ids = torch.as_tensor(tokenize(prompt), dtype=torch.long)
        if self.text_ids.ndim != 1 or not len(self.text_ids):
            raise ValueError("The prompt must tokenize to a nonempty vector")

    def _image(self, value):
        value = torch.as_tensor(value).detach().cpu()
        if value.dtype != torch.uint8 or tuple(value.shape) != self.image_shape:
            raise ValueError("Observed images must preserve the prepared uint8 CHW geometry")
        return value

    def __call__(
        self,
        *,
        state,
        state_mask,
        action_mask,
        image,
        tick,
        history=(),
        first_image=None,
        revision=0,
        prepared_vision_prefix=None,
    ):
        tick, revision = _integer(tick, "tick"), _integer(revision, "revision")
        sm = self.template.validate_valid_mask(state_mask).detach().cpu().clone()
        am = self.template.validate_valid_mask(action_mask).detach().cpu().clone()
        self.statistics.validate(self.template.width, sm, am)
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Current actions require a valid measured state anchor")
        state = torch.as_tensor(state).detach().float().cpu()
        if state.shape != (self.template.width,):
            raise ValueError("State must be one fresh physical template-width measurement")
        state = self.template.sanitize(state, sm)
        observed, previous = {}, -1
        for record in history:
            index = _integer(record.tick, "history tick")
            if not previous < index < tick:
                raise ValueError("Historical ticks must be unique, ordered and before the current tick")
            observed[index], previous = self._image(record.image), index
        observed[tick] = self._image(image)
        if first_image is not None:
            first_image = self._image(first_image)
            if 0 in observed and not torch.equal(observed[0], first_image):
                raise ValueError("The reset image is immutable")
            observed[0] = first_image
        window = build_history_window(
            tick,
            history_steps=self.history_steps,
            actions_per_block=self.action_horizon,
            video_stride=self.video_stride,
            history_padding="repeat_first",
        )
        missing = [index for index in window.required_observation_ticks if index not in observed]
        if missing:
            raise ValueError(f"missing real historical video observations at ticks {missing}")
        prefix_count = 1 + self.history_steps // self.actions_per_latent
        if prepared_vision_prefix is None:
            if self.vae is None:
                raise ValueError("Online decisions require a streaming VAE prefix from the measured window")
            device = self.vae.latent_mean.device

            def pixels(ticks):
                return (
                    torch.stack([self._image(observed[index]) for index in ticks], dim=1)
                    .unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                    / 127.5
                    - 1
                )

            stream = self.vae.stream_encode_start(pixels(window.observation_ticks[:1]))
            try:
                for start in range(1, len(window.observation_ticks), self.temporal_compression):
                    self.vae.stream_encode_step(
                        stream, pixels(window.observation_ticks[start : start + self.temporal_compression])
                    )
                prepared_vision_prefix = self.vae.stream_encode_finish(stream).squeeze(0)
            finally:
                self.vae.stream_encode_abort(stream)
        prefix = prepared_vision_prefix.detach()
        if prefix.ndim != 4 or prefix.shape[1] != prefix_count or not torch.isfinite(prefix).all():
            raise ValueError("Streaming prefix does not match the measured history window")
        future = self.action_horizon // self.actions_per_latent
        video = torch.cat((prefix, prefix.new_zeros(prefix.shape[0], future, *prefix.shape[2:])), dim=1)
        states, _ = self.statistics.state.normalize(state[None], sm[None])
        frame_ids = prefix_count + torch.arange(self.action_horizon) // self.actions_per_latent
        metadata = DecisionWindowMetadata(
            block_size=future,
            action_frame_ids=frame_ids,
            action_mask=am.expand(self.action_horizon, -1),
            anchors=state[None],
            state_mask=sm[None],
            template=self.template,
            statistics=self.statistics,
            observed_latents=prefix_count,
        )
        return dict(
            mode="policy",
            layout_mode="decision_window",
            observed_latents=prefix_count,
            text_token_ids=self.text_ids,
            video_latents=video,
            action=torch.zeros(self.action_horizon, self.template.width, device=video.device),
            states=states,
            anchors=state[None],
            action_mask=am,
            state_mask=sm,
            action_frame_ids=frame_ids.tolist(),
            state_latent_indexes=[prefix_count - 1],
            condition_frame_indexes_vision=list(range(prefix_count)),
            condition_frame_indexes_action=[],
            block_size=future,
            history_blocks=prefix_count,
            action_valid_steps=torch.ones(self.action_horizon, dtype=torch.bool),
            num_valid_video_latents=prefix_count + future,
            conditioning_fps=self.controller_hz / self.video_stride,
            conditioning_fps_action=self.controller_hz,
            action_start_frame_offset=self.history_steps + 1,
            tick=tick,
            revision=revision,
            window=window,
            rtc_metadata=metadata,
        )
