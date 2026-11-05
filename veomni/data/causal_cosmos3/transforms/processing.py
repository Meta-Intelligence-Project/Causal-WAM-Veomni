# SPDX-License-Identifier: OpenMDW-1.1
"""Shared video counts and action block encoding/quantile normalization."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .action_template import ActionTemplate55


def video_token_layout(
    video_frames, height, width, text_tokens, *, temporal_compression=4, spatial_compression=16, patch_size=2
):
    """Count text and both video copies from the padded pixel-frame shape."""
    if min(temporal_compression, spatial_compression, patch_size, video_frames) < 1:
        raise ValueError("Video frames and compression/patch sizes must be positive")
    if (video_frames - 1) % temporal_compression or height % spatial_compression or width % spatial_compression:
        raise ValueError("Video frames and spatial dimensions must be VAE aligned")
    h, w = height // spatial_compression, width // spatial_compression
    patches = ((h + patch_size - 1) // patch_size) * ((w + patch_size - 1) // patch_size)
    latents = 1 + (video_frames - 1) // temporal_compression
    return dict(text=text_tokens, clean_video=latents * patches, target_video=latents * patches)


def num_valid_video_latents(real_intervals, video_stride, temporal_compression):
    """Keep mixed real/padded latents; repeated off-grid endpoints are padding."""
    sampled = real_intervals // video_stride
    return 1 + (sampled + temporal_compression - 1) // temporal_compression


@dataclass(frozen=True)
class Geometry:
    actions_per_block: int = 32
    video_stride: int = 4
    temporal_compression: int = 4
    spatial_compression: int = 16
    patch_size: int = 2

    def __post_init__(self):
        if min(vars(self).values()) <= 0 or self.actions_per_block % self.actions_per_latent:
            raise ValueError("Positive geometry and whole latent groups per action block are required")

    @property
    def actions_per_latent(self):
        return self.video_stride * self.temporal_compression

    def token_layout(self, actions, height, width, text_tokens):
        """Full clean/target teacher-forcing video, one action/step and state/block."""
        if actions % self.actions_per_block or height % self.spatial_compression or width % self.spatial_compression:
            raise ValueError("Action/video geometry is not block/VAE aligned")
        layout = video_token_layout(
            1 + actions // self.video_stride,
            height,
            width,
            text_tokens,
            temporal_compression=self.temporal_compression,
            spatial_compression=self.spatial_compression,
            patch_size=self.patch_size,
        )
        layout.update(action=actions, state=actions // self.actions_per_block)
        return layout


class ActionProcessor:
    """Map source channels, encode block-relative actions, apply separate measured stats.

    Return action/frame and state/latent indexes as Python integer lists; payloads
    remain tensors so DiTTrainer can move them without moving index metadata.
    Mapping is a template-width list of source indices; -1 denotes an absent channel.
    Pass statistics=None only for collecting statistics, never a training configuration.
    A replacement template needs width/sanitize/required_anchor_mask/encode/decode methods.
    """

    def __init__(self, statistics, *, template=None, state_mapping=None, action_mapping=None):
        self.template = template or ActionTemplate55()
        self.state_mapping = state_mapping
        self.action_mapping = action_mapping
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text())
        self.statistics = statistics
        self.statistics_geometry = (statistics or {}).get("provenance", {}).get("geometry", {})
        if statistics is not None:
            if statistics.get("provenance", {}).get("kind") == "mock":
                raise ValueError("Production data loading requires measured statistics")
            template_id = statistics.get("template_id", statistics.get("provenance", {}).get("template_id"))
            if template_id is not None and template_id != self.template.template_id:
                raise ValueError("Statistics template_id differs from the configured template")
            self.statistics = {}
            for name in ("state", "action"):
                q = statistics[name]
                low, high, valid = torch.tensor(q["low"]), torch.tensor(q["high"]), torch.tensor(q["valid"]).bool()
                if any(t.shape != (self.template.width,) for t in (low, high, valid)):
                    raise ValueError("Statistics must have template width")
                if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any():
                    raise ValueError("Statistics bounds must be finite and ordered")
                self.statistics[name] = (low.float(), high.float(), valid)

    def _map(self, values, mask, mapping):
        if mapping is None:
            return values.float(), mask
        indexes = torch.as_tensor(mapping, dtype=torch.long)
        if indexes.shape != (self.template.width,) or (indexes < -1).any() or (indexes >= values.shape[-1]).any():
            raise ValueError("Channel mapping must contain one source index (or -1) per template channel")
        valid = (indexes >= 0) & mask[indexes.clamp_min(0)]
        return values[:, indexes.clamp_min(0)].float().masked_fill(~valid, 0), valid

    def normalize(self, values, mask, name):
        if self.statistics is None:
            return values, torch.zeros(values.shape[-1])
        low, high, valid = self.statistics[name]
        if (mask & ~valid).any():
            raise ValueError(f"{name} statistics do not cover active source channels")
        span = high - low
        scaled = 2 * (values - low) / span.clamp_min(1e-12) - 1
        clipped = mask & (span > 0) & (scaled.abs() > 1.5)
        return scaled.clamp(-1.5, 1.5).masked_fill(~mask | (span == 0), 0), clipped.float().mean(0)

    def decode(self, actions, anchors, mask):
        if self.statistics is not None:
            low, high, _ = self.statistics["action"]
            low, high = low.to(actions), high.to(actions)
            actions = ((actions + 1) * 0.5 * (high - low) + low).masked_fill(~mask, 0)
        return self.template.decode_action_delta(actions, anchors, mask)

    def __call__(self, raw, geometry):
        """Encode explicit multi-block diagnostic/rollout inputs; training uses decision_window."""
        state, sm = self._map(raw["state"], raw["state_mask"], self.state_mapping)
        target, am = self._map(raw["action"], raw["action_mask"], self.action_mapping)
        sm, am = self.template.validate_valid_mask(sm), self.template.validate_valid_mask(am)
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Active relative action channels require measured anchor state channels")
        state, target = self.template.sanitize(state, sm), self.template.sanitize(target, am)
        n, block = len(target), geometry.actions_per_block
        if len(state) != n + 1 or n % block:
            raise ValueError("Expected block-aligned T targets and T+1 states")
        starts = torch.arange(0, n, block)
        anchors = state[starts]
        encoded = self.template.encode_action_delta(target, anchors.repeat_interleave(block, 0), am)
        states, state_clip = self.normalize(anchors, sm, "state")
        actions, action_clip = self.normalize(encoded, am, "action")
        real = raw["real_action_steps"]
        return dict(
            action=actions,
            states=states,
            anchors=anchors,
            action_mask=am,
            state_mask=sm,
            state_action_indexes=starts,
            action_frame_ids=(1 + np.arange(n, dtype=np.int64) // geometry.actions_per_latent).tolist(),
            state_latent_indexes=(np.arange(0, n, block, dtype=np.int64) // geometry.actions_per_latent).tolist(),
            action_valid_steps=torch.arange(n) < real,
            num_valid_video_latents=num_valid_video_latents(
                real, geometry.video_stride, geometry.temporal_compression
            ),
            state_clip_fraction=state_clip,
            action_clip_fraction=action_clip,
        )

    def decision_window(self, raw, geometry, history_steps):
        """Encode every future target against the single current measured state."""
        state, sm = self._map(raw["state"][:1], raw["state_mask"], self.state_mapping)
        target, am = self._map(raw["action"], raw["action_mask"], self.action_mapping)
        sm, am = self.template.validate_valid_mask(sm), self.template.validate_valid_mask(am)
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Active relative action channels require measured anchor state channels")
        anchor = self.template.sanitize(state, sm)
        target = self.template.sanitize(target, am)
        encoded = self.template.encode_action_delta(target, anchor, am)
        states, state_clip = self.normalize(anchor, sm, "state")
        actions, action_clip = self.normalize(encoded, am, "action")
        per_latent = geometry.actions_per_latent
        observed = 1 + history_steps // per_latent
        horizon = len(actions)
        return dict(
            action=actions,
            states=states,
            anchors=anchor,
            action_mask=am,
            state_mask=sm,
            state_action_indexes=torch.tensor([0], dtype=torch.long),
            action_frame_ids=(observed + np.arange(horizon, dtype=np.int64) // per_latent).tolist(),
            state_latent_indexes=[observed - 1],
            action_start_frame_offset=history_steps + 1,
            action_valid_steps=torch.arange(horizon) < raw["real_action_steps"],
            num_valid_video_latents=history_steps // per_latent
            + num_valid_video_latents(raw["real_action_steps"], geometry.video_stride, geometry.temporal_compression),
            state_clip_fraction=state_clip,
            action_clip_fraction=action_clip,
        )
