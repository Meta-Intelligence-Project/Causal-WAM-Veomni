# SPDX-License-Identifier: OpenMDW-1.1
"""Volatile per-lane continuous observations and native multi-decision targets."""

import hashlib
import json
from dataclasses import dataclass

import torch

from ..cosmos3_core.episode_vae import BoundedEpisodeVAE
from ..cosmos3_core.segment_inputs import compact_stream_payload


@dataclass
class EpisodeConditionState:
    encoder: BoundedEpisodeVAE
    context_id: str
    segment_id: int
    origin_tick: int
    end_tick: int
    last_anchor: int
    boundary_image: torch.Tensor


def context_identity(segment):
    return hashlib.sha256(json.dumps(segment["stream_id"], sort_keys=True).encode()).hexdigest()


def clear_streams(condition, lane=None):
    lanes = list(condition.episode_streams) if lane is None else [lane]
    for item in lanes:
        state = condition.episode_streams.pop(item, None)
        if state is not None:
            state.encoder.clear()


def _take(encoder, frames):
    if not frames:
        return encoder.latents[0, :, :0]
    first = encoder.latent_ids[0]
    if frames[0] < first or frames[-1] >= encoder.latent_count:
        raise ValueError("The bounded observation ring does not cover this decision")
    return encoder.latents[0, :, frames[0] - first : frames[-1] - first + 1]


@torch.no_grad()
def encode_segment(condition, segment):
    if condition.config.stage != "action":
        raise ValueError("Continuous episode segments currently require Action training")
    lane = segment["lane_id"]
    context_id = context_identity(segment)
    decisions = segment["decisions"]
    if not decisions:
        raise ValueError("An episode segment must contain at least one decision")
    history = 1 + segment["history_steps"] // segment["actions_per_latent"]
    horizon = segment["action_horizon"] // segment["actions_per_latent"]
    slots = segment["segment_latents"] // horizon
    if slots < 1 or len(decisions) > slots:
        raise ValueError("The segment exceeds its configured decision slots")
    video = segment["video"]
    if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("Episode observations must be uint8 [3,T,H,W]")
    frames = video.unsqueeze(0).to(device=condition.device, dtype=torch.float32) / 127.5 - 1.0
    state = condition.episode_streams.get(lane)
    try:
        if segment["reset"]:
            clear_streams(condition, lane)
            if segment["segment_id"] != 0:
                raise ValueError("A reset episode segment must start at segment zero")
            encoder = BoundedEpisodeVAE(
                condition.video_vae, max_latents=history + segment["segment_latents"] + horizon
            )
            encoder.reset(frames)
            first_anchor = history - 1
            previous_anchor = first_anchor
        else:
            if state is None or state.context_id != context_id:
                raise ValueError("A continued segment has no matching episode VAE state")
            if segment["segment_id"] != state.segment_id + 1 or segment["start_tick"] != state.end_tick:
                raise ValueError("Episode segments must be consumed in order without gaps")
            if not torch.equal(video[:, 0].cpu(), state.boundary_image):
                raise ValueError("The continued observation boundary changed")
            encoder = state.encoder
            first_anchor = encoder.latent_count - 1
            expected = history - 1 + (segment["start_tick"] - state.origin_tick) // segment["actions_per_latent"]
            if first_anchor != expected:
                raise ValueError("Episode VAE clock and source ticks disagree")
            previous_anchor = state.last_anchor
            encoder.append(frames[:, :, 1:])
        anchors = tuple(first_anchor + int(row["decision_offset_latents"]) for row in decisions)
        if anchors[0] != first_anchor or anchors[-1] + horizon >= encoder.latent_count:
            raise ValueError("Decision anchors and encoded future observations disagree")
        samples = []
        for sample, anchor in zip(decisions, anchors, strict=True):
            observed = tuple(range(anchor - history + 1, anchor + horizon + 1))
            samples.append({**sample, "video_latents": _take(encoder, observed)})
        prefix_ids = tuple(range(first_anchor - history + 1, first_anchor + 1)) if segment["reset"] else ()
        clean_ids = tuple(range(previous_anchor + 1, anchors[-1] + 1))
        encoded = dict(
            samples=samples,
            prefix_latents=_take(encoder, prefix_ids),
            prefix_frame_ids=prefix_ids,
            clean_latents=_take(encoder, clean_ids),
            clean_frame_ids=clean_ids,
            anchor_frame_ids=anchors,
            context_id=context_id,
            history_latents=history,
            actual_decisions=len(decisions),
            nominal_decisions=slots,
            lane_id=lane,
            final=segment["final"],
        )
        condition.episode_streams[lane] = EpisodeConditionState(
            encoder=encoder,
            context_id=context_id,
            segment_id=segment["segment_id"],
            origin_tick=segment["start_tick"] if segment["reset"] else state.origin_tick,
            end_tick=segment["end_tick"],
            last_anchor=anchors[-1],
            boundary_image=video[:, -1].cpu().clone(),
        )
        return encoded
    except BaseException:
        clear_streams(condition, lane)
        # A cold encoder may not yet have been committed to the lane map.
        if "encoder" in locals() and (state is None or encoder is not state.encoder):
            encoder.clear()
        raise


@torch.no_grad()
def prepare_segment(condition, episode, **noise_options):
    try:
        inputs, targets = condition.prepare(episode["samples"], **noise_options)
        payload = compact_stream_payload(
            episode["samples"],
            inputs,
            targets["spans"],
            targets=targets,
            prefix_latents=episode["prefix_latents"],
            prefix_frame_ids=episode["prefix_frame_ids"],
            clean_latents=episode["clean_latents"],
            clean_frame_ids=episode["clean_frame_ids"],
            anchor_frame_ids=episode["anchor_frame_ids"],
            context_id=episode["context_id"],
            history_latents=episode["history_latents"],
            position_options={**condition.position_options, "patch_size": condition.patch_size},
        )
        payload["attention_backend"] = condition.config.attention_backend
        return payload
    except BaseException:
        clear_streams(condition, episode["lane_id"])
        raise
