# SPDX-License-Identifier: OpenMDW-1.1
"""Compact native decision payloads onto one shared observation time axis."""

import math

import torch

from . import mrope
from .packing import SampleSpans
from .patching import patchify_video


def observation_positions(frame_ids, latent_shape, text_length, sample, options, device):
    """Use stable stream-origin temporal coordinates and native spatial indices."""
    if not frame_ids:
        return torch.empty((3, 0), dtype=torch.float32, device=device)
    patch = options.get("patch_size", 2)
    _, _, height, width = latent_shape
    compression = options.get("temporal_compression", 4)
    positions, _ = mrope.get_3d_mrope_ids_vae_tokens(
        len(frame_ids),
        math.ceil(height / patch),
        math.ceil(width / patch),
        temporal_offset=text_length + options.get("temporal_modality_margin", 15000),
        fps=sample["conditioning_fps"],
        base_fps=options.get("base_fps", 24.0),
        temporal_compression_factor=compression,
        actual_temporal_compression_factor=compression,
        temporal_positions=torch.tensor(frame_ids, dtype=torch.float32, device="cpu"),
    )
    return positions.to(device=device)


def compact_stream_payload(
    samples,
    inputs,
    spans,
    *,
    prefix_latents,
    prefix_frame_ids,
    clean_latents,
    clean_frame_ids,
    anchor_frame_ids,
    context_id,
    history_latents,
    targets=None,
    position_options=None,
):
    """Retain every decision's native noise/masks while sharing clean encoding.

    Input spans refer to ordinary decision-window packing. Output target spans
    refer only to the compact future payload; unobserved/condition supervision
    rules are copied, not inferred again. No noise or task is sampled here.
    """
    if not samples or len(samples) != len(spans) or len(samples) != len(anchor_frame_ids):
        raise ValueError("A stream segment requires one span and anchor per decision")
    options = dict(position_options or {})
    patch = options.get("patch_size", 2)
    device, dtype = inputs["video_tokens"].device, inputs["video_tokens"].dtype
    text = samples[0]["text_token_ids"]
    if any(not torch.equal(sample["text_token_ids"].cpu(), text.cpu()) for sample in samples):
        raise ValueError("A shared observation stream requires identical final prompt tokens")
    if len(prefix_frame_ids) != prefix_latents.shape[1] or len(clean_frame_ids) != clean_latents.shape[1]:
        raise ValueError("Shared observation payload and frame IDs disagree")
    _, _, height, width = prefix_latents.shape
    patches = math.ceil(height / patch) * math.ceil(width / patch)
    payload = dict(
        input_ids=inputs["input_ids"][: len(text)],
        context_id=str(context_id),
        history_latents=history_latents,
        patches_per_frame=patches,
        prefix_video_tokens=patchify_video(prefix_latents.to(device=device, dtype=dtype), patch),
        prefix_position_ids=observation_positions(
            prefix_frame_ids, prefix_latents.shape, len(text), samples[0], options, device
        ),
        prefix_frame_ids=tuple(prefix_frame_ids),
        clean_video_tokens=patchify_video(clean_latents.to(device=device, dtype=dtype), patch),
        clean_position_ids=observation_positions(
            clean_frame_ids, clean_latents.shape, len(text), samples[0], options, device
        ),
        clean_frame_ids=tuple(clean_frame_ids),
    )
    values = {
        name: []
        for name in (
            "future_video_tokens",
            "future_video_timesteps",
            "future_position_ids",
            "action_tokens",
            "action_timesteps",
            "action_mask",
            "action_position_ids",
            "state_tokens",
            "state_mask",
            "state_position_ids",
        )
    }
    decisions, compact_spans = [], []
    video_offset = action_offset = 0
    target_video, target_times, video_masks, action_masks = [], [], [], []
    for index, (sample, span, anchor) in enumerate(zip(samples, spans, anchor_frame_ids, strict=True)):
        observed = sample["observed_latents"]
        if observed != history_latents or len(sample["states"]) != 1:
            raise ValueError("Streaming decisions require one anchor and the configured history")
        future_frames = sample["video_latents"].shape[1] - observed
        count = future_frames * patches
        video_rows = slice(span.current_video_rows.stop - count, span.current_video_rows.stop)
        future_global = slice(span.current_video.stop - count, span.current_video.stop)
        delta = (anchor - (observed - 1)) * options.get("base_fps", 24.0) / sample["conditioning_fps"]

        def shifted(section, delta=delta):
            positions = inputs["position_ids"][:, section].clone()
            positions[0] += delta
            return positions

        values["future_video_tokens"].append(inputs["video_tokens"][video_rows])
        values["future_video_timesteps"].append(inputs["video_timesteps"][video_rows])
        values["future_position_ids"].append(shifted(future_global))
        values["action_tokens"].append(inputs["action_tokens"][span.action_rows])
        values["action_timesteps"].append(inputs["action_timesteps"][span.action_rows])
        values["action_mask"].append(inputs["action_mask"][span.action_rows])
        values["action_position_ids"].append(shifted(span.action))
        values["state_tokens"].append(inputs["state_tokens"][index : index + 1])
        values["state_mask"].append(inputs["state_mask"][index : index + 1])
        values["state_position_ids"].append(shifted(span.state))
        action_stop = action_offset + len(sample["action"])
        video_stop = video_offset + count
        conditioned_video = set(sample["condition_frame_indexes_vision"])
        conditioned_action = set(sample["condition_frame_indexes_action"])
        decisions.append(
            dict(
                anchor_frame=int(anchor),
                video_start=video_offset,
                video_stop=video_stop,
                action_start=action_offset,
                action_stop=action_stop,
                state_row=index,
                video_condition=tuple(
                    frame in conditioned_video
                    for frame in range(observed, observed + future_frames)
                    for _ in range(patches)
                ),
                action_condition=tuple(step in conditioned_action for step in range(len(sample["action"]))),
            )
        )
        if targets is not None:
            target_video.append(targets["video"][index][:, observed:])
            target_times.append(targets["video_times"][index][observed:])
            video_masks.append(targets["loss_masks"]["video_loss_mask"][video_rows])
            action_masks.append(targets["loss_masks"]["action_loss_mask"][span.action_rows])
            empty = slice(0, 0)
            compact_spans.append(
                SampleSpans(
                    text=empty,
                    clean_video=empty,
                    current_video=slice(video_offset, video_stop),
                    action=slice(action_offset, action_stop),
                    state=slice(index, index + 1),
                    current_video_rows=slice(video_offset, video_stop),
                    action_rows=slice(action_offset, action_stop),
                )
            )
        video_offset, action_offset = video_stop, action_stop
    for name, parts in values.items():
        payload[name] = torch.cat(parts, dim=1 if name.endswith("position_ids") else 0)
    payload["decisions"] = tuple(decisions)
    if targets is not None:
        payload["targets"] = dict(
            video=target_video,
            video_times=target_times,
            action=targets["action"],
            action_times=targets["action_times"],
            spans=tuple(compact_spans),
            loss_masks=dict(video_loss_mask=torch.cat(video_masks), action_loss_mask=torch.cat(action_masks)),
        )
    return payload
