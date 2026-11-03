# SPDX-License-Identifier: OpenMDW-1.1
"""Independent causal Cosmos3 teacher forcing and pixel-feedback policy rollout.

CausalCosmos3Pipeline loads complete native/HF weights and native data assets.
The same block sampler serves FD/ID/Policy teacher forcing and Policy rollout.
Result helpers export physical actions, RGB video and block feedback records.
Legacy checkpoint conversion is a separate tool executed before inference.
"""

import copy
import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.causal_cosmos3.readers.observation import (
    make_sample,
    read_initial_observation,
    read_npz_observation,
    read_teacher_window,
)
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry
from veomni.data.causal_cosmos3.transforms.tokenization import build_prompt_tokenizer, tokenize_prompt
from veomni.models import build_foundation_model
from veomni.models.checkpoint.weights import load_model_weights

from .cosmos3_core.fixed_step import FixedStepSampler
from .cosmos3_core.packing import pack_model_inputs
from .cosmos3_core.patching import unpatchify_video
from .cosmos3_core.unipc import FlowUniPCMultistepScheduler
from .cosmos3_core.vae import Cosmos3VideoVAE
from .cosmos3_transformer.configuration_cosmos3_transformer import Cosmos3Config


__all__ = ["CausalCosmos3Pipeline", "InferenceResult", "save_result"]


# Model loading


def _empty_model(config, dtype, ops_implementation=None):
    config.weight_loading = {"initialize_missing_patterns": []}
    ops = dict(attn_implementation="eager", rotary_pos_emb_implementation="eager", rms_norm_implementation="eager")
    ops.update(ops_implementation or {})
    return build_foundation_model(
        config,
        torch_dtype=str(dtype).split(".")[-1],
        init_device="meta",
        ops_implementation=OpsImplementationConfig(**ops),
    )


def load_trained_model(checkpoint, *, config_path=None, device="cpu", dtype=torch.bfloat16, ops_implementation=None):
    """Missing trained parameters, including action/state, are always errors."""
    checkpoint = pathlib.Path(checkpoint)
    config = Cosmos3Config.from_json_file(str(config_path or checkpoint / "config.json"))
    if config.action_interface != "shared" or config.action_dim != 55 or config.state_dim != 55:
        raise ValueError("Causal action inference requires the trained shared 55D action/state configuration")
    model = _empty_model(config, dtype, ops_implementation)
    load_model_weights(model, str(checkpoint), init_device=device)
    if model.weight_loading_report["mode"] != "complete":
        raise ValueError("Inference requires complete trained weights")
    return model.eval().requires_grad_(False)


# Inference pipeline


class CausalCosmos3Pipeline:
    """Reuse native weights, preprocessing and packing for explicit generation."""

    def __init__(self, model, vae, processor, tokenize, config, *, device="cpu"):
        self.model = model.eval().requires_grad_(False)
        self.vae, self.processor, self.tokenize = vae, processor, tokenize
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        self.source = self.config["source"]
        self.geometry = Geometry(**config.get("geometry", {}))
        options = self.config.setdefault("sampling", {})
        fixed_step = options.get("fixed_step_sampler_config") or getattr(
            model.config, "fixed_step_sampler_config", None
        )
        if fixed_step is not None:
            options["fixed_step_sampler_config"] = copy.deepcopy(fixed_step)
            # The source inference entry uses distilled guidance, already learned by the student.
            options["guidance"] = 1.0
        self.history = options.get("history", 32)
        if type(self.history) is not int or self.history < 1:
            raise ValueError("history must be a positive integer")
        self.sampling_options = dict(
            steps=options.get("steps", 20),
            shift=float(options.get("shift", 1.0)),
            guidance=float(options.get("guidance", 1.0)),
            guidance_interval=options.get("guidance_interval"),
            normalize_cfg=options.get("normalize_cfg", False),
            fixed_step_sampler_config=fixed_step,
            seed=options.get("seed", 42),
            attention_backend=config.get("attention_backend", "grouped"),
            negative_text_ids=torch.as_tensor(tokenize(options.get("negative_prompt", "")), dtype=torch.long),
            position_options=dict(
                base_fps=config.get("base_fps", 24.0),
                temporal_compression=self.geometry.temporal_compression,
                temporal_modality_margin=config.get("temporal_modality_margin", 15000),
            ),
        )
        if model.config.patch_size != self.geometry.patch_size:
            raise ValueError("Model and source patch sizes differ")

    @classmethod
    def from_pretrained(cls, checkpoint, config):
        device = config.get("device", "cpu")
        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        model = load_trained_model(
            checkpoint,
            config_path=config.get("model_config"),
            device=device,
            dtype=dtype,
            ops_implementation=config.get("ops_implementation"),
        )
        vae = Cosmos3VideoVAE(config["vae"], device=device, dtype=dtype)
        processor = ActionProcessor(
            config["statistics"],
            state_mapping=config["source"].get("state_mapping"),
            action_mapping=config["source"].get("action_mapping"),
        )
        tokenizer = build_prompt_tokenizer(config["tokenizer"])
        pipe = cls(model, vae, processor, partial(tokenize_prompt, tokenizer=tokenizer), config, device=device)
        pipe.loading_report = {
            **model.weight_loading_report,
            "vae": dict(source=config["vae"], loader="diffusers_pretrained", dtype=str(dtype)),
        }
        return pipe

    def teacher_forcing(self, dataset, episode, start, horizon, *, task="policy", prompt=None):
        sample = read_teacher_window(
            dataset,
            episode,
            start,
            horizon,
            source=self.source,
            processor=self.processor,
            geometry=self.geometry,
            tokenize=self.tokenize,
            task=task,
            history=self.history,
            prompt=prompt,
        )
        return teacher_forcing(self, sample)

    def rollout(self, *, dataset=None, input_npz=None, episode=0, start=0, horizon=32, fps=None, prompt=None):
        if (dataset is None) == (input_npz is None):
            raise ValueError("Choose exactly one dataset or initial NPZ input")
        common = dict(source=self.source, processor=self.processor, geometry=self.geometry, prompt=prompt)
        if input_npz is not None:
            if fps is None:
                raise ValueError("An initial NPZ requires explicit --fps")
            initial = read_npz_observation(input_npz, fps=fps, horizon=horizon, **common)
        else:
            initial = read_initial_observation(dataset, episode, start, horizon, **common)
        result = rollout(self, initial)
        self.last_input_identity = initial.identity
        return result


# Block sampling


def noise_seed(seed, block, stream):
    return int(hashlib.sha256(f"cosmos3-inference-v1:{seed}:{block}:{stream}".encode()).hexdigest()[:16], 16) % 2**63


@torch.no_grad()
def sample_block(
    model,
    sample,
    block,
    *,
    steps=20,
    shift=1.0,
    guidance=1.0,
    guidance_interval=None,
    normalize_cfg=False,
    fixed_step_sampler_config=None,
    seed=42,
    seed_block=None,
    negative_text_ids=None,
    attention_backend="grouped",
    initial_noise=None,
    position_options=None,
):
    """Sample one block; the clean sample supplies only causally visible context.

    Teacher forcing keeps the original clean video across calls. Rollout supplies
    decoded/re-encoded historical pixels and zero future pixels instead.
    ``initial_noise`` contains actual FP32 video/action tensors for strict oracles.
    """
    if steps < 1 or not all(math.isfinite(x) and x > 0 for x in (shift, guidance)):
        raise ValueError("steps, shift and guidance must be finite and positive")
    if guidance_interval is not None and (
        len(guidance_interval) != 2
        or not all(math.isfinite(t) for t in guidance_interval)
        or not 0 <= guidance_interval[0] < guidance_interval[1]
    ):
        raise ValueError("guidance_interval must be finite nonnegative [lo, hi] in model time with lo < hi")
    video = sample["video_latents"].float()
    action = sample["action"].to(video.device).float()
    size = sample["block_size"]
    start, stop = 1 + block * size, min(1 + (block + 1) * size, video.shape[1])
    if block < 0 or start >= stop:
        raise ValueError("Requested prediction block is outside the sample")
    channels = sample["action_mask"].to(video.device).bool().expand_as(action)
    vm = torch.zeros_like(video, dtype=torch.bool)
    vm[:, start:stop] = True
    vm[:, sample["condition_frame_indexes_vision"]] = False
    af = torch.as_tensor(sample["action_frame_ids"], device=video.device)
    am = ((af >= start) & (af < stop))[:, None] & channels
    am[sample["condition_frame_indexes_action"]] = False
    active = torch.cat((vm.flatten(), am.flatten()))
    reference = torch.cat((video.flatten(), action.masked_fill(~channels, 0).flatten()))
    sid = block if seed_block is None else seed_block
    seeds = {name: noise_seed(seed, sid, name) for name in ("video", "action")}
    if initial_noise is None:
        noises = [
            torch.randn(t.shape, generator=torch.Generator(device=t.device).manual_seed(seeds[name]), device=t.device)
            for name, t in (("video", video), ("action", action))
        ]
    else:
        noises = [initial_noise[name].to(device=video.device, dtype=torch.float32) for name in ("video", "action")]
        if any(n.shape != t.shape for n, t in zip(noises, (video, action), strict=True)):
            raise ValueError("Initial noise shapes must match video/action")
    latent = torch.where(active, torch.cat([n.flatten() for n in noises]), reference)
    if not active.any():
        return dict(video_latents=video.clone(), action=action.masked_fill(~channels, 0), seeds=seeds)
    if guidance != 1 and negative_text_ids is None:
        raise ValueError("CFG requires explicit negative/empty-prompt token IDs")
    dtype = next(model.parameters()).dtype

    def predict(current, timestep, text_ids):
        branch = dict(sample)
        if text_ids is not None:
            branch["text_token_ids"] = text_ids
            branch["token_length"] += len(text_ids) - len(sample["text_token_ids"])
        frame_times = torch.zeros(video.shape[1], device=video.device, dtype=torch.float32)
        frame_times[start:stop] = timestep.float()
        inputs, spans, _ = pack_model_inputs(
            [branch],
            [current[: video.numel()].reshape_as(video)],
            [current[video.numel() :].reshape_as(action)],
            timesteps=[frame_times],
            patch_size=model.config.patch_size,
            **(position_options or {}),
        )
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            inputs[key] = inputs[key].to(dtype=dtype)
        inputs["action_tokens"] = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        outputs = model(**inputs, attention_backend=attention_backend)
        pred_v = unpatchify_video(outputs.video[spans[0].current_video_rows], video.shape, model.config.patch_size)
        pred_a = outputs.action[spans[0].action_rows].masked_fill(~channels, 0)
        return torch.cat((pred_v.float().flatten(), pred_a.float().flatten())).masked_fill(~active, 0)

    if fixed_step_sampler_config is None:
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        step_options = {}
    else:
        scheduler = FixedStepSampler(**fixed_step_sampler_config)
        seeds["sampler"] = noise_seed(seed, sid, "sampler")
        step_options = dict(generator=torch.Generator(device=video.device).manual_seed(seeds["sampler"]))
    scheduler.set_timesteps(steps, device=video.device, shift=float(shift))
    for timestep in scheduler.timesteps:
        current = torch.where(active, latent, reference)
        velocity = predict(current, timestep, None)
        use_cfg = guidance != 1 and (
            guidance_interval is None or guidance_interval[0] < float(timestep) < guidance_interval[1]
        )
        if use_cfg:
            unconditional = predict(current, timestep, negative_text_ids)
            conditional = velocity
            velocity = unconditional + guidance * (conditional - unconditional)
            if normalize_cfg:
                velocity *= (torch.norm(conditional) / (torch.norm(velocity) + 1e-8)).clamp(0, 1)
        latent = scheduler.step(velocity, timestep, current.unsqueeze(0), return_dict=False, **step_options)[0]
        latent = latent.squeeze(0).float()
        latent = torch.where(active, latent, reference)
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite sampling state")
    return dict(
        video_latents=latent[: video.numel()].reshape_as(video),
        action=latent[video.numel() :].reshape_as(action).masked_fill(~channels, 0),
        seeds=seeds,
    )


# Teacher forcing and rollout


@dataclass
class InferenceResult:
    video: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    action_timestamps: torch.Tensor
    video_timestamps: torch.Tensor
    semantics: str
    blocks: list[dict] = field(default_factory=list)
    ground_truth_video: torch.Tensor | None = None
    ground_truth_actions: torch.Tensor | None = None
    reconstruction: torch.Tensor | None = None
    video_fps: float | None = None
    identity: dict = field(default_factory=dict)


def _times(count, fps, stride, offset):
    return offset + torch.arange(count, dtype=torch.float64) * stride / fps


@torch.no_grad()
def teacher_forcing(pipe, sample):
    """Generate each block using measured clean history and measured block state."""
    encoded = pipe.vae.encode_pack([sample])[0]
    for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
        encoded[key] = encoded[key].to(pipe.device)
    block = pipe.geometry.actions_per_block
    frames = block // pipe.geometry.video_stride
    real = sample["real_action_steps"]
    actions, videos, records = [], [sample["video"][:, :1].cpu()], []
    for index in range((real + block - 1) // block):
        begin = time.monotonic()
        generated = sample_block(pipe.model, encoded, index, seed_block=index, **pipe.sampling_options)
        physical = pipe.processor.decode(
            generated["action"].cpu(), sample["anchors"].repeat_interleave(block, 0), sample["action_mask"]
        )
        valid = min(block, real - index * block)
        actions.append(physical[index * block : index * block + valid])
        pixels = pipe.vae.decode(generated["video_latents"])
        videos.append(pixels[:, 1 + index * frames : 1 + index * frames + valid // pipe.geometry.video_stride])
        records.append(
            dict(
                block=index,
                valid_steps=valid,
                seeds=generated["seeds"],
                anchor=sample["anchors"][index].tolist(),
                history_blocks=min(1 + index * encoded["block_size"], pipe.history),
                input_pixel_frames=sample["video"].shape[1],
                latent_shape=list(encoded["video_latents"].shape),
                padded_action_steps=len(sample["action"]) - real,
                temporal_padding_frames=sample["video"].shape[1] - (1 + real // pipe.geometry.video_stride),
                seconds=time.monotonic() - begin,
                feedback="measured history and measured block state",
            )
        )
    fps = sample["conditioning_fps_action"]
    offset = float(sample["timestamps"][0])
    actions = torch.cat(actions)
    video = torch.cat(videos, dim=1)
    # Preserve measured targets before normalization clipping for honest GT curves.
    gt = sample["ground_truth_actions"][:real].cpu()
    reconstruction = pipe.vae.decode(encoded["video_latents"])[:, : video.shape[1]]
    return InferenceResult(
        video,
        actions,
        sample["action_mask"].cpu(),
        _times(real, fps, 1, offset + 1 / fps),
        _times(video.shape[1], fps, pipe.geometry.video_stride, offset),
        f"teacher_forcing_{sample['mode']}_measured_history_and_state",
        records,
        sample["video"][:, : video.shape[1]].cpu(),
        gt,
        reconstruction,
        video_fps=fps / pipe.geometry.video_stride,
        identity=sample.get("identity", {}),
    )


def rollout_sample(initial, state, history, *, source, processor, geometry, tokenize, history_blocks, valid):
    """Build synthetic unknown targets without reading any episode data."""
    frame_count = geometry.actions_per_block // geometry.video_stride
    context, remainder = divmod(history.shape[1] - 1, frame_count)
    if remainder or not 1 <= valid <= geometry.actions_per_block:
        raise ValueError("History must end on a complete action-block boundary")
    total = (context + 1) * geometry.actions_per_block
    future = torch.zeros((3, frame_count, *history.shape[-2:]), dtype=torch.uint8)
    state = processor.template.sanitize(state.cpu(), initial.state_mask)
    if not torch.isfinite(state).all():
        raise ValueError("Nonfinite physical feedback state")
    raw = dict(
        video=torch.cat((history, future), dim=1),
        state=state.expand(total + 1, -1).clone(),
        action=state.expand(total, -1).clone(),
        state_mask=initial.state_mask,
        action_mask=initial.action_mask,
        real_action_steps=context * geometry.actions_per_block + valid,
    )
    # Initial observations and decoded actions are already in template channel order.
    canonical_processor = copy.copy(processor)
    canonical_processor.state_mapping = canonical_processor.action_mapping = None
    sample = make_sample(
        raw,
        source=source,
        processor=canonical_processor,
        geometry=geometry,
        tokenize=tokenize,
        caption=initial.caption,
        fps=initial.fps,
        task="policy",
        history=history_blocks,
    )
    sample["condition_frame_indexes_vision"] = list(
        range(1 + context * geometry.actions_per_block // geometry.actions_per_latent)
    )
    return sample, context


@torch.no_grad()
def rollout(pipe, initial):
    """Model-only open-loop rollout: generated pixels and absolute action feedback."""
    size = pipe.geometry.actions_per_block
    new_frames = size // pipe.geometry.video_stride
    latent_block = size // pipe.geometry.actions_per_latent
    context_limit = (pipe.history + latent_block - 1) // latent_block
    history_pixel_limit = 1 + context_limit * new_frames
    state = initial.state.cpu().clone()
    history = initial.image.cpu().clone()
    videos, actions, records = [history.clone()], [], []
    for index, start in enumerate(range(0, initial.horizon, size)):
        begin = time.monotonic()
        valid = min(size, initial.horizon - start)
        sample, context = rollout_sample(
            initial,
            state,
            history,
            source=pipe.source,
            processor=pipe.processor,
            geometry=pipe.geometry,
            tokenize=pipe.tokenize,
            history_blocks=pipe.history,
            valid=valid,
        )
        encoded = pipe.vae.encode_pack([sample])[0]
        for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
            encoded[key] = encoded[key].to(pipe.device)
        generated = sample_block(pipe.model, encoded, context, seed_block=index, **pipe.sampling_options)
        normalized = generated["action"][-size:].cpu()
        absolute = pipe.processor.decode(normalized, state.expand(size, -1), initial.action_mask)
        if not torch.isfinite(absolute).all():
            raise ValueError("Nonfinite decoded action")
        anchor = state.clone()
        actions.append(absolute[:valid])
        # Feedback always selects the final valid action, never a padded tail.
        state = absolute[valid - 1].clone()
        pixels = pipe.vae.decode(generated["video_latents"])
        if pixels.shape != (3, history.shape[1] + new_frames, *history.shape[-2:]):
            raise ValueError("Decoded pixels do not match the native input geometry")
        added = pixels[:, -new_frames:]
        videos.append(added[:, : valid // pipe.geometry.video_stride])
        history = torch.cat((history, added), dim=1)
        history = history[:, -history_pixel_limit:].contiguous()
        records.append(
            dict(
                block=index,
                valid_steps=valid,
                seeds=generated["seeds"],
                anchor=anchor.tolist(),
                feedback_state=state.tolist(),
                history_blocks=min(1 + context * latent_block, pipe.history),
                context_blocks=context,
                history_pixel_frames=history.shape[1],
                input_pixel_frames=sample["video"].shape[1],
                latent_shape=list(encoded["video_latents"].shape),
                padded_action_steps=size - valid,
                temporal_padding_frames=new_frames - valid // pipe.geometry.video_stride,
                state_clip_fraction=sample["state_clip_fraction"].tolist(),
                seconds=time.monotonic() - begin,
                feedback="generated uint8 pixels and final valid absolute action",
            )
        )
    video = torch.cat(videos, dim=1)
    return InferenceResult(
        video,
        torch.cat(actions),
        initial.action_mask.cpu(),
        _times(initial.horizon, initial.fps, 1, initial.timestamp + 1 / initial.fps),
        _times(video.shape[1], initial.fps, pipe.geometry.video_stride, initial.timestamp),
        "model_only_open_loop_action_as_state_pixel_feedback",
        records,
        video_fps=initial.fps / pipe.geometry.video_stride,
        identity=initial.identity,
    )


# Output artifacts


def write_video(path, video, fps):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(float(fps)).limit_denominator(10000))
        stream.height, stream.width = video.shape[-2:]
        stream.pix_fmt = "yuv420p"
        for pixels in video.permute(1, 2, 3, 0).numpy():
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def action_plot(path, actions, mask, timestamps, truth=None):
    """Write a portable SVG with one physical-unit curve per valid channel."""
    channels = np.flatnonzero(mask)
    rows = (len(channels) + 3) // 4
    width, height = 1200, 150 * rows
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for plot, channel in enumerate(channels):
        x, y = 300 * (plot % 4) + 38, 150 * (plot // 4) + 25
        values = actions[:, channel]
        bounds = values if truth is None else np.concatenate((values, truth[:, channel]))
        lo, hi = float(bounds.min()), float(bounds.max())
        scale = max(hi - lo, 1e-6)
        parts.append(f'<text x="{x}" y="{y - 7}" font-size="12">channel {channel}: {lo:.4g} .. {hi:.4g}</text>')
        parts.append(f'<path d="M{x},{y}v100h240" fill="none" stroke="#aaa"/>')
        for array, color in ((values, "#1261a0"), (None if truth is None else truth[:, channel], "#e38225")):
            if array is None:
                continue
            points = " ".join(
                f"{x + 240 * i / max(1, len(array) - 1):.2f},{y + 100 - 100 * (float(v) - lo) / scale:.2f}"
                for i, v in enumerate(array)
            )
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2"/>')
        parts.append(
            f'<text x="{x}" y="{y + 118}" font-size="10">{timestamps[0]:.3f} .. {timestamps[-1]:.3f} seconds</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def save_result(result, output, *, config, checkpoint, identity, loading_report=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(f"Inference output already exists: {output}")
    arrays = dict(
        actions=result.actions.numpy(),
        action_mask=result.action_mask.numpy(),
        action_timestamps=result.action_timestamps.numpy(),
        video_timestamps=result.video_timestamps.numpy(),
    )
    if result.ground_truth_actions is not None:
        arrays["ground_truth_actions"] = result.ground_truth_actions.numpy()
    np.savez_compressed(output / "actions.npz", **arrays)
    fps = result.video_fps
    if fps is None and result.video.shape[1] > 1:
        fps = 1 / float(result.video_timestamps[1] - result.video_timestamps[0])
    if fps is None:
        raise ValueError("A one-frame result requires explicit video FPS")
    write_video(output / "generated.mp4", result.video, fps)
    for name, video in (("ground_truth", result.ground_truth_video), ("vae_reconstruction", result.reconstruction)):
        if video is not None:
            write_video(output / f"{name}.mp4", video, fps)
    action_plot(
        output / "actions.svg",
        arrays["actions"],
        arrays["action_mask"],
        arrays["action_timestamps"],
        arrays.get("ground_truth_actions"),
    )
    (output / "blocks.jsonl").write_text("".join(json.dumps(block, allow_nan=False) + "\n" for block in result.blocks))
    metadata = dict(
        semantics=result.semantics,
        checkpoint=str(Path(checkpoint).resolve()),
        config=config,
        identity={**identity, **result.identity},
        weight_loading=loading_report,
        valid_actions=len(result.actions),
        video_frames=result.video.shape[1],
        image_size=list(result.video.shape[-2:]),
        video_fps=fps,
        duration_seconds=sum(block.get("seconds", 0) for block in result.blocks),
        preprocessing="native VeOmni resolution buckets, camera rectangles and bicubic resize",
        evaluation_scope="training-data replay unless the selected input is explicitly held out",
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return output
