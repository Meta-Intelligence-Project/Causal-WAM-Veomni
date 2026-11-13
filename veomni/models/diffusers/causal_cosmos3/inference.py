# SPDX-License-Identifier: OpenMDW-1.1
"""Independent causal Cosmos3 generation and legacy pixel-feedback rollout.

CausalCosmos3Pipeline loads complete native/HF weights and native data assets.
The same block sampler serves FD/ID/Policy teacher forcing and Policy rollout.
Triple experts support video, forward dynamics and cached inverse dynamics.
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


def load_trained_model(
    checkpoint, *, config_path=None, device="cpu", dtype=torch.bfloat16, ops_implementation=None, task=None
):
    """Missing trained parameters, including action/state, are always errors."""
    checkpoint = pathlib.Path(checkpoint)
    config = Cosmos3Config.from_json_file(str(config_path or checkpoint / "config.json"))
    video_only = task == "video" and getattr(config, "expert_layout", "dual") == "triple"
    if not video_only and (config.action_interface != "shared" or config.action_dim != 55 or config.state_dim != 55):
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
        if "expert_layout" in config and config["expert_layout"] != getattr(model.config, "expert_layout", "dual"):
            raise ValueError("Inference configuration expert_layout differs from the trained checkpoint")
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
        if getattr(model.config, "expert_layout", "dual") == "triple":
            if fixed_step is not None:
                raise ValueError(
                    "Triple inference uses independent UniPC/Euler schedules, not a distilled joint schedule"
                )
            for name, default in (
                ("video_steps", 20),
                ("action_steps", 20),
                ("partial_video_steps", 4),
                ("video_stop_sigma", 0.9),
                ("profile_memory", True),
            ):
                self.sampling_options[name] = options.setdefault(name, default)
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
            **({"task": "video"} if config.get("task") == "video" else {}),
        )
        vae = Cosmos3VideoVAE(config["vae"], device=device, dtype=dtype)
        processor = (
            None
            if config.get("task") == "video"
            else ActionProcessor(
                config["statistics"],
                state_mapping=config["source"].get("state_mapping"),
                action_mapping=config["source"].get("action_mapping"),
            )
        )
        tokenizer = build_prompt_tokenizer(config["tokenizer"])
        pipe = cls(model, vae, processor, partial(tokenize_prompt, tokenizer=tokenizer), config, device=device)
        pipe.loading_report = {
            **model.weight_loading_report,
            "vae": dict(source=config["vae"], loader="diffusers_pretrained", dtype=str(dtype)),
        }
        return pipe

    def teacher_forcing(self, dataset, episode, start, horizon, *, task="policy", prompt=None):
        triple = getattr(self.model.config, "expert_layout", "dual") == "triple"
        if triple and task not in ("video", "forward_dynamics", "inverse_dynamics"):
            raise ValueError(
                "Triple inference supports video, forward_dynamics and inverse_dynamics; policy is disabled"
            )
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
        if getattr(self.model.config, "expert_layout", "dual") == "triple":
            raise ValueError("Triple experts do not support policy rollout")
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


def _synchronize_profile(device):
    if torch.device(device).type != "cpu":
        torch.get_device_module(device).synchronize(device)


def _timed_profile(device, operation):
    _synchronize_profile(device)
    begin = time.monotonic()
    result = operation()
    _synchronize_profile(device)
    return result, time.monotonic() - begin


def _peak_memory_profile(device, enabled, scope):
    if not enabled or torch.device(device).type == "cpu":
        return None
    stats = torch.get_device_module(device).memory_stats(device)
    return dict(
        device=str(device),
        scope=scope,
        peak_allocated_bytes=stats.get("allocated_bytes.all.peak"),
        peak_reserved_bytes=stats.get("reserved_bytes.all.peak"),
    )


def _encode_pack_profile(pipe, sample):
    enabled = pipe.sampling_options.get("profile_memory", True)
    _synchronize_profile(pipe.device)
    if enabled and pipe.device.type != "cpu":
        torch.get_device_module(pipe.device).reset_peak_memory_stats(pipe.device)
    pack, seconds = _timed_profile(pipe.device, lambda: pipe.vae.encode_pack([sample]))
    memory = _peak_memory_profile(pipe.device, enabled, "entire_pack_vae_encode")
    return pack, seconds, memory


def _chunk_profile(device, generated, chunk_seconds, decode_seconds, profile_memory):
    return dict(
        chunk_seconds=chunk_seconds,
        solver_seconds=generated["solver_seconds"],
        vae_decode_seconds=decode_seconds,
        memory=generated["memory_profile"],
        chunk_memory=_peak_memory_profile(device, profile_memory, "solver_start_through_chunk_end"),
        stage_seconds=generated["stage_seconds"],
    )


def _pipeline_profile(device, begin, encode_seconds, reconstruction_seconds, records, encode_memory):
    _synchronize_profile(device)
    chunk_profiles = [dict(block=record["block"], **record["profile"]) for record in records]
    return dict(
        device=str(device),
        pipeline_seconds=time.monotonic() - begin,
        vae_encode_pack_seconds=encode_seconds,
        vae_encode_scope="entire_pack_once",
        vae_encode_calls=1,
        vae_encode_memory=encode_memory,
        vae_reconstruction_decode_pack_seconds=reconstruction_seconds,
        vae_reconstruction_decode_calls=int(reconstruction_seconds is not None),
        chunk_seconds=sum(p["chunk_seconds"] for p in chunk_profiles),
        solver_seconds=sum(p["solver_seconds"] for p in chunk_profiles),
        vae_chunk_decode_calls=sum(p["vae_decode_seconds"] is not None for p in chunk_profiles),
        chunks=chunk_profiles,
    )


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
    video_steps=20,
    action_steps=20,
    partial_video_steps=4,
    video_stop_sigma=0.9,
    profile_memory=True,
):
    """Sample one block; the clean sample supplies only causally visible context.

    Teacher forcing keeps the original clean video across calls. Rollout supplies
    decoded/re-encoded historical pixels and zero future pixels instead.
    ``initial_noise`` contains actual FP32 video/action tensors for strict oracles.
    """
    if getattr(model.config, "expert_layout", "dual") == "triple":
        return _sample_triple_block(
            model,
            sample,
            block,
            video_steps=video_steps,
            action_steps=action_steps,
            partial_video_steps=partial_video_steps,
            video_stop_sigma=video_stop_sigma,
            shift=shift,
            guidance=guidance,
            guidance_interval=guidance_interval,
            normalize_cfg=normalize_cfg,
            fixed_step_sampler_config=fixed_step_sampler_config,
            seed=seed,
            seed_block=seed_block,
            negative_text_ids=negative_text_ids,
            attention_backend=attention_backend,
            initial_noise=initial_noise,
            position_options=position_options,
            profile_memory=profile_memory,
        )
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


@torch.no_grad()
def _sample_triple_block(
    model,
    sample,
    block,
    *,
    video_steps,
    action_steps,
    partial_video_steps,
    video_stop_sigma,
    shift,
    guidance,
    guidance_interval,
    normalize_cfg,
    fixed_step_sampler_config,
    seed,
    seed_block,
    negative_text_ids,
    attention_backend,
    initial_noise,
    position_options,
    profile_memory,
):
    """Run independent video/FD sampling or video-prefill followed by IDM action sampling.

    Every expert operation enters through the model's forward, including under
    FSDP. Caches are scoped to this block and prompt branch. Inverse dynamics
    never reads measured future video/action and never decodes its noisy future.
    """
    mode = sample.get("mode")
    if mode not in ("video", "forward_dynamics", "inverse_dynamics"):
        raise ValueError("Triple inference supports video, forward_dynamics and inverse_dynamics; policy is disabled")
    for name, value in (
        ("video_steps", video_steps),
        ("action_steps", action_steps),
        ("partial_video_steps", partial_video_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(video_stop_sigma) or not 0 < video_stop_sigma < 1:
        raise ValueError("video_stop_sigma must be strictly between zero and one")
    if not all(math.isfinite(x) and x > 0 for x in (shift, guidance)):
        raise ValueError("shift and guidance must be finite and positive")
    if guidance_interval is not None and (
        len(guidance_interval) != 2
        or not all(math.isfinite(t) for t in guidance_interval)
        or not 0 <= guidance_interval[0] < guidance_interval[1]
    ):
        raise ValueError("guidance_interval must be finite nonnegative [lo, hi] in model time with lo < hi")
    if fixed_step_sampler_config is not None:
        raise ValueError("Triple inference does not accept a distilled joint fixed-step schedule")
    if guidance != 1 and negative_text_ids is None:
        raise ValueError("CFG requires explicit negative/empty-prompt token IDs")
    if not isinstance(profile_memory, bool):
        raise ValueError("profile_memory must be a boolean")
    device = sample["video_latents"].device
    _synchronize_profile(device)
    if profile_memory and device.type != "cpu":
        torch.get_device_module(device).reset_peak_memory_stats(device)
    solve_begin = time.monotonic()
    source_video = sample["video_latents"].float()
    device, dtype = source_video.device, next(model.parameters()).dtype
    start = 1 + block * sample["block_size"]
    stop = min(start + sample["block_size"], source_video.shape[1])
    if block < 0 or start >= stop:
        raise ValueError("Requested prediction block is outside the sample")
    # Measured future is supervision, never a generation input, including the
    # duplicated clean-history stream retained by the legacy packing format.
    reference_video = source_video.clone()
    reference_video[:, start:] = 0
    video_active = torch.zeros_like(reference_video, dtype=torch.bool)
    video_active[:, start:stop] = True
    has_action = mode != "video"
    channels = action_active = reference_action = None
    if has_action:
        channels = sample["action_mask"].to(device=device, dtype=torch.bool).expand_as(sample["action"])
        reference_action = sample["action"].to(device=device, dtype=torch.float32).masked_fill(~channels, 0)
        if mode == "inverse_dynamics":
            reference_action = torch.zeros_like(reference_action)
        action_frames = torch.as_tensor(sample["action_frame_ids"], device=device)
        action_active = ((action_frames >= start) & (action_frames < stop))[:, None] & channels
    sid = block if seed_block is None else seed_block
    seeds = {name: noise_seed(seed, sid, name) for name in (("video", "action") if has_action else ("video",))}

    def noise(name, reference):
        if initial_noise is None:
            return torch.randn(
                reference.shape,
                dtype=torch.float32,
                device=device,
                generator=torch.Generator(device=device).manual_seed(seeds[name]),
            )
        result = initial_noise[name].to(device=device, dtype=torch.float32)
        if result.shape != reference.shape or not torch.isfinite(result).all():
            raise ValueError(f"Initial {name} noise must be finite and match its payload")
        return result

    video = torch.where(video_active, noise("video", reference_video), reference_video)
    action = reference_action
    branches = []
    for text_ids in [None, negative_text_ids] if guidance != 1 else [None]:
        branch_sample = dict(sample, video_latents=reference_video)
        if mode == "inverse_dynamics":
            branch_sample["action"] = reference_action
            branch_sample["condition_frame_indexes_vision"] = list(range(source_video.shape[1]))
            branch_sample["condition_frame_indexes_action"] = []
        else:
            branch_sample["condition_frame_indexes_vision"] = list(range(start))
        if text_ids is not None:
            branch_sample["text_token_ids"] = text_ids
            branch_sample["token_length"] += len(text_ids) - len(sample["text_token_ids"])
        branches.append(dict(sample=branch_sample, text_cache=None, action_condition_cache=None, video_cache=None))
    calls = dict.fromkeys(("text", "action_condition", "video", "action"), 0)
    seconds = {}

    def inputs_for(branch, current_video, current_action, video_time, action_time):
        vt = torch.zeros(source_video.shape[1], device=device, dtype=torch.float32)
        vt[start:stop] = video_time
        options = dict(position_options or {})
        if has_action:
            at = torch.zeros(len(reference_action), device=device, dtype=torch.float32)
            at[(action_frames >= start) & (action_frames < stop)] = action_time
            options["action_timesteps"] = [at]
        if mode == "inverse_dynamics":
            options.update(video_condition_payloads=[current_video], video_condition_timesteps=[vt])
        inputs, spans, _ = pack_model_inputs(
            [branch["sample"]],
            [current_video],
            [current_action] if has_action else None,
            timesteps=[vt],
            has_action=has_action,
            patch_size=model.config.patch_size,
            **options,
        )
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=dtype)
        inputs["task_modes"] = (mode,)
        return inputs, spans[0]

    def invoke(branch, operation, current_video, current_action, video_time, action_time):
        inputs, spans = inputs_for(branch, current_video, current_action, video_time, action_time)
        calls[operation] += 1
        output = model(
            **inputs,
            expert_operation=operation,
            attention_backend=attention_backend,
            text_cache=branch["text_cache"],
            video_cache=branch["video_cache"],
            action_condition_cache=branch["action_condition_cache"],
        )
        return output, spans

    def timed(name, fn):
        result, seconds[name] = _timed_profile(device, fn)
        return result

    def prefill_text():
        for branch in branches:
            output, _ = invoke(branch, "text", video, action, 1000.0, 0.0)
            branch["text_cache"] = output.text_cache

    timed("text_prefill", prefill_text)
    if has_action:

        def prefill_conditions():
            for branch in branches:
                output, _ = invoke(branch, "action_condition", video, action, 1000.0, 0.0)
                branch["action_condition_cache"] = output.action_condition_cache

        timed("action_condition_prefill", prefill_conditions)

    def velocity(operation, current, timestep):
        use_cfg = guidance != 1 and (
            guidance_interval is None or guidance_interval[0] < float(timestep) < guidance_interval[1]
        )
        predictions = []
        for branch in branches[: 2 if use_cfg else 1]:
            output, spans = invoke(
                branch,
                operation,
                current if operation == "video" else video,
                current if operation == "action" else action,
                timestep if operation == "video" else video_stop_sigma * 1000,
                timestep if operation == "action" else 0.0,
            )
            prediction = (
                unpatchify_video(output.video[spans.current_video_rows], video.shape, model.config.patch_size)
                if operation == "video"
                else output.action[spans.action_rows]
            )
            predictions.append(
                prediction.float().masked_fill(~(video_active if operation == "video" else action_active), 0)
            )
        conditional = predictions[0]
        if not use_cfg:
            return conditional
        guided = predictions[1] + guidance * (conditional - predictions[1])
        if normalize_cfg:
            guided *= (torch.norm(conditional) / (torch.norm(guided) + 1e-8)).clamp(0, 1)
        return guided

    def unipc(initial, active, reference, count, operation):
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(count, device=device, shift=float(shift))
        latent = initial
        for timestep in scheduler.timesteps:
            prediction = velocity(operation, latent, timestep)
            latent = scheduler.step(prediction.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False)[0][
                0
            ].float()
            latent = torch.where(active, latent, reference)
            if not torch.isfinite(latent).all():
                raise ValueError("Nonfinite triple sampling state")
        return latent

    if mode == "inverse_dynamics":

        def partial_video():
            latent = video
            schedule = torch.linspace(1.0, float(video_stop_sigma), partial_video_steps + 1, device=device)
            for sigma, next_sigma in zip(schedule[:-1], schedule[1:], strict=True):
                latent = latent + (next_sigma - sigma) * velocity("video", latent, sigma * 1000)
                latent = torch.where(video_active, latent, reference_video)
                if not torch.isfinite(latent).all():
                    raise ValueError("Nonfinite partial-video state")
            return latent

        video = timed("partial_video", partial_video)

        def prefill_video():
            for branch in branches:
                output, _ = invoke(branch, "video", video, action, video_stop_sigma * 1000, 0.0)
                branch["video_cache"] = output.video_cache

        # The last Euler forward encoded the pre-update latent. Encode the
        # final x_sigma again, including its actual nonzero noise timestep.
        timed("video_cache_prefill", prefill_video)
        action = torch.where(action_active, noise("action", reference_action), reference_action)
        action = timed(
            "action_denoise", lambda: unipc(action, action_active, reference_action, action_steps, "action")
        )
    else:
        video = timed("video_denoise", lambda: unipc(video, video_active, reference_video, video_steps, "video"))
    _synchronize_profile(device)
    solver_seconds = time.monotonic() - solve_begin
    return dict(
        video_latents=video,
        action=action,
        seeds=seeds,
        video_is_clean=mode != "inverse_dynamics",
        video_sigma=float(video_stop_sigma) if mode == "inverse_dynamics" else 0.0,
        action_is_prediction=mode == "inverse_dynamics",
        expert_calls=calls,
        stage_seconds=seconds,
        solver_seconds=solver_seconds,
        memory_profile=_peak_memory_profile(device, profile_memory, "solver"),
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
    video_role: str = "generated"
    action_role: str = "generated"
    profile: dict = field(default_factory=dict)


def _times(count, fps, stride, offset):
    return offset + torch.arange(count, dtype=torch.float64) * stride / fps


@torch.no_grad()
def teacher_forcing(pipe, sample):
    """Generate each block using measured clean history and measured block state."""
    if sample["mode"] == "video":
        return _teacher_forcing_video(pipe, sample)
    triple = getattr(pipe.model.config, "expert_layout", "dual") == "triple"
    if triple:
        _synchronize_profile(pipe.device)
        pipeline_begin = time.monotonic()
        pack, encode_seconds, encode_memory = _encode_pack_profile(pipe, sample)
        encoded = pack[0]
    else:
        encoded = pipe.vae.encode_pack([sample])[0]
    for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
        if key in encoded:
            encoded[key] = encoded[key].to(pipe.device)
    noisy_future = triple and sample["mode"] == "inverse_dynamics"
    has_action = "action" in sample
    block = pipe.geometry.actions_per_block
    frames = block // pipe.geometry.video_stride
    real = sample["real_action_steps"]
    actions, videos, records = [], [sample["video"][:, :1].cpu()], []
    for index in range((real + block - 1) // block):
        if triple:
            _synchronize_profile(pipe.device)
        begin = time.monotonic()
        generated = sample_block(pipe.model, encoded, index, seed_block=index, **pipe.sampling_options)
        valid = min(block, real - index * block)
        if has_action:
            physical = pipe.processor.decode(
                generated["action"].cpu(), sample["anchors"].repeat_interleave(block, 0), sample["action_mask"]
            )
            actions.append(physical[index * block : index * block + valid])
        decode_seconds = None
        if not noisy_future:
            if triple:
                pixels, decode_seconds = _timed_profile(
                    pipe.device, lambda latent=generated["video_latents"]: pipe.vae.decode(latent)
                )
            else:
                pixels = pipe.vae.decode(generated["video_latents"])
            videos.append(pixels[:, 1 + index * frames : 1 + index * frames + valid // pipe.geometry.video_stride])
        if triple:
            _synchronize_profile(pipe.device)
        chunk_seconds = time.monotonic() - begin
        records.append(
            dict(
                block=index,
                valid_steps=valid,
                seeds=generated["seeds"],
                anchor=sample["anchors"][index].tolist() if has_action else None,
                history_blocks=min(1 + index * encoded["block_size"], pipe.history),
                input_pixel_frames=sample["video"].shape[1],
                latent_shape=list(encoded["video_latents"].shape),
                padded_action_steps=len(sample["action"]) - real if has_action else 0,
                temporal_padding_frames=sample["video"].shape[1] - (1 + real // pipe.geometry.video_stride),
                seconds=chunk_seconds,
                feedback="measured history and measured block state",
                **{
                    name: generated[name]
                    for name in (
                        "expert_calls",
                        "stage_seconds",
                        "video_sigma",
                        "video_is_clean",
                        "action_is_prediction",
                    )
                    if name in generated
                },
            )
        )
        if triple:
            records[-1]["profile"] = _chunk_profile(
                pipe.device,
                generated,
                chunk_seconds,
                decode_seconds,
                pipe.sampling_options.get("profile_memory", True),
            )
    fps = sample["conditioning_fps_action"]
    offset = float(sample["timestamps"][0])
    width = sample["action"].shape[-1] if has_action else pipe.processor.template.width
    actions = torch.cat(actions) if actions else torch.empty(0, width)
    video = torch.cat(videos, dim=1)
    # Preserve measured targets before normalization clipping for honest GT curves.
    gt = sample["ground_truth_actions"][:real].cpu() if has_action else None
    reconstruction_seconds = None
    if noisy_future:
        reconstruction = None
    elif triple:
        reconstruction, reconstruction_seconds = _timed_profile(
            pipe.device, lambda: pipe.vae.decode(encoded["video_latents"])
        )
        reconstruction = reconstruction[:, : video.shape[1]]
    else:
        reconstruction = pipe.vae.decode(encoded["video_latents"])[:, : video.shape[1]]
    profile = (
        _pipeline_profile(pipe.device, pipeline_begin, encode_seconds, reconstruction_seconds, records, encode_memory)
        if triple
        else {}
    )
    return InferenceResult(
        video,
        actions,
        sample["action_mask"].cpu() if has_action else torch.zeros(width, dtype=torch.bool),
        _times(len(actions), fps, 1, offset + 1 / fps),
        _times(video.shape[1], fps, pipe.geometry.video_stride, offset),
        f"teacher_forcing_{sample['mode']}_measured_history_and_state",
        records,
        sample["video"][:, : video.shape[1]].cpu(),
        gt,
        reconstruction,
        video_fps=fps / pipe.geometry.video_stride,
        identity=sample.get("identity", {}),
        video_role="observation" if noisy_future else "generated",
        action_role="none"
        if not has_action
        else "conditioned"
        if triple and sample["mode"] == "forward_dynamics"
        else "generated",
        profile=profile,
    )


@torch.no_grad()
def _teacher_forcing_video(pipe, sample):
    """Replay a native video-only window without processors or action columns."""
    _synchronize_profile(pipe.device)
    pipeline_begin = time.monotonic()
    pack, encode_seconds, encode_memory = _encode_pack_profile(pipe, sample)
    encoded = pack[0]
    encoded["video_latents"] = encoded["video_latents"].to(pipe.device)
    stride = sample["window"].video_stride
    intervals = sample.get("real_frame_intervals")
    if intervals is None:
        intervals = sample["window"].real_intervals
    real_frames = intervals // stride
    frames_per_block = sample["block_size"] * pipe.geometry.temporal_compression
    videos, records = [sample["video"][:, :1].cpu()], []
    for index in range(math.ceil(real_frames / frames_per_block)):
        _synchronize_profile(pipe.device)
        begin = time.monotonic()
        generated = sample_block(pipe.model, encoded, index, seed_block=index, **pipe.sampling_options)
        pixels, decode_seconds = _timed_profile(
            pipe.device, lambda latent=generated["video_latents"]: pipe.vae.decode(latent)
        )
        valid = min(frames_per_block, real_frames - index * frames_per_block)
        start = 1 + index * frames_per_block
        videos.append(pixels[:, start : start + valid])
        _synchronize_profile(pipe.device)
        chunk_seconds = time.monotonic() - begin
        records.append(
            dict(
                block=index,
                valid_video_frames=valid,
                seeds=generated["seeds"],
                latent_shape=list(encoded["video_latents"].shape),
                seconds=chunk_seconds,
                feedback="measured clean video history",
                expert_calls=generated["expert_calls"],
                stage_seconds=generated["stage_seconds"],
                video_sigma=0.0,
                video_is_clean=True,
                profile=_chunk_profile(
                    pipe.device,
                    generated,
                    chunk_seconds,
                    decode_seconds,
                    pipe.sampling_options.get("profile_memory", True),
                ),
            )
        )
    video = torch.cat(videos, dim=1)
    width = getattr(pipe.model.config, "action_dim", 55)
    fps = sample["conditioning_fps"]
    offset = float(sample["timestamps"][0])
    reconstruction, reconstruction_seconds = _timed_profile(
        pipe.device, lambda: pipe.vae.decode(encoded["video_latents"])
    )
    profile = _pipeline_profile(
        pipe.device, pipeline_begin, encode_seconds, reconstruction_seconds, records, encode_memory
    )
    return InferenceResult(
        video=video,
        actions=torch.empty(0, width),
        action_mask=torch.zeros(width, dtype=torch.bool),
        action_timestamps=torch.empty(0, dtype=torch.float64),
        video_timestamps=_times(video.shape[1], fps, 1, offset),
        semantics="teacher_forcing_video_measured_history",
        blocks=records,
        ground_truth_video=sample["video"][:, : video.shape[1]].cpu(),
        reconstruction=reconstruction[:, : video.shape[1]],
        video_fps=fps,
        identity=sample.get("identity", {}),
        action_role="none",
        profile=profile,
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
    write_video(output / f"{result.video_role}.mp4", result.video, fps)
    for name, video in (("ground_truth", result.ground_truth_video), ("vae_reconstruction", result.reconstruction)):
        if video is not None:
            write_video(output / f"{name}.mp4", video, fps)
    if len(result.actions) and result.action_mask.any():
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
        video_role=result.video_role,
        action_role=result.action_role,
        profile=result.profile,
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
