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
from veomni.data.causal_cosmos3.transforms.layout import format_prompt
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


def _resolve_normalize_cfg(value):
    """Accept legacy bool (False->0.0, True->1.0) or soft interpolation rho in [0, 1]."""
    if type(value) is bool:
        return value, (1.0 if value else 0.0)
    if isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= float(value) <= 1.0:
        rho = float(value)
        return rho, rho
    raise ValueError("normalize_cfg must be a boolean or a finite float in [0, 1]")


def _apply_cfg_modality(cond, uncond, guidance_scale, rho, has_active):
    """Apply per-modality CFG with soft norm interpolation and measure pre/post guidance residuals."""
    diff = cond - uncond
    vel_cfg = uncond + guidance_scale * diff
    if rho == 0.0:
        scale_val = 1.0
        vel_out = vel_cfg
    else:
        hard_scale = (torch.norm(cond) / (torch.norm(vel_cfg) + 1e-8)).clamp(0.0, 1.0)
        scale_t = hard_scale if rho == 1.0 else (1.0 - rho) + rho * hard_scale
        scale_val = float(scale_t.item())
        vel_out = vel_cfg * scale_t
    if not has_active:
        return vel_out, None
    raw_delta = vel_cfg - cond
    post_delta = vel_out - cond
    cond_norm = float(torch.norm(cond).item())
    pre_norm = float(torch.norm(vel_cfg).item())
    post_norm = float(torch.norm(vel_out).item())
    delta_pre_sq = float(torch.dot(raw_delta, raw_delta).item())
    delta_pre_norm = math.sqrt(delta_pre_sq)
    delta_post_norm = float(torch.norm(post_delta).item())
    if delta_pre_sq > 1e-20:
        retention_ratio = delta_post_norm / delta_pre_norm
        aligned_retention = float(torch.dot(post_delta, raw_delta).item()) / delta_pre_sq
    else:
        retention_ratio = 1.0
        aligned_retention = 1.0
    diff_norm = float(torch.norm(diff).item())
    cond_diff_denom = cond_norm * diff_norm
    cond_diff_cos = (
        max(-1.0, min(1.0, float(torch.dot(cond, diff).item()) / cond_diff_denom))
        if cond_diff_denom > 1e-20
        else 0.0
    )
    return vel_out, dict(
        cond_norm=cond_norm,
        pre_cfg_norm=pre_norm,
        post_cfg_norm=post_norm,
        scale_factor=scale_val,
        delta_pre_norm=delta_pre_norm,
        delta_post_norm=delta_post_norm,
        delta_retention_ratio=retention_ratio,
        delta_aligned_retention=aligned_retention,
        cond_diff_cosine=cond_diff_cos,
    )


def _summarize_cfg_modality(items, modality):
    rows = [item[modality] for item in items if item.get(modality) is not None]
    if not rows:
        return None
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


class CausalCosmos3Pipeline:
    """Reuse native weights, preprocessing and packing for explicit generation."""

    def __init__(self, model, vae, processor, tokenize, config, *, device="cpu"):
        self.model = model.eval().requires_grad_(False)
        self.vae, self.processor, self.tokenize = vae, processor, tokenize
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        self.source = self.config["source"]
        self.geometry = Geometry(**config.get("geometry", {}))
        options = self.config.setdefault("sampling", {})
        enable_cfg = options.get("enable_cfg", False)
        if type(enable_cfg) is not bool:
            raise ValueError("sampling.enable_cfg must be a boolean")
        fixed_step = options.get("fixed_step_sampler_config") or getattr(
            model.config, "fixed_step_sampler_config", None
        )
        if fixed_step is not None:
            options["fixed_step_sampler_config"] = copy.deepcopy(fixed_step)
        if not enable_cfg:
            # CFG is off by default (effective guidance = 1.0), including distilled checkpoints.
            guidance = video_guidance = action_guidance = 1.0
        else:
            guidance = float(3.0 if options.get("guidance") is None else options["guidance"])
            video_guidance = float(guidance if options.get("video_guidance") is None else options["video_guidance"])
            action_guidance = float(guidance if options.get("action_guidance") is None else options["action_guidance"])
        options.update(
            enable_cfg=enable_cfg,
            guidance=guidance,
            video_guidance=video_guidance,
            action_guidance=action_guidance,
        )
        raw_enable_txt_steering = options.get("enable_txt_steering")
        if raw_enable_txt_steering is None:
            enable_txt_steering = bool(getattr(model.config, "enable_txt_steering", False))
        elif type(raw_enable_txt_steering) is bool:
            enable_txt_steering = raw_enable_txt_steering
        else:
            raise ValueError("sampling.enable_txt_steering must be None or a boolean")
        txt_steering_method = (
            getattr(model.config, "txt_steering_method", "multiplication")
            if options.get("txt_steering_method") is None
            else options["txt_steering_method"]
        )
        if txt_steering_method not in ("multiplication", "addition"):
            raise ValueError("txt_steering_method must be 'multiplication' or 'addition'")
        txt_steering_scale = (
            getattr(model.config, "txt_steering_scale", 1.3)
            if options.get("txt_steering_scale") is None
            else options["txt_steering_scale"]
        )
        txt_steering_min_timestep = (
            getattr(model.config, "txt_steering_min_timestep", 250.0)
            if options.get("txt_steering_min_timestep") is None
            else options["txt_steering_min_timestep"]
        )
        txt_steering_max_timestep = (
            getattr(model.config, "txt_steering_max_timestep", 1000.0)
            if options.get("txt_steering_max_timestep") is None
            else options["txt_steering_max_timestep"]
        )
        if (
            any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (txt_steering_scale, txt_steering_min_timestep, txt_steering_max_timestep)
            )
            or float(txt_steering_scale) <= 0.0
            or not 0.0 <= float(txt_steering_min_timestep) < float(txt_steering_max_timestep) <= 1000.0
        ):
            raise ValueError(
                "txt_steering_scale must be finite positive and 0 <= txt_steering_min_timestep < txt_steering_max_timestep <= 1000"
            )
        txt_steering_scale = float(txt_steering_scale)
        txt_steering_min_timestep = float(txt_steering_min_timestep)
        txt_steering_max_timestep = float(txt_steering_max_timestep)
        options.update(
            enable_txt_steering=enable_txt_steering,
            txt_steering_method=txt_steering_method,
            txt_steering_scale=txt_steering_scale,
            txt_steering_min_timestep=txt_steering_min_timestep,
            txt_steering_max_timestep=txt_steering_max_timestep,
        )
        negative_prompt = options.get("negative_prompt")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError("sampling.negative_prompt must be None or a string")
        negative_text_ids = (
            torch.as_tensor(tokenize(negative_prompt), dtype=torch.long)
            if enable_cfg and negative_prompt is not None and negative_prompt.strip()
            else None
        )
        normalize_cfg, _ = _resolve_normalize_cfg(options.get("normalize_cfg", False))
        options["normalize_cfg"] = normalize_cfg
        self.history = options.get("history", 32)
        if type(self.history) is not int or self.history < 1:
            raise ValueError("history must be a positive integer")
        self.sampling_options = dict(
            steps=options.get("steps", 20),
            shift=float(options.get("shift", 1.0)),
            enable_cfg=enable_cfg,
            guidance=guidance,
            video_guidance=video_guidance,
            action_guidance=action_guidance,
            guidance_interval=options.get("guidance_interval"),
            normalize_cfg=normalize_cfg,
            enable_txt_steering=enable_txt_steering,
            txt_steering_method=txt_steering_method,
            txt_steering_scale=txt_steering_scale,
            txt_steering_min_timestep=txt_steering_min_timestep,
            txt_steering_max_timestep=txt_steering_max_timestep,
            fixed_step_sampler_config=fixed_step,
            seed=options.get("seed", 42),
            attention_backend=config.get("attention_backend", "grouped"),
            negative_text_ids=negative_text_ids,
            tokenize=tokenize,
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
    enable_cfg=None,
    guidance=1.0,
    video_guidance=None,
    action_guidance=None,
    guidance_interval=None,
    normalize_cfg=False,
    enable_txt_steering=None,
    txt_steering_method=None,
    txt_steering_scale=None,
    txt_steering_min_timestep=None,
    txt_steering_max_timestep=None,
    fixed_step_sampler_config=None,
    seed=42,
    seed_block=None,
    negative_text_ids=None,
    tokenize=None,
    attention_backend="grouped",
    initial_noise=None,
    position_options=None,
):
    """Sample one block; the clean sample supplies only causally visible context.

    Teacher forcing keeps the original clean video across calls. Rollout supplies
    decoded/re-encoded historical pixels and zero future pixels instead.
    ``initial_noise`` contains actual FP32 video/action tensors for strict oracles.
    """
    if enable_cfg is not None and type(enable_cfg) is not bool:
        raise ValueError("enable_cfg must be a boolean")
    if enable_txt_steering is not None and type(enable_txt_steering) is not bool:
        raise ValueError("enable_txt_steering must be a boolean")
    txt_steering_overrides = {}
    if enable_txt_steering is not None:
        txt_steering_overrides["enable_txt_steering"] = enable_txt_steering
    if txt_steering_method is not None:
        if txt_steering_method not in ("multiplication", "addition"):
            raise ValueError("txt_steering_method must be 'multiplication' or 'addition'")
        txt_steering_overrides["txt_steering_method"] = txt_steering_method
    for key, raw in (
        ("txt_steering_scale", txt_steering_scale),
        ("txt_steering_min_timestep", txt_steering_min_timestep),
        ("txt_steering_max_timestep", txt_steering_max_timestep),
    ):
        if raw is not None:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                raise ValueError(f"{key} must be a finite number")
            txt_steering_overrides[key] = float(raw)
    if txt_steering_overrides:
        eff_scale = txt_steering_overrides.get(
            "txt_steering_scale", float(getattr(model.config, "txt_steering_scale", 1.3))
        )
        eff_min = txt_steering_overrides.get(
            "txt_steering_min_timestep", float(getattr(model.config, "txt_steering_min_timestep", 250.0))
        )
        eff_max = txt_steering_overrides.get(
            "txt_steering_max_timestep", float(getattr(model.config, "txt_steering_max_timestep", 1000.0))
        )
        if eff_scale <= 0.0 or not 0.0 <= eff_min < eff_max <= 1000.0:
            raise ValueError(
                "txt_steering_scale must be finite positive and 0 <= txt_steering_min_timestep < txt_steering_max_timestep <= 1000"
            )
    _, cfg_norm_rho = _resolve_normalize_cfg(normalize_cfg)
    vg = 1.0 if enable_cfg is False else float(guidance if video_guidance is None else video_guidance)
    ag = 1.0 if enable_cfg is False else float(guidance if action_guidance is None else action_guidance)
    if steps < 1 or not all(math.isfinite(x) and x > 0 for x in (shift, guidance, vg, ag)):
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
    cfg_enabled = (vg != 1.0 or ag != 1.0) if enable_cfg is None else (enable_cfg and (vg != 1.0 or ag != 1.0))
    if cfg_enabled and negative_text_ids is None:
        if tokenize is None:
            raise ValueError("CFG requires explicit negative/empty-prompt token IDs or a tokenize callback")
        negative_text_ids = torch.as_tensor(tokenize(format_prompt(sample, unconditional=True)), dtype=torch.long)
    dtype = next(model.parameters()).dtype
    v_len = video.numel()

    def predict(current, timestep, text_ids):
        branch = dict(sample)
        if text_ids is not None:
            branch["text_token_ids"] = text_ids
            branch["token_length"] += len(text_ids) - len(sample["text_token_ids"])
            if "text_steering_mask" in sample:
                branch["text_steering_mask"] = torch.zeros(len(text_ids), dtype=torch.bool, device=text_ids.device)
        frame_times = torch.zeros(video.shape[1], device=video.device, dtype=torch.float32)
        frame_times[start:stop] = timestep.float()
        inputs, spans, _ = pack_model_inputs(
            [branch],
            [current[:v_len].reshape_as(video)],
            [current[v_len:].reshape_as(action)],
            timesteps=[frame_times],
            patch_size=model.config.patch_size,
            **(position_options or {}),
        )
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            inputs[key] = inputs[key].to(dtype=dtype)
        inputs["action_tokens"] = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        outputs = model(**inputs, attention_backend=attention_backend, **txt_steering_overrides)
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
    has_active_video = bool(vm.any().item())
    has_active_action = bool(am.any().item())
    cfg_step_records = []
    for timestep in scheduler.timesteps:
        current = torch.where(active, latent, reference)
        velocity = predict(current, timestep, None)
        use_cfg = cfg_enabled and (
            guidance_interval is None or guidance_interval[0] < float(timestep) < guidance_interval[1]
        )
        if use_cfg:
            unconditional = predict(current, timestep, negative_text_ids)
            conditional = velocity
            vel_v, stats_v = _apply_cfg_modality(
                conditional[:v_len], unconditional[:v_len], vg, cfg_norm_rho, has_active_video
            )
            vel_a, stats_a = _apply_cfg_modality(
                conditional[v_len:], unconditional[v_len:], ag, cfg_norm_rho, has_active_action
            )
            velocity = torch.cat((vel_v, vel_a))
            cfg_step_records.append(dict(timestep=float(timestep), video=stats_v, action=stats_a))
        latent = scheduler.step(velocity, timestep, current.unsqueeze(0), return_dict=False, **step_options)[0]
        latent = latent.squeeze(0).float()
        latent = torch.where(active, latent, reference)
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite sampling state")
    cfg_diagnostics = (
        dict(
            normalize_cfg_rho=cfg_norm_rho,
            guided_steps=len(cfg_step_records),
            video=_summarize_cfg_modality(cfg_step_records, "video"),
            action=_summarize_cfg_modality(cfg_step_records, "action"),
            steps=cfg_step_records,
        )
        if cfg_step_records
        else None
    )
    return dict(
        video_latents=latent[:v_len].reshape_as(video),
        action=latent[v_len:].reshape_as(action).masked_fill(~channels, 0),
        seeds=seeds,
        cfg_diagnostics=cfg_diagnostics,
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
        record = dict(
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
        if generated.get("cfg_diagnostics") is not None:
            record["cfg_diagnostics"] = generated["cfg_diagnostics"]
        records.append(record)
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


def rollout_sample(
    initial, state, latent_history, *, source, processor, geometry, tokenize, history_blocks, valid
):
    """Build synthetic unknown targets directly from historical video latents."""
    latent_block = geometry.actions_per_block // geometry.actions_per_latent
    if latent_history.ndim != 4 or latent_history.shape[1] < 1:
        raise ValueError("Latent history must be a [C, T, H, W] tensor with at least the initial latent")
    context, remainder = divmod(latent_history.shape[1] - 1, latent_block)
    if remainder or not 1 <= valid <= geometry.actions_per_block:
        raise ValueError("Latent history must end on a complete action-block boundary")
    total = (context + 1) * geometry.actions_per_block
    state = processor.template.sanitize(state.cpu(), initial.state_mask)
    if not torch.isfinite(state).all():
        raise ValueError("Nonfinite physical feedback state")
    raw = dict(
        video=initial.image[:, :0],
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
    sample.pop("video")
    future_latents = latent_history.new_zeros((latent_history.shape[0], latent_block, *latent_history.shape[2:]))
    sample["video_latents"] = torch.cat((latent_history, future_latents), dim=1)
    sample["condition_frame_indexes_vision"] = list(range(1 + context * latent_block))
    return sample, context


@torch.no_grad()
def rollout(pipe, initial):
    """Model-only open-loop rollout: generated video latents and absolute action feedback."""
    size = pipe.geometry.actions_per_block
    new_frames = size // pipe.geometry.video_stride
    latent_block = size // pipe.geometry.actions_per_latent
    context_limit = (pipe.history + latent_block - 1) // latent_block
    history_latent_limit = 1 + context_limit * latent_block
    state = initial.state.cpu().clone()
    initial_image = initial.image.cpu().clone()
    if initial_image.ndim != 4 or initial_image.shape[:2] != (3, 1):
        raise ValueError("Initial rollout observation requires a single RGB frame [3, 1, H, W]")
    encode_begin = time.monotonic()
    initial_latent = pipe.vae.encode_pack([{"video": initial_image}])[0]["video_latents"].to(pipe.device)
    encode_seconds = time.monotonic() - encode_begin
    if initial_latent.ndim != 4 or initial_latent.shape[1] != 1:
        raise ValueError("Encoded initial observation must contain exactly one video latent frame")
    latent_history = initial_latent
    full_latents = [initial_latent.cpu()]
    actions, records = [], []
    for index, start in enumerate(range(0, initial.horizon, size)):
        begin = time.monotonic()
        valid = min(size, initial.horizon - start)
        sample, context = rollout_sample(
            initial,
            state,
            latent_history,
            source=pipe.source,
            processor=pipe.processor,
            geometry=pipe.geometry,
            tokenize=pipe.tokenize,
            history_blocks=pipe.history,
            valid=valid,
        )
        for key in ("action", "states", "action_mask", "state_mask"):
            sample[key] = sample[key].to(pipe.device)
        generated = sample_block(pipe.model, sample, context, seed_block=index, **pipe.sampling_options)
        normalized = generated["action"][-size:].cpu()
        absolute = pipe.processor.decode(normalized, state.expand(size, -1), initial.action_mask)
        if not torch.isfinite(absolute).all():
            raise ValueError("Nonfinite decoded action")
        anchor = state.clone()
        actions.append(absolute[:valid])
        # Feedback always selects the final valid action, never a padded tail.
        state = absolute[valid - 1].clone()
        added_latents = generated["video_latents"][:, -latent_block:]
        if not torch.isfinite(added_latents).all():
            raise ValueError("Nonfinite generated video latents")
        full_latents.append(added_latents.cpu())
        latent_history = torch.cat((latent_history, added_latents), dim=1)
        latent_history = latent_history[:, -history_latent_limit:].contiguous()
        step_seconds = time.monotonic() - begin + (encode_seconds if index == 0 else 0.0)
        record = dict(
            block=index,
            valid_steps=valid,
            seeds=generated["seeds"],
            anchor=anchor.tolist(),
            feedback_state=state.tolist(),
            history_blocks=min(1 + context * latent_block, pipe.history),
            context_blocks=context,
            history_latent_frames=latent_history.shape[1],
            history_pixel_frames=1 + (latent_history.shape[1] - 1) * pipe.geometry.temporal_compression,
            latent_shape=list(sample["video_latents"].shape),
            padded_action_steps=size - valid,
            temporal_padding_frames=new_frames - valid // pipe.geometry.video_stride,
            state_clip_fraction=sample["state_clip_fraction"].tolist(),
            seconds=step_seconds,
            feedback="generated video latents and final valid absolute action",
        )
        if generated.get("cfg_diagnostics") is not None:
            record["cfg_diagnostics"] = generated["cfg_diagnostics"]
        records.append(record)
    decode_begin = time.monotonic()
    pixels = pipe.vae.decode(torch.cat(full_latents, dim=1))
    decode_seconds = time.monotonic() - decode_begin
    if records:
        records[-1]["vae_decode_seconds"] = decode_seconds
        records[-1]["seconds"] += decode_seconds
    if pixels.shape != (3, 1 + len(records) * new_frames, *initial_image.shape[-2:]):
        raise ValueError("Decoded pixels do not match the native input geometry")
    video = torch.cat((initial_image, pixels[:, 1 : 1 + initial.horizon // pipe.geometry.video_stride]), dim=1)
    return InferenceResult(
        video,
        torch.cat(actions),
        initial.action_mask.cpu(),
        _times(initial.horizon, initial.fps, 1, initial.timestamp + 1 / initial.fps),
        _times(video.shape[1], initial.fps, pipe.geometry.video_stride, initial.timestamp),
        "model_only_open_loop_action_as_state_latent_feedback",
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
    cfg_blocks = [block["cfg_diagnostics"] for block in result.blocks if block.get("cfg_diagnostics") is not None]
    if cfg_blocks:
        metadata["cfg_diagnostics_summary"] = dict(
            normalize_cfg_rho=cfg_blocks[0]["normalize_cfg_rho"],
            blocks_recorded=len(cfg_blocks),
            guided_steps_per_block=[block["guided_steps"] for block in cfg_blocks],
            video=_summarize_cfg_modality(cfg_blocks, "video"),
            action=_summarize_cfg_modality(cfg_blocks, "action"),
        )
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return output
