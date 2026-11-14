# SPDX-License-Identifier: OpenMDW-1.1
"""Independent causal Cosmos3 teacher forcing and pixel-feedback Policy/IDM rollout.

CausalCosmos3Pipeline loads complete VeOmni DCP/HF weights and native data assets.
The same block sampler serves FD/ID/Policy teacher forcing and Policy/IDM rollout.
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
from veomni.distributed.torch_parallelize import _to_empty_preserving_nonpersistent_buffers
from veomni.models import build_foundation_model
from veomni.models.checkpoint.weights import load_model_weights

from .cosmos3_core.fixed_step import FixedStepSampler
from .cosmos3_core.inference_cache import ObservationKVCache
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


def _load_dcp_model(model, checkpoint, device):
    """Load a native VeOmni model/ckpt directory without training state or collectives."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(checkpoint)
    metadata = reader.read_metadata().state_dict_metadata
    expected = {f"model.{name}": tensor for name, tensor in model.state_dict().items()}
    missing = sorted(expected.keys() - metadata.keys())
    if missing:
        raise ValueError(f"Missing Cosmos3 weights in DCP: {missing}")
    unexpected = sorted(metadata.keys() - expected.keys())
    if unexpected:
        raise ValueError(f"Unexpected Cosmos3 weights in DCP: {unexpected}; use the model/ckpt directory")
    for name, tensor in expected.items():
        stored = metadata[name]
        if not isinstance(stored, TensorStorageMetadata) or tuple(stored.size) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {getattr(stored, 'size', None)}, model {tuple(tensor.shape)}"
            )
    _to_empty_preserving_nonpersistent_buffers(model, device)
    state = {"model": model.state_dict()}
    dcp.load(state_dict=state, storage_reader=reader, no_dist=True)
    model.load_state_dict(state["model"], strict=True)
    model.config.weight_loading = {}
    model.weight_loading_report = {
        "mode": "complete",
        "format": "dcp",
        "loaded_count": sum(1 for _ in model.parameters()),
        "initialized": [],
        "skipped": [],
    }


def load_trained_model(checkpoint, *, device="cpu", dtype=torch.bfloat16, ops_implementation=None):
    """Load complete trained weights using the checkpoint's own config.json."""
    checkpoint = pathlib.Path(checkpoint)
    is_dcp = (checkpoint / ".metadata").is_file()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing trained model configuration: {config_path}. "
            "Copy the matching training run's model_assets/config.json into the checkpoint directory."
        )
    config = Cosmos3Config.from_json_file(str(config_path))
    if config.action_interface != "shared" or config.action_dim != 55 or config.state_dim != 55:
        raise ValueError("Causal action inference requires the trained shared 55D action/state configuration")
    model = _empty_model(config, dtype, ops_implementation)
    if is_dcp:
        _load_dcp_model(model, checkpoint, device)
    else:
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
        self.kv_cache = options.setdefault("kv_cache", True)
        if type(self.kv_cache) is not bool:
            raise ValueError("sampling.kv_cache must be a boolean")
        self.observation_mode = options.setdefault("observation_mode", "streaming")
        if self.observation_mode not in ("streaming", "window"):
            raise ValueError("sampling.observation_mode must be streaming or window")
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
            blockwise_noise=True,
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
        if config.get("model_config") is not None:
            raise ValueError(
                "model_config is no longer supported in the inference YAML. "
                "Save the trained configuration as config.json inside the checkpoint directory."
            )
        device = config.get("device", "cpu")
        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        model = load_trained_model(
            checkpoint,
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

    def rollout(
        self, *, dataset=None, input_npz=None, episode=0, start=0, horizon=32, fps=None, prompt=None, task="policy"
    ):
        validate_rollout_task(task)
        if (dataset is None) == (input_npz is None):
            raise ValueError("Choose exactly one dataset or initial NPZ input")
        common = dict(source=self.source, processor=self.processor, geometry=self.geometry, prompt=prompt)
        if input_npz is not None:
            if fps is None:
                raise ValueError("An initial NPZ requires explicit --fps")
            initial = read_npz_observation(input_npz, fps=fps, horizon=horizon, **common)
        else:
            initial = read_initial_observation(dataset, episode, start, horizon, **common)
        result = rollout(self, initial, task=task)
        self.last_input_identity = initial.identity
        return result


# Block sampling


def create_observation_caches(pipe, initial_latent, *, text_ids, conditioning_fps, conditioning_fps_action):
    """Initialize branches from one real latent, text and fixed geometry only."""
    options = pipe.sampling_options
    branches = {"cond": text_ids}
    if options["guidance"] != 1:
        branches["uncond"] = options["negative_text_ids"]
    return {
        name: ObservationKVCache(
            pipe.model,
            initial_latent,
            text_ids=text,
            block_size=pipe.geometry.actions_per_block // pipe.geometry.actions_per_latent,
            actions_per_block=pipe.geometry.actions_per_block,
            history=pipe.history,
            conditioning_fps=conditioning_fps,
            conditioning_fps_action=conditioning_fps_action,
            attention_backend=options["attention_backend"],
            position_options=options["position_options"],
        )
        for name, text in branches.items()
    }


def compact_prediction_sample(sample, measured, block, *, actions_per_block, patch_size):
    """Expose only the current measured state and unknown targets to the cache."""
    size = sample["block_size"]
    if measured.shape[1] != (1 if block == 0 else size):
        raise ValueError("Policy replay requires an initial latent, then complete measured blocks")
    video = measured.new_zeros((measured.shape[0], 1 + size, *measured.shape[2:]))
    if block == 0:
        video[:, :1] = measured
    begin = block * actions_per_block
    compact = {
        key: sample[key]
        for key in (
            "mode",
            "text_token_ids",
            "block_size",
            "history_blocks",
            "conditioning_fps",
            "conditioning_fps_action",
        )
    }
    compact.update(
        cache_revision=block,
        video_latents=video,
        action=measured.new_zeros((actions_per_block, sample["action"].shape[1])),
        states=sample["states"][block : block + 1].to(measured.device).clone(),
        action_mask=sample["action_mask"].to(measured.device),
        state_mask=sample["state_mask"].to(measured.device),
        action_frame_ids=[int(frame) for frame in sample["action_frame_ids"][:actions_per_block]],
        state_latent_indexes=[0],
        condition_frame_indexes_vision=[0],
        condition_frame_indexes_action=[],
        action_valid_steps=sample["action_valid_steps"][begin : begin + actions_per_block].clone(),
        num_valid_video_latents=max(1, min(1 + size, sample["num_valid_video_latents"] - block * size)),
    )
    if sample.get("mode") == "forward_dynamics":
        compact["action"] = sample["action"][begin : begin + actions_per_block].to(measured.device).clone()
        compact["condition_frame_indexes_action"] = list(range(actions_per_block))
    patches = ((video.shape[2] + patch_size - 1) // patch_size) * ((video.shape[3] + patch_size - 1) // patch_size)
    compact["token_length"] = len(compact["text_token_ids"]) + 2 * (1 + size) * patches + actions_per_block + 1
    return compact


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
    kv_cache=None,
    blockwise_noise=False,
):
    """Sample one block; the clean sample supplies only causally visible context.

    Teacher forcing keeps the original clean video across calls. Rollout supplies
    decoded/re-encoded historical pixels and zero future pixels instead.
    ``initial_noise`` contains actual FP32 video/action tensors for strict oracles.
    """
    if kv_cache is not None and block != 0:
        raise ValueError("Online cache predicts compact block zero only")
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
    if initial_noise is None and blockwise_noise:
        # Identical target noise regardless of full-window or compact layout.
        video_noise, action_noise = torch.zeros_like(video), torch.zeros_like(action)
        video_noise[:, start:stop] = torch.randn(
            (video.shape[0], stop - start, *video.shape[2:]),
            device=video.device,
            generator=torch.Generator(device=video.device).manual_seed(seeds["video"]),
        )
        action_rows = (af >= start) & (af < stop)
        action_noise[action_rows] = torch.randn(
            (int(action_rows.sum()), action.shape[1]),
            device=video.device,
            generator=torch.Generator(device=video.device).manual_seed(seeds["action"]),
        )
        noises = [video_noise, action_noise]
    elif initial_noise is None:
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
    idm_caches = {}
    if kv_cache is not None and sample.get("mode") == "inverse_dynamics":
        from .cosmos3_core.inference_cache import IDMBlockKVCache

        for name, cache in kv_cache.items():
            cache.validate_branch(
                model,
                sample,
                attention_backend=attention_backend,
                position_options=position_options,
                text_ids=None if name == "cond" else negative_text_ids,
                mode="inverse_dynamics",
            )
            idm_caches[name] = IDMBlockKVCache(cache, sample)

    def predict(current, timestep, text_ids):
        if kv_cache is not None:
            branch_name = "cond" if text_ids is None else "uncond"
            if branch_name not in kv_cache:
                raise ValueError("Initialize and update ObservationKVCache branches before sampling")
            kv_cache[branch_name].validate_branch(
                model,
                sample,
                attention_backend=attention_backend,
                position_options=position_options,
                text_ids=text_ids,
                mode="inverse_dynamics" if idm_caches else "policy",
            )
            velocity_v, velocity_a = torch.zeros_like(video), torch.zeros_like(action)
            current_action = current[video.numel() :].reshape_as(action)
            if idm_caches:
                pred_a = idm_caches[branch_name].predict(current_action, timestep)
            else:
                pred_v, pred_a = kv_cache[branch_name].predict(
                    current[: video.numel()].reshape_as(video)[:, start:stop],
                    current_action,
                    sample["states"],
                    timestep,
                    action_mask=sample["action_mask"],
                    state_mask=sample["state_mask"],
                    revision=sample["cache_revision"],
                )
                velocity_v[:, start:stop] = pred_v.float()
            velocity_a[:] = pred_a.float()
            return torch.cat((velocity_v.flatten(), velocity_a.flatten())).masked_fill(~active, 0)
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
        if blockwise_noise:
            # SDE draws must also be independent of reference/context padding.
            updated = scheduler.step(
                velocity[active], timestep, current[active].unsqueeze(0), return_dict=False, **step_options
            )[0]
            latent = reference.clone()
            latent[active] = updated.squeeze(0).float()
        else:
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
    if pipe.kv_cache and sample["mode"] not in ("policy", "inverse_dynamics"):
        raise ValueError("KV cache supports policy and inverse_dynamics; disable it for forward_dynamics")
    streaming = pipe.observation_mode == "streaming"
    caches = {}
    encoder = pipe.vae.stream_encoder() if streaming else None
    measured_history = []
    pending_observed = None
    if not streaming:
        encoded = pipe.vae.encode_pack([sample])[0]
        for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
            encoded[key] = encoded[key].to(pipe.device)
    block = pipe.geometry.actions_per_block
    frames = block // pipe.geometry.video_stride
    real = sample["real_action_steps"]
    actions, videos, records = [], [sample["video"][:, :1].cpu()], []
    for index in range((real + block - 1) // block):
        begin = time.monotonic()
        if streaming:
            pixel_start = 0 if index == 0 else 1 + (index - 1) * frames
            pixel_stop = 1 + index * frames
            observed = pending_observed
            if observed is None:
                observed = encoder.encode(sample["video"][:, pixel_start:pixel_stop]).to(pipe.device)
            measured_history.append(observed.cpu())
            if pipe.kv_cache and index == 0:
                caches = create_observation_caches(
                    pipe,
                    observed,
                    text_ids=sample["text_token_ids"],
                    conditioning_fps=sample["conditioning_fps"],
                    conditioning_fps_action=sample["conditioning_fps_action"],
                )
            elif pipe.kv_cache:
                for cache in caches.values():
                    cache.update(observed, revision=index)
            encoded = compact_prediction_sample(
                sample, observed, index, actions_per_block=block, patch_size=pipe.model.config.patch_size
            )
            if sample["mode"] == "inverse_dynamics":
                # IDM knows this block's video, but it becomes history only next iteration.
                pending_observed = encoder.encode(sample["video"][:, pixel_stop : pixel_stop + frames]).to(pipe.device)
                encoded["video_latents"][:, 1:] = pending_observed
                encoded["condition_frame_indexes_vision"] = list(range(1 + encoded["block_size"]))
            target_block = 0
            if not pipe.kv_cache:
                encoded, target_block = expand_observation_sample(
                    encoded, torch.cat(measured_history, dim=1).to(pipe.device)
                )
            generated = sample_block(
                pipe.model,
                encoded,
                target_block,
                seed_block=index,
                kv_cache=caches if pipe.kv_cache else None,
                **pipe.sampling_options,
            )
            physical = pipe.processor.decode(
                generated["action"][-block:].cpu(), sample["anchors"][index].expand(block, -1), sample["action_mask"]
            )
            predicted = physical
            # Decode with actual measured context, never the compact zero prefix.
            decode_latents = torch.cat(
                (*measured_history, generated["video_latents"][:, -encoded["block_size"] :].cpu()), dim=1
            )
        else:
            generated = sample_window_block(pipe, encoded, index, revision=index)
            physical = pipe.processor.decode(
                generated["action"].cpu(), sample["anchors"].repeat_interleave(block, 0), sample["action_mask"]
            )
            predicted = physical[index * block : (index + 1) * block]
            decode_latents = generated["video_latents"]
        valid = min(block, real - index * block)
        actions.append(predicted[:valid])
        pixels = pipe.vae.decode(decode_latents)
        videos.append(pixels[:, 1 + index * frames : 1 + index * frames + valid // pipe.geometry.video_stride])
        records.append(
            dict(
                block=index,
                valid_steps=valid,
                seeds=generated["seeds"],
                anchor=sample["anchors"][index].tolist(),
                history_blocks=min(1 + index * encoded["block_size"], pipe.history),
                input_pixel_frames=(
                    pixel_stop - pixel_start + (frames if index == 0 else 0)
                    if streaming and sample["mode"] == "inverse_dynamics"
                    else pixel_stop - pixel_start
                    if streaming
                    else sample["video"].shape[1]
                ),
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
    # Full-window targets are encoded only after predictions, for export/evaluation.
    reconstruction_latents = (
        pipe.vae.encode_pack([sample])[0]["video_latents"] if streaming else encoded["video_latents"]
    )
    reconstruction = pipe.vae.decode(reconstruction_latents)[:, : video.shape[1]]
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
    initial, state, history, *, source, processor, geometry, tokenize, history_blocks, valid, task="policy"
):
    """Build synthetic unknown targets without reading any episode data."""
    validate_rollout_task(task)
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
        task=task,
        history=history_blocks,
    )
    sample["condition_frame_indexes_vision"] = list(
        range(1 + context * geometry.actions_per_block // geometry.actions_per_latent)
    )
    return sample, context


def validate_rollout_task(task):
    if task not in ("policy", "inverse_dynamics"):
        raise ValueError("Rollout/serving supports policy or inverse_dynamics")


def encode_observation_block(pipe, sample, pixels, *, encoder, caches, revision):
    """Append arrived RGB to caller-owned VAE/KV sessions and build one target block.

    Both live feedback and model-only rollout call this function. The caller
    decides where pixels come from and must discard the session on failure.
    """
    observed = encoder.encode(pixels).to(pipe.device)
    block = pipe.geometry.actions_per_block // pipe.geometry.actions_per_latent
    expected = block if revision else 1
    if observed.shape[1] != expected:
        raise ValueError(f"Expected {expected} observed latent frames, got {observed.shape[1]}")
    if pipe.kv_cache and revision == 0:
        if caches:
            raise ValueError("Reset caches before starting a new episode")
        caches.update(
            create_observation_caches(
                pipe,
                observed,
                text_ids=sample["text_token_ids"],
                conditioning_fps=sample["conditioning_fps"],
                conditioning_fps_action=sample["conditioning_fps_action"],
            )
        )
    elif pipe.kv_cache:
        if not caches:
            raise ValueError("Initialize observation caches before advancing")
        for cache in caches.values():
            cache.update(observed, revision=revision)
    video = observed.new_zeros((observed.shape[0], 1 + block, *observed.shape[2:]))
    if revision == 0:
        video[:, :1] = observed
    encoded = {key: value for key, value in sample.items() if key != "video"}
    encoded.update(video_latents=video, cache_revision=revision)
    for key in ("action", "states", "action_mask", "state_mask"):
        encoded[key] = encoded[key].to(pipe.device)
    return encoded, observed


def expand_observation_sample(sample, observed_history):
    """Full-pack oracle with the same streaming latents and absolute positions.

    Keep the complete encoded prefix: truncating it would change deep clean-row
    representations. Attention still enforces the configured history budget.
    Historical action/state rows are inert for the current block's prediction.
    """
    size = sample["block_size"]
    block = (observed_history.shape[1] - 1) // size
    actions = len(sample["action"])
    video = torch.cat((observed_history, sample["video_latents"][:, 1:]), dim=1)
    past = block * actions
    current_conditions = [block * size + f for f in sample["condition_frame_indexes_vision"] if f > 0]
    expanded = dict(
        sample,
        video_latents=video,
        action=torch.cat((sample["action"].new_zeros(past, sample["action"].shape[1]), sample["action"])),
        states=sample["states"].expand(block + 1, -1).clone(),
        action_frame_ids=[1 + i // (actions // size) for i in range(past + actions)],
        state_latent_indexes=[i * size for i in range(block + 1)],
        condition_frame_indexes_vision=list(range(observed_history.shape[1])) + current_conditions,
        condition_frame_indexes_action=[past + i for i in sample["condition_frame_indexes_action"]],
        action_valid_steps=torch.cat((sample["action_valid_steps"].new_ones(past), sample["action_valid_steps"])),
        num_valid_video_latents=block * size + sample["num_valid_video_latents"],
    )
    # Recover patch count from the compact layout, independently of model width.
    patches = (sample["token_length"] - len(sample["text_token_ids"]) - actions - 1) // (2 * (1 + size))
    expanded["token_length"] = (
        len(sample["text_token_ids"]) + 2 * video.shape[1] * patches + past + actions + block + 1
    )
    return expanded, block


class ObservationSequence:
    """One streaming RGB/latent sequence; KV reuse is an independent option."""

    def __init__(self, pipe, *, retain_latents=False):
        self.pipe = pipe
        self.encoder = pipe.vae.stream_encoder()
        self.caches = {}
        self.latents = []
        self.retain_latents = retain_latents or not pipe.kv_cache
        self.revision = 0

    def prepare(self, sample, pixels, revision):
        if revision != self.revision:
            raise ValueError("Observation sequence revisions must advance by one")
        encoded, observed = encode_observation_block(
            self.pipe, sample, pixels, encoder=self.encoder, caches=self.caches, revision=revision
        )
        if self.retain_latents:
            self.latents.append(observed.cpu())
        self.revision += 1
        if self.pipe.kv_cache:
            return encoded, 0
        return expand_observation_sample(encoded, torch.cat(self.latents, dim=1).to(self.pipe.device))


def sample_window_block(pipe, sample, block, *, revision, rollout_task=None):
    """Run the same encoded window with or without KV; never retain window KV.

    A moving window changes its VAE context and local positions. Rebuild its
    clean prefix explicitly instead of borrowing an incompatible episode cache.
    """

    def sample_one(encoded, index, caches):
        if rollout_task is not None:
            return sample_rollout_block(pipe, encoded, index, revision=revision, task=rollout_task, kv_cache=caches)
        return sample_block(pipe.model, encoded, index, seed_block=revision, kv_cache=caches, **pipe.sampling_options)

    if not pipe.kv_cache:
        return sample_one(sample, block, None)
    size = sample["block_size"]
    video = sample["video_latents"]
    observed = video[:, :1]
    caches = create_observation_caches(
        pipe,
        observed,
        text_ids=sample["text_token_ids"],
        conditioning_fps=sample["conditioning_fps"],
        conditioning_fps_action=sample["conditioning_fps_action"],
    )
    for index in range(1, block + 1):
        observed = video[:, 1 + (index - 1) * size : 1 + index * size]
        for cache in caches.values():
            cache.update(observed, revision=index)
    compact = compact_prediction_sample(
        sample,
        observed,
        block,
        actions_per_block=pipe.geometry.actions_per_block,
        patch_size=pipe.model.config.patch_size,
    )
    if rollout_task is None and sample["mode"] == "inverse_dynamics":
        compact["video_latents"][:, 1:] = video[:, 1 + block * size : 1 + (block + 1) * size]
        compact["condition_frame_indexes_vision"] = list(range(1 + size))
    result = sample_one(compact, 0, caches)
    full_video, full_action = video.float().clone(), sample["action"].float().clone()
    full_video[:, 1 + block * size : 1 + (block + 1) * size] = result["video_latents"][:, 1:]
    start = block * pipe.geometry.actions_per_block
    full_action[start : start + pipe.geometry.actions_per_block] = result["action"]
    return dict(result, video_latents=full_video, action=full_action.masked_fill(~sample["action_mask"], 0))


def sample_rollout_block(pipe, sample, block, *, revision, task="policy", kv_cache=None):
    """Shared online/offline transition: joint Policy prior, then optional IDM.

    No future observation exists here. First generate a video with the Policy
    prior (discard its actions for IDM), then condition on that completed video
    and resample actions. Scheme B conditions are declared by indexes, never by
    changing training masks. Prediction video is never committed as measured KV.
    """
    validate_rollout_task(task)
    prior = dict(sample, mode="policy")
    generated = sample_block(pipe.model, prior, block, seed_block=revision, kv_cache=kv_cache, **pipe.sampling_options)
    if task == "policy":
        return generated
    conditioned = dict(
        sample,
        mode="inverse_dynamics",
        video_latents=generated["video_latents"],
        condition_frame_indexes_vision=list(range(generated["video_latents"].shape[1])),
    )
    # The prior's video depends on its noisy actions. Use an independent stream
    # for conditional action sampling instead of reusing that same action noise.
    action_options = dict(pipe.sampling_options)
    action_options["seed"] = noise_seed(action_options["seed"], revision, "idm-action-stage")
    action = sample_block(pipe.model, conditioned, block, seed_block=revision, kv_cache=kv_cache, **action_options)
    action["seeds"] = dict(video=generated["seeds"], action=action["seeds"])
    return action


@torch.no_grad()
def rollout(pipe, initial, *, task="policy"):
    """Model-only open-loop rollout: generated pixels and absolute action feedback."""
    validate_rollout_task(task)
    size = pipe.geometry.actions_per_block
    new_frames = size // pipe.geometry.video_stride
    latent_block = size // pipe.geometry.actions_per_latent
    context_limit = (pipe.history + latent_block - 1) // latent_block
    history_pixel_limit = 1 + context_limit * new_frames
    state = initial.state.cpu().clone()
    history = initial.image.cpu().clone()
    videos, actions, records = [history.clone()], [], []
    streaming = pipe.observation_mode == "streaming"
    sequence = ObservationSequence(pipe, retain_latents=True) if streaming else None
    arrived_pixels = history
    for index, start in enumerate(range(0, initial.horizon, size)):
        begin = time.monotonic()
        valid = min(size, initial.horizon - start)
        sample, context = rollout_sample(
            initial,
            state,
            arrived_pixels[:, :1] if streaming else history,
            source=pipe.source,
            processor=pipe.processor,
            geometry=pipe.geometry,
            tokenize=pipe.tokenize,
            history_blocks=pipe.history,
            valid=valid,
            task=task,
        )
        if streaming:
            encoded, context = sequence.prepare(sample, arrived_pixels, index)
        else:
            encoded = pipe.vae.encode_pack([sample])[0]
        for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
            encoded[key] = encoded[key].to(pipe.device)
        if streaming:
            generated = sample_rollout_block(
                pipe, encoded, context, revision=index, task=task, kv_cache=sequence.caches if pipe.kv_cache else None
            )
        else:
            generated = sample_window_block(pipe, encoded, context, revision=index, rollout_task=task)
        normalized = generated["action"][-size:].cpu()
        absolute = pipe.processor.decode(normalized, state.expand(size, -1), initial.action_mask)
        if not torch.isfinite(absolute).all():
            raise ValueError("Nonfinite decoded action")
        anchor = state.clone()
        actions.append(absolute[:valid])
        # Feedback always selects the final valid action, never a padded tail.
        state = absolute[valid - 1].clone()
        decode_latents = (
            torch.cat((*sequence.latents, generated["video_latents"][:, -latent_block:].cpu()), dim=1)
            if streaming
            else generated["video_latents"]
        )
        pixels = pipe.vae.decode(decode_latents)
        expected_frames = 1 + (index + 1) * new_frames if streaming else history.shape[1] + new_frames
        if pixels.shape != (3, expected_frames, *history.shape[-2:]):
            raise ValueError("Decoded pixels do not match the native input geometry")
        added = pixels[:, -new_frames:]
        input_pixel_frames = arrived_pixels.shape[1] if streaming else sample["video"].shape[1]
        arrived_pixels = added
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
                history_blocks=min(1 + (index if streaming else context) * latent_block, pipe.history),
                context_blocks=min(index, context_limit) if streaming else context,
                history_pixel_frames=history.shape[1],
                input_pixel_frames=input_pixel_frames,
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
        "model_only_open_loop_action_as_state_pixel_feedback"
        if task == "policy"
        else "inverse_dynamics_video_then_action_pixel_feedback",
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


# Compatibility aliases; observation history is shared by Policy and IDM.
create_policy_caches = create_observation_caches
compact_policy_sample = compact_prediction_sample

encode_cached_observation = encode_observation_block
