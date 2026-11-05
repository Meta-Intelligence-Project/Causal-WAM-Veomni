# SPDX-License-Identifier: OpenMDW-1.1
"""Joint native video/action sampling with optional input-VJP prefix guidance."""

import math
from contextlib import nullcontext

import torch

from ..cosmos3_core.attention import prepare_attention
from ..cosmos3_core.fixed_step import FixedStepSampler
from ..cosmos3_core.packing import pack_model_inputs
from ..cosmos3_core.patching import patchify_video, unpatchify_video
from ..cosmos3_core.unipc import FlowUniPCMultistepScheduler
from ..inference import noise_seed
from .npu_replay_graph import graph_request
from .prefix_cache import PrefixCache
from .rtc import guided_velocity, prepare_prefix


@torch.no_grad()
def sample_decision(
    model,
    sample,
    *,
    steps=10,
    shift=1.0,
    guidance=1.0,
    seed=42,
    negative_text_ids=None,
    attention_backend="grouped",
    cache_mode="full",
    rtc=None,
    initial_noise=None,
    fixed_step_sampler_config=None,
    guidance_interval=None,
    normalize_cfg=False,
    position_options=None,
    npu_replay_graph=False,
    episode_cache=None,
):
    """Generate only this decision's future; keep measured conditions fixed at every step."""
    if type(npu_replay_graph) is not bool:
        raise ValueError("npu_replay_graph must be a bool")
    if npu_replay_graph:
        if rtc is not None:
            raise ValueError("NPU replay graphs and RTC are mutually exclusive; disable RTC")
        if cache_mode != "full" or attention_backend != "npu_tnd" or guidance != 1:
            raise ValueError("NPU replay graphs require full cache, npu_tnd attention and guidance=1")
    with (
        episode_cache.request() if episode_cache is not None else nullcontext(),
        graph_request(model, npu_replay_graph) as graph_cache,
    ):
        return _sample_decision(
            model,
            sample,
            steps=steps,
            shift=shift,
            guidance=guidance,
            seed=seed,
            negative_text_ids=negative_text_ids,
            attention_backend=attention_backend,
            cache_mode=cache_mode,
            rtc=rtc,
            initial_noise=initial_noise,
            fixed_step_sampler_config=fixed_step_sampler_config,
            guidance_interval=guidance_interval,
            normalize_cfg=normalize_cfg,
            position_options=position_options,
            graph_cache=graph_cache,
            episode_cache=episode_cache,
        )


def _sample_decision(
    model,
    sample,
    *,
    steps,
    shift,
    guidance,
    seed,
    negative_text_ids,
    attention_backend,
    cache_mode,
    rtc,
    initial_noise,
    fixed_step_sampler_config,
    guidance_interval,
    normalize_cfg,
    position_options,
    graph_cache,
    episode_cache,
):
    if cache_mode not in ("off", "plans", "full"):
        raise ValueError("cache_mode must be off, plans or full")
    if type(steps) is not int or steps < 1 or not all(math.isfinite(x) and x > 0 for x in (shift, guidance)):
        raise ValueError("steps, shift and guidance must be positive and finite")
    if rtc is not None and torch.is_inference_mode_enabled():
        raise RuntimeError("RTC requires torch.no_grad(), not torch.inference_mode()")
    video, action = sample["video_latents"].float(), sample["action"].float()
    observed = sample["observed_latents"]
    channels = sample["action_mask"].to(video.device).expand_as(action)
    vm = torch.zeros_like(video, dtype=torch.bool)
    vm[:, observed:] = True
    active = torch.cat((vm.flatten(), channels.flatten()))
    reference = torch.cat((video.flatten(), action.masked_fill(~channels, 0).flatten()))
    identity = sample.get("tick", 0)
    seeds = {name: noise_seed(seed, identity, name) for name in ("video", "action")}
    noises = [
        torch.randn(
            tensor.shape, device=video.device, generator=torch.Generator(device=video.device).manual_seed(seeds[name])
        )
        if initial_noise is None
        else initial_noise[name].to(video.device).float()
        for name, tensor in (("video", video), ("action", action))
    ]
    if any(noise.shape != target.shape for noise, target in zip(noises, (video, action), strict=True)):
        raise ValueError("Initial noise must match the prepared video/action shapes")
    latent = torch.where(active, torch.cat([part.flatten() for part in noises]), reference)
    dtype, patch = next(model.parameters()).dtype, model.config.patch_size
    sample = dict(sample)
    patches = math.ceil(video.shape[2] / patch) * math.ceil(video.shape[3] / patch)
    sample["token_length"] = len(sample["text_token_ids"]) + video.shape[1] * patches + len(action) + 1
    templates, plans, caches = {}, {}, {}
    for branch, text in (("cond", None), ("uncond", negative_text_ids)):
        if branch == "uncond" and guidance == 1:
            continue
        if branch == "uncond" and text is None:
            raise ValueError("CFG requires explicit negative prompt tokens")
        row = dict(sample)
        if text is not None:
            row["text_token_ids"] = text
            row["token_length"] += len(text) - len(sample["text_token_ids"])
        inputs, spans, _ = pack_model_inputs(
            [row],
            [video],
            [action],
            timesteps=[torch.zeros(video.shape[1], device=video.device)],
            patch_size=patch,
            **(position_options or {}),
        )
        for field in ("video_tokens", "action_tokens", "state_tokens"):
            inputs[field] = inputs[field].to(dtype=dtype)
        templates[branch] = inputs, spans[0]
        if cache_mode == "full":
            options = dict(differentiable=rtc is not None, graph_cache=graph_cache)
            if episode_cache is not None:
                options.update(episode_cache=episode_cache, branch=branch)
            caches[branch] = PrefixCache(model, inputs, attention_backend, **options)
        elif cache_mode == "plans":
            plans[branch] = prepare_attention(inputs["layout"], video.device, attention_backend)
    prepared = prepare_prefix(rtc, sample["rtc_metadata"], observed, device=video.device) if rtc is not None else None
    target, weights = torch.zeros_like(reference), torch.zeros_like(reference)
    if prepared is not None:
        target[video.numel() :] = prepared.normalized_targets.flatten()
        weights[video.numel() :] = prepared.weights.flatten()
    vjp_calls = 0

    def predict(current, timestep, branch):
        # Measured history is constant even in the uncached input-gradient oracle.
        current = torch.where(active, current, reference)
        inputs, span = templates[branch]
        inputs = dict(inputs)
        inputs["video_tokens"] = patchify_video(current[: video.numel()].reshape_as(video), patch).to(dtype)
        inputs["action_tokens"] = current[video.numel() :].reshape_as(action).to(dtype).masked_fill(~channels, 0)
        frame_times = torch.zeros(video.shape[1], device=video.device)
        frame_times[observed:] = timestep.float()
        inputs["video_timesteps"] = frame_times.repeat_interleave(patches)
        inputs["action_timesteps"] = timestep.float().expand(len(action))
        output = (
            caches[branch](inputs)
            if branch in caches
            else model(**inputs, attention_backend=attention_backend, attention_plan=plans.get(branch))
        )
        pred_v = unpatchify_video(output.video[span.current_video_rows], video.shape, patch)
        pred_a = output.action[span.action_rows].masked_fill(~channels, 0)
        return torch.cat((pred_v.float().flatten(), pred_a.float().flatten())).masked_fill(~active, 0)

    if fixed_step_sampler_config is None:
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        step_options = {}
    else:
        scheduler = FixedStepSampler(**fixed_step_sampler_config)
        seeds["sampler"] = noise_seed(seed, identity, "sampler")
        step_options = dict(generator=torch.Generator(device=video.device).manual_seed(seeds["sampler"]))
    scheduler.set_timesteps(steps, device=video.device, shift=float(shift))
    for index, timestep in enumerate(scheduler.timesteps):

        def velocity(value, timestep=timestep):
            conditional = predict(value, timestep, "cond")
            use_cfg = guidance != 1 and (
                guidance_interval is None or guidance_interval[0] < float(timestep) < guidance_interval[1]
            )
            if not use_cfg:
                return conditional
            guided = predict(value, timestep, "uncond")
            guided = guided + guidance * (conditional - guided)
            if normalize_cfg:
                guided = guided * (conditional.norm() / (guided.norm() + 1e-8)).clamp(0, 1)
            return guided

        current = torch.where(active, latent, reference)
        # Use the solver's true sigma, independently of model-time quantization.
        sigma = scheduler.sigmas[index]
        sigma = float(sigma)
        if prepared is not None:
            prediction = guided_velocity(current, velocity, sigma, target, weights, prepared.beta)
            vjp_calls += int(sigma > 0 and prepared.beta > 0 and bool(weights.any()))
        else:
            prediction = velocity(current)
        latent = (
            scheduler.step(prediction, timestep, current.unsqueeze(0), return_dict=False, **step_options)[0]
            .squeeze(0)
            .float()
        )
        latent = torch.where(active, latent, reference)
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite online sampling state")
    actions = latent[video.numel() :].reshape_as(action).masked_fill(~channels, 0)
    if prepared is not None and prepared.delay_steps:
        actions[: prepared.delay_steps] = prepared.normalized_targets[: prepared.delay_steps]
    return dict(
        video_latents=latent[: video.numel()].reshape_as(video),
        action=actions,
        seeds=seeds,
        prepared_prefix=prepared,
        rtc_vjp_calls=vjp_calls,
        inference_cache={
            branch: {
                **cache.stats,
                **({"npu_replay_graph_stats": graph_cache.stats} if graph_cache is not None else {}),
            }
            for branch, cache in caches.items()
        },
    )
