# SPDX-License-Identifier: OpenMDW-1.1
"""Current-block sampling from real conditions; no full-window sampling state."""

import hashlib
import math
from dataclasses import replace

import torch

from .fixed_step import FixedStepSampler
from .inference_inputs import InferenceInputs
from .patching import unpatchify_video
from .unipc import FlowUniPCMultistepScheduler


def noise_seed(seed, block, stream):
    return int(hashlib.sha256(f"cosmos3-inference-v1:{seed}:{block}:{stream}".encode()).hexdigest()[:16], 16) % 2**63


@torch.no_grad()
def generate_block(
    model,
    context,
    stage,
    *,
    conditioned_video=None,
    conditioned_action=None,
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
    """Sample only current targets. Conditions never enter the solver state.

    initial_noise, when supplied, contains exactly the active modalities with
    current-block shapes. Its draws are independent of history/window length.
    """
    if type(steps) is not int or steps < 1 or not all(math.isfinite(x) and x > 0 for x in (shift, guidance)):
        raise ValueError("steps, shift and guidance must be finite and positive")
    if guidance_interval is not None and (
        len(guidance_interval) != 2
        or not all(math.isfinite(t) for t in guidance_interval)
        or not 0 <= guidance_interval[0] < guidance_interval[1]
    ):
        raise ValueError("guidance_interval must be finite nonnegative [lo, hi]")
    if guidance != 1 and negative_text_ids is None:
        raise ValueError("CFG requires explicit negative/empty-prompt token IDs")
    options = dict(
        patch_size=model.config.patch_size,
        conditioned_video=conditioned_video,
        conditioned_action=conditioned_action,
        **(position_options or {}),
    )
    branches = [InferenceInputs(context, stage, **options)]
    if guidance != 1:
        branches.append(InferenceInputs(replace(context, text_ids=negative_text_ids), stage, **options))
    inputs = branches[0]
    device = context.history_video.device
    sid = context.block if seed_block is None else seed_block
    shapes = {}
    if inputs.video_active:
        shapes["video"] = context.video_shape
    if inputs.action_active:
        shapes["action"] = context.action_shape
    if initial_noise is not None and set(initial_noise) != set(shapes):
        raise ValueError("initial_noise must contain exactly this stage's target modalities")
    seeds = {name: noise_seed(seed, sid, name) for name in shapes}
    noises = []
    for name, shape in shapes.items():
        if initial_noise is None:
            noise = torch.randn(
                shape,
                dtype=torch.float32,
                device=device,
                generator=torch.Generator(device=device).manual_seed(seeds[name]),
            )
        else:
            noise = initial_noise[name].to(device=device, dtype=torch.float32)
            if tuple(noise.shape) != shape or not torch.isfinite(noise).all():
                raise ValueError(f"Initial {name} noise must be finite with current-block shape {shape}")
        noises.append(noise.flatten())
    latent = torch.cat(noises)
    video_numel = math.prod(context.video_shape) if inputs.video_active else 0
    valid = torch.ones_like(latent, dtype=torch.bool)
    if inputs.action_active:
        valid[video_numel:] = context.action_mask.to(device).expand(context.action_shape).flatten()
    latent = latent.masked_fill(~valid, 0)
    if fixed_step_sampler_config is None:
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        step_options = {}
    else:
        scheduler = FixedStepSampler(**fixed_step_sampler_config)
        seeds["sampler"] = noise_seed(seed, sid, f"{stage}-sampler")
        step_options = dict(generator=torch.Generator(device=device).manual_seed(seeds["sampler"]))
    scheduler.set_timesteps(steps, device=device, shift=float(shift))
    dtype = next(model.parameters()).dtype
    probe = getattr(model, "_attention_probe", None)
    if probe is not None:
        # Probe metadata describes the actual history + current block, never
        # an extra model input or a second copy of the observed video.
        video = latent[:video_numel].reshape(context.video_shape) if video_numel else conditioned_video
        probe.begin_sample(
            dict(
                mode=inputs.mode, caption=context.caption, video_latents=torch.cat((context.history_video, video), 1)
            ),
            context.block,
            sid,
            list(range(context.history_video.shape[1], context.history_video.shape[1] + context.block_size))
            if video_numel
            else [],
            inputs.action_active,
            len(scheduler.timesteps),
        )

    def predict(branch, timestep, step):
        video = latent[:video_numel].reshape(context.video_shape) if video_numel else None
        action = latent[video_numel:].reshape(context.action_shape) if inputs.action_active else None
        payload = branches[branch].model_inputs(video, action, timestep, dtype)
        if probe is not None:
            probe.before_forward(payload, step, float(timestep), branch == 0)
        try:
            output = model(**payload, attention_backend=attention_backend)
        finally:
            if probe is not None:
                probe.after_forward()
        velocity = []
        if video_numel:
            velocity.append(
                unpatchify_video(output.video[inputs.video_target_rows], context.video_shape, model.config.patch_size)
                .float()
                .flatten()
            )
        if inputs.action_active:
            velocity.append(output.action.float().flatten())
        return torch.cat(velocity).masked_fill(~valid, 0)

    for step, timestep in enumerate(scheduler.timesteps):
        velocity = predict(0, timestep, step)
        if guidance != 1 and (
            guidance_interval is None or guidance_interval[0] < float(timestep) < guidance_interval[1]
        ):
            unconditional = predict(1, timestep, step)
            conditional = velocity
            velocity = unconditional + guidance * (conditional - unconditional)
            if normalize_cfg:
                velocity *= (torch.norm(conditional) / (torch.norm(velocity) + 1e-8)).clamp(0, 1)
        latent = scheduler.step(velocity, timestep, latent.unsqueeze(0), return_dict=False, **step_options)[0]
        latent = latent.squeeze(0).float().masked_fill(~valid, 0)
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite sampling state")
    return dict(
        video_latents=latent[:video_numel].reshape(context.video_shape) if video_numel else conditioned_video,
        action=latent[video_numel:].reshape(context.action_shape) if inputs.action_active else conditioned_action,
        seeds=seeds,
    )


def generate_video_block(model, context, **options):
    """IDM video prior: observed history and internal video noise only."""
    return generate_block(model, context, "video", **options)


def generate_action_block(model, context, conditioned_video, **options):
    """IDM actions: one fixed visual stream and internal action noise only."""
    return generate_block(model, context, "action", conditioned_video=conditioned_video, **options)
