"""Request-local packing templates for repeated denoising of one block."""

import torch

from .attention import select_attention_queries
from .packing import pack_model_inputs
from .patching import patchify_video


class DenoisePacking:
    """Reuse immutable packing metadata; each CFG branch owns its own plan."""

    def __init__(
        self,
        model,
        sample,
        block,
        attention_backend,
        position_options=None,
        *,
        prune_attention=True,
        cache_fixed_attention=True,
    ):
        self.patch_size = model.config.patch_size
        self.dtype = next(model.parameters()).dtype
        self.video = sample["video_latents"]
        self.action = sample["action"].to(self.video.device)
        start = 1 + block * sample["block_size"]
        stop = min(start + sample["block_size"], self.video.shape[1])
        times = torch.zeros(self.video.shape[1], device=self.video.device, dtype=torch.float32)
        times[start:stop] = 1
        self.inputs, spans, _ = pack_model_inputs(
            [sample],
            [self.video],
            [self.action],
            timesteps=[times],
            patch_size=self.patch_size,
            **(position_options or {}),
        )
        self.span = spans[0]
        self.video_condition = torch.zeros(self.video.shape[1], device=self.video.device, dtype=torch.bool)
        self.video_condition[sample["condition_frame_indexes_vision"]] = True
        self.action_condition = torch.zeros(len(self.action), device=self.video.device, dtype=torch.bool)
        self.action_condition[sample["condition_frame_indexes_action"]] = True
        self.video_times = self.inputs["video_timesteps"]
        self.action_times = self.inputs["action_timesteps"]
        for name in ("video_tokens", "action_tokens", "state_tokens"):
            self.inputs[name] = self.inputs[name].to(self.dtype)
        self.clean_tokens = self.inputs["video_tokens"][: self.span.current_video_rows.start]
        self.inputs["attention_backend"] = attention_backend
        if hasattr(model, "prepare_denoise_packing"):
            self.inputs["denoise_packing"] = model.prepare_denoise_packing(self.inputs, attention_backend)
            if prune_attention:
                layout = self.inputs["layout"]
                # Keep the complete earlier clean prefix, including indirect
                # dependencies outside the target's immediate history window.
                keep = tuple(
                    (role == 0 and frame < start) or (role != 0 and bid == block)
                    for role, bid, frame in zip(layout.roles, layout.block_ids, layout.frame_ids, strict=True)
                    if role != -1
                )
                prepared = self.inputs["denoise_packing"]
                prepared.attention = select_attention_queries(prepared.attention, keep)
                if cache_fixed_attention:
                    target = tuple(
                        role == 1 and bid == block
                        for role, bid in zip(layout.roles, layout.block_ids, strict=True)
                        if role != -1
                    )
                    target_plan = select_attention_queries(prepared.attention, target)
                    prepared.fixed_attention = {"target": target_plan, "layers": {}}

    def update(self, video, action, timestep):
        current = torch.where(self.video_condition[None, :, None, None], self.video, video)
        self.inputs["video_tokens"] = torch.cat(
            (self.clean_tokens, patchify_video(current, self.patch_size).to(self.dtype))
        )
        self.inputs["action_tokens"] = (
            torch.where(self.action_condition[:, None], self.action, action)
            .to(self.dtype)
            .masked_fill(~self.inputs["action_mask"], 0)
        )
        self.inputs["video_timesteps"] = self.video_times * timestep.float()
        self.inputs["action_timesteps"] = self.action_times * timestep.float()
        return self.inputs
