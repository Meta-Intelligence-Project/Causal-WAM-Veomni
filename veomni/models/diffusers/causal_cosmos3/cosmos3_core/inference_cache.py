"""Packing metadata and fixed computation caches for no-grad Cosmos3 inference."""

from dataclasses import dataclass, field

import torch

from .attention import select_attention_queries
from .packing import pack_model_inputs
from .patching import patchify_video


@dataclass
class LayerCache:
    text_hidden: torch.Tensor
    hidden: torch.Tensor
    fixed_k: torch.Tensor
    fixed_v: torch.Tensor


@dataclass
class BlockCache:
    """Keep full hidden layouts; recompute only the current target group.

    Clean video, conditions, state and other blocks have fixed inputs and
    cannot attend the current noisy targets. Only target-visible fixed KV is
    retained; all fixed hidden rows are restored for the following layer.
    """

    dynamic: torch.Tensor
    fixed_global: torch.Tensor
    target_plan: object
    layers: dict = field(default_factory=dict)

    @classmethod
    def prepare(cls, plan, layout, block):
        gen = plan.gen_indexes.tolist()
        keep = [layout.roles[i] == 1 and layout.block_ids[i] == block for i in gen]
        dynamic = plan.gen_indexes.new_tensor([i for i, selected in enumerate(keep) if selected])
        visible = set()
        for chunk in plan.chunks:
            qs = ks = 0
            for qe, ke in zip(chunk.query_ends, chunk.kv_ends, strict=True):
                if any(keep[i] for i in chunk.query_indexes[qs:qe].tolist()):
                    visible.update(chunk.kv_indexes[ks:ke].tolist())
                qs, ks = qe, ke
        fixed = sorted(visible - {gen[i] for i in dynamic.tolist()})
        return cls(dynamic, plan.gen_indexes.new_tensor(fixed), select_attention_queries(plan, keep))


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
                prepared = self.inputs["denoise_packing"]
                if cache_fixed_attention:
                    # The cache computes/restores every row, replacing query-only
                    # pruning without changing the packed arrays or visibility.
                    prepared.fixed_attention = BlockCache.prepare(prepared.attention, layout, block)
                else:
                    keep = tuple(
                        (role == 0 and frame < start) or (role != 0 and bid == block)
                        for role, bid, frame in zip(layout.roles, layout.block_ids, layout.frame_ids, strict=True)
                        if role != -1
                    )
                    prepared.attention = select_attention_queries(prepared.attention, keep)

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
