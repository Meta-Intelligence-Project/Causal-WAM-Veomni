"""Packing metadata and fixed computation caches for no-grad Cosmos3 inference."""

from dataclasses import dataclass, field, replace

import torch

from .attention import TNDChunk, select_attention_queries
from .packing import pack_model_inputs
from .patching import patchify_video


@dataclass
class LayerCache:
    text_hidden: torch.Tensor
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class BlockCache:
    """Keep full hidden layouts; recompute only the current target group.

    Clean video, conditions, state and other blocks have fixed inputs and
    cannot attend the current noisy targets. Target-visible KV is retained in
    shared storage with mutable target slots; all fixed hidden rows are restored
    for the following layer.
    """

    dynamic: torch.Tensor
    kv_global: torch.Tensor
    dynamic_kv: torch.Tensor
    compact_plan: object
    dynamic_tables: tuple[torch.Tensor, torch.Tensor]
    layers: dict = field(default_factory=dict)
    history: object = None

    @classmethod
    def prepare(cls, plan, layout, block, tables):
        gen = plan.gen_indexes.tolist()
        keep = [layout.roles[i] == 1 and layout.block_ids[i] == block for i in gen]
        dynamic_rows = [i for i, selected in enumerate(keep) if selected]
        target = select_attention_queries(plan, keep)
        group_indexes = [(chunk.query_indexes.tolist(), chunk.kv_indexes.tolist()) for chunk in target.chunks]
        visible = sorted({index for _, keys in group_indexes for index in keys})
        tensor = plan.gen_indexes.new_tensor
        dynamic = tensor(dynamic_rows)
        kv_global = tensor(visible)
        key_rows = {index: row for row, index in enumerate(visible)}
        query_rows = {index: row for row, index in enumerate(dynamic_rows)}
        chunks = tuple(
            TNDChunk(
                tensor([query_rows[i] for i in queries]),
                tensor([key_rows[i] for i in keys]),
                chunk.query_ends,
                chunk.kv_ends,
            )
            for chunk, (queries, keys) in zip(target.chunks, group_indexes, strict=True)
        )
        compact = replace(
            target,
            chunks=chunks,
            query_selection=tensor([query_rows[i] for i in target.query_selection.tolist()]),
            mask=None if target.mask is None else target.mask.index_select(0, dynamic).index_select(1, kv_global),
        )
        return cls(
            dynamic,
            kv_global,
            tensor([key_rows[gen[i]] for i in dynamic_rows]),
            compact,
            tuple(table[dynamic] for table in tables),
        )


@dataclass
class HistoryLayer:
    text_hidden: torch.Tensor
    text_key: torch.Tensor
    text_value: torch.Tensor
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class HistoryBinding:
    """A block-local view of the previous and next immutable history snapshots."""

    previous: dict
    following: dict
    clean_rows: torch.Tensor
    reused_rows: torch.Tensor
    previous_rows: torch.Tensor
    fresh_rows: torch.Tensor
    fresh_plan: object


class HistoryCache:
    """Episode-owned clean/text cache, separated by CFG branch.

    Callers promise immutable historical latents and model weights. Re-encoding
    pixels or changing the history window can violate that promise; reuse then
    intentionally approximates a streaming latent history. Keys are stored
    before RoPE so local coordinate shifts never reuse an old rotation.
    """

    def __init__(self):
        self.branches = {}

    def bind(self, model, sample, block, sequence_block, branch, plan, layout, position_options):
        video = sample["video_latents"]
        signature = (
            id(model),
            str(video.device),
            next(model.parameters()).dtype,
            tuple(sample["text_token_ids"].tolist()),
            video.shape[0],
            tuple(video.shape[2:]),
            model.config.patch_size,
            sample["block_size"],
            sample["history_blocks"],
            sample["conditioning_fps"],
            sample["conditioning_fps_action"],
            tuple(sorted((position_options or {}).items())),
        )
        old = self.branches.get(branch)
        if old is None or old[0] != signature or old[1] + 1 != sequence_block:
            old_ids, previous = (), {}
        else:
            old_ids, previous = old[2:]
        gen = plan.gen_indexes.tolist()
        # Never persist clean placeholders for the current or future blocks.
        start = 1 + block * sample["block_size"]
        offset = (sequence_block - block) * sample["block_size"]
        clean = [i for i, g in enumerate(gen) if layout.roles[g] == 0 and layout.frame_ids[g] < start]
        patches = sum(layout.roles[g] == 0 and layout.frame_ids[g] == 0 for g in gen)
        ids = [(layout.frame_ids[gen[i]] + offset, i % patches) for i in clean]
        lookup = {key: i for i, key in enumerate(old_ids)}
        matched = [(row, lookup[key]) for row, key in zip(clean, ids, strict=True) if key in lookup]
        reused = {row for row, _ in matched}
        keep = [i not in reused for i in range(len(gen))]
        following = {}
        self.branches[branch] = (signature, sequence_block, ids, following)
        tensor = plan.gen_indexes.new_tensor
        return HistoryBinding(
            previous,
            following,
            tensor(clean),
            tensor([row for row, _ in matched]),
            tensor([row for _, row in matched]),
            tensor([i for i, selected in enumerate(keep) if selected]),
            select_attention_queries(plan, keep, allow_partial=True),
        )


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
        history_cache=None,
        sequence_block=None,
        branch=True,
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
                    prepared.fixed_attention = BlockCache.prepare(
                        prepared.attention, layout, block, prepared.tables[1]
                    )
                    if history_cache is not None:
                        prepared.fixed_attention.history = history_cache.bind(
                            model,
                            sample,
                            block,
                            block if sequence_block is None else sequence_block,
                            branch,
                            prepared.attention,
                            layout,
                            position_options,
                        )
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
