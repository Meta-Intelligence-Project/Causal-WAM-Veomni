"""Packing metadata and fixed computation caches for no-grad Cosmos3 inference."""

from dataclasses import dataclass, field, replace
from itertools import chain

import torch

from .attention import _pack_attention_groups, select_attention_queries
from .packing import pack_model_inputs
from .patching import patchify_video


def residual_refresh_steps(full_steps, steps):
    """Uniform interval starts, retaining every solver update (20F is an oracle)."""
    if full_steps is None:
        return ()
    if type(full_steps) is not int or full_steps not in (2, 4, 6, 8, 20) or steps != 20:
        raise ValueError("Residual reuse requires 20 solver steps and full_steps in {2,4,6,8,20}")
    return tuple(i * steps // full_steps for i in range(full_steps))


@dataclass
class ResidualCache:
    """Owned by one sample_block call; FP32 residual before final GEN norm."""

    residual: torch.Tensor | None = None
    signature: tuple | None = None
    refresh_count: int = 0
    reuse_count: int = 0

    def update(self, before, after, owner, layout):
        self.residual = (after.float() - before.float()).detach()
        self.signature = (id(owner), layout, before.shape, before.dtype, before.device)
        self.refresh_count += 1

    def apply(self, current, owner, layout):
        if self.residual is None or self.signature != (
            id(owner),
            layout,
            current.shape,
            current.dtype,
            current.device,
        ):
            raise ValueError("Residual cache is empty or does not match this block layout/model/precision")
        self.reuse_count += 1
        return (current.float() + self.residual).to(current.dtype)


@dataclass
class LayerCache:
    text_hidden: torch.Tensor
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


def _attention_groups(plan):
    """Read ordered group metadata on the host once, outside decoder layers."""
    groups = []
    for chunk in plan.chunks:
        queries, keys = chunk.query_indexes.tolist(), chunk.kv_indexes.tolist()
        qs = ks = 0
        for qe, ke in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            groups.append((queries[qs:qe], keys[ks:ke]))
            qs, ks = qe, ke
    return groups


def _compact_attention_plan(plan, rows, kv_global, groups):
    """Remap host groups, joining adjacent queries with identical ordered KV."""
    query_rows = {index: row for row, index in enumerate(rows.tolist())}
    key_rows = {index: row for row, index in enumerate(kv_global.tolist())}
    merged = []
    for queries, keys in groups:
        q, k = [query_rows[i] for i in queries], [key_rows[i] for i in keys]
        if merged and merged[-1][1] == k and len(merged[-1][0]) + len(q) <= 1_000_000:
            merged[-1][0].extend(q)
        else:
            merged.append((q, k))
    tensor = plan.gen_indexes.new_tensor
    chunks = _pack_attention_groups(
        merged, plan.max_kv_tokens, concatenate=lambda parts: tensor(list(chain.from_iterable(parts)))
    )
    return replace(
        plan,
        chunks=chunks,
        query_selection=tensor([query_rows[i] for queries, _ in groups for i in queries]),
        mask=None if plan.mask is None else plan.mask.index_select(0, rows).index_select(1, kv_global),
    )


@dataclass
class PrefillPlan:
    """Dependency-closed prefill queries and their compact, temporary KV storage."""

    rows: torch.Tensor
    kv_global: torch.Tensor
    text_rows: torch.Tensor
    text_kv: torch.Tensor
    gen_kv: torch.Tensor
    warm_kv: torch.Tensor
    plan: object
    tables: tuple[torch.Tensor, torch.Tensor]

    @classmethod
    def prepare(cls, plan, dynamic, warm_kv, tables, history=None):
        gen = plan.gen_indexes.tolist()
        inverse = {g: i for i, g in enumerate(gen)}
        remaining = [(set(q), keys) for q, keys in _attention_groups(plan)]
        cached = set() if history is None else set(history.reused_rows.tolist())
        retained = set(dynamic.tolist())
        if history is not None:
            # Newly observed clean rows are outputs too: the next block needs
            # their snapshots even when this block's targets do not read them.
            retained.update(history.clean_rows.tolist())
        retained.difference_update(cached)
        # Direct target-visible KV is insufficient at a truncated history edge:
        # fresh queries need their dependencies, stopping at cached clean rows.
        while remaining:
            pending = []
            for queries, keys in remaining:
                if queries & retained:
                    retained.update(inverse[k] for k in keys if k in inverse and inverse[k] not in cached)
                else:
                    pending.append((queries, keys))
            if len(pending) == len(remaining):
                break
            remaining = pending
        tensor = plan.gen_indexes.new_tensor
        selected = select_attention_queries(plan, [i in retained for i in range(len(gen))], allow_partial=True)
        rows = selected.query_selection
        groups = _attention_groups(selected)
        visible = sorted(set(warm_kv.tolist()) | {k for _, keys in groups for k in keys})
        key_rows = {g: i for i, g in enumerate(visible)}
        text = [(i, key_rows[g]) for i, g in enumerate(plan.modality_indexes[0].tolist()) if g in key_rows]
        kv_global = tensor(visible)
        if history is not None:
            history.prepare(rows, kv_global, plan, tables)
        return cls(
            rows,
            kv_global,
            tensor([i for i, _ in text]),
            tensor([k for _, k in text]),
            tensor([key_rows[gen[i]] for i in rows.tolist()]),
            tensor([key_rows[g] for g in warm_kv.tolist()]),
            _compact_attention_plan(selected, rows, kv_global, groups),
            tuple(table[rows] for table in tables),
        )


@dataclass
class BlockCache:
    """Keep full hidden layouts; recompute only the current target group.

    Clean video, conditions, state and other blocks have fixed inputs and
    cannot attend the current noisy targets. Target-visible KV is retained in
    shared storage with mutable target slots. Without cross-block history,
    only dependency-closed hidden rows are computed; unused rows remain zero.
    """

    dynamic: torch.Tensor
    kv_global: torch.Tensor
    dynamic_kv: torch.Tensor
    compact_plan: object
    dynamic_tables: tuple[torch.Tensor, torch.Tensor]
    layers: dict = field(default_factory=dict)
    history: object = None
    prefill: PrefillPlan | None = None

    @classmethod
    def prepare(cls, plan, layout, block, tables, *, history=None):
        gen = plan.gen_indexes.tolist()
        keep = [layout.roles[i] == 1 and layout.block_ids[i] == block for i in gen]
        dynamic_rows = [i for i, selected in enumerate(keep) if selected]
        target = select_attention_queries(plan, keep)
        groups = _attention_groups(target)
        visible = sorted({index for _, keys in groups for index in keys})
        tensor = plan.gen_indexes.new_tensor
        dynamic = tensor(dynamic_rows)
        kv_global = tensor(visible)
        key_rows = {index: row for row, index in enumerate(visible)}
        return cls(
            dynamic,
            kv_global,
            tensor([key_rows[gen[i]] for i in dynamic_rows]),
            _compact_attention_plan(target, dynamic, kv_global, groups),
            tuple(table[dynamic] for table in tables),
            history=history,
            prefill=PrefillPlan.prepare(plan, dynamic, kv_global, tables, history),
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
    fresh_clean: torch.Tensor = field(init=False)
    fresh_queries: torch.Tensor = field(init=False)
    reused_clean: torch.Tensor = field(init=False)
    reused_kv: torch.Tensor = field(init=False)
    previous_kv: torch.Tensor = field(init=False)
    reused_tables: tuple[torch.Tensor, torch.Tensor] = field(init=False)

    def prepare(self, rows, kv_global, plan, tables):
        """Resolve compact read/write and snapshot slots once, outside layers."""
        tensor = rows.new_tensor
        clean = {row: i for i, row in enumerate(self.clean_rows.tolist())}
        reused = dict(zip(self.reused_rows.tolist(), self.previous_rows.tolist(), strict=True))
        fresh = [(i, clean[row]) for i, row in enumerate(rows.tolist()) if row in clean]
        self.fresh_queries = tensor([i for i, _ in fresh])
        self.fresh_clean = tensor([i for _, i in fresh])
        self.reused_clean = tensor([clean[row] for row in reused])
        gen = plan.gen_indexes.tolist()
        kv = {g: i for i, g in enumerate(kv_global.tolist())}
        reads = [(row, old, kv[gen[row]]) for row, old in reused.items() if gen[row] in kv]
        self.reused_kv = tensor([slot for _, _, slot in reads])
        self.previous_kv = tensor([old for _, old, _ in reads])
        read_rows = tensor([row for row, _, _ in reads])
        self.reused_tables = tuple(table[read_rows] for table in tables)

    def save(self, layer, text, text_key, text_value, hidden, raw_key, value):
        """Build an owned clean snapshot without allocating full-layout raw KV."""
        old = self.previous.get(layer)
        keys = raw_key.new_empty((len(self.clean_rows), *raw_key.shape[1:]))
        values = value.new_empty((len(self.clean_rows), *value.shape[1:]))
        keys.index_copy_(0, self.fresh_clean, raw_key[self.fresh_queries])
        values.index_copy_(0, self.fresh_clean, value[self.fresh_queries])
        if old is not None:
            keys.index_copy_(0, self.reused_clean, old.key[self.previous_rows])
            values.index_copy_(0, self.reused_clean, old.value[self.previous_rows])
        self.following[layer] = HistoryLayer(text, text_key, text_value, hidden[self.clean_rows], keys, values)
        # Only the prefill evaluation reads the previous snapshot. Release each
        # consumed layer immediately instead of retaining two full histories
        # throughout the rest of this block's denoising steps.
        self.previous.pop(layer, None)


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
            # A partially executed/failed forward is not a reusable snapshot.
            if any(layer not in previous for layer in model.layers):
                old_ids, previous = (), {}
        gen = plan.gen_indexes.tolist()
        # Never persist clean placeholders for the current or future blocks.
        start = 1 + block * sample["block_size"]
        offset = (sequence_block - block) * sample["block_size"]
        clean = [i for i, g in enumerate(gen) if layout.roles[g] == 0 and layout.frame_ids[g] < start]
        patches = sum(layout.roles[g] == 0 and layout.frame_ids[g] == 0 for g in gen)
        ids = [(layout.frame_ids[gen[i]] + offset, i % patches) for i in clean]
        lookup = {key: i for i, key in enumerate(old_ids)}
        matched = [(row, lookup[key]) for row, key in zip(clean, ids, strict=True) if key in lookup]
        following = {}
        self.branches[branch] = (signature, sequence_block, ids, following)
        tensor = plan.gen_indexes.new_tensor
        return HistoryBinding(
            dict(previous),
            following,
            tensor(clean),
            tensor([row for row, _ in matched]),
            tensor([row for _, row in matched]),
        )


@dataclass
class BoundaryRows:
    """Modality payload and local GEN indexes for a sampler-only row selection."""

    rows: torch.Tensor
    payload: tuple[torch.Tensor, ...]
    local: tuple[torch.Tensor, ...]

    @classmethod
    def prepare(cls, modality_rows, rows):
        selected = {row: i for i, row in enumerate(rows.tolist())}
        payload, local = [], []
        for indexes in modality_rows:
            pairs = [(i, selected[g]) for i, g in enumerate(indexes) if g in selected]
            payload.append(rows.new_tensor([i for i, _ in pairs]))
            local.append(rows.new_tensor([i for _, i in pairs]))
        return cls(rows, tuple(payload), tuple(local))


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
        compact_boundary=False,
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
                    history = None
                    if history_cache is not None:
                        history = history_cache.bind(
                            model,
                            sample,
                            block,
                            block if sequence_block is None else sequence_block,
                            branch,
                            prepared.attention,
                            layout,
                            position_options,
                        )
                    prepared.fixed_attention = BlockCache.prepare(
                        prepared.attention, layout, block, prepared.tables[1], history=history
                    )
                    if compact_boundary:
                        cache = prepared.fixed_attention
                        modality_rows = tuple(indexes.tolist() for indexes in prepared.rows)
                        prepared.compact_boundary = (
                            BoundaryRows.prepare(modality_rows, cache.prefill.rows),
                            BoundaryRows.prepare(modality_rows, cache.dynamic),
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
