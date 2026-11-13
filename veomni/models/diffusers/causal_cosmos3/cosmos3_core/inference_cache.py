# SPDX-License-Identifier: OpenMDW-1.1
"""Request-owned fixed-window and verified history KV reuse, isolated by CFG branch."""

from dataclasses import dataclass, replace

import torch

from .attention import AttentionPlan, TNDChunk, prepare_attention


@dataclass
class TextLayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    hidden: torch.Tensor | None = None


class TextKVCache:
    """Verify text identity and coordinates across rollout stages/windows."""

    def __init__(self):
        self.signature = None
        self.ids = self.positions = None
        self.layers = []

    def prepare(self, model, ids, positions, backend, sample_ids):
        signature = (id(model), ids.device, next(model.parameters()).dtype, backend, sample_ids)
        same = signature == self.signature and torch.equal(ids, self.ids) and torch.equal(positions, self.positions)
        if not same:
            self.signature = signature
            self.ids, self.positions = ids.clone(), positions.clone()
            self.layers = [TextLayerCache() for _ in model.layers]
        return self.layers


class InferenceKVCache:
    """One request, immutable weights. Never attach this object to the model.

    Teacher forcing owns one immutable encoded sample, allowing full fixed-row
    reuse. Rollout verifies clean history prefixes before reusing their layers.
    Sample identity is an ownership contract, not a content hash:
    the teacher-forcing caller must not mutate its encoded sample in place.
    """

    def __init__(self, *, reuse_window=False):
        self.reuse_window = reuse_window
        self.sample = None
        self.blocks = None
        self.text = {branch: TextKVCache() for branch in ("conditional", "unconditional")}
        self.history = {branch: VideoHistoryCache() for branch in self.text}

    def for_sample(self, sample, *, history_frames=0, window_start=0):
        if self.reuse_window and sample is self.sample:
            return self.blocks
        blocks = {
            branch: BlockKVCache(
                text=text,
                history=None if self.reuse_window else self.history[branch],
                history_frames=history_frames,
                window_start=window_start,
            )
            for branch, text in self.text.items()
        }
        if self.reuse_window:
            self.sample, self.blocks = sample, blocks
        return blocks

    def history_stats(self):
        """Cumulative token counts across CFG branches, once per sampling prefill."""
        return {
            name: sum(cache.stats[name] for cache in self.history.values())
            for name in next(iter(self.history.values())).stats
        }


@dataclass
class HistoryLayer:
    key: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor


@dataclass
class HistorySnapshot:
    signature: tuple
    text_layers: list
    tokens: torch.Tensor
    times: torch.Tensor
    positions: torch.Tensor
    frames: tuple[int, ...]
    layers: list


@dataclass
class HistoryBinding:
    owner: object
    following: HistorySnapshot
    previous: HistorySnapshot | None
    rows: torch.Tensor
    kv_rows: torch.Tensor
    reused_rows: torch.Tensor
    reused_kv: torch.Tensor
    fresh_rows: torch.Tensor
    plan: AttentionPlan

    def save(self, index, key, value, hidden):
        # Advanced indexing owns storage; never retain the mutable full KV workspace.
        self.following.layers[index] = HistoryLayer(key[self.kv_rows], value[self.kv_rows], hidden[self.rows])

    def commit(self):
        if all(layer is not None for layer in self.following.layers):
            self.owner.snapshot = self.following
        self.previous = None


class VideoHistoryCache:
    """Reuse only a dependency-closed, exactly matching clean history prefix.

    No decoded/re-encoded latent is assumed immutable. A changed frame invalidates
    its whole causal block and every subsequent block. A shifted local window
    invalidates all visual history, even when some frame IDs overlap.
    """

    def __init__(self):
        self.snapshot = None
        self.stats = dict(prefill_calls=0, candidate_tokens=0, reused_tokens=0)

    def bind(self, model, plan, layout, text_layers, positions, tokens, times, history_frames, window_start):
        gen = [i for i, role in enumerate(layout.roles) if role != -1]
        rows = [i for i, g in enumerate(gen) if layout.roles[g] == 0 and layout.frame_ids[g] < history_frames]
        tensor = plan.gen_indexes.new_tensor
        rows = tensor(rows)
        kv_rows = plan.gen_indexes[rows]
        video_rows = torch.searchsorted(plan.modality_indexes[1], kv_rows)
        clean, time, pos = tokens[video_rows], times[video_rows], positions[:, kv_rows]
        frames = tuple(
            layout.frame_ids[g] for g in gen if layout.roles[g] == 0 and layout.frame_ids[g] < history_frames
        )
        signature = (
            id(model),
            clean.device,
            clean.dtype,
            time.dtype,
            pos.dtype,
            plan.backend,
            layout.modes,
            layout.block_sizes,
            layout.history_blocks,
            window_start,
            clean.shape[1:],
            sum(frame == 0 for frame in frames),
        )
        previous = self.snapshot
        reused = 0
        if previous is not None and previous.signature == signature and previous.text_layers is text_layers:
            count = min(len(frames), len(previous.frames))
            if frames[:count] == previous.frames[:count]:
                same = (
                    (clean[:count] == previous.tokens[:count]).all(dim=1)
                    & (time[:count] == previous.times[:count])
                    & (pos[:, :count] == previous.positions[:, :count]).all(dim=0)
                )
                matches = same.cpu().tolist()  # One host transfer per new window, never per layer/step.
                first_bad = next((i for i, equal in enumerate(matches) if not equal), count)
                if first_bad == count:
                    reused = count
                else:
                    frame = frames[first_bad]
                    # Clean frames within the same block attend each other.
                    block_start = 0 if frame == 0 else 1 + (frame - 1) // layout.block_sizes[0] * layout.block_sizes[0]
                    reused = sum(frame < block_start for frame in frames[:count])
        # These advanced-index results already own storage independently of inputs.
        following = HistorySnapshot(signature, text_layers, clean, time, pos, frames, [None for _ in model.layers])
        # Failed/partial prefills must never become the source of a later reuse.
        self.snapshot = None
        self.stats["prefill_calls"] += 1
        self.stats["candidate_tokens"] += len(rows)
        self.stats["reused_tokens"] += reused
        reused_rows = rows[:reused]
        skipped = set(reused_rows.tolist())
        fresh = tensor([i for i in range(len(gen)) if i not in skipped])
        return HistoryBinding(
            self,
            following,
            previous if reused else None,
            rows,
            kv_rows,
            reused_rows,
            kv_rows[:reused],
            fresh,
            _select_queries(plan, fresh) if reused else plan,
        )


@dataclass
class LayerKVCache:
    dynamic_rows: torch.Tensor
    fixed_rows: torch.Tensor
    dynamic_kv_rows: torch.Tensor
    plan: AttentionPlan
    text: TextLayerCache | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    und: torch.Tensor | None = None
    fixed_gen: torch.Tensor | None = None
    history: HistoryBinding | None = None
    layer_index: int = 0

    @property
    def ready(self):
        return self.fixed_gen is not None


def _dynamic_plan(plan, layout):
    """Keep the original KV ordering, but remove fixed GEN query groups.

    Clean video (role 0) and state (role 3) only depend on text and other fixed
    rows in all supported task masks. Conditions (role 2) stay dynamic: IDM
    condition queries can depend on target rows, so caching all conditions is
    not generally valid. No quadratic mask is introduced for grouped/TND.
    """
    gen = [i for i, role in enumerate(layout.roles) if role != -1]
    dynamic = [i for i, packed in enumerate(gen) if layout.roles[packed] not in (0, 3)]
    fixed = [i for i, packed in enumerate(gen) if layout.roles[packed] in (0, 3)]
    dynamic_rows = plan.gen_indexes.new_tensor(dynamic)
    fixed_rows = plan.gen_indexes.new_tensor(fixed)
    return dynamic_rows, fixed_rows, _select_queries(plan, dynamic_rows)


def _select_queries(plan, rows):
    """Select GEN queries without changing their ordered, full-layout KV keys."""
    remap = {row: i for i, row in enumerate(rows.tolist())}
    device = plan.gen_indexes.device

    def indexes(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    chunks = []
    for chunk in plan.chunks:
        old_queries = chunk.query_indexes.tolist()
        queries, kv, qends, kends = [], [], [], []
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            selected = [remap[row] for row in old_queries[qstart:qend] if row in remap]
            if selected:
                queries.extend(selected)
                kv.append(chunk.kv_indexes[kstart:kend])
                qends.append(len(queries))
                kends.append((kends[-1] if kends else 0) + kend - kstart)
            qstart, kstart = qend, kend
        if queries:
            chunks.append(TNDChunk(indexes(queries), torch.cat(kv), tuple(qends), tuple(kends)))
    return replace(
        plan,
        gen_indexes=plan.gen_indexes[rows],
        chunks=tuple(chunks),
        mask=plan.mask[rows] if plan.mask is not None else None,
    )


class BlockKVCache:
    """Private inference state for one fixed encoded window and text branch.

    The request owner guarantees fixed text, first video copy, states, positions
    and model weights for this lifetime. The first forward prefills each layer;
    later forwards reuse fixed K/V and fixed layer outputs. Dynamic K/V slots
    are overwritten every step. This stores full K/V workspaces as well as
    fixed hidden states, not just a compact static-token KV tensor.
    """

    def __init__(self, text=None, history=None, history_frames=0, window_start=0):
        self.plan = None
        self.layers = []
        self.signature = None
        self.text = text or TextKVCache()
        self.history = history
        self.history_frames = history_frames
        self.window_start = window_start
        self.binding = None

    def prepare(self, model, layout, device, backend, input_ids, position_ids, video_tokens, video_timesteps):
        if model.training or torch.is_grad_enabled():
            raise ValueError("KV reuse requires model.eval() and disabled gradients")
        signature = (id(model), layout, device, backend)
        if self.plan is None:
            self.signature = signature
            self.plan = prepare_attention(layout, device, backend)
            dynamic, fixed, dynamic_plan = _dynamic_plan(self.plan, layout)
            text_samples = tuple(s for s, r in zip(layout.sample_ids, layout.roles, strict=True) if r == -1)
            text_layers = self.text.prepare(
                model, input_ids, position_ids[:, self.plan.modality_indexes[0]], backend, text_samples
            )
            if self.history is not None and self.history_frames:
                self.binding = self.history.bind(
                    model,
                    self.plan,
                    layout,
                    text_layers,
                    position_ids,
                    video_tokens,
                    video_timesteps,
                    self.history_frames,
                    self.window_start,
                )
            self.layers = [
                LayerKVCache(
                    dynamic,
                    fixed,
                    dynamic_plan.gen_indexes,
                    dynamic_plan,
                    text=text,
                    history=self.binding,
                    layer_index=index,
                )
                for index, text in enumerate(text_layers)
            ]
        elif signature != self.signature:
            raise ValueError("A block KV cache cannot be reused with another model, layout, device or backend")
        return self.plan

    def finish_prefill(self):
        if self.binding is not None:
            self.binding.commit()
            for layer in self.layers:
                layer.history = None
            self.binding = None
