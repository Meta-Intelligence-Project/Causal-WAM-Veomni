# SPDX-License-Identifier: OpenMDW-1.1
"""Packed independent episode streams through one decoder-layer traversal.

Payloads are modality-major, with each modality retaining stream order. Attention
indexes carry stream isolation; repeated frame IDs and RoPE coordinates across
episodes are valid. Only metadata preparation and snapshot splitting loop over
streams. Projections, attention dispatch and MLPs share each layer invocation.
"""

from dataclasses import dataclass, replace
from itertools import groupby

import torch

from .layers import rotary_tables
from .segment import SegmentLayerContext, SegmentMemory, _attention_groups, _plan, _prepare


def _slices(lengths, start=0):
    spans = []
    for length in lengths:
        spans.append(slice(start, start + length))
        start += length
    return tuple(spans), start


def _rows(span):
    return list(range(span.start, span.stop))


@dataclass(frozen=True)
class _StreamRows:
    text: slice
    old: slice
    clean: slice
    future: slice
    action: slice
    state: slice
    cold: slice

    @property
    def active(self):
        return _rows(self.clean) + _rows(self.future) + _rows(self.action) + _rows(self.state)


def _layout(metas, segments, memories):
    text, text_count = _slices(len(meta["input_ids"]) for meta in metas)
    old, old_count = _slices(len(meta["old_frames"]) * meta["patches"] for meta in metas)
    clean, clean_count = _slices(len(segment["clean_video_tokens"]) for segment in segments)
    future, future_end = _slices((len(segment["future_video_tokens"]) for segment in segments), clean_count)
    action, action_end = _slices((len(segment["action_tokens"]) for segment in segments), future_end)
    state, active_count = _slices((len(segment["state_tokens"]) for segment in segments), action_end)
    cold, cold_count = _slices(
        span.stop - span.start if memory is None else 0 for span, memory in zip(old, memories, strict=True)
    )
    streams = tuple(_StreamRows(*items) for items in zip(text, old, clean, future, action, state, cold, strict=True))
    return streams, text_count, old_count, clean_count, future_end, action_end, active_count, cold_count


def _remap(groups, query_map, key_map):
    result = []
    for queries, keys in groups:
        queries = [query_map[index] for index in queries]
        keys = [key_map[index] for index in keys]
        # A B1 group can span future/action modality boundaries. Packing another
        # stream between those rows splits it into separate contiguous groups.
        for _, entries in groupby(enumerate(queries), key=lambda item: item[1] - item[0]):
            result.append(([row for _, row in entries], keys))
    return result


def _ordered(groups, count):
    groups.sort(key=lambda group: group[0][0])
    # Both grouped and NPU attention concatenate outputs in query-group order.
    if [row for queries, _ in groups for row in queries] != list(range(count)):
        raise ValueError("Packed segment attention must cover each query once in original row order")
    return groups


def _text_boundaries(plan, streams):
    return replace(
        plan,
        text_samples=tuple(plan.text_indexes[stream.text] for stream in streams),
        text_ends=tuple(stream.text.stop for stream in streams),
    )


def _plans(metas, streams, memories, text_count, old_count, active_count, cold_count, backend):
    active_groups, cold_groups, cold_rows = [], [], []
    for meta, stream, memory in zip(metas, streams, memories, strict=True):
        groups, prefix = _attention_groups(meta)
        key_map = (
            _rows(stream.text)
            + [text_count + row for row in _rows(stream.old)]
            + [text_count + old_count + row for row in stream.active]
        )
        active_groups.extend(_remap(groups, stream.active, key_map))
        if memory is None:
            cold_groups.extend(_remap(prefix, _rows(stream.cold), key_map))
            cold_rows.extend(_rows(stream.old))
    device = metas[0]["device"]
    active = _plan(
        _ordered(active_groups, active_count), text_count + old_count + active_count, text_count, device, backend
    )
    active = _text_boundaries(active, streams)
    cold = None
    if cold_count:
        cold = _plan(_ordered(cold_groups, cold_count), text_count + old_count, text_count, device, backend)
        cold = _text_boundaries(cold, streams)
    return active, cold, torch.tensor(cold_rows, dtype=torch.long, device=device)


def forward_segment_batch(model, segments, memories):
    """Run independent prepared streams together, returning stream-major outputs.

    ``segments`` and ``memories`` are equally sized nonempty tuples. A stream
    owns one segment, its text/context and its immutable history snapshot. Mixed
    cold/continued streams and different decision/text lengths are supported.
    Returned memories are detached candidates; the caller commits them only
    after the complete packed forward/backward succeeds. No loss is computed.
    """
    if not isinstance(segments, tuple) or not segments:
        raise ValueError("A packed episode batch requires a nonempty tuple of segments")
    if not isinstance(memories, tuple) or len(memories) != len(segments):
        raise ValueError("A packed episode batch requires one memory entry per segment")
    metas = tuple(_prepare(model, segment, memory) for segment, memory in zip(segments, memories, strict=True))
    if len({meta["context"] for meta in metas}) != len(metas):
        raise ValueError("A packed episode batch may contain only one segment per stream context")
    backend = segments[0].get("attention_backend", "dense")
    device = metas[0]["device"]
    dtype = segments[0]["future_video_tokens"].dtype
    if any(
        meta["device"] != device
        or segment.get("attention_backend", "dense") != backend
        or segment["future_video_tokens"].dtype != dtype
        for segment, meta in zip(segments, metas, strict=True)
    ):
        raise ValueError("Packed episode streams must share device, attention backend and payload dtype")
    streams, text_count, old_count, clean_count, future_end, action_end, active_count, cold_count = _layout(
        metas, segments, memories
    )
    active_plan, cold_plan, cold_rows = _plans(
        metas, streams, memories, text_count, old_count, active_count, cold_count, backend
    )

    def cat(field, *, positions=False):
        return torch.cat([segment[field] for segment in segments], dim=1 if positions else 0)

    vc = torch.tensor(
        [condition for meta in metas for condition in meta["video_conditions"]], dtype=torch.bool, device=device
    )
    ac = torch.tensor(
        [condition for meta in metas for condition in meta["action_conditions"]], dtype=torch.bool, device=device
    )
    future = model.proj_in(cat("future_video_tokens"))
    future = future + model.time_embedder(
        cat("future_video_timesteps").masked_fill(vc, 0) * model.config.timestep_scale
    )
    clean = future[:0]
    if clean_count:
        clean = model.proj_in(cat("clean_video_tokens"))
        clean = clean + model.time_embedder(torch.zeros(clean_count, dtype=torch.float32, device=device))
    action_mask = cat("action_mask")
    action = model.action_proj_in(cat("action_tokens").masked_fill(~action_mask, 0))
    action = (
        action
        + model.action_modality_embed
        + model.time_embedder(cat("action_timesteps").masked_fill(ac, 0) * model.config.timestep_scale)
    )
    state = model.state_proj_in(cat("state_tokens").masked_fill(~cat("state_mask"), 0))
    state = state + model.state_modality_embed
    state = state + model.time_embedder(torch.zeros(len(state), dtype=torch.float32, device=device))
    gen = torch.cat((clean, future, action, state))
    und = model.embed_tokens(torch.cat([meta["input_ids"] for meta in metas]))
    positions = torch.cat(
        (
            torch.cat([meta["text_positions"] for meta in metas], dim=1),
            torch.cat([meta["old_positions"] for meta in metas], dim=1),
            cat("clean_position_ids", positions=True),
            cat("future_position_ids", positions=True),
            cat("action_position_ids", positions=True),
            cat("state_position_ids", positions=True),
        ),
        dim=1,
    )
    cos, sin = rotary_tables(positions, model.config, gen.dtype)
    active_start = text_count + old_count
    tables = ((cos[:text_count], sin[:text_count]), (cos[active_start:], sin[active_start:]))
    memory_tables = (cos[text_count:active_start], sin[text_count:active_start])
    with torch.no_grad():
        prefix = (
            torch.cat(
                [
                    segment["prefix_video_tokens"]
                    for segment, memory in zip(segments, memories, strict=True)
                    if memory is None
                ],
                dim=0,
            )
            if any(memory is None for memory in memories)
            else segments[0]["prefix_video_tokens"][:0]
        )
        cold = model.proj_in(prefix) if cold_count else gen[:0].detach()
        if cold_count:
            cold = cold + model.time_embedder(torch.zeros(cold_count, dtype=torch.float32, device=device))
    for memory in memories:
        if memory is not None and any(value.dtype != gen.dtype for value in memory.layer_inputs):
            raise ValueError("Episode layer memory must match packed hidden dtype")
    snapshots = [[] for _ in segments]
    unchanged = tuple(
        memory is not None and not meta["clean_frames"] for meta, memory in zip(metas, memories, strict=True)
    )
    for layer_index, layer in enumerate(model.layers):
        past = torch.cat(
            [
                cold[stream.cold] if memory is None else memory.layer_inputs[layer_index]
                for stream, memory in zip(streams, memories, strict=True)
            ]
        )
        # These clones are outside checkpointed layer execution. Recomputation
        # observes immutable layer input/context and never edits a stream cache.
        for index, (meta, stream) in enumerate(zip(metas, streams, strict=True)):
            if not unchanged[index]:
                keep = min(meta["history"], len(meta["frames"])) * meta["patches"]
                snapshots[index].append(
                    torch.cat((past[stream.old].detach(), gen[stream.clean].detach()))[-keep:].clone()
                )
        context = SegmentLayerContext(past, *memory_tables, cold_plan, cold_rows)
        if model.gradient_checkpointing and model.training and torch.is_grad_enabled():
            und, gen, cold = model._gradient_checkpointing_func(
                layer, und, gen, tables, active_plan, None, None, context
            )
        else:
            und, gen, cold = layer(und, gen, tables, active_plan, segment_context=context)
    gen = model.norm_moe_gen(gen)
    candidates, statistics = [], []
    for index, (segment, meta, memory) in enumerate(zip(segments, metas, memories, strict=True)):
        retained = min(meta["history"], len(meta["frames"]))
        keep = retained * meta["patches"]
        candidate = memory
        if not unchanged[index]:
            candidate = SegmentMemory(
                frame_ids=meta["frames"][-retained:],
                position_ids=torch.cat((meta["old_positions"], segment["clean_position_ids"]), dim=1)[:, -keep:]
                .detach()
                .clone(),
                layer_inputs=tuple(snapshots[index]),
                input_ids=meta["input_ids"].detach().cpu().clone(),
                text_position_ids=meta["text_positions"].detach().cpu().clone(),
                context_id=meta["context"],
                history_latents=meta["history"],
                patches_per_frame=meta["patches"],
                model_id=id(model),
            )
        candidates.append(candidate)
        statistics.append(
            dict(
                decisions=len(meta["decisions"]),
                new_clean_latents=len(meta["clean_frames"]),
                cold_prefix_latents=len(meta["old_frames"]) if memory is None else 0,
                reused_history_latents=len(meta["old_frames"]) if memory is not None else 0,
                retained_history_latents=retained,
            )
        )
    return dict(
        video=model.proj_out(gen[clean_count:future_end]),
        action=model.action_proj_out(gen[future_end:action_end]).masked_fill(~action_mask, 0),
        segment_memories=tuple(candidates),
        segment_stats=tuple(statistics),
    )
