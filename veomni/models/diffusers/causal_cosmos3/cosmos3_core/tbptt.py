# SPDX-License-Identifier: OpenMDW-1.1
"""Forward-local metadata for Action and Video history-gradient truncation."""

from dataclasses import dataclass, replace

import torch

from .attention import AttentionPlan, TNDChunk, contiguous_span


@dataclass(frozen=True)
class HistoryPartition:
    """Disjoint GEN query rows; each group retains its full ordered K/V set."""

    old: torch.Tensor
    active: torch.Tensor
    old_plan: AttentionPlan
    active_plan: AttentionPlan


@dataclass(frozen=True)
class VideoHistoryPartition:
    """Route cross-chunk clean keys to readouts of detached hidden states."""

    memory_rows: torch.Tensor
    plan: AttentionPlan


def _partition_video_history(plan, chunk_latents):
    """Preserve every teacher-forcing target and cut clean cross-chunk edges.

    Complete prediction blocks belong to one temporal chunk; the independent
    first latent belongs to chunk zero. All rows keep their ordinary residual,
    Q and MLP gradients. Only a read of a prior chunk's clean hidden state uses
    detached input to the still-trainable norm/K/V projections.
    """
    if type(chunk_latents) is not int or chunk_latents < 1:
        raise ValueError("tbptt_keep_history_latents must be a positive integer")
    layout = plan.layout
    if layout is None or any(modality > 1 for modality in layout.modalities):
        raise ValueError("TBPTT requires Action decision-window or video-only layouts")
    if any(chunk_latents % size for size in layout.block_sizes):
        raise ValueError("tbptt_keep_history_latents must be divisible by each video's latent block size")
    chunk_ids = [
        max(0, block) // (chunk_latents // layout.block_sizes[sample])
        for sample, block in zip(layout.sample_ids, layout.block_ids, strict=True)
    ]
    gen_rows = [i for i, role in enumerate(layout.roles) if role != -1]
    memory_keys = set()
    routed = []
    for chunk in plan.chunks:
        queries = chunk.query_indexes.detach().cpu().tolist()
        keys = chunk.kv_indexes.detach().cpu().tolist()
        qstart = kstart = 0
        detached = []
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            query = gen_rows[queries[qstart]]
            for index in range(kstart, kend):
                key = keys[index]
                if layout.roles[key] == 0 and chunk_ids[key] < chunk_ids[query]:
                    detached.append(index)
                    memory_keys.add(key)
            qstart, kstart = qend, kend
        routed.append((chunk, keys, detached))
    if not memory_keys:
        return None
    memory_keys = sorted(memory_keys)
    appended = {key: len(layout.roles) + offset for offset, key in enumerate(memory_keys)}
    chunks = []
    for chunk, keys, detached in routed:
        for index in detached:
            keys[index] = appended[keys[index]]
        chunks.append(
            TNDChunk.from_lists(
                chunk.query_indexes.detach().cpu().tolist(),
                keys,
                chunk.query_ends,
                chunk.kv_ends,
                device=plan.gen_indexes.device,
            )
        )
    mask = None
    if plan.mask is not None:
        # Keep dense as an independent attention backend with the same routed
        # visibility; do not create a dense mask for grouped/NPU TND execution.
        mask = torch.zeros(
            (len(gen_rows), len(layout.roles) + len(memory_keys)), dtype=torch.bool, device=plan.mask.device
        )
        for chunk in chunks:
            qstart = kstart = 0
            for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
                queries = chunk.query_indexes[qstart:qend]
                keys = chunk.kv_indexes[kstart:kend]
                mask[queries[:, None], keys[None, :]] = True
                qstart, kstart = qend, kend
    row_lookup = {packed: row for row, packed in enumerate(gen_rows)}
    return VideoHistoryPartition(
        memory_rows=torch.tensor(
            [row_lookup[key] for key in memory_keys], dtype=torch.long, device=plan.gen_indexes.device
        ),
        plan=replace(plan, chunks=tuple(chunks), mask=mask),
    )


def _query_subset(plan, rows):
    device = plan.gen_indexes.device
    row_lookup = {original: selected for selected, original in enumerate(rows)}
    chunks = []
    for chunk in plan.chunks:
        original_queries = chunk.query_indexes.detach().cpu().tolist()
        query_rows, key_parts, query_ends, key_ends = [], [], [], []
        query_start = key_start = total_keys = 0
        for query_end, key_end in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            chosen = [row_lookup[row] for row in original_queries[query_start:query_end] if row in row_lookup]
            if chosen:
                query_rows.extend(chosen)
                key_parts.append(chunk.kv_indexes[key_start:key_end])
                total_keys += key_end - key_start
                query_ends.append(len(query_rows))
                key_ends.append(total_keys)
            query_start, key_start = query_end, key_end
        if query_rows:
            chunks.append(
                TNDChunk(
                    query_indexes=torch.tensor(query_rows, dtype=torch.long, device=device),
                    kv_indexes=key_parts[0] if len(key_parts) == 1 else torch.cat(key_parts),
                    query_ends=tuple(query_ends),
                    kv_ends=tuple(key_ends),
                    query_span=contiguous_span(query_rows),
                    # K/V rows can repeat across groups; preserve the gather.
                    kv_span=None,
                )
            )
    selected = torch.tensor(rows, dtype=torch.long, device=device)
    return replace(
        plan,
        gen_indexes=plan.gen_indexes.index_select(0, selected),
        chunks=tuple(chunks),
        mask=None if plan.mask is None else plan.mask.index_select(0, selected),
    )


def partition_history(plan, keep_history_latents):
    """Partition one packed forward; never retain historical hidden states/KV.

    Action detaches the earliest observed encoder rows beyond recent K. Video
    retains every teacher-forcing query and detaches cross-temporal-chunk memory
    reads instead. Neither policy changes attention visibility or target masks.
    Return None when no history edge needs truncation, preserving the native path.
    """
    if type(keep_history_latents) is not int or keep_history_latents < 1:
        raise ValueError("tbptt_keep_history_latents must be a positive integer")
    layout = plan.layout
    windows = None if layout is None else layout.decision_windows
    if not windows or not any(windows):
        return _partition_video_history(plan, keep_history_latents)
    if any(value <= 0 for value in windows):
        raise ValueError("TBPTT cannot mix Action decision-window and block layouts")
    old, active = [], []
    for row, packed in enumerate(i for i, role in enumerate(layout.roles) if role != -1):
        sample = layout.sample_ids[packed]
        is_old = layout.roles[packed] == 0 and layout.frame_ids[packed] < max(
            0, windows[sample] - keep_history_latents
        )
        (old if is_old else active).append(row)
    if not old:
        return None
    device = plan.gen_indexes.device
    return HistoryPartition(
        old=torch.tensor(old, dtype=torch.long, device=device),
        active=torch.tensor(active, dtype=torch.long, device=device),
        old_plan=_query_subset(plan, old),
        active_plan=_query_subset(plan, active),
    )
