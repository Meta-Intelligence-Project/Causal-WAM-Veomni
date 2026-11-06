# SPDX-License-Identifier: OpenMDW-1.1
"""Forward-local row metadata for truncated Action history gradients."""

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

    Every sample still exposes its complete decision-window history. Only the
    earliest observed latent rows beyond K are marked for encoder stop-gradient.
    Current state, text, task conditions and future predictions remain active.
    Return None when all observed history fits in K, preserving the native path.
    """
    if type(keep_history_latents) is not int or keep_history_latents < 1:
        raise ValueError("tbptt_keep_history_latents must be a positive integer")
    layout = plan.layout
    windows = None if layout is None else layout.decision_windows
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("TBPTT requires Action decision-window layouts")
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
