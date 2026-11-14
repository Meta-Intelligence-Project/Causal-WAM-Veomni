# SPDX-License-Identifier: OpenMDW-1.1
"""Per-expert causal visibility with packed KV and compact, ordered queries.

The dense reference builds its mask independently. Grouped/NPU execution uses
CPU token buckets and contiguous query runs, never a quadratic pair mask.
"""

from collections import defaultdict
from itertools import accumulate, groupby

import torch

from veomni.utils.import_utils import is_torch_npu_available

from .attention import AttentionPlan, TNDChunk


def _dense_branch_mask(
    layout, query_indexes, task_modes, branch, include_conditioning, device, video_conditioning, text_conditioning
):
    sample, block, role, modality, frame = (
        torch.tensor(field, dtype=torch.long, device=device)
        for field in (layout.sample_ids, layout.block_ids, layout.roles, layout.modalities, layout.frame_ids)
    )
    qs, qb, qr, qm = (field[query_indexes, None] for field in (sample, block, role, modality))
    ks, kb, kr, km = (field[None, :] for field in (sample, block, role, modality))
    same_block = kb == qb
    text = (km == 0) & torch.tensor(text_conditioning, dtype=torch.bool, device=device)[qs]
    state = same_block & (km == 3)
    known_action = same_block & (km == 2) & (kr == 2)
    clean_same = same_block & (km == 1) & (kr == 0)
    video_same = same_block & (km == 1) & ((kr == 1) | (kr == 2))
    action_same = same_block & (km == 2) & ((kr == 1) | (kr == 2))
    history = torch.tensor(layout.history_blocks, device=device)[qs]
    block_size = torch.tensor(layout.block_sizes, device=device)[qs]
    start = (1 + qb * block_size).clamp_min(0)
    past = (km == 1) & (kr == 0) & (kb < qb) & (frame[None, :] >= start - history) & (frame[None, :] < start)
    if branch == "video":
        current = text | past | video_same
        if include_conditioning:
            fd = torch.tensor([mode == "forward_dynamics" for mode in task_modes], dtype=torch.bool, device=device)[qs]
            enabled = torch.tensor(video_conditioning, dtype=torch.bool, device=device)[qs]
            current = current | (enabled & (state | (fd & known_action)))
        visible = torch.where(qr == 0, text | past | clean_same, current)
    elif branch == "condition":
        visible = text | state | ((qm == 2) & known_action)
    else:
        target = text | past | video_same | action_same | state
        condition = text | known_action | state
        visible = torch.where(qm == 3, text | state, torch.where(qr == 2, condition, target))
    return (qs == ks) & visible


def prepare_branch_attention(
    layout,
    device,
    backend,
    query_modalities: tuple[int, ...],
    task_modes: tuple[str, ...],
    branch: str,
    include_conditioning: bool = True,
    video_conditioning: tuple[bool, ...] | None = None,
    text_conditioning: tuple[bool, ...] | None = None,
):
    """Prepare expert queries against full packed KV for the existing attention kernel.

    ``gen_indexes`` contains packed-global query indexes. Query tensors passed
    to ``generation_attention`` must follow that order; chunk query indexes are
    relative to this compact tensor. The returned modality indexes remain global.
    Video accepts all video rows. Condition accepts state and known FD actions.
    Action accepts action/state rows of inverse-dynamics samples only.
    video_conditioning gates external action/state KV for video queries only;
    text_conditioning gates text KV for every generation branch, including
    state/condition encoding, so an action-only FDM cannot recover text indirectly.
    """
    if backend not in ("dense", "grouped", "npu_tnd"):
        raise ValueError("attention_backend must be dense, grouped, or npu_tnd")
    if backend == "npu_tnd" and (torch.device(device).type != "npu" or not is_torch_npu_available()):
        raise ValueError("npu_tnd requires an Ascend NPU; use grouped for a CPU reference")
    if branch not in ("video", "condition", "action"):
        raise ValueError("branch must be video, condition, or action")
    if len(task_modes) != len(layout.history_blocks) or any(
        mode not in ("video", "forward_dynamics", "inverse_dynamics") for mode in task_modes
    ):
        raise ValueError("task_modes must provide video/forward_dynamics/inverse_dynamics for every sample")
    if not isinstance(query_modalities, tuple) or any(m not in (1, 2, 3) for m in query_modalities):
        raise ValueError("query_modalities must be a tuple containing only video/action/state modality IDs")

    def conditioning_flags(values, name):
        if values is None:
            return (True,) * len(task_modes)
        if len(values) != len(task_modes) or any(type(value) is not bool for value in values):
            raise ValueError(f"{name} must provide one boolean per sample")
        return values

    video_conditioning = conditioning_flags(video_conditioning, "video_conditioning")
    text_conditioning = conditioning_flags(text_conditioning, "text_conditioning")

    def indexes(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    modalities = [[], [], [], []]
    buckets, clean_frames, text, text_local = (defaultdict(list) for _ in range(4))
    queries, text_count = [], 0
    for i, (sample, block, role, modality, frame) in enumerate(
        zip(
            layout.sample_ids,
            layout.block_ids,
            layout.roles,
            layout.modalities,
            layout.frame_ids,
            strict=True,
        )
    ):
        modalities[modality].append(i)
        if modality == 0:
            text[sample].append(i)
            text_local[sample].append(text_count)
            text_count += 1
        else:
            buckets[sample, modality, role, block].append(i)
            if modality == 1 and role == 0:
                clean_frames[sample, frame].append(i)
        if modality not in query_modalities:
            continue
        if branch == "video":
            selected = modality == 1
        elif branch == "condition":
            selected = modality == 3 or (modality == 2 and role == 2 and task_modes[sample] == "forward_dynamics")
        else:
            selected = modality in (2, 3) and task_modes[sample] == "inverse_dynamics"
        if selected:
            queries.append(i)

    modality_indexes = tuple(indexes(items) for items in modalities)
    text_samples = tuple(indexes(items) for items in text_local.values())
    text_indexes = indexes([index for items in text_local.values() for index in items])
    text_ends = tuple(accumulate(map(len, text_local.values())))
    if backend == "npu_tnd" and (len(text_ends) > 1000 or len(text_indexes) > 1_000_000):
        raise ValueError("UND attention exceeds the NPU TND token/sequence limits")

    key_cache = {}

    def signature(pair):
        i = pair[1]
        return layout.sample_ids[i], layout.roles[i], layout.block_ids[i], layout.modalities[i]

    def visible_keys(sample, role, block, modality):
        signature = sample, role, block, modality
        if signature in key_cache:
            return key_cache[signature]
        keys = list(text[sample]) if text_conditioning[sample] else []

        def same(m, roles):
            for r in roles:
                keys.extend(buckets[sample, m, r, block])

        def history():
            start = max(0, 1 + block * layout.block_sizes[sample])
            for frame in range(max(0, start - layout.history_blocks[sample]), start):
                keys.extend(clean_frames[sample, frame])

        if branch == "video":
            history()
            same(1, (0,) if role == 0 else (1, 2))
            if role != 0 and include_conditioning and video_conditioning[sample]:
                same(3, (3,))
                if task_modes[sample] == "forward_dynamics":
                    same(2, (2,))
        elif modality == 3:
            same(3, (3,))
        elif branch == "condition" or role == 2:
            same(2, (2,))
            same(3, (3,))
        else:
            history()
            same(1, (1, 2))
            same(2, (1, 2))
            same(3, (3,))
        keys.sort()
        key_cache[signature] = keys
        return keys

    chunks, qi, ki, qe, ke = [], [], [], [], []

    def flush():
        if qe:
            chunks.append(TNDChunk(indexes(qi), indexes(ki), tuple(qe), tuple(ke)))
            qi.clear()
            ki.clear()
            qe.clear()
            ke.clear()

    # Group contiguous runs only. Regrouping all equal signatures would reorder
    # outputs because generation_attention concatenates groups without scattering.
    for (sample, role, block, modality), items in groupby(enumerate(queries), key=signature):
        compact_queries = [offset for offset, _ in items]
        keys = visible_keys(sample, role, block, modality)
        if len(keys) > 1_000_000 or len(compact_queries) > 1_000_000:
            raise ValueError("One attention group exceeds the NPU TND token limit")
        if qe and (len(ki) + len(keys) > 131072 or len(qi) + len(compact_queries) > 1_000_000 or len(qe) == 1000):
            flush()
        qi.extend(compact_queries)
        ki.extend(keys)
        qe.append(len(qi))
        ke.append(len(ki))
    flush()
    query_indexes = indexes(queries)
    mask = (
        _dense_branch_mask(
            layout,
            query_indexes,
            task_modes,
            branch,
            include_conditioning,
            device,
            video_conditioning,
            text_conditioning,
        )
        if backend == "dense"
        else None
    )
    return AttentionPlan(
        modality_indexes,
        query_indexes,
        text_samples,
        text_indexes,
        text_ends,
        tuple(chunks),
        mask,
        backend,
    )
