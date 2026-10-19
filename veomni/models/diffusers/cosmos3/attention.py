# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scheme B visibility, independent dense reference, and maskless TND groups.

Adapted from Cosmos Ascend main@291e994. Layout metadata lives on the CPU;
indices and masks are prepared once per forward, outside transformer layers.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate, groupby

import torch
from torch.nn import functional as F

from veomni.utils.import_utils import is_torch_npu_available


@dataclass(frozen=True)
class CausalLayout:
    """One entry per packed token, in the order used by position_ids.

    Modalities: text=0, video=1, action=2, state=3.
    Roles: text=-1, clean video=0, target=1, condition=2, state=3.
    Blocks: first latent=-1, then (latent_index - 1) // block_size.
    Each modality's payload follows its occurrence order in this layout.
    """

    sample_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    roles: tuple[int, ...]
    modalities: tuple[int, ...]
    history_blocks: tuple[int, ...]

    def __post_init__(self):
        n = len(self.sample_ids)
        if not n or any(len(x) != n for x in (self.block_ids, self.roles, self.modalities)):
            raise ValueError("Layout fields must have the same nonzero length")
        if any(h < 0 for h in self.history_blocks):
            raise ValueError("history_blocks must be nonnegative")
        for s, b, r, m in zip(self.sample_ids, self.block_ids, self.roles, self.modalities, strict=True):
            if not 0 <= s < len(self.history_blocks) or b < -1:
                raise ValueError("Invalid sample/block index")
            if (m == 0 and b != -1) or (m in (2, 3) and b < 0):
                raise ValueError("Text uses block -1; action/state must follow the independent first latent")
            if (m, r) not in ((0, -1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)):
                raise ValueError("Invalid modality/role combination")


def dense_generation_mask(layout, device):
    """Vectorized oracle, intentionally independent of the grouped planner."""
    s, b, r = (torch.tensor(x, device=device) for x in (layout.sample_ids, layout.block_ids, layout.roles))
    gen = r != -1
    qs, qb, qr = s[gen, None], b[gen, None], r[gen, None]
    ks, kb, kr = s[None, :], b[None, :], r[None, :]
    history = torch.tensor(layout.history_blocks, device=device)[qs]
    past = (kr == 0) & (kb < qb) & (kb >= qb - history)
    same = (kb == qb) & torch.where(qr == 0, kr == 0, torch.where(qr == 2, kr == 2, (kr == 1) | (kr == 2)))
    visible = (kr == -1) | past | same | ((qr != 0) & (kr == 3) & (kb == qb))
    visible = torch.where(qr == 3, (kr == -1) | ((kr == 3) & (kb == qb)), visible)
    return (qs == ks) & visible


@dataclass(frozen=True)
class TNDChunk:
    query_indexes: torch.Tensor
    kv_indexes: torch.Tensor
    query_ends: tuple[int, ...]
    kv_ends: tuple[int, ...]


@dataclass
class AttentionPlan:
    modality_indexes: tuple[torch.Tensor, ...]
    gen_indexes: torch.Tensor
    text_samples: tuple[torch.Tensor, ...]
    text_indexes: torch.Tensor
    text_ends: tuple[int, ...]
    chunks: tuple[TNDChunk, ...]
    mask: torch.Tensor | None
    backend: str


def prepare_attention(layout, device, backend="dense", max_kv_tokens=131072):
    """Group equal (sample, role, block) queries; never allocate a pair mask for TND."""
    if backend not in ("dense", "grouped", "npu_tnd"):
        raise ValueError("attention_backend must be dense, grouped, or npu_tnd")
    if backend == "npu_tnd" and (torch.device(device).type != "npu" or not is_torch_npu_available()):
        raise ValueError("npu_tnd requires an Ascend NPU; use grouped for a CPU reference")
    if not 1 <= max_kv_tokens <= 1_000_000:
        raise ValueError("max_kv_tokens must be in [1, 1000000]")

    def indices(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    modalities = tuple(indices([i for i, m in enumerate(layout.modalities) if m == j]) for j in range(4))
    gen = [i for i, r in enumerate(layout.roles) if r != -1]
    buckets = defaultdict(list)
    text = defaultdict(list)
    for i, (s, r, b) in enumerate(zip(layout.sample_ids, layout.roles, layout.block_ids, strict=True)):
        if r == -1:
            text[s].append(i)
        else:
            buckets[s, r, b].append(i)
    # Text sample indices are relative to the text-only projections.
    text_offsets = {index: offset for offset, index in enumerate(i for i, m in enumerate(layout.modalities) if m == 0)}
    text_samples = tuple(indices([text_offsets[i] for i in ids]) for ids in text.values())
    text_indexes = indices([text_offsets[i] for ids in text.values() for i in ids])
    text_ends = tuple(accumulate(len(ids) for ids in text.values()))
    if backend == "npu_tnd" and (len(text_ends) > 1000 or len(text_indexes) > 1_000_000):
        raise ValueError("UND attention exceeds the NPU TND token/sequence limits")
    chunks, qi, ki, qe, ke = [], [], [], [], []

    def flush():
        if qe:
            chunks.append(TNDChunk(indices(qi), indices(ki), tuple(qe), tuple(ke)))
            qi.clear()
            ki.clear()
            qe.clear()
            ke.clear()

    def signature(pair):
        i = pair[1]
        return layout.sample_ids[i], layout.roles[i], layout.block_ids[i]

    for (sample, role, block), items in groupby(enumerate(gen), key=signature):
        queries = [offset for offset, _ in items]
        keys = list(text[sample])
        if role != 3:
            for past in range(max(-1, block - layout.history_blocks[sample]), block):
                keys.extend(buckets[sample, 0, past])
        current_roles = {0: (0,), 1: (1, 2, 3), 2: (2, 3), 3: (3,)}[role]
        for current in current_roles:
            keys.extend(buckets[sample, current, block])
        keys.sort()
        if len(keys) > 1_000_000 or len(queries) > 1_000_000:
            raise ValueError("One attention group exceeds the NPU TND token limit")
        if qe and (len(ki) + len(keys) > max_kv_tokens or len(qi) + len(queries) > 1_000_000 or len(qe) == 1000):
            flush()
        qi.extend(queries)
        ki.extend(keys)
        qe.append(len(qi))
        ke.append(len(ki))
    flush()
    mask = dense_generation_mask(layout, device) if backend == "dense" else None
    return AttentionPlan(modalities, indices(gen), text_samples, text_indexes, text_ends, tuple(chunks), mask, backend)


def sdpa(query, key, value, mask=None, causal=False):
    repeats = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query.transpose(0, 1),
        key.repeat_interleave(repeats, dim=1).transpose(0, 1),
        value.repeat_interleave(repeats, dim=1).transpose(0, 1),
        attn_mask=mask,
        is_causal=causal,
        dropout_p=0.0,
    ).transpose(0, 1)


@lru_cache(maxsize=8)
def _compressed_causal_mask(device):
    # Ascend sparse_mode=2 uses this fixed-size template, including for T > 2048.
    return torch.ones(2048, 2048, dtype=torch.bool, device=device).triu_(diagonal=1)


def understanding_attention(query, key, value, plan):
    """Packed causal UND attention with native GQA on Ascend."""
    if not query.shape[0]:
        return query
    output = torch.zeros_like(query)
    if plan.backend == "npu_tnd":
        import torch_npu

        indexes = plan.text_indexes
        result = torch_npu.npu_fusion_attention(
            query.index_select(0, indexes),
            key.index_select(0, indexes),
            value.index_select(0, indexes),
            head_num=query.shape[1],
            input_layout="TND",
            atten_mask=_compressed_causal_mask(query.device),
            scale=query.shape[-1] ** -0.5,
            keep_prob=1.0,
            sparse_mode=2,
            actual_seq_qlen=plan.text_ends,
            actual_seq_kvlen=plan.text_ends,
        )[0]
        return output.index_copy(0, indexes, result)
    for indexes in plan.text_samples:
        output = output.index_copy(0, indexes, sdpa(query[indexes], key[indexes], value[indexes], causal=True))
    return output


class _GatherGroups(torch.autograd.Function):
    """Accumulate repeated KV gradients before casting once to the input dtype."""

    @staticmethod
    def forward(ctx, tensor, *indexes):
        ctx.save_for_backward(*indexes)
        ctx.shape, ctx.dtype = tensor.shape, tensor.dtype
        return tuple(tensor.index_select(0, index) for index in indexes)

    @staticmethod
    def backward(ctx, *gradients):
        dtype = torch.float32 if ctx.dtype in (torch.bfloat16, torch.float16) else ctx.dtype
        result = torch.zeros(ctx.shape, dtype=dtype, device=gradients[0].device)
        for index, gradient in zip(ctx.saved_tensors, gradients, strict=True):
            result.index_add_(0, index, gradient.to(dtype))
        return (result.to(ctx.dtype),) + (None,) * len(gradients)


class _NPUTeacherForcing(torch.autograd.Function):
    """Recompute gathered KV per chunk in backward; first-order, no dropout."""

    @staticmethod
    def forward(ctx, query, key, value, chunks):
        import torch_npu

        outputs, statistics, rng = [], [], []
        for chunk in chunks:
            q = query.index_select(0, chunk.query_indexes)
            k = key.index_select(0, chunk.kv_indexes)
            v = value.index_select(0, chunk.kv_indexes)
            result = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=q.shape[1],
                input_layout="TND",
                scale=q.shape[-1] ** -0.5,
                keep_prob=1.0,
                sparse_mode=0,
                actual_seq_qlen=chunk.query_ends,
                actual_seq_kvlen=chunk.kv_ends,
            )
            outputs.append(result[0])
            statistics.extend(result[1:3])
            rng.append(result[4:7])
        output = torch.cat(outputs)
        ctx.save_for_backward(query, key, value, output, *statistics)
        ctx.chunks, ctx.rng = chunks, rng
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        query, key, value, output, *statistics = ctx.saved_tensors
        dq = torch.empty_like(query)
        dk, dv = torch.zeros_like(key, dtype=torch.float32), torch.zeros_like(value, dtype=torch.float32)
        offset = 0
        for i, chunk in enumerate(ctx.chunks):
            end = offset + chunk.query_ends[-1]
            q = query.index_select(0, chunk.query_indexes)
            k = key.index_select(0, chunk.kv_indexes)
            v = value.index_select(0, chunk.kv_indexes)
            seed, rng_offset, numels = ctx.rng[i]
            gradients = torch.ops.npu.npu_fusion_attention_grad.default(
                q,
                k,
                v,
                grad_output[offset:end].contiguous(),
                q.shape[1],
                "TND",
                atten_mask=None,
                softmax_max=statistics[2 * i],
                softmax_sum=statistics[2 * i + 1],
                attention_in=output[offset:end].contiguous(),
                scale_value=q.shape[-1] ** -0.5,
                keep_prob=1.0,
                sparse_mode=0,
                seed=seed,
                offset=rng_offset,
                numels=numels,
                actual_seq_qlen=chunk.query_ends,
                actual_seq_kvlen=chunk.kv_ends,
            )
            dq.index_copy_(0, chunk.query_indexes, gradients[0])
            dk.index_add_(0, chunk.kv_indexes, gradients[1].float())
            dv.index_add_(0, chunk.kv_indexes, gradients[2].float())
            offset = end
        return dq, dk.to(key.dtype), dv.to(value.dtype), None


def generation_attention(query, key, value, plan):
    if not query.shape[0]:
        return query
    if plan.backend == "dense":
        return sdpa(query, key, value, plan.mask)
    if plan.backend == "npu_tnd":
        return _NPUTeacherForcing.apply(query, key, value, plan.chunks)
    indexes = tuple(chunk.kv_indexes for chunk in plan.chunks)
    keys, values = _GatherGroups.apply(key, *indexes), _GatherGroups.apply(value, *indexes)
    outputs = []
    for chunk, k, v in zip(plan.chunks, keys, values, strict=True):
        q = query.index_select(0, chunk.query_indexes)
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            outputs.append(sdpa(q[qstart:qend], k[kstart:kend], v[kstart:kend]))
            qstart, kstart = qend, kend
    return torch.cat(outputs)
