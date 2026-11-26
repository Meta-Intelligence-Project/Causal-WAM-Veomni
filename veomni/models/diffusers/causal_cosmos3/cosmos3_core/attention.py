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
import math

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
    frame_ids: tuple[int, ...]
    block_sizes: tuple[int, ...]

    def __post_init__(self):
        n = len(self.sample_ids)
        if not n or any(len(x) != n for x in (self.block_ids, self.roles, self.modalities, self.frame_ids)):
            raise ValueError("Layout fields must have the same nonzero length")
        if any(h < 0 for h in self.history_blocks):
            raise ValueError("history_blocks must be nonnegative")
        if len(self.block_sizes) != len(self.history_blocks) or any(size < 1 for size in self.block_sizes):
            raise ValueError("Each sample requires a positive latent block size")
        for s, b, r, m, frame in zip(
            self.sample_ids, self.block_ids, self.roles, self.modalities, self.frame_ids, strict=True
        ):
            if not 0 <= s < len(self.history_blocks) or b < -1:
                raise ValueError("Invalid sample/block index")
            if (m == 0 and b != -1) or (m in (2, 3) and b < 0):
                raise ValueError("Text uses block -1; action/state must follow the independent first latent")
            if (m, r) not in ((0, -1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)):
                raise ValueError("Invalid modality/role combination")
            if (m == 1 and (frame < 0 or b != (frame - 1) // self.block_sizes[s])) or (m != 1 and frame != -1):
                raise ValueError("Video frame_ids must match block IDs; other modalities use -1")


def dense_generation_mask(layout, device):
    """Vectorized oracle, intentionally independent of the grouped planner."""
    s, b, r = (torch.tensor(x, device=device) for x in (layout.sample_ids, layout.block_ids, layout.roles))
    gen = r != -1
    qs, qb, qr = s[gen, None], b[gen, None], r[gen, None]
    ks, kb, kr = s[None, :], b[None, :], r[None, :]
    history = torch.tensor(layout.history_blocks, device=device)[qs]
    block_sizes = torch.tensor(layout.block_sizes, device=device)[qs]
    start = (1 + qb * block_sizes).clamp_min(0)
    frames = torch.tensor(layout.frame_ids, device=device)[None, :]
    past = (kr == 0) & (kb < qb) & (frames >= start - history) & (frames < start)
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
    kv_scales: torch.Tensor | None = None


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
    dense_logit_scales: torch.Tensor | None = None


def compute_txt_steering_scale(timesteps, *, scale, min_timestep=250.0, max_timestep=1000.0):
    """Piecewise-linear cross-modal text steering temperature gamma(t); t <= min_timestep stays 1."""
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (scale, min_timestep, max_timestep)
        )
        or float(scale) <= 0.0
        or not 0.0 <= float(min_timestep) < float(max_timestep) <= 1000.0
    ):
        raise ValueError("scale must be finite positive and 0 <= min_timestep < max_timestep <= 1000")
    ramp = ((timesteps.float() - float(min_timestep)) / float(max_timestep - min_timestep)).clamp(0.0, 1.0)
    return 1.0 + (float(scale) - 1.0) * ramp


def apply_txt_steering_to_plan(plan, layout, gen_txt_steering_scales, *, text_steering_mask=None):
    """Attach per-group cross-modal text steering scales to an existing AttentionPlan."""
    ti = plan.modality_indexes[0]
    device = plan.gen_indexes.device
    if text_steering_mask is not None:
        if (
            not isinstance(text_steering_mask, torch.Tensor)
            or text_steering_mask.dtype != torch.bool
            or text_steering_mask.shape != (len(ti),)
        ):
            raise ValueError("text_steering_mask must be a 1D boolean tensor matching text tokens")
        if text_steering_mask.device != device:
            raise ValueError("text_steering_mask must reside on the same device as the attention plan")
    if gen_txt_steering_scales is None:
        return plan
    num_gen = len(plan.gen_indexes)
    if (
        gen_txt_steering_scales.ndim != 1
        or len(gen_txt_steering_scales) != num_gen
        or not gen_txt_steering_scales.is_floating_point()
    ):
        raise ValueError("gen_txt_steering_scales must be a 1D floating tensor matching gen tokens")
    if gen_txt_steering_scales.device != device:
        raise ValueError("gen_txt_steering_scales must reside on the same device as the attention plan")
    if not torch.isfinite(gen_txt_steering_scales).all() or (gen_txt_steering_scales <= 0).any():
        raise ValueError("gen_txt_steering_scales must be finite and positive")
    if not num_gen:
        return plan
    scales_cpu = gen_txt_steering_scales.detach().float().cpu()
    gen_roles = [r for r in layout.roles if r != -1]
    has_steerable_text = (
        len(ti) > 0 if text_steering_mask is None else bool(text_steering_mask.detach().cpu().any())
    )
    is_steered_text = None

    def steered_mask():
        nonlocal is_steered_text
        if is_steered_text is None:
            if text_steering_mask is None:
                is_steered_text = torch.tensor([m == 0 for m in layout.modalities], dtype=torch.bool, device=device)
            else:
                is_steered_text = torch.zeros(len(layout.modalities), dtype=torch.bool, device=device)
                if len(ti):
                    is_steered_text = is_steered_text.index_copy(0, ti, text_steering_mask)
        return is_steered_text

    updated_chunks = []
    for chunk in plan.chunks:
        q_idx_cpu = chunk.query_indexes.cpu()
        group_scales = []
        kv_lengths = []
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            group_q = q_idx_cpu[qstart:qend]
            values = scales_cpu.index_select(0, group_q)
            scale = float(values[0].item())
            if not torch.all(values == values[0]):
                raise ValueError("txt_steering requires uniform scales within each (sample, role, block) query group")
            if gen_roles[int(group_q[0].item())] != 1 and scale != 1.0:
                raise ValueError("Clean, condition and state query groups must use unit txt_steering scale")
            group_scales.append(scale)
            kv_lengths.append(kend - kstart)
            qstart, kstart = qend, kend
        if has_steerable_text and any(scale != 1.0 for scale in group_scales):
            expanded = torch.repeat_interleave(
                torch.tensor(group_scales, dtype=torch.float32, device=device),
                torch.tensor(kv_lengths, dtype=torch.long, device=device),
            )
            kv_scales = torch.where(steered_mask().index_select(0, chunk.kv_indexes), expanded, 1.0)
        else:
            kv_scales = None
        updated_chunks.append(
            TNDChunk(chunk.query_indexes, chunk.kv_indexes, chunk.query_ends, chunk.kv_ends, kv_scales)
        )
    plan.chunks = tuple(updated_chunks)
    if plan.backend == "dense":
        if has_steerable_text and not bool(torch.all(scales_cpu == 1.0)):
            plan.dense_logit_scales = torch.where(
                steered_mask()[None, :], gen_txt_steering_scales.detach().float()[:, None], 1.0
            )
        else:
            plan.dense_logit_scales = None
    return plan


def prepare_attention(
    layout, device, backend="dense", max_kv_tokens=131072, *, gen_txt_steering_scales=None, text_steering_mask=None
):
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
    clean_frames = defaultdict(list)
    text = defaultdict(list)
    for i, (s, r, b) in enumerate(zip(layout.sample_ids, layout.roles, layout.block_ids, strict=True)):
        if r == -1:
            text[s].append(i)
        else:
            buckets[s, r, b].append(i)
            if r == 0:
                clean_frames[s, layout.frame_ids[i]].append(i)
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
            start = max(0, 1 + block * layout.block_sizes[sample])
            for frame in range(max(0, start - layout.history_blocks[sample]), start):
                keys.extend(clean_frames[sample, frame])
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
    plan = AttentionPlan(modalities, indices(gen), text_samples, text_indexes, text_ends, tuple(chunks), mask, backend)
    return apply_txt_steering_to_plan(
        plan, layout, gen_txt_steering_scales, text_steering_mask=text_steering_mask
    )


def _scale_kv(tensor, kv_scales):
    if kv_scales is None:
        return tensor
    compute_dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    return (tensor.to(compute_dtype) * kv_scales.to(compute_dtype)[:, None, None]).to(tensor.dtype)


def sdpa(query, key, value, mask=None, causal=False, logit_scales=None):
    repeats = query.shape[1] // key.shape[1]
    if logit_scales is None:
        return F.scaled_dot_product_attention(
            query.transpose(0, 1),
            key.repeat_interleave(repeats, dim=1).transpose(0, 1),
            value.repeat_interleave(repeats, dim=1).transpose(0, 1),
            attn_mask=mask,
            is_causal=causal,
            dropout_p=0.0,
        ).transpose(0, 1)
    if not query.shape[0]:
        return query
    compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    q = query.transpose(0, 1).to(compute_dtype)
    k = key.repeat_interleave(repeats, dim=1).transpose(0, 1).to(compute_dtype)
    v = value.repeat_interleave(repeats, dim=1).transpose(0, 1).to(compute_dtype)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
    scores = scores * logit_scales.to(compute_dtype).unsqueeze(0)
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v).to(query.dtype).transpose(0, 1)


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
            k = _scale_kv(key.index_select(0, chunk.kv_indexes), chunk.kv_scales)
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
            k = _scale_kv(key.index_select(0, chunk.kv_indexes), chunk.kv_scales)
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
            dk_chunk = gradients[1].float()
            if chunk.kv_scales is not None:
                dk_chunk = dk_chunk * chunk.kv_scales[:, None, None]
            dk.index_add_(0, chunk.kv_indexes, dk_chunk)
            dv.index_add_(0, chunk.kv_indexes, gradients[2].float())
            offset = end
        return dq, dk.to(key.dtype), dv.to(value.dtype), None


def generation_attention(query, key, value, plan):
    if not query.shape[0]:
        return query
    if plan.backend == "dense":
        return sdpa(query, key, value, plan.mask, logit_scales=plan.dense_logit_scales)
    if plan.backend == "npu_tnd":
        return _NPUTeacherForcing.apply(query, key, value, plan.chunks)
    indexes = tuple(chunk.kv_indexes for chunk in plan.chunks)
    keys, values = _GatherGroups.apply(key, *indexes), _GatherGroups.apply(value, *indexes)
    outputs = []
    for chunk, k, v in zip(plan.chunks, keys, values, strict=True):
        k = _scale_kv(k, chunk.kv_scales)
        q = query.index_select(0, chunk.query_indexes)
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            outputs.append(sdpa(q[qstart:qend], k[kstart:kend], v[kstart:kend]))
            qstart, kstart = qend, kend
    return torch.cat(outputs)
