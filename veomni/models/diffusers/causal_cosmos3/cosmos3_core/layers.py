# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 dense MoT layers with original safetensors parameter names."""

import math

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.nn import functional as F

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils import logging

from .attention import generation_attention, understanding_attention


logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = config.rms_norm_eps
        self.scale_in_fp32 = config.norm_scale_in_fp32
        self.op = VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation"))

    def forward(self, x):
        # Nemotron scales before casting; Qwen casts before scaling.
        if self.scale_in_fp32 and self.op.impl != "torch_npu":
            return self.op(x.float(), self.weight.float(), eps=self.eps).to(x.dtype)
        return self.op(x, self.weight, eps=self.eps)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            if config.hidden_act == "silu"
            else None
        )

    def forward(self, x):
        up = self.up_proj(x)
        return self.down_proj(F.relu(up).square() if self.gate_proj is None else F.silu(self.gate_proj(x)) * up)


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, times):
        freqs = torch.exp(-math.log(10000) * torch.arange(128, device=times.device, dtype=torch.float32) / 128)
        phases = times.float()[:, None] * freqs[None, :]
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        # Match the source's FP32 autocast scope, casting only the MLP output.
        if embedding.device.type in ("cuda", "npu"):
            with torch.autocast(device_type=embedding.device.type, dtype=torch.float32):
                output = self.linear_2(F.silu(self.linear_1(embedding)))
            return output.to(self.linear_1.weight.dtype)
        else:
            # CPU autocast does not support FP32; retain its homogeneous Linear path.
            logger.warning(
                "TimeEmbedder entered the Linear fallback on device=%s with weight dtype=%s.",
                embedding.device,
                self.linear_1.weight.dtype,
            )
            embedding = embedding.to(self.linear_1.weight.dtype)
            return self.linear_2(F.silu(self.linear_1(embedding)))


class DomainLinear(nn.Module):
    """Checkpoint stores input-major matrices flattened into domain embeddings."""

    def __init__(self, input_size, output_size, num_domains):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)

    def forward(self, x, domains):
        weights = self.fc(domains).view(-1, self.input_size, self.output_size)
        return torch.bmm(x[:, None, :], weights).squeeze(1) + self.bias(domains)


def rotary_tables(position_ids, config, dtype):
    # 提前降精度会合并大偏移下的相邻时间坐标，后续 .float() 无法恢复，必须直接报错。
    if position_ids.dtype in (torch.float16, torch.bfloat16):
        raise ValueError("position_ids must retain float32 precision before RoPE; FP16/BF16 coordinates are invalid")
    # All three variants use interleaved temporal/height/width frequencies.
    inv_freq = 1.0 / config.rope_theta ** (
        torch.arange(0, config.head_dim, 2, device=position_ids.device, dtype=torch.float32) / config.head_dim
    )
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
    mixed = freqs[0].clone()
    for axis in (1, 2):
        section = slice(axis, config.mrope_section[axis] * 3, 3)
        mixed[..., section] = freqs[axis, ..., section]
    mixed = torch.cat((mixed, mixed), dim=-1)
    return mixed.cos().to(dtype), mixed.sin().to(dtype)


class Cosmos3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.num_heads, self.num_kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, d
        q, kv = self.num_heads * d, self.num_kv_heads * d
        self.to_q = nn.Linear(h, q, bias=config.attention_bias)
        self.to_k = nn.Linear(h, kv, bias=config.attention_bias)
        self.to_v = nn.Linear(h, kv, bias=config.attention_bias)
        self.to_out = nn.Linear(q, h, bias=config.attention_bias)
        self.add_q_proj = nn.Linear(h, q, bias=config.attention_bias)
        self.add_k_proj = nn.Linear(h, kv, bias=config.attention_bias)
        self.add_v_proj = nn.Linear(h, kv, bias=config.attention_bias)
        self.to_add_out = nn.Linear(q, h, bias=config.attention_bias)
        self.norm_q = RMSNorm(d, config) if config.qk_norm_for_text else nn.Identity()
        self.norm_k = RMSNorm(d, config) if config.qk_norm_for_text else nn.Identity()
        self.norm_added_q, self.norm_added_k = RMSNorm(d, config), RMSNorm(d, config)
        self.k_norm_und_for_gen = RMSNorm(d, config) if config.use_und_k_norm_for_gen else None
        self.rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))

    def apply_rope(self, query, key, cos, sin):
        # The Ascend fused operator consumes [B,T,H,D], including packed inputs.
        query, key = self.rope(
            query.unsqueeze(0), key.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=2
        )
        return query.squeeze(0), key.squeeze(0)

    def forward(self, und, gen, tables, plan, cache_capture=None):
        ui, gi = plan.modality_indexes[0], plan.gen_indexes
        (cu, su), (cg, sg) = tables
        qu = self.norm_q(self.to_q(und).view(-1, self.num_heads, self.head_dim))
        raw_ku = self.to_k(und).view(-1, self.num_kv_heads, self.head_dim)
        ku = self.norm_k(raw_ku)
        vu = self.to_v(und).view(-1, self.num_kv_heads, self.head_dim)
        qg = self.norm_added_q(self.add_q_proj(gen).view(-1, self.num_heads, self.head_dim))
        kg = self.norm_added_k(self.add_k_proj(gen).view(-1, self.num_kv_heads, self.head_dim))
        vg = self.add_v_proj(gen).view(-1, self.num_kv_heads, self.head_dim)
        qu, ku = self.apply_rope(qu, ku, cu, su)
        qg, kg = self.apply_rope(qg, kg, cg, sg)
        cross_ku = ku
        if self.k_norm_und_for_gen is not None:
            cross_ku = self.k_norm_und_for_gen(raw_ku)
            _, cross_ku = self.apply_rope(cross_ku, cross_ku, cu, su)
        n = len(ui) + len(gi)
        k = kg.new_zeros(n, self.num_kv_heads, self.head_dim).index_copy(0, ui, cross_ku).index_copy(0, gi, kg)
        v = vg.new_zeros(n, self.num_kv_heads, self.head_dim).index_copy(0, ui, vu).index_copy(0, gi, vg)
        if cache_capture is not None:
            cache_capture["key"], cache_capture["value"] = k.detach(), v.detach()
        ou = understanding_attention(qu, ku, vu, plan)
        og = generation_attention(qg, k, v, plan)
        return self.to_out(ou.flatten(1)), self.to_add_out(og.flatten(1))


class Cosmos3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Cosmos3Attention(config)
        self.mlp, self.mlp_moe_gen = MLP(config), MLP(config)
        self.input_layernorm, self.input_layernorm_moe_gen = (
            RMSNorm(config.hidden_size, config),
            RMSNorm(config.hidden_size, config),
        )
        self.post_attention_layernorm, self.post_attention_layernorm_moe_gen = (
            RMSNorm(config.hidden_size, config),
            RMSNorm(config.hidden_size, config),
        )

    def forward(self, und, gen, tables, plan, cache_capture=None, history_partition=None):
        if history_partition is not None:
            if cache_capture is not None:
                raise ValueError("TBPTT does not capture inference K/V")
            if isinstance(self.self_attn, FSDPModule):
                raise ValueError(
                    "TBPTT requires decoder-layer FSDP wrapping; separately sharded attention is unsupported"
                )
            return self.forward_tbptt(und, gen, tables, plan, history_partition)
        au, ag = self.self_attn(
            self.input_layernorm(und), self.input_layernorm_moe_gen(gen), tables, plan, cache_capture
        )
        und, gen = und + au, gen + ag
        und = und + self.mlp(self.post_attention_layernorm(und))
        gen = gen + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(gen))
        return und, gen

    def forward_tbptt(self, und, gen, tables, plan, partition):
        """Stop old encoder gradients while retaining trainable memory readout."""
        a = self.self_attn
        ui, gi = plan.modality_indexes[0], plan.gen_indexes
        (cu, su), (cg, sg) = tables
        old, active = partition.old, partition.active

        # Keep UND and its GEN cross-normalization on their native gradient path.
        norm_und = self.input_layernorm(und)
        qu = a.norm_q(a.to_q(norm_und).view(-1, a.num_heads, a.head_dim))
        raw_ku = a.to_k(norm_und).view(-1, a.num_kv_heads, a.head_dim)
        ku = a.norm_k(raw_ku)
        vu = a.to_v(norm_und).view(-1, a.num_kv_heads, a.head_dim)
        qu, ku = a.apply_rope(qu, ku, cu, su)
        cross_ku = ku
        if a.k_norm_und_for_gen is not None:
            cross_ku = a.k_norm_und_for_gen(raw_ku)
            _, cross_ku = a.apply_rope(cross_ku, cross_ku, cu, su)
        attended_und = a.to_out(understanding_attention(qu, ku, vu, plan).flatten(1))
        und = und + attended_und
        und = und + self.mlp(self.post_attention_layernorm(und))

        current = gen.index_select(0, active)
        memory = gen.detach().index_copy(0, active, current)
        # Old hidden states are detached, but their norm/K/V projections still
        # learn how active queries should read that memory.
        normalized = self.input_layernorm_moe_gen(memory)
        kg = a.norm_added_k(a.add_k_proj(normalized).view(-1, a.num_kv_heads, a.head_dim))
        vg = a.add_v_proj(normalized).view(-1, a.num_kv_heads, a.head_dim)
        current_q = a.norm_added_q(a.add_q_proj(normalized.index_select(0, active)).view(-1, a.num_heads, a.head_dim))
        with torch.no_grad():
            old_q = a.norm_added_q(a.add_q_proj(normalized.index_select(0, old)).view(-1, a.num_heads, a.head_dim))
        qg = current_q.new_zeros(len(gi), a.num_heads, a.head_dim)
        qg = qg.index_copy(0, old, old_q).index_copy(0, active, current_q)
        qg, kg = a.apply_rope(qg, kg, cg, sg)
        count = len(ui) + len(gi)
        keys = kg.new_zeros(count, a.num_kv_heads, a.head_dim).index_copy(0, ui, cross_ku).index_copy(0, gi, kg)
        values = vg.new_zeros(count, a.num_kv_heads, a.head_dim).index_copy(0, ui, vu).index_copy(0, gi, vg)

        # Keep every autograd save, including index_copy's indices, before the
        # old no-grad branch so non-reentrant checkpoint/SAC can stop early.
        current_attention = generation_attention(qg.index_select(0, active), keys, values, partition.active_plan)
        current = current + a.to_add_out(current_attention.flatten(1))
        current = current + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(current))
        current_full = gen.new_zeros(gen.shape).index_copy(0, active, current)
        with torch.no_grad():
            old_attention = generation_attention(qg.index_select(0, old), keys, values, partition.old_plan)
            past = gen.index_select(0, old).detach() + a.to_add_out(old_attention.flatten(1))
            past = past + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(past))
            past_full = gen.new_zeros(gen.shape).index_copy(0, old, past)
        # The disjoint supports require no shared-storage mutation. AddBackward
        # saves no tensor values, letting recompute omit the old attention/MLP.
        return und, current_full + past_full
