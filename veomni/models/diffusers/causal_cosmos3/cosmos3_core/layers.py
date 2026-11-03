# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 dense MoT layers with original safetensors parameter names."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils import logging

from .attention import generation_attention, selected_generation_attention, understanding_attention
from .inference_cache import HistoryLayer, LayerCache


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

    def _project_und(self, und):
        qu = self.norm_q(self.to_q(und).view(-1, self.num_heads, self.head_dim))
        raw_ku = self.to_k(und).view(-1, self.num_kv_heads, self.head_dim)
        ku = self.norm_k(raw_ku)
        vu = self.to_v(und).view(-1, self.num_kv_heads, self.head_dim)
        return qu, ku, vu, raw_ku

    def _project_gen(self, gen):
        qg = self.norm_added_q(self.add_q_proj(gen).view(-1, self.num_heads, self.head_dim))
        kg = self.norm_added_k(self.add_k_proj(gen).view(-1, self.num_kv_heads, self.head_dim))
        vg = self.add_v_proj(gen).view(-1, self.num_kv_heads, self.head_dim)
        return qg, kg, vg

    def forward(self, und, gen, tables, plan):
        ui, gi = plan.modality_indexes[0], plan.gen_indexes
        (cu, su), (cg, sg) = tables
        qu, ku, vu, raw_ku = self._project_und(und)
        qg, kg, vg = self._project_gen(gen)
        qu, ku = self.apply_rope(qu, ku, cu, su)
        qg, kg = self.apply_rope(qg, kg, cg, sg)
        cross_ku = ku
        if self.k_norm_und_for_gen is not None:
            cross_ku = self.k_norm_und_for_gen(raw_ku)
            _, cross_ku = self.apply_rope(cross_ku, cross_ku, cu, su)
        n = len(ui) + len(gi)
        k = kg.new_zeros(n, self.num_kv_heads, self.head_dim).index_copy(0, ui, cross_ku).index_copy(0, gi, kg)
        v = vg.new_zeros(n, self.num_kv_heads, self.head_dim).index_copy(0, ui, vu).index_copy(0, gi, vg)
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

    def forward(self, und, gen, tables, plan, fixed_attention=None):
        if fixed_attention is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Fixed layer caches require eval mode and disabled gradients")
            return self.forward_cached(und, gen, tables, plan, fixed_attention)
        au, ag = self.self_attn(self.input_layernorm(und), self.input_layernorm_moe_gen(gen), tables, plan)
        return self._finish_und(und, au), self._finish_gen(gen, ag)

    def _finish_und(self, hidden, attention):
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def _finish_gen(self, hidden, attention):
        hidden = hidden + attention
        return hidden + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(hidden))

    def text_forward(self, und, tables, plan):
        a = self.self_attn
        cu, su = tables
        qu, ku, vu, raw = a._project_und(self.input_layernorm(und))
        cross = ku if a.k_norm_und_for_gen is None else a.k_norm_und_for_gen(raw)
        qu, ku = a.apply_rope(qu, ku, cu, su)
        attention = a.to_out(understanding_attention(qu, ku, vu, plan).flatten(1))
        text = self._finish_und(und, attention)
        return text, cross, vu

    def forward_cached(self, und, gen, tables, plan, cache):
        a = self.self_attn
        entry = cache.layers.get(self)
        history = cache.history if entry is None else None
        old = None if history is None else history.previous.get(self)
        (cu, su), (cg, sg) = tables
        if entry is None:
            if old is None:
                text, text_key, vu = self.text_forward(und, (cu, su), plan)
            else:
                text, text_key, vu = old.text_hidden, old.text_key, old.text_value
            _, cross = a.apply_rope(text_key, text_key, cu, su)
            rows = torch.arange(len(gen), device=gen.device) if old is None else history.fresh_rows
        else:
            text = entry.text_hidden
            rows = cache.dynamic
        hidden = gen[rows]
        q, raw_key, v = a._project_gen(self.input_layernorm_moe_gen(hidden))
        q, k = a.apply_rope(q, raw_key, cg[rows], sg[rows])
        n = len(plan.gen_indexes) + len(plan.modality_indexes[0])
        keys = k.new_zeros(n, a.num_kv_heads, a.head_dim)
        values = torch.zeros_like(keys)
        if entry is None:
            keys.index_copy_(0, plan.modality_indexes[0], cross)
            values.index_copy_(0, plan.modality_indexes[0], vu)
            if old is not None and len(history.reused_rows):
                reused = history.reused_rows
                raw_old = old.key[history.previous_rows]
                _, rotated = a.apply_rope(raw_old, raw_old, cg[reused], sg[reused])
                keys.index_copy_(0, plan.gen_indexes[reused], rotated)
                values.index_copy_(0, plan.gen_indexes[reused], old.value[history.previous_rows])
        else:
            keys.index_copy_(0, cache.fixed_global, entry.fixed_k)
            values.index_copy_(0, cache.fixed_global, entry.fixed_v)
        keys.index_copy_(0, plan.gen_indexes[rows], k)
        values.index_copy_(0, plan.gen_indexes[rows], v)
        if entry is None and old is None:
            attention = generation_attention(q, keys, values, plan)
        else:
            # Selection changes only queries. Each query keeps its full ordered KV.
            fullq = q.new_zeros(len(gen), a.num_heads, a.head_dim).index_copy_(0, rows, q)
            selected = cache.target_plan if entry is not None else history.fresh_plan
            attention = selected_generation_attention(fullq, keys, values, selected)
        hidden = self._finish_gen(hidden, a.to_add_out(attention.flatten(1)))
        if entry is None:
            result = torch.empty_like(gen).index_copy_(0, rows, hidden)
            if old is not None:
                result.index_copy_(0, history.reused_rows, old.hidden[history.previous_rows])
            cache.layers[self] = LayerCache(text, result, keys[cache.fixed_global], values[cache.fixed_global])
            if history is not None:
                raw = raw_key.new_empty(len(gen), a.num_kv_heads, a.head_dim).index_copy_(0, rows, raw_key)
                if old is not None:
                    raw.index_copy_(0, history.reused_rows, old.key[history.previous_rows])
                clean = history.clean_rows
                history.following[self] = HistoryLayer(
                    text, text_key, vu, result[clean], raw[clean], values[plan.gen_indexes[clean]]
                )
        else:
            result = entry.hidden.index_copy(0, rows, hidden)
        return text, result
