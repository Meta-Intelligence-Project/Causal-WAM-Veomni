# SPDX-License-Identifier: OpenMDW-1.1
"""Independent action transformer with detached text/video attention sources."""

from copy import copy

import torch
from torch import nn

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl

from .attention import generation_attention
from .layers import MLP, RMSNorm, TimeEmbedder


class Cosmos3ActionLayer(nn.Module):
    """One action block; all parameter reads stay inside the FSDP forward scope.

    ``plan.gen_indexes`` selects precisely ``indexes``. Text keys are raw,
    unrotated UND projections; video keys arrive normalized and rotated. Each
    cross-expert source is detached before action-private processing.
    """

    def __init__(self, config):
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = d
        self.input_layernorm = RMSNorm(h, config)
        self.post_attention_layernorm = RMSNorm(h, config)
        self.to_q = nn.Linear(h, self.num_heads * d, bias=config.attention_bias)
        self.to_k = nn.Linear(h, self.num_kv_heads * d, bias=config.attention_bias)
        self.to_v = nn.Linear(h, self.num_kv_heads * d, bias=config.attention_bias)
        self.to_out = nn.Linear(self.num_heads * d, h, bias=config.attention_bias)
        self.norm_q = RMSNorm(d, config)
        self.norm_k = RMSNorm(d, config)
        self.text_k_norm = RMSNorm(d, config)
        self.mlp = MLP(config)
        self.rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))

    def _rotate(self, query, key, tables):
        cos, sin = tables
        query, key = self.rope(
            query.unsqueeze(0), key.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=2
        )
        return query.squeeze(0), key.squeeze(0)

    def forward(self, hidden, tables, plan, indexes, packed_length, text_kv, text_tables, video_kv=None):
        residual = hidden
        normalized = self.input_layernorm(hidden)
        query = self.norm_q(self.to_q(normalized).view(-1, self.num_heads, self.head_dim))
        key = self.norm_k(self.to_k(normalized).view(-1, self.num_kv_heads, self.head_dim))
        value = self.to_v(normalized).view(-1, self.num_kv_heads, self.head_dim)
        query, key = self._rotate(query, key, tables)
        raw_text_key, text_value = text_kv
        text_key = self.text_k_norm(raw_text_key.detach().to(key))
        _, text_key = self._rotate(text_key, text_key, text_tables)
        text_indexes = plan.modality_indexes[0]
        packed_key = key.new_zeros(packed_length, self.num_kv_heads, self.head_dim)
        packed_value = value.new_zeros(packed_length, self.num_kv_heads, self.head_dim)
        packed_key = packed_key.index_copy(0, text_indexes, text_key).index_copy(0, indexes, key)
        packed_value = packed_value.index_copy(0, text_indexes, text_value.detach().to(value))
        packed_value = packed_value.index_copy(0, indexes, value)
        if video_kv is not None:
            video_indexes, video_key, video_value = video_kv
            packed_key = packed_key.index_copy(0, video_indexes, video_key.detach().to(key))
            packed_value = packed_value.index_copy(0, video_indexes, video_value.detach().to(value))
        attended = generation_attention(query, packed_key, packed_value, plan)
        hidden = residual + self.to_out(attended.flatten(1))
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, key, value


class Cosmos3ActionExpert(nn.Module):
    """Action/state inputs, private diffusion conditioning, and action decoder."""

    def __init__(self, config):
        super().__init__()
        action_config = copy(config)
        action_config.hidden_size = config.action_hidden_size
        action_config.intermediate_size = config.action_intermediate_size
        h = action_config.hidden_size
        self.proj_in = nn.Linear(config.action_dim, h)
        self.proj_out = nn.Linear(h, config.action_dim)
        self.time_embedder = TimeEmbedder(h)
        self.modality_embed = nn.Parameter(torch.empty(h))
        nn.init.normal_(self.modality_embed, std=h**-0.5)
        if config.state_dim:
            self.state_proj_in = nn.Linear(config.state_dim, h)
            self.state_modality_embed = nn.Parameter(torch.empty(h))
            nn.init.normal_(self.state_modality_embed, std=h**-0.5)
        self.norm = RMSNorm(h, action_config)
        self.layers = nn.ModuleList(Cosmos3ActionLayer(action_config) for _ in range(config.action_num_hidden_layers))
