# SPDX-License-Identifier: OpenMDW-1.1
"""Request-local native MoT K/V with a fresh differentiable future-token replay."""

from dataclasses import replace

import torch

from ..cosmos3_core.attention import TNDChunk, generation_attention, prepare_attention
from ..cosmos3_core.layers import rotary_tables
from ..cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Output


def current_plan(layout, plan):
    """Keep future queries and their complete, unchanged K/V sets."""
    device = plan.gen_indexes.device
    selected = [i for i, packed in enumerate(plan.gen_indexes.tolist()) if layout.roles[packed] == 1]
    target = torch.tensor(selected, device=device, dtype=torch.long)
    mapping = torch.full((len(plan.gen_indexes),), -1, device=device, dtype=torch.long)
    mapping[target] = torch.arange(len(target), device=device)
    chunks = []
    for chunk in plan.chunks:
        queries, keys, qe, ke = [], [], [], []
        qs = ks = 0
        for endq, endk in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            old = chunk.query_indexes[qs:endq]
            chosen = mapping[old]
            keep = chosen >= 0
            group_keys = chunk.kv_indexes[ks:endk]
            if keep.any():
                queries.append(chosen[keep])
                keys.append(group_keys)
                qe.append(sum(len(value) for value in queries))
                ke.append(sum(len(value) for value in keys))
            elif any(layout.roles[k] == 1 for k in group_keys.tolist()):
                raise ValueError("A cached fixed-prefix query must not depend on current noisy inputs")
            qs, ks = endq, endk
        if queries:
            chunks.append(TNDChunk(torch.cat(queries), torch.cat(keys), tuple(qe), tuple(ke)))
    packed = plan.gen_indexes[target]
    compact = replace(
        plan, gen_indexes=packed, chunks=tuple(chunks), mask=plan.mask[target] if plan.mask is not None else None
    )
    return target, packed, compact


class PrefixCache:
    """One CFG branch of one decision, owned by the sampler rather than the model.

    Inputs except current future video/action and their times remain fixed for
    this object's lifetime. No entry survives a new observation or request.
    The first call prefills without gradients, then replays the future inputs
    with a new graph, so RTC can differentiate its very first evaluation too.
    """

    def __init__(self, model, inputs, backend):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Prefix caching requires frozen eval-mode model weights")
        if model.config.action_interface != "shared":
            raise ValueError("Online prefix caching requires the shared action interface")
        self.model, self.inputs = model, inputs
        self.plan = prepare_attention(inputs["layout"], inputs["video_tokens"].device, backend)
        layout = inputs["layout"]
        if len(layout.history_blocks) != 1 or not layout.decision_windows or not layout.decision_windows[0]:
            raise ValueError("Prefix caching requires one explicit decision window")
        self.target, self.packed, self.compact = current_plan(layout, self.plan)
        self.video_rows = torch.where(
            torch.tensor(
                [layout.roles[i] == 1 for i in self.plan.modality_indexes[1].tolist()], device=self.packed.device
            )
        )[0]
        self.action_rows = torch.where(
            torch.tensor(
                [layout.roles[i] == 1 for i in self.plan.modality_indexes[2].tolist()], device=self.packed.device
            )
        )[0]
        self.video_gen = torch.searchsorted(self.packed, self.plan.modality_indexes[1][self.video_rows])
        self.action_gen = torch.searchsorted(self.packed, self.plan.modality_indexes[2][self.action_rows])
        cos, sin = rotary_tables(inputs["position_ids"], model.config, inputs["video_tokens"].dtype)
        self.tables = cos[self.packed], sin[self.packed]
        self.layers, self.reference = {}, None
        self.stats = dict(prefill_calls=0, replay_calls=0, active_tokens=len(self.packed))

    def __call__(self, inputs):
        model = self.model
        if self.reference is None:
            with torch.no_grad():
                self.reference = model(**inputs, attention_plan=self.plan, cache_capture=self.layers)
            self.stats["prefill_calls"] += 1
        video = model.proj_in(inputs["video_tokens"][self.video_rows])
        video = video + model.time_embedder(inputs["video_timesteps"][self.video_rows] * model.config.timestep_scale)
        gen = video.new_zeros(len(self.packed), model.config.hidden_size).index_copy(0, self.video_gen, video)
        action = inputs["action_tokens"][self.action_rows].masked_fill(~inputs["action_mask"][self.action_rows], 0)
        action = model.action_proj_in(action) + model.action_modality_embed
        action = action + model.time_embedder(
            inputs["action_timesteps"][self.action_rows] * model.config.timestep_scale
        )
        gen = gen.index_copy(0, self.action_gen, action)
        cos, sin = self.tables
        for index, layer in enumerate(model.layers):
            attention = layer.self_attn
            normalized = layer.input_layernorm_moe_gen(gen)
            query = attention.norm_added_q(
                attention.add_q_proj(normalized).view(-1, attention.num_heads, attention.head_dim)
            )
            key = attention.norm_added_k(
                attention.add_k_proj(normalized).view(-1, attention.num_kv_heads, attention.head_dim)
            )
            value = attention.add_v_proj(normalized).view(-1, attention.num_kv_heads, attention.head_dim)
            query, key = attention.apply_rope(query, key, cos, sin)
            keys = self.layers[index]["key"].index_copy(0, self.packed, key)
            values = self.layers[index]["value"].index_copy(0, self.packed, value)
            attended = generation_attention(query, keys, values, self.compact)
            gen = gen + attention.to_add_out(attended.flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        video = self.reference.video.detach().index_copy(0, self.video_rows, model.proj_out(gen[self.video_gen]))
        action = self.reference.action.detach().index_copy(
            0, self.action_rows, model.action_proj_out(gen[self.action_gen])
        )
        self.stats["replay_calls"] += 1
        return Cosmos3Output(video=video, action=action.masked_fill(~inputs["action_mask"], 0))
