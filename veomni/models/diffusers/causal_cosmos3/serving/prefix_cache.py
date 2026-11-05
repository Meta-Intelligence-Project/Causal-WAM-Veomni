# SPDX-License-Identifier: OpenMDW-1.1
"""Request-local native MoT K/V with a fresh differentiable future-token replay."""

from dataclasses import replace

import torch

from ..cosmos3_core.attention import TNDChunk, generation_attention, prepare_attention
from ..cosmos3_core.layers import rotary_tables
from ..cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Output


def query_plan(layout, plan, selected):
    """Keep selected GEN queries and their complete, unchanged K/V sets."""
    device = plan.gen_indexes.device
    target = torch.tensor(selected, device=device, dtype=torch.long)
    changing = set(plan.gen_indexes[target].tolist())
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
            elif any(k in changing for k in group_keys.tolist()):
                raise ValueError("A cached fixed-prefix query must not depend on current noisy inputs")
            qs, ks = endq, endk
        if queries:
            chunks.append(
                TNDChunk.from_lists(torch.cat(queries).tolist(), torch.cat(keys).tolist(), qe, ke, device=device)
            )
    packed = plan.gen_indexes[target]
    compact = replace(
        plan, gen_indexes=packed, chunks=tuple(chunks), mask=plan.mask[target] if plan.mask is not None else None
    )
    return target, packed, compact


def current_plan(layout, plan):
    """Keep future queries and their complete, unchanged K/V sets."""
    return query_plan(
        layout, plan, [i for i, packed in enumerate(plan.gen_indexes.tolist()) if layout.roles[packed] == 1]
    )


class PrefixCache:
    """One CFG branch of one decision, owned by the sampler rather than the model.

    Inputs except current future video/action and their times remain fixed for
    this object's lifetime. No entry survives a new observation or request.
    Ordinary inference returns the first prefill output and updates K/V in place.
    Differentiable inference replays even that first evaluation with a fresh
    graph, so RTC can differentiate its very first velocity too.
    """

    def __init__(
        self, model, inputs, backend, *, differentiable=False, graph_cache=None, episode_cache=None, branch="cond"
    ):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Prefix caching requires frozen eval-mode model weights")
        if model.config.action_interface != "shared":
            raise ValueError("Online prefix caching requires the shared action interface")
        self.model, self.inputs = model, inputs
        self.episode_cache, self.branch = episode_cache, branch
        self.differentiable, self.graph_cache = differentiable, graph_cache
        if differentiable and graph_cache is not None:
            raise ValueError("NPU replay graphs and RTC input gradients are mutually exclusive")
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
        self.full_tables = (cos, sin) if episode_cache is not None else None
        self.layers, self.reference = {}, None
        self.stats = dict(prefill_calls=0, replay_calls=0, active_tokens=len(self.packed))

    def __call__(self, inputs):
        if not self.differentiable and torch.is_grad_enabled():
            raise RuntimeError("Ordinary prefix caching requires no-grad inference")
        if self.reference is None:
            with torch.no_grad():
                source, shared, signature = (
                    self.episode_cache.match(self, self.branch) if self.episode_cache is not None else (None, 0, None)
                )
                if shared:
                    self.reference = self.prefill_suffix(inputs, source, shared)
                else:
                    self.reference = self.model(**inputs, attention_plan=self.plan, cache_capture=self.layers)
                if self.episode_cache is not None:
                    self.episode_cache.remember(self, self.branch, signature)
                    self.stats["reused_history_tokens"] = shared
                    self.stats["prefilled_history_tokens"] = sum(r == 0 for r in inputs["layout"].roles) - shared
            self.stats["prefill_calls"] += 1
            if not self.differentiable:
                return self.reference
        compact = {
            "video_tokens": inputs["video_tokens"][self.video_rows],
            "video_timesteps": inputs["video_timesteps"][self.video_rows],
            "action_tokens": inputs["action_tokens"][self.action_rows],
            "action_timesteps": inputs["action_timesteps"][self.action_rows],
            "action_mask": inputs["action_mask"][self.action_rows],
        }
        video, action = (
            self.graph_cache.run(self, compact) if self.graph_cache is not None else self.current_forward(compact)
        )
        self.stats["replay_calls"] += 1
        if self.differentiable:
            video = self.reference.video.detach().index_copy(0, self.video_rows, video)
            action = self.reference.action.detach().index_copy(0, self.action_rows, action)
            return Cosmos3Output(video=video, action=action.masked_fill(~inputs["action_mask"], 0))
        self.reference.video.index_copy_(0, self.video_rows, video)
        self.reference.action.index_copy_(0, self.action_rows, action).masked_fill_(~inputs["action_mask"], 0)
        return self.reference

    def prefill_suffix(self, inputs, source, shared):
        """Recompute every GEN token after an identical whole-latent prefix."""
        model, layout = self.model, inputs["layout"]
        end = len(inputs["input_ids"]) + shared
        selected = [i for i, packed in enumerate(self.plan.gen_indexes.tolist()) if packed >= end]
        _, packed, plan = query_plan(layout, self.plan, selected)
        vi, ai, si = self.plan.modality_indexes[1:]
        video_rows = torch.where(vi >= end)[0]
        video_gen = torch.searchsorted(packed, vi[video_rows])
        action_gen, state_gen = torch.searchsorted(packed, ai), torch.searchsorted(packed, si)
        times = inputs["video_timesteps"][video_rows].clone()
        roles = torch.tensor([layout.roles[i] for i in vi[video_rows].tolist()], device=times.device)
        times = times.masked_fill(roles != 1, 0)
        video = model.proj_in(inputs["video_tokens"][video_rows]) + model.time_embedder(
            times * model.config.timestep_scale
        )
        gen = video.new_zeros(len(packed), model.config.hidden_size).index_copy(0, video_gen, video)
        action = model.action_proj_in(inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0))
        action = (
            action
            + model.action_modality_embed
            + model.time_embedder(inputs["action_timesteps"] * model.config.timestep_scale)
        )
        gen = gen.index_copy(0, action_gen, action)
        state = (
            model.state_proj_in(inputs["state_tokens"].masked_fill(~inputs["state_mask"], 0))
            + model.state_modality_embed
        )
        state = state + model.time_embedder(torch.zeros(len(si), device=state.device))
        gen = gen.index_copy(0, state_gen, state)
        cos, sin = (table[packed] for table in self.full_tables)
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
            keys = key.new_zeros(len(layout.roles), *key.shape[1:])
            values = value.new_zeros(len(layout.roles), *value.shape[1:])
            keys[:end].copy_(source["layers"][index]["key"][:end])
            values[:end].copy_(source["layers"][index]["value"][:end])
            keys.index_copy_(0, packed, key)
            values.index_copy_(0, packed, value)
            self.layers[index] = dict(key=keys, value=values)
            attended = generation_attention(query, keys, values, plan)
            gen = gen + attention.to_add_out(attended.flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        predicted_video = model.proj_out(gen[video_gen])
        video_output = predicted_video.new_zeros(len(vi), predicted_video.shape[-1])
        video_output[:shared].copy_(source["video_output"][:shared])
        video_output.index_copy_(0, video_rows, predicted_video)
        return Cosmos3Output(
            video=video_output, action=model.action_proj_out(gen[action_gen]).masked_fill(~inputs["action_mask"], 0)
        )

    def current_forward(self, inputs):
        """Compute compact future predictions; ordinary K/V owners remain stable."""
        model = self.model
        video = model.proj_in(inputs["video_tokens"])
        video = video + model.time_embedder(inputs["video_timesteps"] * model.config.timestep_scale)
        gen = video.new_zeros(len(self.packed), model.config.hidden_size).index_copy(0, self.video_gen, video)
        action = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        action = model.action_proj_in(action) + model.action_modality_embed
        action = action + model.time_embedder(inputs["action_timesteps"] * model.config.timestep_scale)
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
            fixed = self.layers[index]
            if self.differentiable:
                keys = fixed["key"].index_copy(0, self.packed, key)
                values = fixed["value"].index_copy(0, self.packed, value)
            else:
                keys = fixed["key"].index_copy_(0, self.packed, key)
                values = fixed["value"].index_copy_(0, self.packed, value)
            attended = generation_attention(query, keys, values, self.compact)
            gen = gen + attention.to_add_out(attended.flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        return model.proj_out(gen[self.video_gen]), model.action_proj_out(gen[self.action_gen])
