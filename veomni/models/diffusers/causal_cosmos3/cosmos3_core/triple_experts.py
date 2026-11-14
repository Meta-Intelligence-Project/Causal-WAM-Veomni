# SPDX-License-Identifier: OpenMDW-1.1
"""Three-expert execution and invocation-local, detached conditioning caches.

Video/text blocks keep their original checkpoint names. Every block computation
goes through its module forward so composable FSDP owns materialization scopes.
"""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .attention import generation_attention, understanding_attention
from .layers import rotary_tables
from .triple_attention import prepare_branch_attention


@dataclass(frozen=True)
class TextCache:
    layout: object
    layers: tuple
    hidden: torch.Tensor


@dataclass(frozen=True)
class ExpertCache:
    layout: object
    indexes: torch.Tensor
    layers: tuple
    text_cache: TextCache
    prediction: torch.Tensor | None = None


def text_block(layer, hidden, tables, plan):
    attn = layer.self_attn
    x = layer.input_layernorm(hidden)
    query = attn.norm_q(attn.to_q(x).view(-1, attn.num_heads, attn.head_dim))
    raw_key = attn.to_k(x).view(-1, attn.num_kv_heads, attn.head_dim)
    value = attn.to_v(x).view(-1, attn.num_kv_heads, attn.head_dim)
    query, key = attn.apply_rope(query, attn.norm_k(raw_key), *tables)
    hidden = hidden + attn.to_out(understanding_attention(query, key, value, plan).flatten(1))
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, raw_key, value


def video_block(layer, hidden, tables, plan, context):
    attn = layer.self_attn
    x = layer.input_layernorm_moe_gen(hidden)
    query = attn.norm_added_q(attn.add_q_proj(x).view(-1, attn.num_heads, attn.head_dim))
    key = attn.norm_added_k(attn.add_k_proj(x).view(-1, attn.num_kv_heads, attn.head_dim))
    value = attn.add_v_proj(x).view(-1, attn.num_kv_heads, attn.head_dim)
    query, key = attn.apply_rope(query, key, *tables)
    raw_text_key, text_value = context["text_kv"]
    raw_text_key = raw_text_key.detach().to(key)
    text_key = (
        attn.k_norm_und_for_gen(raw_text_key) if attn.k_norm_und_for_gen is not None else attn.norm_k(raw_text_key)
    )
    _, text_key = attn.apply_rope(text_key, text_key, *context["text_tables"])
    ti, vi = plan.modality_indexes[:2]
    packed_key = key.new_zeros(context["packed_length"], attn.num_kv_heads, attn.head_dim)
    packed_value = value.new_zeros(packed_key.shape)
    packed_key = packed_key.index_copy(0, ti, text_key).index_copy(0, vi, key)
    packed_value = packed_value.index_copy(0, ti, text_value.detach().to(value)).index_copy(0, vi, value)
    conditioning = context.get("conditioning")
    if conditioning is not None:
        indexes, ck, cv = conditioning
        packed_key = packed_key.index_copy(0, indexes, ck.detach().to(key))
        packed_value = packed_value.index_copy(0, indexes, cv.detach().to(value))
    hidden = hidden + attn.to_add_out(generation_attention(query, packed_key, packed_value, plan).flatten(1))
    hidden = hidden + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(hidden))
    return hidden, key, value


class TripleExecution:
    """One packed invocation; cache objects never become model attributes."""

    def __init__(self, model, inputs, task_modes, backend):
        self.model, self.inputs = model, inputs
        self.layout = inputs["layout"]
        self.device = inputs["video_tokens"].device
        self.n = len(self.layout.sample_ids)
        self.modes = tuple(task_modes) if task_modes is not None else self._infer_modes()
        if len(self.modes) != len(self.layout.history_blocks) or any(
            mode not in ("video", "forward_dynamics", "inverse_dynamics") for mode in self.modes
        ):
            raise ValueError("Triple task_modes support only video, forward_dynamics and inverse_dynamics; not policy")
        self.plan = self.branch("video", include_conditioning=False)
        self.ti, self.vi, self.ai, self.si = self.plan.modality_indexes
        self._validate_inputs()
        self.cos, self.sin = rotary_tables(inputs["position_ids"], model.config, inputs["video_tokens"].dtype)
        self.text_tables = (self.cos[self.ti], self.sin[self.ti])
        self.video_tables = (self.cos[self.vi], self.sin[self.vi])
        self.mapping = {layer - 1: i for i, layer in enumerate(model.config.action_layer_map)}

    def _infer_modes(self):
        modes = []
        for sample in range(len(self.layout.history_blocks)):
            actions = [
                r
                for s, m, r in zip(self.layout.sample_ids, self.layout.modalities, self.layout.roles, strict=True)
                if s == sample and m == 2
            ]
            if not actions:
                modes.append("video")
            elif all(r == 2 for r in actions):
                modes.append("forward_dynamics")
            elif any(
                s == sample and m == 1 and r == 1
                for s, m, r in zip(self.layout.sample_ids, self.layout.modalities, self.layout.roles, strict=True)
            ):
                raise ValueError("Triple experts do not support policy; supply explicit task_modes for staged IDM")
            else:
                modes.append("inverse_dynamics")
        return tuple(modes)

    def _validate_inputs(self):
        inputs, config = self.inputs, self.model.config
        if inputs["position_ids"].shape != (3, self.n):
            raise ValueError("position_ids must be [3, packed_length]")
        if inputs["input_ids"].shape != (len(self.ti),) or inputs["video_tokens"].shape != (
            len(self.vi),
            config.patch_dim,
        ):
            raise ValueError("Text/video payload shapes do not match layout/config")
        if inputs["video_timesteps"].shape != (len(self.vi),):
            raise ValueError("One video timestep is required per video token")
        for name, indexes, width in (("action", self.ai, config.action_dim), ("state", self.si, config.state_dim)):
            payload, mask = inputs.get(name + "_tokens"), inputs.get(name + "_mask")
            if len(indexes) and (
                payload is None
                or payload.shape != (len(indexes), width)
                or mask is None
                or mask.shape != payload.shape
                or mask.dtype != torch.bool
            ):
                raise ValueError(f"{name} payload and boolean mask must match layout/config")
        if len(self.ai) and not hasattr(self.model, "action_expert"):
            raise ValueError("Action payload requires an enabled action expert")
        if len(self.ai) and (
            inputs.get("action_timesteps") is None or inputs["action_timesteps"].shape != (len(self.ai),)
        ):
            raise ValueError("One independent action timestep is required per action token")
        for sample, mode in enumerate(self.modes):
            roles = [
                r
                for s, m, r in zip(self.layout.sample_ids, self.layout.modalities, self.layout.roles, strict=True)
                if s == sample and m == 2
            ]
            has_state = any(
                s == sample and m == 3 for s, m in zip(self.layout.sample_ids, self.layout.modalities, strict=True)
            )
            if mode == "video" and (roles or has_state):
                raise ValueError("Pure video input must not contain action/state tokens")
            if mode == "forward_dynamics" and (not roles or any(r != 2 for r in roles)):
                raise ValueError("FDM requires clean, conditioned action tokens")

    def branch(self, name, include_conditioning=True, modes=None):
        modalities = (1,) if name == "video" else (2, 3)
        return prepare_branch_attention(
            self.layout,
            self.device,
            self.inputs.get("attention_backend", "dense"),
            query_modalities=modalities,
            task_modes=self.modes if modes is None else modes,
            branch=name,
            include_conditioning=include_conditioning,
            video_conditioning=self.inputs.get("video_conditioning"),
            text_conditioning=self.inputs.get("text_conditioning"),
        )

    def check_cache(self, cache):
        if cache.layout != self.layout:
            raise ValueError("Conditioning cache belongs to a different packed layout")

    def text(self, cache=None):
        if cache is not None:
            self.check_cache(cache)
            return cache
        with torch.no_grad():
            hidden = self.model.embed_tokens(self.inputs["input_ids"])
            entries = []
            for layer in self.model.layers:
                hidden, key, value = layer(hidden, None, self.text_tables, self.plan, {"operation": "text"})
                entries.append((key.detach(), value.detach()))
            return TextCache(self.layout, tuple(entries), hidden.detach())

    def _action_hidden(self, plan):
        expert, inputs = self.model.action_expert, self.inputs
        # Query selection is CPU layout metadata, not a device-to-host read.
        global_ids = [
            i
            for i, (s, m, r) in enumerate(
                zip(self.layout.sample_ids, self.layout.modalities, self.layout.roles, strict=True)
            )
            if m in (2, 3)
            and (getattr(self, "_all_actions", False) or m == 3 or (r == 2 and self.modes[s] == "forward_dynamics"))
        ]
        rows = {index: row for row, index in enumerate(global_ids)}
        hidden = inputs["video_tokens"].new_zeros(len(global_ids), self.model.config.action_hidden_size)
        for modality, name in ((2, "action"), (3, "state")):
            global_modality = [i for i, m in enumerate(self.layout.modalities) if m == modality]
            selected = [(p, rows[index]) for p, index in enumerate(global_modality) if index in rows]
            if not selected:
                continue
            payload_rows = torch.tensor([p for p, _ in selected], device=self.device)
            hidden_rows = torch.tensor([q for _, q in selected], device=self.device)
            payload = inputs[name + "_tokens"].masked_fill(~inputs[name + "_mask"], 0)[payload_rows]
            if name == "action":
                times = inputs["action_timesteps"][payload_rows]
                roles = [self.layout.roles[global_modality[p]] for p, _ in selected]
                times = times.masked_fill(torch.tensor(roles, device=self.device) != 1, 0)
                embedded = expert.proj_in(payload) + expert.modality_embed
            else:
                times = torch.zeros(len(payload), device=self.device)
                embedded = expert.state_proj_in(payload) + expert.state_modality_embed
            embedded = embedded + expert.time_embedder(times * self.model.config.timestep_scale)
            hidden = hidden.index_copy(0, hidden_rows, embedded)
        return hidden

    def conditions(self, text_cache, cache=None):
        if cache is not None:
            self.check_cache(cache)
            return cache
        if not hasattr(self.model, "action_expert") or not len(self.si) + len(self.ai):
            return None
        plan = self.branch("condition")
        with torch.no_grad():
            hidden = self._action_hidden(plan)
            tables = (self.cos[plan.gen_indexes], self.sin[plan.gen_indexes])
            entries = []
            for j, layer in enumerate(self.model.action_expert.layers):
                source = self.model.config.action_layer_map[j] - 1
                hidden, key, value = layer(
                    hidden,
                    tables,
                    plan,
                    plan.gen_indexes,
                    self.n,
                    text_cache.layers[source],
                    self.text_tables,
                )
                entries.append((key.detach(), value.detach()))
            return ExpertCache(self.layout, plan.gen_indexes, tuple(entries), text_cache)

    def video(self, text_cache, conditions=None):
        inputs = self.inputs
        # Fixed noisy conditions preserve their actual sigma. Only the clean
        # teacher-forcing stream is unconditionally at time zero.
        roles = torch.tensor(self.layout.roles, device=self.device)[self.vi]
        times = inputs["video_timesteps"].masked_fill(roles == 0, 0)
        hidden = self.model.proj_in(inputs["video_tokens"])
        hidden = hidden + self.model.time_embedder(times * self.model.config.timestep_scale)
        simple = self.branch("video", include_conditioning=False)
        conditioned = self.branch("video", include_conditioning=conditions is not None)
        entries = []
        for i, layer in enumerate(self.model.layers):
            bridge = self.mapping.get(i)
            external = None
            if bridge is not None and conditions is not None:
                ck, cv = conditions.layers[bridge]
                external = (conditions.indexes, ck, cv)
            plan = conditioned if external is not None else simple
            context = dict(
                operation="video",
                packed_length=self.n,
                text_kv=text_cache.layers[i],
                text_tables=self.text_tables,
                conditioning=external,
            )
            args = (None, hidden, self.video_tables, plan, context)
            if self.model.gradient_checkpointing and self.model.training and torch.is_grad_enabled():
                hidden, key, value = checkpoint(layer, *args, use_reentrant=False)
            else:
                hidden, key, value = layer(*args)
            if bridge is not None:
                entries.append((key.detach(), value.detach()))
        normalized = self.model.norm_moe_gen(hidden)
        prediction = self.model.proj_out(normalized)
        return (
            prediction,
            ExpertCache(self.layout, self.vi, tuple(entries), text_cache, prediction.detach()),
            normalized,
        )

    def action(self, text_cache, video_cache):
        if not hasattr(self.model, "action_expert") or not len(self.ai):
            return None, None
        self.check_cache(video_cache)
        # FDM rows execute the same modules (with their condition mask and zero
        # supervision), keeping FSDP collectives aligned with IDM on other ranks.
        all_modes = tuple("inverse_dynamics" if m != "video" else m for m in self.modes)
        plan = self.branch("action", modes=all_modes)
        self._all_actions = True
        hidden = self._action_hidden(plan)
        tables = (self.cos[plan.gen_indexes], self.sin[plan.gen_indexes])
        for j, layer in enumerate(self.model.action_expert.layers):
            source = self.model.config.action_layer_map[j] - 1
            key, value = video_cache.layers[j]
            args = (
                hidden,
                tables,
                plan,
                plan.gen_indexes,
                self.n,
                text_cache.layers[source],
                self.text_tables,
                (self.vi, key, value),
            )
            if self.model.gradient_checkpointing and self.model.training and torch.is_grad_enabled():
                hidden, _, _ = checkpoint(layer, *args, use_reentrant=False)
            else:
                hidden, _, _ = layer(*args)
        hidden = self.model.action_expert.norm(hidden)
        action_rows = torch.searchsorted(plan.gen_indexes, self.ai)
        prediction = self.model.action_expert.proj_out(hidden[action_rows])
        return prediction.masked_fill(~self.inputs["action_mask"], 0), hidden

    def rollout_inputs(self, text_cache, conditions, metadata, targets):
        probability = float(metadata.get("rollout_cache_probability", 0))
        if not 0 <= probability <= 1:
            raise ValueError("rollout_cache_probability must be within [0,1]")
        if probability == 0:
            return
        stop = float(metadata.get("video_stop_sigma", 0.9))
        if not 0 < stop < 1:
            raise ValueError("video_stop_sigma must be strictly between zero and one")
        seed = int(metadata.get("seed", 0))
        if "mix_seed" in metadata:
            # Batch-level selection is deterministic in optimizer step and
            # rank-independent. FSDP ranks must either all enter the extra
            # video forwards or all skip them, even with different local tasks.
            mix_seed = int(metadata["mix_seed"]) + 1000003 * int(metadata.get("optimizer_step", 0))
            choose_rng = torch.Generator(device="cpu").manual_seed(mix_seed)
            if float(torch.rand((), generator=choose_rng, device="cpu")) >= probability:
                return
            selected = torch.ones(len(self.modes), dtype=torch.bool, device="cpu")
        else:
            choose_rng = torch.Generator(device="cpu").manual_seed(seed)
            selected = torch.rand(len(self.modes), generator=choose_rng, device="cpu") < probability
        active = torch.tensor(
            [
                self.modes[s] == "inverse_dynamics" and bool(selected[s]) and m == 1 and r in (1, 2) and b >= 0
                for s, m, r, b in zip(
                    self.layout.sample_ids,
                    self.layout.modalities,
                    self.layout.roles,
                    self.layout.block_ids,
                    strict=True,
                )
                if m == 1
            ],
            device=self.device,
        )
        # Every rank executes the same rollout trace once mixing is enabled;
        # local FDM-only packs are masked out, rather than skipping FSDP calls.
        generator = torch.Generator(device=self.device).manual_seed(seed + 32768)
        original, original_times = self.inputs["video_tokens"], self.inputs["video_timesteps"]
        current = original.float().clone()
        noise = torch.randn(current.shape, device=self.device, dtype=torch.float32, generator=generator)
        if targets is not None:
            from .patching import patchify_video

            for span, target in zip(targets["spans"], targets["video"], strict=True):
                epsilon = torch.randn(target.shape, device=self.device, dtype=torch.float32, generator=generator)
                noise[span.current_video_rows] = patchify_video(epsilon, self.model.config.patch_size)
        current = torch.where(active[:, None], noise, current)
        steps = metadata.get("partial_video_steps", 4)
        if type(steps) is not int or steps < 1:
            raise ValueError("partial_video_steps must be a positive integer")
        sigmas = torch.linspace(1, stop, steps + 1, device=self.device)
        with torch.no_grad():
            for start, end in zip(sigmas[:-1], sigmas[1:], strict=True):
                self.inputs["video_tokens"] = current.to(original.dtype)
                self.inputs["video_timesteps"] = torch.where(active, start * 1000, original_times)
                prediction, _, _ = self.video(text_cache, conditions)
                updated = current + (end - start) * prediction.float()
                current = torch.where(active[:, None], updated, current)
        self.inputs["video_tokens"] = current.to(original.dtype)
        self.inputs["video_timesteps"] = torch.where(active, stop * 1000, original_times)


def forward_triple(
    model,
    inputs,
    *,
    task_modes,
    operation,
    text_cache=None,
    video_cache=None,
    action_condition_cache=None,
    training_metadata=None,
    idm_video_inputs=None,
    targets=None,
    output_text_logits=False,
    output_hidden_states=False,
):
    from ..cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Output

    if operation not in ("full", "text", "action_condition", "video", "action"):
        raise ValueError("Unknown triple expert_operation")
    execution = TripleExecution(model, inputs, task_modes, inputs["attention_backend"])
    text_cache = execution.text(text_cache)
    output = Cosmos3Output(text_cache=text_cache)

    def finish():
        return Cosmos3Output(**{name: getattr(output, name) for name in Cosmos3Output.__dataclass_fields__})

    if output_text_logits:
        output.text_logits = model.lm_head(model.norm(text_cache.hidden))
    if operation == "text":
        return finish()
    if operation == "action":
        if video_cache is None:
            raise ValueError("Action-only execution requires a fixed video_cache")
        output.action, action_hidden = execution.action(text_cache, video_cache)
        output.video = video_cache.prediction
        output.video_cache = video_cache
        return finish()
    action_condition_cache = execution.conditions(text_cache, action_condition_cache)
    output.action_condition_cache = action_condition_cache
    if operation == "action_condition":
        return finish()
    action_video_cache = None
    if idm_video_inputs is not None:
        if operation != "full" or not len(execution.ai):
            raise ValueError("idm_video_inputs require full action-stage execution")
        if not isinstance(idm_video_inputs, dict) or set(idm_video_inputs) != {"video_tokens", "video_timesteps"}:
            raise ValueError("idm_video_inputs must contain only video_tokens and video_timesteps")
        # The supervised video velocity uses an ordinary GT-noised FM pair.
        # Action's fixed future is encoded independently, so neither a stopped
        # timestep nor a model rollout can change that supervised input/target.
        enabled = inputs.get("video_conditioning")
        enabled = (True,) * len(execution.modes) if enabled is None else enabled
        cache_inputs = {
            **inputs,
            **idm_video_inputs,
            "video_conditioning": tuple(
                mode == "inverse_dynamics" or condition
                for mode, condition in zip(execution.modes, enabled, strict=True)
            ),
        }
        cache_execution = TripleExecution(model, cache_inputs, task_modes, inputs["attention_backend"])
        with torch.no_grad():
            if training_metadata is not None:
                cache_execution.rollout_inputs(text_cache, action_condition_cache, training_metadata, targets)
            _, action_video_cache, _ = cache_execution.video(text_cache, action_condition_cache)
    elif operation == "full" and training_metadata is not None:
        # Standalone callers may still supply the original fixed-condition
        # input directly, without requesting a separate supervised video pass.
        execution.rollout_inputs(text_cache, action_condition_cache, training_metadata, targets)
    output.video, video_cache, video_hidden = execution.video(text_cache, action_condition_cache)
    if action_video_cache is not None:
        video_cache = action_video_cache
    output.video_cache = video_cache
    if operation == "full":
        output.action, action_hidden = execution.action(text_cache, video_cache)
        if output_hidden_states:
            output.expert_hidden_states = {"text": text_cache.hidden, "video": video_hidden, "action": action_hidden}
        if targets is not None:
            output.loss = model.loss(output, targets)
    return finish()
