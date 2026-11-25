# SPDX-License-Identifier: OpenMDW-1.1
"""Window-scoped text/observed-video KV; current targets and state stay dynamic.

Text and clean-block prefill follow PR #21's ObservationKVCache. The dynamic
path implements this branch's visibility, without PR #21's fixed state cache.
Caches are local to one encoded observation window, never stored on the model.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .attention import _compressed_causal_mask, sdpa
from .layers import rotary_tables


@dataclass(frozen=True)
class LayerKV:
    text_k: torch.Tensor
    text_v: torch.Tensor
    video_k: torch.Tensor
    video_v: torch.Tensor


class ObservationKVCache:
    """Prefill one CFG branch's text and real visual prefix in local coordinates.

    Historical clean blocks are evaluated in order before evicting old KV, so
    their deep representations match a full-prefix forward with sliding attention.
    A new RGB/VAE window requires a new instance. No state or prediction is retained.
    """

    def __init__(self, model, inputs, *, attention_backend):
        self.model = model
        self.check_inference()
        if getattr(model.config, "action_interface", None) != "shared":
            raise ValueError("Observation KV cache requires shared Cosmos3 projections")
        if getattr(model, "_attention_probe", None) is not None:
            raise ValueError("Attention probing requires kv_cache=false")
        if attention_backend not in ("dense", "grouped", "npu_tnd"):
            raise ValueError("Unknown cached attention backend")
        context = inputs.context
        self.backend = attention_backend
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        if context.history_video.device != self.device:
            raise ValueError("Observation KV cache requires history on the model device")
        if self.backend == "npu_tnd" and self.device.type != "npu":
            raise ValueError("npu_tnd requires an Ascend NPU")
        self.text_ids = context.text_ids.detach().cpu().clone()
        self.observed = context.history_video.detach().clone()
        self.metadata = self._metadata(context)
        self.history_rows = inputs.history_rows
        self.prefix_length = len(self.text_ids) + self.history_rows
        self.positions = inputs.positions[:, : self.prefix_length].detach().clone()
        self.layers = []
        self._prefill_text()
        patches = self.history_rows // context.history_video.shape[1]
        tokens = inputs.fixed_video_tokens[: self.history_rows].to(self.dtype)
        positions = self.positions[:, len(self.text_ids) :]
        # The independent first latent is followed by complete clean blocks.
        for first in range(0, context.history_video.shape[1], context.block_size):
            start = 0 if first == 0 else (first - context.block_size + 1) * patches
            stop = patches if first == 0 else (first + 1) * patches
            self._append_clean(tokens[start:stop], positions[:, start:stop], context.history_frames * patches)

    @staticmethod
    def _metadata(context):
        return (
            context.block_size,
            context.actions_per_block,
            context.history_frames,
            context.video_fps,
            context.action_fps,
        )

    def check_inference(self):
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("Observation KV cache requires eval() and no_grad()")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError("Observation KV cache requires single-process inference")

    def attend(self, query, key, value, *, causal=False):
        if self.backend != "npu_tnd":
            return sdpa(query, key, value, causal=causal)
        import torch_npu

        return torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=query.shape[1],
            input_layout="TND",
            scale=query.shape[-1] ** -0.5,
            keep_prob=1.0,
            atten_mask=_compressed_causal_mask(query.device) if causal else None,
            sparse_mode=2 if causal else 0,
            actual_seq_qlen=(len(query),),
            actual_seq_kvlen=(len(key),),
        )[0]

    def _prefill_text(self):
        model = self.model
        und = model.embed_tokens(self.text_ids.to(self.device))
        tables = rotary_tables(self.positions[:, : len(self.text_ids)], model.config, self.dtype)
        for layer in model.layers:
            attn = layer.self_attn
            q, k, v, cross_k = attn.project_text(layer.input_layernorm(und), tables)
            und = und + attn.to_out(self.attend(q, k, v, causal=True).flatten(1))
            und = und + layer.mlp(layer.post_attention_layernorm(und))
            self.layers.append(LayerKV(cross_k.detach().clone(), v.detach().clone(), k[:0], v[:0]))

    def _append_clean(self, tokens, positions, keep):
        model = self.model
        gen = model.proj_in(tokens) + model.time_embedder(torch.zeros(len(tokens), device=self.device))
        tables = rotary_tables(positions, model.config, self.dtype)
        staged = []
        for layer, entry in zip(model.layers, self.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), tables)
            keys = torch.cat((entry.text_k, entry.video_k, k))
            values = torch.cat((entry.text_v, entry.video_v, v))
            gen = gen + attn.to_add_out(self.attend(q, keys, values).flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
            staged.append(
                LayerKV(
                    entry.text_k,
                    entry.text_v,
                    torch.cat((entry.video_k, k))[-keep:].detach().clone(),
                    torch.cat((entry.video_v, v))[-keep:].detach().clone(),
                )
            )
        self.layers = staged

    def bind(self, inputs, *, model, attention_backend):
        """Bind a stage to this exact observed prefix; never borrow another window."""
        self.check_inference()
        context = inputs.context
        if (
            model is not self.model
            or attention_backend != self.backend
            or next(model.parameters()).dtype != self.dtype
            or next(model.parameters()).device != self.device
            or context.history_video.device != self.device
            or self.metadata != self._metadata(context)
            or inputs.history_rows != self.history_rows
            or not torch.equal(context.text_ids.cpu(), self.text_ids)
            or not torch.equal(context.history_video, self.observed)
            or not torch.equal(inputs.positions[:, : self.prefix_length], self.positions)
        ):
            raise ValueError("Observation KV cache requires the same model, text, observed window and positions")
        return _CachedBlock(self, inputs)


class _CachedBlock:
    """Run current-block queries with native task visibility and cached prefix KV."""

    def __init__(self, cache, inputs):
        self.cache, self.inputs = cache, inputs
        self.tables = rotary_tables(inputs.positions[:, cache.prefix_length :], cache.model.config, cache.dtype)

    def predict(self, payload):
        cache, inputs = self.cache, self.inputs
        cache.check_inference()
        model = cache.model
        tokens = payload["video_tokens"][inputs.history_rows :]
        times = payload["video_timesteps"][inputs.history_rows :]
        gen = model.proj_in(tokens) + model.time_embedder(times * model.config.timestep_scale)
        nv = len(tokens)
        if inputs.stage != "video":
            action = payload["action_tokens"].masked_fill(~payload["action_mask"], 0)
            action = model.action_proj_in(action) + model.action_modality_embed
            action = action + model.time_embedder(payload["action_timesteps"] * model.config.timestep_scale)
            state = payload["state_tokens"].masked_fill(~payload["state_mask"], 0)
            state = model.state_proj_in(state) + model.state_modality_embed
            state = state + model.time_embedder(torch.zeros(1, device=cache.device))
            gen = torch.cat((gen, action, state))
        for layer, entry in zip(model.layers, cache.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), self.tables)
            pk, pv = torch.cat((entry.text_k, entry.video_k)), torch.cat((entry.text_v, entry.video_v))
            # Video/Policy targets (including Policy state) read the entire current stream.
            if inputs.stage in ("video", "policy"):
                output = cache.attend(q, torch.cat((pk, k)), torch.cat((pv, v)))
            elif inputs.stage == "action":
                # IDM clean video is independent of action/state; state follows action queries.
                video = cache.attend(q[:nv], torch.cat((pk, k[:nv])), torch.cat((pv, v[:nv])))
                action_state = cache.attend(q[nv:], torch.cat((pk, k)), torch.cat((pv, v)))
                output = torch.cat((video, action_state))
            else:
                # FD measured action/state cannot read the noisy video targets.
                video = cache.attend(q[:nv], torch.cat((pk, k)), torch.cat((pv, v)))
                conditions = cache.attend(q[nv:], torch.cat((pk, k[nv:])), torch.cat((pv, v[nv:])))
                output = torch.cat((video, conditions))
            gen = gen + attn.to_add_out(output.flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        return SimpleNamespace(
            video=model.proj_out(gen[:nv]),
            action=model.action_proj_out(gen[nv:-1]).masked_fill(~payload["action_mask"], 0)
            if inputs.action_active
            else None,
        )
