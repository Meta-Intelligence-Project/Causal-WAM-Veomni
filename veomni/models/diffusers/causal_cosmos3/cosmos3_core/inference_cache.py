# SPDX-License-Identifier: OpenMDW-1.1
"""Session-owned Observation KV cache for live observations and offline measured replay."""

import math
from dataclasses import dataclass

import torch

from . import mrope
from .attention import _compressed_causal_mask, sdpa
from .layers import rotary_tables
from .patching import patchify_video, unpatchify_video


@dataclass
class LayerKV:
    text_k: torch.Tensor
    text_v: torch.Tensor
    video_k: torch.Tensor
    video_v: torch.Tensor


class ObservationKVCache:
    """Own immutable text/geometry and measured KV, independently of predictions.

    Initialization consumes exactly one real latent and no prediction sample.
    Updates append complete measured blocks. Each predict call supplies noisy
    targets, the current measured state and masks without retaining them.
    """

    def __init__(
        self,
        model,
        initial_latent,
        *,
        text_ids,
        block_size,
        actions_per_block,
        history,
        conditioning_fps,
        conditioning_fps_action,
        attention_backend="grouped",
        position_options=None,
    ):
        self.model = model
        self.check_inference()
        if model.config.action_interface != "shared":
            raise ValueError("Observation KV cache requires shared action projections")
        if initial_latent.ndim != 4 or initial_latent.shape[1] != 1 or min(initial_latent.shape) < 1:
            raise ValueError("Online cache initialization requires exactly one real latent [C,1,H,W]")
        if initial_latent.shape[0] != model.config.latent_channels or not initial_latent.is_floating_point():
            raise ValueError("Initial latent must have floating-point model latent channels")
        if initial_latent.device != next(model.parameters()).device:
            raise ValueError("Initial latent must be on the model device")
        if any(type(value) is not int or value < 1 for value in (block_size, actions_per_block, history)):
            raise ValueError("Cache block size, actions per block and history must be positive integers")
        if actions_per_block % block_size:
            raise ValueError("Cache requires a whole number of actions per latent")
        if text_ids.ndim != 1 or not text_ids.numel() or text_ids.dtype != torch.long:
            raise ValueError("Cache text IDs must be a nonempty one-dimensional long tensor")
        if attention_backend not in ("dense", "grouped", "npu_tnd"):
            raise ValueError("Unknown cached attention backend")
        if attention_backend == "npu_tnd" and initial_latent.device.type != "npu":
            raise ValueError("npu_tnd requires an Ascend NPU")
        self.backend = attention_backend
        self.position_options = dict(position_options or {})
        if self.position_options.keys() - {"base_fps", "temporal_compression", "temporal_modality_margin"}:
            raise ValueError("Unknown cache position options")
        self.base_fps = self.position_options.get("base_fps", 24.0)
        self.temporal_compression = self.position_options.get("temporal_compression", 4)
        margin = self.position_options.get("temporal_modality_margin", 15000)
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                conditioning_fps,
                conditioning_fps_action,
                self.base_fps,
                self.temporal_compression,
            )
        ) or not math.isfinite(margin):
            raise ValueError("Cache FPS/temporal compression must be finite and positive, with finite position margin")
        self.block_size = block_size
        self.actions_per_block = actions_per_block
        self.history = history
        self.conditioning_fps = conditioning_fps
        self.conditioning_fps_action = conditioning_fps_action
        self.dtype = next(model.parameters()).dtype
        self.device = initial_latent.device
        self.frame_shape = (initial_latent.shape[0], *initial_latent.shape[2:])
        self.patch_size = model.config.patch_size
        self.grid_height = (initial_latent.shape[2] + self.patch_size - 1) // self.patch_size
        self.grid_width = (initial_latent.shape[3] + self.patch_size - 1) // self.patch_size
        self.patches = self.grid_height * self.grid_width
        self.text_ids = text_ids.detach().cpu().clone()
        self.position_offset = len(self.text_ids) + margin
        self.layers = []
        self.next_frame = 0
        self.frame_ids = ()
        self.revision = 0
        self._prefill_text()
        self._append_clean(
            patchify_video(initial_latent, self.patch_size).to(self.dtype),
            self._video_positions(0, 1),
            (0,),
        )
        self.prediction_tables = self._prediction_tables(0)

    def validate_branch(self, model, sample, *, attention_backend, position_options, text_ids, mode="policy"):
        """Validate current prediction metadata without binding or retaining a sample."""
        if mode not in ("policy", "inverse_dynamics"):
            raise ValueError("Observation cache supports policy or inverse_dynamics")
        if self.model is not model:
            raise ValueError("A Observation KV cache cannot be reused across models")
        ids = sample["text_token_ids"] if text_ids is None else text_ids
        if (
            self.backend != attention_backend
            or self.position_options != dict(position_options or {})
            or not torch.equal(self.text_ids, ids.detach().cpu())
        ):
            raise ValueError("A Observation KV cache requires unchanged text, position options and attention backend")
        video = sample["video_latents"]
        expected_shape = (self.frame_shape[0], 1 + self.block_size, *self.frame_shape[1:])
        expected_action_frames = tuple(
            1 + i // (self.actions_per_block // self.block_size) for i in range(self.actions_per_block)
        )
        if (
            sample.get("mode") != mode
            or list(sample["condition_frame_indexes_vision"])
            != (list(range(1 + self.block_size)) if mode == "inverse_dynamics" else [0])
            or len(sample["condition_frame_indexes_action"])
            or tuple(video.shape) != expected_shape
            or video.device != self.device
            or sample["block_size"] != self.block_size
            or sample["history_blocks"] != self.history
            or sample["conditioning_fps"] != self.conditioning_fps
            or sample["conditioning_fps_action"] != self.conditioning_fps_action
            or tuple(int(frame) for frame in sample["action_frame_ids"]) != expected_action_frames
            or list(sample["state_latent_indexes"]) != [0]
            or sample.get("cache_revision") != self.revision
        ):
            raise ValueError(
                "Observation KV cache prediction requires matching geometry, task conditions and current revision"
            )

    def check_inference(self):
        """Validate the shared execution contract for history and condition caches."""
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("Observation KV cache requires eval() and no_grad()")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError("Observation KV cache supports single-process inference only")

    def attend(self, q, k, v, *, causal=False):
        """Shared inference attention primitive; preserves backend and Q/K normalization."""
        if self.backend != "npu_tnd":
            return sdpa(q, k, v, causal=causal)
        import torch_npu

        return torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=q.shape[1],
            input_layout="TND",
            scale=q.shape[-1] ** -0.5,
            keep_prob=1.0,
            atten_mask=_compressed_causal_mask(q.device) if causal else None,
            sparse_mode=2 if causal else 0,
            actual_seq_qlen=(len(q),),
            actual_seq_kvlen=(len(k),),
        )[0]

    def _tables(self, positions):
        return rotary_tables(positions, self.model.config, self.dtype)

    def _prefill_text(self):
        und = self.model.embed_tokens(self.text_ids.to(self.device))
        with torch.device("cpu"):
            positions, _ = mrope.get_3d_mrope_ids_text_tokens(
                len(self.text_ids), temporal_offset=0, use_float_positions=True
            )
        tables = self._tables(positions.to(self.device))
        staged = []
        for layer in self.model.layers:
            attn = layer.self_attn
            q, k, v, cross_k = attn.project_text(layer.input_layernorm(und), tables)
            output = attn.to_out(self.attend(q, k, v, causal=True).flatten(1))
            und = und + output
            und = und + layer.mlp(layer.post_attention_layernorm(und))
            staged.append(LayerKV(cross_k.detach().clone(), v.detach().clone(), k[:0].clone(), v[:0].clone()))
        self.layers = staged

    def _append_clean(self, tokens, positions, frame_ids):
        """Jointly prefill one measured block, then atomically commit all layers."""
        self.check_inference()
        zeros = torch.zeros(len(tokens), device=tokens.device)
        gen = self.model.proj_in(tokens) + self.model.time_embedder(zeros)
        tables = self._tables(positions)
        staged = []
        keep = self.history * self.patches
        for layer, entry in zip(self.model.layers, self.layers, strict=True):
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
        # No partial layer updates survive a failed prefill.
        self.layers = staged
        self.frame_ids = (*self.frame_ids, *frame_ids)[-self.history :]
        self.next_frame = frame_ids[-1] + 1

    def predict(self, video, action, state, timestep, *, action_mask, state_mask, revision):
        """Predict velocities for explicit current targets; measured history is read-only."""
        self.check_inference()
        if revision != self.revision:
            raise ValueError("Observation KV cache prediction requires the current revision")
        expected_shape = (self.frame_shape[0], self.block_size, *self.frame_shape[1:])
        if tuple(video.shape) != expected_shape or video.device != self.device:
            raise ValueError("Prediction video must contain only the current noisy block on the model device")
        if (
            tuple(action.shape) != (self.actions_per_block, self.model.config.action_dim)
            or action.device != self.device
        ):
            raise ValueError("Prediction action must match the configured block and model device")
        if tuple(state.shape) != (1, self.model.config.state_dim):
            raise ValueError("Prediction requires one current measured state")
        for mask, width in ((action_mask, action.shape[1]), (state_mask, state.shape[1])):
            if mask.dtype != torch.bool or tuple(mask.shape) != (width,):
                raise ValueError("Prediction masks must be one-dimensional boolean valid-feature masks")
        model = self.model
        action_mask, state_mask = action_mask.to(self.device), state_mask.to(self.device)
        vt = patchify_video(video, self.patch_size).to(self.dtype)
        at = action.to(self.dtype).masked_fill(~action_mask, 0)
        st = state.to(device=self.device, dtype=self.dtype).masked_fill(~state_mask, 0)
        tv = timestep.float().expand(len(vt))
        ta = timestep.float().expand(len(at))
        gv = model.proj_in(vt) + model.time_embedder(tv * model.config.timestep_scale)
        ga = (
            model.action_proj_in(at)
            + model.action_modality_embed
            + model.time_embedder(ta * model.config.timestep_scale)
        )
        gs = (
            model.state_proj_in(st)
            + model.state_modality_embed
            + model.time_embedder(torch.zeros(1, device=video.device))
        )
        gen = torch.cat((gv, ga, gs))
        tables = self.prediction_tables
        for layer, entry in zip(model.layers, self.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), tables)
            # Preserve packed key order: text, clean history, video, action, state.
            target = self.attend(
                q[:-1], torch.cat((entry.text_k, entry.video_k, k)), torch.cat((entry.text_v, entry.video_v, v))
            )
            state = self.attend(q[-1:], torch.cat((entry.text_k, k[-1:])), torch.cat((entry.text_v, v[-1:])))
            gen = gen + attn.to_add_out(torch.cat((target, state)).flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        pred_v = unpatchify_video(model.proj_out(gen[: len(vt)]), video.shape, self.patch_size)
        pred_a = model.action_proj_out(gen[len(vt) : -1]).masked_fill(~action_mask, 0)
        return pred_v, pred_a

    def update(self, measured, *, revision):
        """Append one new complete real latent block, independently of prediction inputs."""
        self.check_inference()
        if type(revision) is not int or revision != self.revision + 1:
            raise ValueError("Online cache revisions must advance by one")
        expected_shape = (self.frame_shape[0], self.block_size, *self.frame_shape[1:])
        if (
            tuple(measured.shape) != expected_shape
            or measured.device != self.device
            or not measured.is_floating_point()
        ):
            raise ValueError("Online cache requires a complete measured latent block on the model device")
        # Prepare metadata first; a failed prefill leaves tables, revision and all layers unchanged.
        tables = self._prediction_tables(revision)
        first = self.next_frame
        self._append_clean(
            patchify_video(measured, self.patch_size).to(self.dtype),
            self._video_positions(first, self.block_size),
            tuple(range(first, first + self.block_size)),
        )
        self.prediction_tables = tables
        self.revision = revision

    def _video_positions(self, start, count):
        """Use the full-pack FP32 mRoPE arithmetic for absolute latent indexes."""
        with torch.device("cpu"):
            positions, _ = mrope.get_3d_mrope_ids_vae_tokens(
                count,
                self.grid_height,
                self.grid_width,
                self.position_offset,
                fps=self.conditioning_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression,
                start_frame_offset=start,
            )
        return positions.to(self.device)

    def _prediction_tables(self, revision):
        """Build target video/action/state RoPE tables once per observation revision."""
        video = self._video_positions(1 + revision * self.block_size, self.block_size)
        state = self._video_positions(revision * self.block_size, 1)[:, :1]
        with torch.device("cpu"):
            action, _ = mrope.get_3d_mrope_ids_vae_tokens(
                self.actions_per_block,
                1,
                1,
                self.position_offset,
                fps=self.conditioning_fps_action,
                base_fps=self.base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=self.temporal_compression,
                start_frame_offset=1 + revision * self.actions_per_block,
            )
        return self._tables(torch.cat((video, action.to(self.device), state), dim=1))


class IDMBlockKVCache:
    """One immutable IDM sampling pass, borrowing an episode's text/history.

    Condition video reads text, clean history, itself and state (Scheme B).
    State reads text and itself. Neither reads target actions, so their layer
    K/V can be prefetched once. These condition K/V are NOT clean-history K/V:
    they depend on the current state and must never be appended with update().
    """

    def __init__(self, history, sample):
        history.check_inference()
        self.history, self.revision = history, history.revision
        model = history.model
        self.action_mask = sample["action_mask"].detach().clone()
        vt = patchify_video(sample["video_latents"][:, 1:], history.patch_size).to(history.dtype)
        st = (
            sample["states"].to(history.device, history.dtype).masked_fill(~sample["state_mask"].to(history.device), 0)
        )
        if tuple(st.shape) != (1, model.config.state_dim):
            raise ValueError("IDM prefill requires one current state")
        self.layers = []
        zeros = torch.zeros(len(vt), device=history.device)
        video = model.proj_in(vt) + model.time_embedder(zeros)
        state = model.state_proj_in(st) + model.state_modality_embed + model.time_embedder(zeros[:1])
        gen = torch.cat((video, state))
        tables = tuple(torch.cat((table[: len(vt)], table[-1:])) for table in history.prediction_tables)
        self.action_tables = tuple(table[len(vt) : -1] for table in history.prediction_tables)
        for layer, entry in zip(model.layers, history.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), tables)
            self.layers.append((k.detach().clone(), v.detach().clone()))
            video_out = history.attend(
                q[:-1], torch.cat((entry.text_k, entry.video_k, k)), torch.cat((entry.text_v, entry.video_v, v))
            )
            state_out = history.attend(q[-1:], torch.cat((entry.text_k, k[-1:])), torch.cat((entry.text_v, v[-1:])))
            gen = gen + attn.to_add_out(torch.cat((video_out, state_out)).flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))

    def predict(self, action, timestep):
        """Only noisy actions change within this pass; all conditions are bound."""
        history = self.history
        history.check_inference()
        if history.revision != self.revision:
            raise ValueError("IDM condition cache expired after history advanced")
        model = history.model
        if (
            tuple(action.shape) != (history.actions_per_block, model.config.action_dim)
            or action.device != history.device
        ):
            raise ValueError("IDM action must match the block shape and model device")
        action_mask = self.action_mask.to(history.device)
        action = action.to(history.dtype).masked_fill(~action_mask, 0)
        times = timestep.float().expand(len(action)) * model.config.timestep_scale
        gen = model.action_proj_in(action) + model.action_modality_embed + model.time_embedder(times)
        for layer, entry, (fixed_k, fixed_v) in zip(model.layers, history.layers, self.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), self.action_tables)
            # Full-pack order: text, clean history, condition video, noisy action, state.
            keys = torch.cat((entry.text_k, entry.video_k, fixed_k[:-1], k, fixed_k[-1:]))
            values = torch.cat((entry.text_v, entry.video_v, fixed_v[:-1], v, fixed_v[-1:]))
            gen = gen + attn.to_add_out(history.attend(q, keys, values).flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        return model.action_proj_out(model.norm_moe_gen(gen)).masked_fill(~action_mask, 0)


# Compatibility for callers of the former online-Policy-only API.
OnlinePolicyKVCache = ObservationKVCache
