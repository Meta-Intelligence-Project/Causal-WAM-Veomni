# SPDX-License-Identifier: OpenMDW-1.1
"""Session-owned Policy KV cache for live observations and offline measured replay."""

from dataclasses import dataclass

import torch

from . import mrope
from .attention import _compressed_causal_mask, sdpa
from .layers import rotary_tables
from .packing import pack_model_inputs
from .patching import patchify_video, unpatchify_video


@dataclass
class LayerKV:
    text_k: torch.Tensor
    text_v: torch.Tensor
    video_k: torch.Tensor
    video_v: torch.Tensor


class OnlinePolicyKVCache:
    """Bind one text branch and rolling measured history to a single eval model.

    K is stored after normalization/RoPE, V after projection. Historical GEN
    rows and UND rows never run again during denoising. Only measured clean
    video enters the rolling cache; state and noisy targets remain block-local.
    Each sample contains one replaceable prediction block. Only explicit
    observation updates append history; absolute positions survive eviction.
    """

    def __init__(self, model, sample, *, attention_backend="grouped", position_options=None, text_ids=None):
        self.model = model
        self._check_inference()
        if model.config.action_interface != "shared":
            raise ValueError("Policy KV cache requires shared action projections")
        if sample.get("mode", "policy") != "policy" or list(sample["condition_frame_indexes_vision"]) != [0]:
            raise ValueError("KV cache requires Policy with only the initial video condition")
        if len(sample["condition_frame_indexes_action"]):
            raise ValueError("Policy KV cache does not accept conditioned actions")
        if sample["video_latents"].shape[1] != 1 + sample["block_size"] or len(sample["states"]) != 1:
            raise ValueError("Online cache requires one compact prediction block")
        if attention_backend not in ("dense", "grouped", "npu_tnd"):
            raise ValueError("Unknown cached attention backend")
        if attention_backend == "npu_tnd" and sample["video_latents"].device.type != "npu":
            raise ValueError("npu_tnd requires an Ascend NPU")
        self.backend = attention_backend
        self.position_options = dict(position_options or {})
        self.sample = sample
        self.block_size = sample["block_size"]
        self.history = sample["history_blocks"]
        self.dtype = next(model.parameters()).dtype
        self.patch_size = model.config.patch_size
        self.layers = []
        self.next_frame = 0
        self.frame_ids = ()
        self.revision = 0
        self._signature = self._sample_signature(sample)
        inputs = self._bind_sample(sample, text_ids)
        self._prefill_text(inputs["input_ids"])
        self._append_clean(
            inputs["video_tokens"][: self.patches].to(self.dtype),
            self.positions[:, self.span.clean_video][:, : self.patches],
            (0,),
        )

    def _bind_sample(self, sample, text_ids=None):
        """Prepare compact inputs without altering any layer's cached history."""
        self.sample = sample
        branch = dict(sample)
        if text_ids is not None:
            branch["text_token_ids"] = text_ids
            branch["token_length"] += len(text_ids) - len(sample["text_token_ids"])
        self.text_ids = branch["text_token_ids"].detach().cpu().clone()
        video = sample["video_latents"]
        inputs, (span,), _ = pack_model_inputs(
            [branch],
            [video],
            [sample["action"]],
            timesteps=[torch.zeros(video.shape[1], device=video.device)],
            patch_size=self.patch_size,
            **self.position_options,
        )
        self.positions = inputs["position_ids"]
        self.span = span
        self.patches = (span.clean_video.stop - span.clean_video.start) // video.shape[1]
        return inputs

    def validate_branch(self, model, sample, *, attention_backend, position_options, text_ids):
        """Reject stale KV if a low-level caller changes branch inputs or options."""
        if self.model is not model or self.sample is not sample:
            raise ValueError("A Policy KV cache cannot be reused across models or encoded samples")
        ids = sample["text_token_ids"] if text_ids is None else text_ids
        if (
            self.backend != attention_backend
            or self.position_options != dict(position_options or {})
            or not torch.equal(self.text_ids, ids.detach().cpu())
        ):
            raise ValueError("A Policy KV cache requires unchanged text, position options and attention backend")

    def _check_inference(self):
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("Policy KV cache requires eval() and no_grad()")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError("Policy KV cache supports single-process inference only")

    def _attend(self, q, k, v, *, causal=False):
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

    def _prefill_text(self, ids):
        und = self.model.embed_tokens(ids)
        tables = self._tables(self.positions[:, self.span.text])
        staged = []
        for layer in self.model.layers:
            attn = layer.self_attn
            q, k, v, cross_k = attn.project_text(layer.input_layernorm(und), tables)
            output = attn.to_out(self._attend(q, k, v, causal=True).flatten(1))
            und = und + output
            und = und + layer.mlp(layer.post_attention_layernorm(und))
            staged.append(LayerKV(cross_k.detach().clone(), v.detach().clone(), k[:0].clone(), v[:0].clone()))
        self.layers = staged

    def _append_clean(self, tokens, positions, frame_ids):
        """Jointly prefill one measured block, then atomically commit all layers."""
        self._check_inference()
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
            gen = gen + attn.to_add_out(self._attend(q, keys, values).flatten(1))
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

    def predict(self, video, action, timestep):
        """Return current-block velocities without modifying measured history KV."""
        self._check_inference()
        model = self.model
        start, stop = 1, 1 + self.block_size
        arows = torch.arange(len(self.sample["action"]), device=video.device)
        mask = self.sample["action_mask"].to(video.device)
        state_mask = self.sample["state_mask"].to(video.device)
        vt = patchify_video(video[:, start:stop], self.patch_size).to(self.dtype)
        at = action[arows].to(self.dtype).masked_fill(~mask, 0)
        st = self.sample["states"].to(device=video.device, dtype=self.dtype).masked_fill(~state_mask, 0)
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
        vpos = self.positions[:, self.span.current_video][:, start * self.patches : stop * self.patches]
        apos = self.positions[:, self.span.action][:, arows]
        spos = self.positions[:, self.span.state]
        tables = self._tables(torch.cat((vpos, apos, spos), dim=1))
        for layer, entry in zip(model.layers, self.layers, strict=True):
            attn = layer.self_attn
            q, k, v = attn.project_generation(layer.input_layernorm_moe_gen(gen), tables)
            # Preserve packed key order: text, clean history, video, action, state.
            target = self._attend(
                q[:-1], torch.cat((entry.text_k, entry.video_k, k)), torch.cat((entry.text_v, entry.video_v, v))
            )
            state = self._attend(q[-1:], torch.cat((entry.text_k, k[-1:])), torch.cat((entry.text_v, v[-1:])))
            gen = gen + attn.to_add_out(torch.cat((target, state)).flatten(1))
            gen = gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(gen))
        gen = model.norm_moe_gen(gen)
        pred_v = unpatchify_video(model.proj_out(gen[: len(vt)]), video[:, start:stop].shape, self.patch_size)
        pred_a = model.action_proj_out(gen[len(vt) : -1]).masked_fill(~mask, 0)
        return pred_v, pred_a, arows

    @staticmethod
    def _sample_signature(sample):
        video = sample["video_latents"]
        return (
            tuple(video.shape),
            video.device,
            sample["block_size"],
            sample["history_blocks"],
            sample["conditioning_fps"],
            sample["conditioning_fps_action"],
            tuple(sample["action_frame_ids"]),
            tuple(sample["state_latent_indexes"]),
        )

    def update(self, sample, measured, *, revision):
        """Append one new real block and bind the next state/target at absolute time."""
        self._check_inference()
        if revision != self.revision + 1:
            raise ValueError("Online cache revisions must advance by one")
        if (
            self._sample_signature(sample) != self._signature
            or sample.get("mode") != "policy"
            or list(sample["condition_frame_indexes_vision"]) != [0]
            or len(sample["condition_frame_indexes_action"])
            or not torch.equal(sample["text_token_ids"], self.sample["text_token_ids"])
            or any(not torch.equal(sample[key], self.sample[key]) for key in ("action_mask", "state_mask"))
        ):
            raise ValueError("Online cache geometry, text and masks require reset before changing")
        shape = sample["video_latents"].shape
        if (
            measured.shape != (shape[0], self.block_size, *shape[2:])
            or measured.device != sample["video_latents"].device
        ):
            raise ValueError("Online cache requires a complete measured latent block on the model device")
        # Preserve each CFG branch's text length and its modality position offset.
        previous = self.sample, self.positions, self.span, self.patches, self.text_ids
        try:
            self._bind_sample(sample, self.text_ids)
            self._position_at(revision - 1)
            observed_positions = self.positions[:, self.span.clean_video][:, self.patches :].clone()
            self._position_at(revision)
            first = self.next_frame
            self._append_clean(
                patchify_video(measured, self.patch_size).to(self.dtype),
                observed_positions,
                tuple(range(first, first + self.block_size)),
            )
        except Exception:
            self.sample, self.positions, self.span, self.patches, self.text_ids = previous
            raise
        self.revision = revision

    def _position_at(self, revision):
        """Reproduce full-pack FP32 arithmetic at absolute frame/action indexes."""
        sample = self.sample
        _, frames, height, width = sample["video_latents"].shape
        base = self.position_options.get("base_fps", 24.0)
        temporal = self.position_options.get("temporal_compression", 4)
        offset = len(self.text_ids) + self.position_options.get("temporal_modality_margin", 15000)
        with torch.device("cpu"):
            video, _ = mrope.get_3d_mrope_ids_vae_tokens(
                frames,
                (height + self.patch_size - 1) // self.patch_size,
                (width + self.patch_size - 1) // self.patch_size,
                offset,
                fps=sample["conditioning_fps"],
                base_fps=base,
                temporal_compression_factor=temporal,
                start_frame_offset=revision * self.block_size,
            )
            action, _ = mrope.get_3d_mrope_ids_vae_tokens(
                len(sample["action"]),
                1,
                1,
                offset,
                fps=sample["conditioning_fps_action"],
                base_fps=base,
                temporal_compression_factor=1,
                base_temporal_compression_factor=temporal,
                start_frame_offset=1 + revision * len(sample["action"]),
            )
        for section in (self.span.clean_video, self.span.current_video):
            self.positions[:, section] = video.to(self.positions.device)
        self.positions[:, self.span.action] = action.to(self.positions.device)
        self.positions[0, self.span.state] = video[0, 0].to(self.positions.device)
