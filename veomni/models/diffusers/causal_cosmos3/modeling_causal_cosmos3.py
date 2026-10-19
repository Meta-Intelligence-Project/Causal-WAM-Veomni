# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Cosmos3 video/action network; no dataset or Cosmos runtime dependency."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from veomni.models.checkpoint.convert import ConvertedCheckpointTensor

from .attention import CausalLayout, prepare_attention
from .configuration_causal_cosmos3 import Cosmos3Config
from .layers import Cosmos3DecoderLayer, DomainLinear, RMSNorm, TimeEmbedder, rotary_tables


def patchify_video(latents, patch_size):
    """[C,T,H,W] -> [T*ceil(H/p)*ceil(W/p),p*p*C], with channels last."""
    c, t, h, w = latents.shape
    p = patch_size
    padded = F.pad(latents, (0, -w % p, 0, -h % p))
    return (
        padded.reshape(c, t, (h + p - 1) // p, p, (w + p - 1) // p, p).permute(1, 2, 4, 3, 5, 0).reshape(-1, p * p * c)
    )


def unpatchify_video(tokens, shape, patch_size):
    """Invert patchify_video and crop spatial padding to the original [C,T,H,W]."""
    c, t, h, w = shape
    p = patch_size
    hp, wp = (h + p - 1) // p, (w + p - 1) // p
    return tokens.reshape(t, hp, wp, p, p, c).permute(5, 0, 1, 3, 2, 4).reshape(c, t, hp * p, wp * p)[:, :, :h, :w]


@dataclass
class Cosmos3Output(ModelOutput):
    """Predictions in modality payload order, including clean/condition positions.

    The caller selects role=1 for supervision and supplies the training loss.
    State tokens are inputs only. Text logits are optional to avoid a large
    vocabulary projection during video/action training.
    """

    video: torch.Tensor | None = None
    action: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


class _WeightAudit:
    """Use VeOmni's streaming converter hook for strict coverage and explicit warm starts."""

    def __init__(self, model):
        self.model = model
        self.expected = {name: tuple(value.shape) for name, value in model.named_parameters()}
        self.loaded, self.skipped = set(), set()
        self.warm_start = model._pretrained_initialization
        model._pretrained_initialization = False

    def can_handle(self, name):
        return True

    def convert(self, name, tensor):
        # Audio belongs to a separate, deliberately unconstructed component.
        if name.startswith(("audio_proj_in.", "audio_proj_out.")) or name == "audio_modality_embed":
            if not self.warm_start:
                raise ValueError("Audio checkpoint tensors require explicit pretrained initialization")
            self.skipped.add(name)
            return None
        action_key = name.startswith(("action_proj_in.", "action_proj_out.")) or name == "action_modality_embed"
        if (
            self.warm_start
            and action_key
            and (
                self.model.config.action_interface == "none"
                or (
                    self.model.config.action_interface == "shared"
                    and (".fc." in name or ".bias.weight" in name or name == "action_modality_embed")
                )
            )
        ):
            self.skipped.add(name)
            return None
        if name not in self.expected:
            raise ValueError(f"Unexpected Cosmos3 weight: {name}; use the transformer component directory")
        if tuple(tensor.shape) != self.expected[name]:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, model {self.expected[name]}"
            )
        if name in self.loaded:
            raise ValueError(f"Duplicate Cosmos3 weight: {name}")
        self.loaded.add(name)
        return ConvertedCheckpointTensor(name, tensor)

    def finalize(self):
        missing = set(self.expected) - self.loaded
        allowed = set()
        if self.warm_start and self.model.config.action_interface == "shared":
            allowed = {
                name
                for name in self.expected
                if name.startswith(("action_proj_in.", "action_proj_out.", "state_proj_in."))
                or name in ("action_modality_embed", "state_modality_embed")
            }
        if missing - allowed:
            raise ValueError(f"Missing Cosmos3 weights: {sorted(missing - allowed)}")
        self.model.weight_loading_report = {
            "mode": "pretrained_initialization" if self.warm_start else "complete",
            "loaded_count": len(self.loaded),
            "initialized": sorted(missing),
            "skipped": sorted(self.skipped),
        }
        initialized = []
        for name in sorted(missing):
            parameter = self.model.get_parameter(name)
            tensor = torch.empty(self.expected[name], dtype=parameter.dtype)
            if name.endswith(".bias"):
                nn.init.zeros_(tensor)
            elif name.endswith(".weight"):
                std = tensor.shape[1] ** -0.5
                nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
            elif name == "action_modality_embed":
                std = self.model.config.hidden_size**-0.5
                nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
            else:
                nn.init.normal_(tensor, std=self.model.config.hidden_size**-0.5)
            initialized.append(ConvertedCheckpointTensor(name, tensor))
        return initialized


class Cosmos3Model(PreTrainedModel):
    config_class = Cosmos3Config
    base_model_prefix = ""
    _no_split_modules = ["Cosmos3DecoderLayer"]
    supports_gradient_checkpointing = True
    _create_checkpoint_tensor_converter = staticmethod(_WeightAudit)

    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.layers = nn.ModuleList(Cosmos3DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm, self.norm_moe_gen = RMSNorm(h, config), RMSNorm(h, config)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)
        self.proj_in, self.proj_out = nn.Linear(config.patch_dim, h), nn.Linear(h, config.patch_dim)
        self.time_embedder = TimeEmbedder(h)
        if config.action_interface != "none":
            self.action_modality_embed = nn.Parameter(torch.empty(h))
            if config.action_interface == "domain":
                self.action_proj_in = DomainLinear(config.action_dim, h, config.num_embodiment_domains)
                self.action_proj_out = DomainLinear(h, config.action_dim, config.num_embodiment_domains)
            else:
                self.action_proj_in, self.action_proj_out = (
                    nn.Linear(config.action_dim, h),
                    nn.Linear(h, config.action_dim),
                )
        if config.state_dim:
            self.state_proj_in = nn.Linear(config.state_dim, h)
            self.state_modality_embed = nn.Parameter(torch.empty(h))
        self.gradient_checkpointing = False
        self._pretrained_initialization = False
        self.weight_loading_report = None
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, DomainLinear):
            nn.init.normal_(module.fc.weight, std=module.input_size**-0.5)
            nn.init.zeros_(module.bias.weight)
        elif isinstance(module, Cosmos3Model):
            std = self.config.hidden_size**-0.5
            if hasattr(module, "action_modality_embed"):
                nn.init.trunc_normal_(module.action_modality_embed, std=std, a=-3 * std, b=3 * std)
            if hasattr(module, "state_modality_embed"):
                nn.init.normal_(module.state_modality_embed, std=std)
            # Match the new shared causal heads' source initialization.
            for name in ("action_proj_in", "action_proj_out", "state_proj_in"):
                projection = getattr(module, name, None)
                if isinstance(projection, nn.Linear):
                    std = projection.in_features**-0.5
                    nn.init.trunc_normal_(projection.weight, std=std, a=-3 * std, b=3 * std)
                    nn.init.zeros_(projection.bias)

    def prepare_pretrained_initialization(self):
        """Allow one load to initialize new shared action/state heads and omit audio.

        This is a transient loading policy, never saved in the structure config.
        Complete model reloads and DCP resumes do not inherit these exceptions.
        """
        self._pretrained_initialization = True

    def forward(
        self,
        input_ids: torch.Tensor,
        video_tokens: torch.Tensor,
        video_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        layout: CausalLayout,
        action_tokens: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_domains: torch.Tensor | None = None,
        state_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        attention_backend: str = "dense",
        output_text_logits: bool = False,
        output_hidden_states: bool = False,
    ) -> Cosmos3Output:
        """Forward prepatchified VAE latents and optional action/state payloads.

        Timesteps use the source 0..1000 convention. Clean/condition/state
        positions use zero time; position_ids [3,N] already include FPS and
        modality offsets. Construct clean/target copies outside the network,
        without detaching them inside the trainable backbone.
        """
        n = len(layout.sample_ids)
        if position_ids.shape != (3, n):
            raise ValueError("position_ids must be [3, packed_length]")
        plan = prepare_attention(layout, video_tokens.device, attention_backend)
        ti, vi, ai, si = plan.modality_indexes
        if (
            input_ids.shape != (len(ti),)
            or video_tokens.shape != (len(vi), self.config.patch_dim)
            or video_timesteps.shape != (len(vi),)
        ):
            raise ValueError("Text/video payload shapes do not match the layout/config")
        for label, indexes, payload, mask, width in (
            ("action", ai, action_tokens, action_mask, self.config.action_dim),
            ("state", si, state_tokens, state_mask, self.config.state_dim),
        ):
            if len(indexes) and (
                payload is None
                or payload.shape != (len(indexes), width)
                or mask is None
                or mask.shape != payload.shape
                or mask.dtype != torch.bool
            ):
                raise ValueError(f"{label} tokens and a boolean valid-dimension mask must match the layout/config")
            if not len(indexes) and payload is not None:
                raise ValueError(f"{label} payload supplied without corresponding layout entries")
        if len(ai) and (
            self.config.action_interface == "none" or action_timesteps is None or action_timesteps.shape != (len(ai),)
        ):
            raise ValueError("Action payload requires an enabled interface and one timestep per action")
        if len(si) and not self.config.state_dim:
            raise ValueError("State payload requires state_dim > 0")
        if (
            len(ai)
            and self.config.action_interface == "domain"
            and (action_domains is None or action_domains.shape != (len(ai),))
        ):
            raise ValueError("Domain action projections require one domain ID per action token")
        roles = torch.tensor(layout.roles, device=video_tokens.device)

        def time_embedding(times, indexes):
            # Conditions and clean history always have time zero.
            times = times.masked_fill(roles[indexes] != 1, 0)
            return self.time_embedder(times * self.config.timestep_scale)

        text = self.embed_tokens(input_ids)
        video = self.proj_in(video_tokens) + time_embedding(video_timesteps, vi)
        hidden = video.new_zeros(n, self.config.hidden_size).index_copy(0, ti, text).index_copy(0, vi, video)
        if len(ai):
            action = action_tokens.masked_fill(~action_mask, 0)
            action = (
                self.action_proj_in(action, action_domains)
                if self.config.action_interface == "domain"
                else self.action_proj_in(action)
            )
            action = action + self.action_modality_embed + time_embedding(action_timesteps, ai)
            hidden = hidden.index_copy(0, ai, action)
        if len(si):
            state = self.state_proj_in(state_tokens.masked_fill(~state_mask, 0)) + self.state_modality_embed
            state = state + self.time_embedder(torch.zeros(len(si), device=state.device))
            hidden = hidden.index_copy(0, si, state)
        tables = rotary_tables(position_ids, self.config, hidden.dtype)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, tables, plan, use_reentrant=False)
            else:
                hidden = layer(hidden, tables, plan)
        normalized = (
            hidden.new_zeros(hidden.shape)
            .index_copy(0, ti, self.norm(hidden[ti]))
            .index_copy(0, plan.gen_indexes, self.norm_moe_gen(hidden[plan.gen_indexes]))
        )
        action_prediction = None
        if len(ai):
            action_prediction = (
                self.action_proj_out(normalized[ai], action_domains)
                if self.config.action_interface == "domain"
                else self.action_proj_out(normalized[ai])
            )
            action_prediction = action_prediction.masked_fill(~action_mask, 0)
        return Cosmos3Output(
            video=self.proj_out(normalized[vi]),
            action=action_prediction,
            text_logits=self.lm_head(normalized[ti]) if output_text_logits else None,
            last_hidden_state=normalized if output_hidden_states else None,
        )
