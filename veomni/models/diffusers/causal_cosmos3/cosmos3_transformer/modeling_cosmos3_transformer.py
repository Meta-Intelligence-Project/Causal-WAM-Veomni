# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Cosmos3 video/action network; no dataset or Cosmos runtime dependency."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from veomni.models.checkpoint.convert import ConvertedCheckpointTensor
from veomni.utils import logging

from ..cosmos3_core.attention import CausalLayout, prepare_attention
from ..cosmos3_core.flow_matching import TrainTimeWeight, compute_flow_matching_loss
from ..cosmos3_core.layers import Cosmos3DecoderLayer, DomainLinear, RMSNorm, TimeEmbedder, rotary_tables
from ..cosmos3_core.patching import unpatchify_video
from .configuration_cosmos3_transformer import Cosmos3Config


logger = logging.get_logger(__name__)


@dataclass
class Cosmos3Output(ModelOutput):
    """Predictions in modality payload order, including clean/condition positions.

    Targets are optional; training returns a video/action loss dictionary.
    State tokens are inputs only. Text logits are optional to avoid a large
    vocabulary projection during video/action training.
    """

    loss: dict[str, torch.Tensor] | None = None
    video: torch.Tensor | None = None
    action: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None


class _WeightAudit:
    """按本次配置筛选并校验参数；缺失参数默认补初始化，也可显式限制范围。"""

    def __init__(self, model):
        self.model = model
        self.expected = {name: tuple(value.shape) for name, value in model.named_parameters()}
        self.loaded, self.skipped = set(), set()
        policy = model.config.weight_loading
        fields = {"skip_patterns", "initialize_missing_patterns"}
        if not isinstance(policy, dict) or set(policy) - fields:
            raise ValueError(f"weight_loading must be a mapping with only {sorted(fields)}")
        for key, patterns in policy.items():
            if not isinstance(patterns, (list, tuple)) or any(not isinstance(p, str) or not p for p in patterns):
                raise ValueError(f"weight_loading.{key} must be a list of nonempty string patterns")
        self.skip_patterns = tuple(policy.get("skip_patterns", ()))
        # 不配置时自动初始化全部缺失参数；显式空列表则要求参数齐全。
        self.initialize_missing_patterns = (
            tuple(policy["initialize_missing_patterns"]) if "initialize_missing_patterns" in policy else None
        )
        # 规则只控制这一次加载；保留调用方训练配置，避免修改传入的字典。
        model.config.weight_loading = {}

    def can_handle(self, name):
        return True

    def should_skip_without_loading(self, name):
        """在读取 Tensor 前匹配参数全名；规则支持 *、? 等 glob 通配符。"""
        return any(fnmatchcase(name, pattern) for pattern in self.skip_patterns)

    def record_skip_without_loading(self, name):
        """保留跳过记录，供最终加载报告和重复参数检查使用。"""
        if not self.should_skip_without_loading(name):
            raise ValueError(f"Cosmos3 checkpoint parameter is not configured to skip: {name}")
        if name in self.loaded or name in self.skipped:
            raise ValueError(f"Duplicate Cosmos3 weight: {name}")
        self.skipped.add(name)

    def convert(self, name, tensor):
        # 同时兼容直接调用 convert 的路径，常规流式加载会在读取前跳过。
        if self.should_skip_without_loading(name):
            self.record_skip_without_loading(name)
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
        if self.initialize_missing_patterns is not None:
            allowed = {
                name
                for name in missing
                if any(fnmatchcase(name, pattern) for pattern in self.initialize_missing_patterns)
            }
            if missing - allowed:
                raise ValueError(f"Missing Cosmos3 weights: {sorted(missing - allowed)}")
        self.model.weight_loading_report = {
            "mode": "partial" if self.skip_patterns or self.initialize_missing_patterns or missing else "complete",
            "loaded_count": len(self.loaded),
            "initialized": sorted(missing),
            "skipped": sorted(self.skipped),
        }
        initialized = []
        for name in sorted(missing):
            parameter = self.model.get_parameter(name)
            tensor = torch.empty(self.expected[name], dtype=parameter.dtype, device="cpu")
            module_name, _, local_name = name.rpartition(".")
            module = self.model.get_submodule(module_name)
            projection = getattr(self.model, name.split(".")[0])
            if isinstance(projection, DomainLinear):
                if module is projection.bias:
                    nn.init.zeros_(tensor)
                else:
                    # 原 Cosmos3VFMNetwork 对 domain 权重同样使用 ±3σ 截断。
                    std = projection.input_size**-0.5
                    nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
            elif local_name == "bias":
                nn.init.zeros_(tensor)
            elif name in (
                "proj_in.weight",
                "proj_out.weight",
                "time_embedder.linear_1.weight",
                "time_embedder.linear_2.weight",
                "action_proj_in.weight",
                "action_proj_out.weight",
                "state_proj_in.weight",
            ):
                # 视频、时间和 shared action/state 投影均按各自输入宽度缩放。
                std = module.in_features**-0.5
                nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
            elif name == "action_modality_embed":
                std = self.model.config.hidden_size**-0.5
                nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
            elif name == "state_modality_embed":
                nn.init.normal_(tensor, std=self.model.config.hidden_size**-0.5)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(tensor)
            else:
                # 原 reasoner 的其余 Linear/Embedding 沿用 HF 的初始化规则。
                nn.init.normal_(tensor, std=self.model.config.initializer_range or 0.02)
            initialized.append(ConvertedCheckpointTensor(name, tensor))
        if initialized:
            logger.warning_rank0(
                "Initialized %d missing Cosmos3 parameter(s) using model initialization rules: %s",
                len(initialized),
                ", ".join(sorted(missing)),
            )
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
        self.weight_loading_report = None
        self.post_init()

    def freeze_model(self):
        """Select the source action recipe's GEN parameters before FSDP wrapping."""
        if not self.config.train_generation_only:
            return
        selected = (
            "moe_gen",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "norm_added_",
            "k_norm_und_for_gen",
            "time_embedder",
            "proj_in",
            "proj_out",
            "action_proj_in",
            "action_proj_out",
            "state_proj_in",
            "action_modality_embed",
            "state_modality_embed",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(any(key in name for key in selected))

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
        targets: dict | None = None,
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
        outputs = Cosmos3Output(
            video=self.proj_out(normalized[vi]),
            action=action_prediction,
            text_logits=self.lm_head(normalized[ti]) if output_text_logits else None,
            last_hidden_state=normalized if output_hidden_states else None,
        )
        if targets is not None:
            outputs["loss"] = self.loss(outputs, targets)
        return outputs

    def loss(self, outputs, targets):
        """用 packing 登记的模态输出切片计算 FP32 video/action loss。

        视频预测和 patch 监督 mask 一起还原到原始空间尺寸，排除 patch padding；
        动作直接保留 [A,D]。只转换已有 mask 的表示，不重算条件、有效帧或有效维度规则。
        每个样本按有效元素数归一化，再对 pack 内样本平均；全条件样本贡献零但仍计数。
        本层不做 microbatch/DP 缩放，也不修改预测或切断其梯度。
        """
        predictions = {"video": [], "action": []}
        condition_masks = {"video": [], "action": []}
        for span, target_video in zip(targets["spans"], targets["video"], strict=True):
            if span.current_video_rows is None or span.action_rows is None:
                raise ValueError("Loss requires modality output rows registered by pack_model_inputs")
            shape = tuple(target_video.shape)
            predictions["video"].append(
                unpatchify_video(outputs.video[span.current_video_rows], shape, self.config.patch_size).float()
            )
            predictions["action"].append(outputs.action[span.action_rows].float())
            # 一维 patch mask 广播到 patch 内的空间点，再复用同一还原/裁剪逻辑。
            # 这样空间补齐不会扩大有效元素分母，也支持后续按空间 patch 屏蔽。
            video_mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows]
            video_mask = unpatchify_video(
                video_mask[:, None].expand(-1, self.config.patch_size**2),
                (1, *shape[1:]),
                self.config.patch_size,
            ).squeeze(0)
            action_mask = targets["loss_masks"]["action_loss_mask"][span.action_rows]
            # 原 loss 接口使用 1=排除；当前 packing 使用 True=监督，仅在这里反转一次。
            condition_masks["video"].append((~video_mask).float())
            condition_masks["action"].append((~action_mask).float())

        losses = {}
        for name, weight in (("video", self.config.video_loss_weight), ("action", self.config.action_loss_weight)):
            value, _ = compute_flow_matching_loss(
                predictions[name],
                targets[name],
                condition_masks[name],
                torch.nn.utils.rnn.pad_sequence(targets[name + "_times"], batch_first=True),
                has_valid_tokens=True,
                rectified_flow=SimpleNamespace(train_time_weight=TrainTimeWeight(None, "uniform")),
                tensor_kwargs_fp32={"device": outputs.video.device, "dtype": torch.float32},
                normalize_by_active=True,
            )
            losses[name + "_loss"] = value * weight
        return losses
