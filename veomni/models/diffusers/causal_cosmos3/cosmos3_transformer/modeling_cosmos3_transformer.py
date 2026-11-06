# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Cosmos3 video/action network; no dataset or Cosmos runtime dependency."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial
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
from ..cosmos3_core.segment import forward_segment
from ..cosmos3_core.tbptt import partition_history
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
    segment_memory: object | None = None
    segment_stats: dict[str, int] | None = None


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
            missing_modules = sorted({name.rpartition(".")[0] or "<root>" for name in missing})
            logger.warning_rank0(
                "Initialized %d missing Cosmos3 parameter(s) using model initialization rules. "
                "These parameters were not loaded from the checkpoint.\n"
                "Affected modules: %s\nMissing parameters (name, shape):\n%s",
                len(initialized),
                ", ".join(missing_modules),
                "\n".join(f"  - {name}: {self.expected[name]}" for name in sorted(missing)),
            )
        return initialized


class Cosmos3Model(PreTrainedModel):
    config_class = Cosmos3Config
    base_model_prefix = ""
    # This DiT uses non-reentrant checkpointing; frozen text embeddings do not
    # need HF's input_ids hook that forces their outputs to require gradients.
    main_input_name = "video_tokens"
    _no_split_modules = ["Cosmos3DecoderLayer"]
    supports_gradient_checkpointing = True
    _create_checkpoint_tensor_converter = staticmethod(_WeightAudit)

    def __init__(self, config):
        config.validate_tbptt()
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
        # Preserve callers that toggle the boolean without calling HF enable.
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self.weight_loading_report = None
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Keep standalone non-reentrant behavior; the runtime can bind its
        # per-layer recomputation/SAC policy onto this HF checkpoint entry.
        options = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

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
        input_ids: torch.Tensor | None = None,
        video_tokens: torch.Tensor | None = None,
        video_timesteps: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        layout: CausalLayout | None = None,
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
        attention_plan=None,
        cache_capture: dict | None = None,
        stream_segment: dict | None = None,
        stream_memory=None,
    ) -> Cosmos3Output:
        """Forward prepatchified VAE latents and optional action/state payloads.

        Timesteps use the source 0..1000 convention. Clean/condition/state
        positions use zero time; position_ids [3,N] already include FPS and
        modality offsets. Construct clean/target copies outside the network.
        Optional training-only TBPTT truncates old historical encoder gradients
        without reducing forward context or detaching its trainable K/V readout.
        """
        if stream_segment is not None:
            if any(
                value is not None
                for value in (
                    input_ids,
                    video_tokens,
                    video_timesteps,
                    position_ids,
                    layout,
                    targets,
                    action_tokens,
                    action_timesteps,
                    action_mask,
                    action_domains,
                    state_tokens,
                    state_mask,
                )
            ):
                raise ValueError("stream_segment cannot be mixed with ordinary packed inputs")
            if cache_capture is not None or attention_plan is not None or output_text_logits or output_hidden_states:
                raise ValueError("Episode segments do not accept ordinary inference/debug cache options")
            if "attention_backend" not in stream_segment:
                stream_segment = {**stream_segment, "attention_backend": attention_backend}
            outputs = Cosmos3Output(**forward_segment(self, stream_segment, stream_memory))
            if stream_segment.get("targets") is not None:
                outputs["loss"] = self.loss(outputs, stream_segment["targets"])
            return outputs
        if stream_memory is not None:
            raise ValueError("stream_memory requires stream_segment")
        if any(value is None for value in (input_ids, video_tokens, video_timesteps, position_ids, layout)):
            raise ValueError("Ordinary forward requires text/video payloads, positions and layout")
        n = len(layout.sample_ids)
        if position_ids.shape != (3, n):
            raise ValueError("position_ids must be [3, packed_length]")
        plan = (
            attention_plan
            if attention_plan is not None
            else prepare_attention(layout, video_tokens.device, attention_backend)
        )
        if plan.layout is not layout and plan.layout != layout:
            raise ValueError("The cached attention plan belongs to a different layout")
        if cache_capture is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("Prefix K/V capture requires eval-mode, no-grad inference")
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

        # GEN rows retain their original packed order, independently of UND.
        video_rows, action_rows, state_rows = (
            torch.searchsorted(plan.gen_indexes, indexes) for indexes in (vi, ai, si)
        )
        und = self.embed_tokens(input_ids)
        video = self.proj_in(video_tokens) + time_embedding(video_timesteps, vi)
        gen = video.new_zeros(len(plan.gen_indexes), self.config.hidden_size).index_copy(0, video_rows, video)
        if len(ai):
            action = action_tokens.masked_fill(~action_mask, 0)
            action = (
                self.action_proj_in(action, action_domains)
                if self.config.action_interface == "domain"
                else self.action_proj_in(action)
            )
            action = action + self.action_modality_embed + time_embedding(action_timesteps, ai)
            gen = gen.index_copy(0, action_rows, action)
        if len(si):
            state = self.state_proj_in(state_tokens.masked_fill(~state_mask, 0)) + self.state_modality_embed
            state = state + self.time_embedder(torch.zeros(len(si), device=state.device))
            gen = gen.index_copy(0, state_rows, state)
        cos, sin = rotary_tables(position_ids, self.config, gen.dtype)
        tables = ((cos[ti], sin[ti]), (cos[plan.gen_indexes], sin[plan.gen_indexes]))
        history_partition = (
            partition_history(plan, self.config.tbptt_keep_history_latents)
            if self.training and torch.is_grad_enabled() and self.config.enable_tbptt
            else None
        )
        for index, layer in enumerate(self.layers):
            if history_partition is not None:
                if self.gradient_checkpointing and self.training:
                    und, gen = self._gradient_checkpointing_func(
                        layer, und, gen, tables, plan, None, history_partition
                    )
                else:
                    und, gen = layer(und, gen, tables, plan, history_partition=history_partition)
            elif self.gradient_checkpointing and self.training:
                und, gen = self._gradient_checkpointing_func(layer, und, gen, tables, plan)
            else:
                if cache_capture is None:
                    und, gen = layer(und, gen, tables, plan)
                else:
                    cache_capture[index] = {}
                    und, gen = layer(und, gen, tables, plan, cache_capture[index])
        und, gen = self.norm(und), self.norm_moe_gen(gen)
        action_prediction = None
        if len(ai):
            action_prediction = (
                self.action_proj_out(gen[action_rows], action_domains)
                if self.config.action_interface == "domain"
                else self.action_proj_out(gen[action_rows])
            )
            action_prediction = action_prediction.masked_fill(~action_mask, 0)
        normalized = None
        if output_hidden_states:
            normalized = (
                gen.new_zeros(n, self.config.hidden_size).index_copy(0, ti, und).index_copy(0, plan.gen_indexes, gen)
            )
        outputs = Cosmos3Output(
            video=self.proj_out(gen[video_rows]),
            action=action_prediction,
            text_logits=self.lm_head(und) if output_text_logits else None,
            last_hidden_state=normalized,
        )
        if targets is not None:
            outputs["loss"] = self.loss(outputs, targets)
        return outputs

    def loss(self, outputs, targets):
        """用 packing 登记的模态输出切片计算 FP32 video/action loss。

        视频预测和 patch 监督 mask 一起还原到原始空间尺寸，排除 patch padding；
        动作直接保留 [A,D]。只转换已有 mask 的表示，不重算条件、有效帧或有效维度规则。
        仅存在 action target 时处理动作监督；纯视频只返回 video_loss。
        每个样本按有效元素数归一化，再对 pack 内样本平均；全条件样本贡献零但仍计数。
        本层不做 microbatch/DP 缩放，也不修改预测或切断其梯度。
        """
        has_action = "action" in targets
        predictions = {"video": [], "action": []}
        condition_masks = {"video": [], "action": []}
        for span, target_video in zip(targets["spans"], targets["video"], strict=True):
            if span.current_video_rows is None or (has_action and span.action_rows is None):
                raise ValueError("Loss requires modality output rows registered by pack_model_inputs")
            shape = tuple(target_video.shape)
            predictions["video"].append(
                unpatchify_video(outputs.video[span.current_video_rows], shape, self.config.patch_size).float()
            )
            # 一维 patch mask 广播到 patch 内的空间点，再复用同一还原/裁剪逻辑。
            # 这样空间补齐不会扩大有效元素分母，也支持后续按空间 patch 屏蔽。
            video_mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows]
            video_mask = unpatchify_video(
                video_mask[:, None].expand(-1, self.config.patch_size**2),
                (1, *shape[1:]),
                self.config.patch_size,
            ).squeeze(0)
            # 原 loss 接口使用 1=排除；当前 packing 使用 True=监督，仅在这里反转一次。
            condition_masks["video"].append((~video_mask).float())
            if has_action:
                if outputs.action is None:
                    raise ValueError("Action targets require action predictions")
                predictions["action"].append(outputs.action[span.action_rows].float())
                action_mask = targets["loss_masks"]["action_loss_mask"][span.action_rows]
                condition_masks["action"].append((~action_mask).float())

        losses = {}
        for name, weight in (("video", self.config.video_loss_weight), ("action", self.config.action_loss_weight)):
            if name not in targets:
                continue
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
