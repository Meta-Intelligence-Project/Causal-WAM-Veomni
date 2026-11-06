# SPDX-License-Identifier: OpenMDW-1.1
"""Blockwise timestep sampling, restorable RNG and FP32 noisy inputs/targets."""

import math

import torch
from transformers import PreTrainedModel

from veomni.distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from veomni.utils.device import get_device_type

from ..cosmos3_core.flow_matching import (
    TrainTimeSampler,
    build_teacher_forcing_frame_block_ids,
)
from ..cosmos3_core.packing import pack_model_inputs
from .configuration_cosmos3_condition import CausalCosmos3ConditionConfig


class CausalCosmos3ConditionModel(PreTrainedModel):
    """Prepare inputs through DiT's condition API, with restorable RNG state.

    Construct on the final training device so the noise generator is created
    there. The VAE is loaded on the first get_condition() call in that process.
    """

    config_class = CausalCosmos3ConditionConfig

    def __init__(self, config=None, *, device=None, meta_init=False, **kwargs):
        # Retain the existing standalone keyword constructor used by validation.
        if config is None:
            if "dtype" in kwargs:
                kwargs["payload_dtype"] = kwargs.pop("dtype")
            config = CausalCosmos3ConditionConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass condition settings through config when a config is supplied")
        super().__init__(config)
        device = torch.device(device if device is not None else get_device_type())
        # No parameters exist before lazy VAE loading. Keep device/dtype explicit
        # without adding weights to the condition checkpoint or consuming RNG.
        self.register_buffer(
            "_input_spec", torch.empty(0, device=device, dtype=getattr(torch, config.payload_dtype)), persistent=False
        )
        # The official DiT runtime passes the same base seed to each rank.
        # Independent DP samples need independent draws; checkpoint restoration
        # subsequently replaces both streams with their rank-local saved state.
        dp_rank = get_parallel_state().dp_rank if is_parallel_state_initialized() else 0
        seed = config.seed + dp_rank
        self.time_generator = torch.Generator(device="cpu").manual_seed(seed)
        # 沿用旧种子偏移和属性名；噪声抽样不消耗时间或全局 RNG。
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 32768)
        self.time_sampler = TrainTimeSampler(config.train_time_distribution)
        self.shift, self.patch_size = config.shift, config.patch_size
        self.action_time_sampler = (
            self.time_sampler if config.action_time_sampler is None else TrainTimeSampler(config.action_time_sampler)
        )
        self.action_shift = config.action_shift
        self.position_options = dict(
            base_fps=config.base_fps,
            temporal_compression=config.temporal_compression,
            temporal_modality_margin=config.temporal_modality_margin,
        )
        # meta_init is the offline runtime's convention; neither mode loads VAE
        # weights until raw video encoding is actually requested.
        self.video_vae = None

    @property
    def device(self):
        return self._input_spec.device

    @property
    def dtype(self):
        return self._input_spec.dtype

    @torch.no_grad()
    def get_condition(self, samples):
        """Encode one list of raw samples; retain CPU indexes and sample metadata."""
        if not samples:
            raise ValueError("A nonempty pack is required")
        if self.video_vae is None:
            from ..cosmos3_core.vae import Cosmos3VideoVAE

            if self.config.vae_path is None:
                raise ValueError("get_condition requires condition config vae_path")
            self.video_vae = Cosmos3VideoVAE(self.config.vae_path, device=self.device, dtype=self.dtype)
            self.video_vae.eval().requires_grad_(False)
        return {"samples": self.video_vae.encode_pack(samples)}

    @torch.no_grad()
    def process_condition(self, samples):
        """Return model kwargs, including targets, for DiTTrainer's forward call."""
        inputs, targets = self.prepare(samples)
        return {**inputs, "targets": targets, "attention_backend": self.config.attention_backend}

    def rng_state_dict(self):
        """时间与噪声状态一起保存；由官方 runtime 写入 checkpoint extra state。"""
        return {"time": self.time_generator.get_state(), "noise": self.generator.get_state()}

    def load_rng_state_dict(self, state):
        """恢复两路状态，兼容 checkpoint 加载时搬到计算设备的状态张量。"""
        self.time_generator.set_state(state["time"].cpu())
        self.generator.set_state(state["noise"].cpu())

    @torch.no_grad()
    def sample_sigmas(self, samples, *, for_action=False, generator=None):
        """返回逐样本 FP32 [T] sigma，范围为 0..1；只读取形状和 CPU 元数据。

        输入样本提供 video_latents [C,T,H,W]、block_size，以及使用分辨率
        shift 时的 resolution。首 latent 单独抽样，之后每个 block 抽样一次。
        for_action=True 选择 ID action 的分布和 shift，沿用同一 block 映射与 RNG。
        本层保留条件位置的原始 sigma；后续乘 1000 传给 pack_model_inputs，
        由组装层统一完成条件时间置零和 action_frame_ids 映射。
        """
        if not samples:
            raise ValueError("A nonempty pack is required")
        sampler = self.action_time_sampler if for_action else self.time_sampler
        shift = self.action_shift if for_action else self.shift
        frame_blocks = [
            build_teacher_forcing_frame_block_ids(s["video_latents"].shape[1], s["block_size"]) for s in samples
        ]
        counts = [1 + math.ceil((s["video_latents"].shape[1] - 1) / s["block_size"]) for s in samples]
        shifts = torch.tensor(
            [shift[str(s["resolution"])] if isinstance(shift, dict) else shift for s in samples],
            dtype=torch.float32,
            device="cpu",
        )
        block_shifts = torch.repeat_interleave(shifts, torch.tensor(counts, dtype=torch.long, device="cpu"))
        values = sampler(
            sum(counts),
            device=self.device,
            dtype=torch.float32,
            generator=self.time_generator if generator is None else generator,
            shifts=block_shifts,
        )
        result, offset = [], 0
        for indexes, count in zip(frame_blocks, counts, strict=True):
            result.append(values[offset : offset + count].index_select(0, indexes.to(self.device)))
            offset += count
        return result

    @torch.no_grad()
    def prepare(
        self,
        samples,
        *,
        sigmas=None,
        action_sigmas=None,
        epsilon_video=None,
        epsilon_action=None,
        augment_video_condition=True,
    ):
        """返回 ``(model_inputs, targets)``，消费已完成 VAE 编码的一个 pack。

        sigmas/action_sigmas 分别提供逐样本 FP32 视频/action 时间，均为视频帧网格 [T] 或 [1]；
        action 按 action_frame_ids 映射到动作步。ID 未提供 action_sigmas 时独立按 block 抽样，
        使用 action 分布（默认继承视频）和 shift（默认 5.0）；其他任务默认沿用视频时间。
        验证需显式传入两组时间以避免训练 RNG 抽样。
        可传入固定 FP32 噪声以独立对照；
        未提供时分别使用时间和噪声 Generator。噪声形状须与 clean 数据相同。
        插值和 target 保持 FP32，只有组装完成的 video/action/state 输入转训练 dtype。
        targets 保存逐样本原始形状的 video/action target、原始扩散时间列表和 action 采样 shift，
        以及 packing 返回的 spans 和 bool loss_masks，供模型的 loss() 消费。
        按整包视频、可选动作、Packing 的顺序处理；stage 来自配置，默认 action。
        ID 条件视频增强在同一样本的视频目标构造之后执行，仅改变第一份视频及其时间标签。
        验证通过 augment_video_condition=False 保持 clean 条件，不消耗增强 RNG。
        Video 不构造动作输入或 target，向 Packing 传 current_actions=None。
        显式将阶段传给 Packing，不根据缺失字段推断阶段。
        """
        if not samples:
            raise ValueError("A nonempty pack is required")
        has_action = self.config.stage == "action"
        for sample in samples:
            mode = sample.get("mode")
            if has_action:
                if mode == "video" or "action" not in sample:
                    raise ValueError("Action stage requires action samples")
            elif mode not in (None, "video") or "action" in sample or "states" in sample:
                raise ValueError("Video stage requires video-only samples")
        if not has_action and (epsilon_action is not None or action_sigmas is not None):
            raise ValueError("Video stage does not accept epsilon_action or action_sigmas")
        for name, values in (
            ("sigmas", sigmas),
            ("action_sigmas", action_sigmas),
            ("epsilon_video", epsilon_video),
            ("epsilon_action", epsilon_action),
        ):
            if values is not None and len(values) != len(samples):
                raise ValueError(f"One {name} tensor per sample is required")
        sigmas = self.sample_sigmas(samples) if sigmas is None else sigmas
        prepared_samples = [
            {**s, "video_latents": s["video_latents"].to(device=self.device, dtype=torch.float32)} for s in samples
        ]
        if has_action and action_sigmas is None:
            action_sigmas = list(sigmas)
            id_indexes = [i for i, s in enumerate(samples) if s.get("mode") == "inverse_dynamics"]
            if id_indexes:
                independent_sigmas = self.sample_sigmas([samples[i] for i in id_indexes], for_action=True)
                for i, sigma in zip(id_indexes, independent_sigmas, strict=True):
                    action_sigmas[i] = sigma
        videos = [s["video_latents"] for s in prepared_samples]
        # 整包视频处理；视频噪声先生成，随后才处理整包动作。
        if epsilon_video is None:
            epsilon_video = [
                torch.randn(v.shape, dtype=torch.float32, device=self.device, generator=self.generator) for v in videos
            ]
        current_videos, targets_video, video_times = [], [], []
        condition_videos, condition_video_timesteps = [], []
        for sample, video, noise, sigma in zip(prepared_samples, videos, epsilon_video, sigmas, strict=True):
            if sigma.dtype != torch.float32 or sigma.ndim != 1 or sigma.numel() not in (1, video.shape[1]):
                raise ValueError("sigmas must be float32 [video_latents] or [1] tensors")
            if noise.dtype != torch.float32 or noise.shape != video.shape:
                raise ValueError("epsilon_video must be float32 and match the clean payload shape")
            sigma = sigma.to(device=self.device).expand(video.shape[1])
            noise = noise.to(device=self.device)
            sv = sigma[None, :, None, None]
            current_videos.append(noise * sv + video * (1 - sv))
            targets_video.append(noise - video)
            video_times.append(sigma * 1000)

            # 先用原始 clean 数据构造视频目标，再独立增强当前样本的条件视频。
            condition_video = video
            condition_time = torch.zeros_like(sigma)
            if (
                augment_video_condition
                and self.config.cond_noise_prob > 0
                and sample.get("mode") == "inverse_dynamics"
                and torch.rand((), dtype=torch.float32, device="cpu", generator=self.time_generator).item()
                < self.config.cond_noise_prob
            ):
                (condition_sigma,) = self.sample_sigmas([sample])
                # 首 latent 保持 clean；其余每个 block 独立采样，block 内共享 sigma。
                condition_sigma[0] = 0
                condition_noise = torch.randn(
                    video.shape, dtype=torch.float32, device=self.device, generator=self.generator
                )
                sv_cond = condition_sigma[None, :, None, None]
                condition_video = video * (1 - sv_cond) + condition_noise * sv_cond
                condition_time = condition_sigma * 1000
            condition_videos.append(condition_video)
            condition_video_timesteps.append(condition_time)
        targets = dict(video=targets_video, video_times=video_times)

        current_actions = None
        action_times = None
        # 整包动作处理；Video 不读取动作字段、不消耗动作噪声 RNG。
        if has_action:
            actions = [s["action"].to(device=self.device, dtype=torch.float32) for s in samples]
            all_actions_conditioned = all(
                len(s["condition_frame_indexes_action"]) == len(s["action"]) for s in samples
            )
            if epsilon_action is None:
                epsilon_action = [
                    torch.zeros_like(a)
                    if all_actions_conditioned
                    else torch.randn(a.shape, dtype=torch.float32, device=self.device, generator=self.generator)
                    for a in actions
                ]
            current_actions, targets_action, action_times = [], [], []
            action_shifts = []
            for sample, action, noise, sigma in zip(
                prepared_samples, actions, epsilon_action, action_sigmas, strict=True
            ):
                frames = sample["video_latents"].shape[1]
                if sigma.dtype != torch.float32 or sigma.ndim != 1 or sigma.numel() not in (1, frames):
                    raise ValueError("action_sigmas must be float32 [video_latents] or [1] tensors")
                if noise.dtype != torch.float32 or noise.shape != action.shape:
                    raise ValueError("epsilon_action must be float32 and match the clean payload shape")
                action_frames = torch.as_tensor(sample["action_frame_ids"], device=self.device, dtype=torch.long)
                action_sigma = sigma.to(device=self.device).expand(frames).index_select(0, action_frames)
                noise = noise.to(device=self.device)
                sa = action_sigma[:, None]
                current_actions.append(noise * sa + action * (1 - sa))
                mask = sample["action_mask"].to(device=self.device, non_blocking=True)
                targets_action.append(
                    (torch.zeros_like(action) if all_actions_conditioned else noise - action).masked_fill(~mask, 0)
                )
                action_times.append(action_sigma * 1000)
                # 记录 action 实际采用的采样 shift，供 loss 归一化；不在这里计算权重。
                shift = self.action_shift if sample["mode"] == "inverse_dynamics" else self.shift
                action_shifts.append(shift[str(sample["resolution"])] if isinstance(shift, dict) else shift)
                sample["action"] = action
            targets.update(action=targets_action, action_times=action_times, action_shifts=action_shifts)

        inputs, spans, loss_masks = pack_model_inputs(
            prepared_samples,
            current_videos,
            current_actions,
            timesteps=video_times,
            action_timesteps=action_times,
            condition_videos=condition_videos,
            condition_video_timesteps=condition_video_timesteps,
            has_action=has_action,
            patch_size=self.patch_size,
            **self.position_options,
        )
        # packing 保留精度；仅转换 payload，坐标、时间与 target 保持 FP32。
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=self.dtype)
        # 条件还原之后统一清零无效 action 通道。
        if "action_tokens" in inputs:
            inputs["action_tokens"] = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        targets.update(spans=spans, loss_masks=loss_masks)
        return inputs, targets
