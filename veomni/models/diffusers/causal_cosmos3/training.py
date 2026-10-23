# SPDX-License-Identifier: OpenMDW-1.1
"""Blockwise timestep sampling, restorable RNG and FP32 noisy inputs/targets."""

import math

import torch
from torch import nn

from .flow_matching import (
    TrainTimeSampler,
    build_teacher_forcing_frame_block_ids,
)
from .packing import pack_model_inputs


class CausalCosmos3ConditionModel(nn.Module):
    """Prepare noisy inputs and targets; expose restorable time/noise RNG state."""

    def __init__(
        self,
        *,
        device,
        seed=42,
        train_time_distribution="waver",
        shift=None,
        dtype=torch.bfloat16,
        patch_size=2,
        base_fps=24.0,
        temporal_compression=4,
        temporal_modality_margin=15000,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.time_generator = torch.Generator(device="cpu").manual_seed(seed)
        # 沿用旧种子偏移和属性名；噪声抽样不消耗时间或全局 RNG。
        self.generator = torch.Generator(device=self.device).manual_seed(seed + 32768)
        self.time_sampler = TrainTimeSampler(train_time_distribution)
        self.shift = {"256": 3, "480": 5} if shift is None else shift
        self.dtype, self.patch_size = dtype, patch_size
        self.position_options = dict(
            base_fps=base_fps,
            temporal_compression=temporal_compression,
            temporal_modality_margin=temporal_modality_margin,
        )

    def rng_state_dict(self):
        """时间与噪声状态一起保存；由官方 runtime 写入 checkpoint extra state。"""
        return {"time": self.time_generator.get_state(), "noise": self.generator.get_state()}

    def load_rng_state_dict(self, state):
        """恢复两路状态，兼容 checkpoint 加载时搬到计算设备的状态张量。"""
        self.time_generator.set_state(state["time"].cpu())
        self.generator.set_state(state["noise"].cpu())

    @torch.no_grad()
    def sample_sigmas(self, samples):
        """返回逐样本 FP32 [T] sigma，范围为 0..1；只读取形状和 CPU 元数据。

        输入样本提供 video_latents [C,T,H,W]、block_size，以及使用分辨率
        shift 时的 resolution。首 latent 单独抽样，之后每个 block 抽样一次。
        本层保留条件位置的原始 sigma；后续乘 1000 传给 pack_model_inputs，
        由组装层统一完成条件时间置零和 action_frame_ids 映射。
        """
        if not samples:
            raise ValueError("A nonempty pack is required")
        frame_blocks = [
            build_teacher_forcing_frame_block_ids(s["video_latents"].shape[1], s["block_size"]) for s in samples
        ]
        counts = [1 + math.ceil((s["video_latents"].shape[1] - 1) / s["block_size"]) for s in samples]
        shifts = torch.tensor(
            [self.shift[str(s["resolution"])] if isinstance(self.shift, dict) else self.shift for s in samples],
            dtype=torch.float32,
            device="cpu",
        )
        block_shifts = torch.repeat_interleave(shifts, torch.tensor(counts, dtype=torch.long, device="cpu"))
        values = self.time_sampler(
            sum(counts),
            device=self.device,
            dtype=torch.float32,
            generator=self.time_generator,
            shifts=block_shifts,
        )
        result, offset = [], 0
        for indexes, count in zip(frame_blocks, counts, strict=True):
            result.append(values[offset : offset + count].index_select(0, indexes.to(self.device)))
            offset += count
        return result

    @torch.no_grad()
    def prepare(self, samples, *, sigmas=None, epsilon_video=None, epsilon_action=None):
        """返回 ``(model_inputs, targets)``，消费已完成 VAE 编码的一个 pack。

        可传入逐样本 FP32 sigma [T] 或 [1]、固定 FP32 噪声以独立对照；
        未提供时分别使用时间和噪声 Generator。噪声形状须与 clean 数据相同。
        插值和 target 保持 FP32，只有组装完成的 video/action/state 输入转训练 dtype。
        targets 保存逐样本原始形状的 video/action target、原始扩散时间列表，
        以及 packing 返回的 spans 和 bool loss_masks，供模型的 loss() 消费。
        """
        if not samples:
            raise ValueError("A nonempty pack is required")
        for name, values in (("sigmas", sigmas), ("epsilon_video", epsilon_video), ("epsilon_action", epsilon_action)):
            if values is not None and len(values) != len(samples):
                raise ValueError(f"One {name} tensor per sample is required")
        sigmas = self.sample_sigmas(samples) if sigmas is None else sigmas
        videos = [s["video_latents"].to(device=self.device, dtype=torch.float32) for s in samples]
        actions = [s["action"].to(device=self.device, dtype=torch.float32) for s in samples]
        # 保留旧 RNG 消耗顺序：整包 video 先抽，再抽整包 action。
        # 仅当整包 action 都是条件时跳过 action 抽样；混合包中的 FD 样本仍占一次。
        if epsilon_video is None:
            epsilon_video = [
                torch.randn(v.shape, dtype=torch.float32, device=self.device, generator=self.generator) for v in videos
            ]
        all_actions_conditioned = all(len(s["condition_frame_indexes_action"]) == len(s["action"]) for s in samples)
        if epsilon_action is None:
            epsilon_action = [
                torch.zeros_like(a)
                if all_actions_conditioned
                else torch.randn(a.shape, dtype=torch.float32, device=self.device, generator=self.generator)
                for a in actions
            ]

        current_videos, current_actions, targets_video, targets_action = [], [], [], []
        video_times, action_times, prepared_samples = [], [], []
        for sample, video, action, ev, ea, sigma in zip(
            samples, videos, actions, epsilon_video, epsilon_action, sigmas, strict=True
        ):
            if sigma.dtype != torch.float32 or sigma.ndim != 1 or sigma.numel() not in (1, video.shape[1]):
                raise ValueError("sigmas must be float32 [video_latents] or [1] tensors")
            for name, noise, clean in (("epsilon_video", ev, video), ("epsilon_action", ea, action)):
                if noise.dtype != torch.float32 or noise.shape != clean.shape:
                    raise ValueError(f"{name} must be float32 and match the clean payload shape")
            sigma = sigma.to(device=self.device).expand(video.shape[1])
            action_frames = sample["action_frame_ids"].to(device=self.device, dtype=torch.long, non_blocking=True)
            action_sigma = sigma.index_select(0, action_frames)
            ev, ea = ev.to(device=self.device), ea.to(device=self.device)
            # 与旧 RectifiedFlow.get_interpolation 相同的运算次序；条件值统一交给 packing 恢复。
            sv, sa = sigma[None, :, None, None], action_sigma[:, None]
            current_videos.append(ev * sv + video * (1 - sv))
            current_actions.append(ea * sa + action * (1 - sa))
            targets_video.append(ev - video)
            mask = sample["action_mask"].to(device=self.device, non_blocking=True)
            # 沿用旧全 FD 分支的零 action target；混合包按真实噪声构造 target。
            targets_action.append(
                (torch.zeros_like(action) if all_actions_conditioned else ea - action).masked_fill(~mask, 0)
            )
            video_times.append(sigma * 1000)
            action_times.append(action_sigma * 1000)
            # 只替换本地副本中的 clean 数据，CPU 索引元数据和调用方样本保持不变。
            prepared_samples.append({**sample, "video_latents": video, "action": action})

        inputs, spans, loss_masks = pack_model_inputs(
            prepared_samples,
            current_videos,
            current_actions,
            timesteps=video_times,
            patch_size=self.patch_size,
            **self.position_options,
        )
        # packing 保留输入精度；训练入口只转换模态 payload，坐标和时间始终保持 FP32。
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            inputs[key] = inputs[key].to(dtype=self.dtype)
        # 条件还原之后统一清零无效 action 通道，避免 FD clean 值重新带入无效维度。
        inputs["action_tokens"] = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        targets = dict(
            video=targets_video,
            action=targets_action,
            video_times=video_times,
            action_times=action_times,
            spans=spans,
            loss_masks=loss_masks,
        )
        return inputs, targets
