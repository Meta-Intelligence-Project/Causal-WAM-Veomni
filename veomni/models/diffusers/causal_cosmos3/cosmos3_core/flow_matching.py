# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""迁自旧 causal 6066bdce 的连续时间采样、block 映射与 Flow Matching loss。"""

from __future__ import annotations

from typing import Any

import torch


class TrainTimeSampler:
    _WAVER_MODE_S = 1.29
    # 标准正态分布的 99.9% 和 0.5% 分位点，用于 ltx2 的区间拉伸。
    _LTX2_NORMAL_999_PCTILE = 3.0902
    _LTX2_NORMAL_005_PCTILE = -2.5758
    _LTX2_UNIFORM_PROB = 0.1

    def __init__(self, distribution: str = "uniform"):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        shifts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在 CPU 用独立 RNG 采样，再将 sigma 搬到指定设备。

        shifts 为逐 block 的一维系数。除 ltx2 外，采样值 t 经过
        sigma = shift * t / (1 + (shift - 1) * t)；ltx2 使用 log(shift)
        作为正态分布均值，不重复做末尾变换。训练调用方显式使用 FP32。
        """
        # 随机数在生成时就指定 FP32，避免默认 BF16 提前损失精度。
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), dtype=torch.float32, device="cpu", generator=generator).to(
                device=device, dtype=dtype
            )
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), dtype=torch.float32, device="cpu", generator=generator)).to(
                device=device, dtype=dtype
            )
        elif self.distribution == "waver":
            u = torch.rand((batch_size,), dtype=torch.float32, device="cpu", generator=generator)
            t = 1.0 - u - self._WAVER_MODE_S * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
            t = t.to(device=device, dtype=dtype)
        elif self.distribution == "ltx2":
            assert shifts is not None, "'ltx2' distribution requires per-sample shifts."
            mu = torch.log(shifts.to(device=torch.device("cpu"), dtype=torch.float32))
            std = 1.0
            eps = 1e-3

            normal_samples = (
                torch.randn((batch_size,), dtype=torch.float32, device="cpu", generator=generator) * std + mu
            )
            logitnormal_samples = torch.sigmoid(normal_samples)
            percentile_999 = torch.sigmoid(mu + self._LTX2_NORMAL_999_PCTILE * std)
            percentile_005 = torch.sigmoid(mu + self._LTX2_NORMAL_005_PCTILE * std)

            zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)
            stretched = torch.where(zero_terminal_raw >= eps, zero_terminal_raw, 2 * eps - zero_terminal_raw)
            stretched = torch.clamp(stretched, 0, 1)

            uniform = (1 - eps) * torch.rand(
                (batch_size,), dtype=torch.float32, device="cpu", generator=generator
            ) + eps
            prob = torch.rand((batch_size,), dtype=torch.float32, device="cpu", generator=generator)
            return torch.where(prob > self._LTX2_UNIFORM_PROB, stretched, uniform).to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Time distribution '{self.distribution}' is not implemented.")

        if shifts is not None:
            shifts = shifts.to(device=device, dtype=dtype)
            t = shifts * t / (1 + (shifts - 1) * t)
        return t


def build_teacher_forcing_frame_block_ids(num_frames: int, block_size: int) -> torch.LongTensor:
    """返回 CPU 采样索引：首 latent 为 0，后续按 block 分组为 1、2……。

    例如 7 个 latent、block_size=2 得到 [0,1,1,2,2,3,3]。
    这是 sigma 数组的下标，不是 CausalLayout 中以 -1 开始的可见性 block ID。
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    block_ids = torch.zeros(num_frames, dtype=torch.long, device="cpu")
    if num_frames > 1:
        block_ids[1:] = 1 + torch.div(
            torch.arange(num_frames - 1, dtype=torch.long, device="cpu"),
            block_size,
            rounding_mode="floor",
        )
    return block_ids


class TrainTimeWeight:
    """保留旧时间权重接口；当前 Rectified Flow 仅支持 uniform 权重。"""

    def __init__(self, noise_scheduler, weight: str = "uniform"):
        if weight == "reweighting":
            weight = "uniform"
        self.weight = weight
        self.noise_scheduler = noise_scheduler
        assert self.weight == "uniform", "Only uniform loss weight is supported in RF"

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")
        return wts


def compute_flow_matching_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    timesteps: torch.Tensor,
    has_valid_tokens: bool,
    rectified_flow: Any,
    tensor_kwargs_fp32: dict,
    raw_action_dim: list[torch.Tensor] | None = None,
    normalize_by_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """迁入旧 loss 计算：逐样本计算，再对样本取平均。

    调用方先将预测、target、condition_mask 和时间转换为 FP32。
    condition_mask 的 1 表示排除、0 表示监督；视频形状为 [T,1,1] 或
    [T,H,W]，动作形状为 [A,D]，均可广播到各自原始形状的平方误差。
    normalize_by_active=True 时除以有效元素数，全条件样本保留零梯度连接，
    并继续计入样本平均。返回加时间权重后的均值和未加权的逐样本 loss。
    """
    if not has_valid_tokens:
        # 保留计算图连接，跨 rank 某模态无监督时仍能产生零梯度。
        dummy_loss = 0.0 * sum(p.sum() for p in pred)
        return dummy_loss, dummy_loss.unsqueeze(0)

    per_instance_losses = []
    per_instance_weighted_losses = []
    for i in range(len(pred)):
        T_i = condition_mask[i].shape[0]
        sqerr_i = (pred[i] - target[i]) ** 2
        noisy_mask_i = 1.0 - condition_mask[i]
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            sqerr_i = sqerr_i[:, : raw_action_dim[i]]
        if normalize_by_active:
            active_count = (noisy_mask_i.sum() * (sqerr_i.numel() // noisy_mask_i.numel())).clamp(min=1)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_count)
        else:
            per_instance_losses.append((sqerr_i * noisy_mask_i).mean())

        ts_i = timesteps[i, :T_i] if timesteps.dim() > 1 else timesteps[i]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32)
        tw_i = tw_i.reshape(-1, *([1] * (condition_mask[i].ndim - 1)))
        if normalize_by_active:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).sum() / active_count)
        else:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).mean())

    per_instance_loss = torch.stack(per_instance_losses)
    per_instance_weighted_loss = torch.stack(per_instance_weighted_losses)
    return per_instance_weighted_loss.mean(), per_instance_loss
