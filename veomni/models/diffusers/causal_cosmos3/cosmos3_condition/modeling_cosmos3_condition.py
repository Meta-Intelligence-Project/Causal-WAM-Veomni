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
    def sample_sigmas(self, samples, *, generator=None):
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
            generator=self.time_generator if generator is None else generator,
            shifts=block_shifts,
        )
        result, offset = [], 0
        for indexes, count in zip(frame_blocks, counts, strict=True):
            result.append(values[offset : offset + count].index_select(0, indexes.to(self.device)))
            offset += count
        return result

    @torch.no_grad()
    def prepare(self, samples, *, sigmas=None, epsilon_video=None, epsilon_action=None, action_sigmas=None):
        """返回 ``(model_inputs, targets)``，消费已完成 VAE 编码的一个 pack。

        可传入逐样本 FP32 sigma [T] 或 [1]、固定 FP32 噪声以独立对照；
        未提供时分别使用时间和噪声 Generator。噪声形状须与 clean 数据相同。
        插值和 target 保持 FP32，只有组装完成的 video/action/state 输入转训练 dtype。
        targets 保存逐样本原始形状的 video/action target、原始扩散时间列表，
        以及 packing 返回的 spans 和 bool loss_masks，供模型的 loss() 消费。
        按整包视频、可选动作、Packing 的顺序处理；stage 来自配置，默认 action。
        Video 不构造动作输入或 target，向 Packing 传 current_actions=None。
        显式将阶段传给 Packing，不根据缺失字段推断阶段。
        """
        if not samples:
            raise ValueError("A nonempty pack is required")
        if self.config.expert_layout == "triple":
            return self._prepare_triple(
                samples,
                sigmas=sigmas,
                action_sigmas=action_sigmas,
                epsilon_video=epsilon_video,
                epsilon_action=epsilon_action,
            )
        if action_sigmas is not None:
            raise ValueError("Independent action_sigmas require expert_layout='triple'")
        has_action = self.config.stage == "action"
        for sample in samples:
            mode = sample.get("mode")
            if has_action:
                if mode == "video" or "action" not in sample:
                    raise ValueError("Action stage requires action samples")
            elif mode not in (None, "video") or "action" in sample or "states" in sample:
                raise ValueError("Video stage requires video-only samples")
        if not has_action and epsilon_action is not None:
            raise ValueError("Video stage does not accept epsilon_action")
        for name, values in (("sigmas", sigmas), ("epsilon_video", epsilon_video), ("epsilon_action", epsilon_action)):
            if values is not None and len(values) != len(samples):
                raise ValueError(f"One {name} tensor per sample is required")
        sigmas = self.sample_sigmas(samples) if sigmas is None else sigmas
        prepared_samples = [
            {**s, "video_latents": s["video_latents"].to(device=self.device, dtype=torch.float32)} for s in samples
        ]
        videos = [s["video_latents"] for s in prepared_samples]
        # 整包视频处理；视频噪声先生成，随后才处理整包动作。
        if epsilon_video is None:
            epsilon_video = [
                torch.randn(v.shape, dtype=torch.float32, device=self.device, generator=self.generator) for v in videos
            ]
        current_videos, targets_video, video_times, frame_sigmas = [], [], [], []
        for video, noise, sigma in zip(videos, epsilon_video, sigmas, strict=True):
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
            frame_sigmas.append(sigma)
        targets = dict(video=targets_video, video_times=video_times)

        current_actions = None
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
            for sample, action, noise, sigma in zip(
                prepared_samples, actions, epsilon_action, frame_sigmas, strict=True
            ):
                if noise.dtype != torch.float32 or noise.shape != action.shape:
                    raise ValueError("epsilon_action must be float32 and match the clean payload shape")
                action_frames = torch.as_tensor(sample["action_frame_ids"], device=self.device, dtype=torch.long)
                action_sigma = sigma.index_select(0, action_frames)
                noise = noise.to(device=self.device)
                sa = action_sigma[:, None]
                current_actions.append(noise * sa + action * (1 - sa))
                mask = sample["action_mask"].to(device=self.device, non_blocking=True)
                targets_action.append(
                    (torch.zeros_like(action) if all_actions_conditioned else noise - action).masked_fill(~mask, 0)
                )
                action_times.append(action_sigma * 1000)
                sample["action"] = action
            targets.update(action=targets_action, action_times=action_times)

        inputs, spans, loss_masks = pack_model_inputs(
            prepared_samples,
            current_videos,
            current_actions,
            timesteps=video_times,
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

    @torch.no_grad()
    def _prepare_triple(self, samples, *, sigmas, action_sigmas, epsilon_video, epsilon_action):
        """Prepare explicit video/FDM/IDM tasks without coupling action and video time.

        IDM's future video is a fixed noisy condition at video_stop_sigma;
        actions always use the ordinary GT forward-noising/FM target. Predicted
        visual-cache mixing is performed by the model, never by altering that target.
        """
        has_action = self.config.stage == "action"
        modes = tuple(s.get("mode", "video" if not has_action else None) for s in samples)
        allowed = ("forward_dynamics", "inverse_dynamics") if has_action else ("video",)
        if any(mode not in allowed for mode in modes):
            raise ValueError(f"Triple {self.config.stage} stage requires explicit tasks from {allowed}")
        if not has_action and any("action" in s or "states" in s for s in samples):
            raise ValueError("Triple video stage requires video-only samples")
        for name, values in (
            ("sigmas", sigmas),
            ("action_sigmas", action_sigmas),
            ("epsilon_video", epsilon_video),
            ("epsilon_action", epsilon_action),
        ):
            if values is not None and len(values) != len(samples):
                raise ValueError(f"One {name} tensor per sample is required")
        if not has_action and (action_sigmas is not None or epsilon_action is not None):
            raise ValueError("Triple video stage does not accept action noise or times")
        sigmas = self.sample_sigmas(samples) if sigmas is None else sigmas
        if has_action and action_sigmas is None:
            # An independent draw, still blockwise aligned to action_frame_ids.
            action_sigmas = [
                value.index_select(0, torch.as_tensor(s["action_frame_ids"], device=self.device))
                for s, value in zip(samples, self.sample_sigmas(samples), strict=True)
            ]
        prepared, current_videos, current_actions = [], [], []
        video_times, action_times, condition_payloads, condition_times = [], [], [], []
        targets = dict(video=[], video_times=[])
        if has_action:
            targets.update(action=[], action_times=[])
        for i, (sample, mode) in enumerate(zip(samples, modes, strict=True)):
            sample = dict(sample)
            video = sample["video_latents"].to(device=self.device, dtype=torch.float32)
            sample["video_latents"] = video
            sigma = sigmas[i]
            if sigma.dtype != torch.float32 or sigma.ndim != 1 or sigma.numel() not in (1, video.shape[1]):
                raise ValueError("sigmas must be float32 [video_latents] or [1] tensors")
            sigma = sigma.to(self.device).expand(video.shape[1]).clone()
            if mode == "inverse_dynamics":
                sigma.fill_(self.config.video_stop_sigma)
                sigma[0] = 0
                sample["condition_frame_indexes_vision"] = list(range(video.shape[1]))
            noise = (
                torch.randn(video.shape, device=self.device, dtype=torch.float32, generator=self.generator)
                if epsilon_video is None
                else epsilon_video[i]
            )
            if noise.dtype != torch.float32 or noise.shape != video.shape:
                raise ValueError("epsilon_video must be float32 and match the clean payload shape")
            noise = noise.to(self.device)
            current_video = (1 - sigma[None, :, None, None]) * video + sigma[None, :, None, None] * noise
            condition_video = current_video.clone() if mode == "inverse_dynamics" else video
            if mode == "inverse_dynamics":
                condition_video[:, 0] = video[:, 0]
            current_videos.append(current_video)
            video_times.append(sigma * 1000)
            condition_payloads.append(condition_video)
            condition_times.append(sigma * 1000 if mode == "inverse_dynamics" else torch.zeros_like(sigma))
            targets["video"].append(noise - video)
            targets["video_times"].append(sigma * 1000)
            if has_action:
                action = sample["action"].to(device=self.device, dtype=torch.float32)
                sample["action"] = action
                if mode == "forward_dynamics":
                    sample["condition_frame_indexes_action"] = list(range(len(action)))
                    sigma_a = torch.zeros(len(action), device=self.device, dtype=torch.float32)
                    noise_a = torch.zeros_like(action)
                else:
                    sample["condition_frame_indexes_action"] = []
                    sigma_a = action_sigmas[i]
                    if sigma_a.dtype != torch.float32 or sigma_a.ndim != 1 or sigma_a.numel() not in (1, len(action)):
                        raise ValueError("action_sigmas must be float32 [actions] or [1] tensors")
                    sigma_a = sigma_a.to(self.device).expand(len(action))
                    noise_a = (
                        torch.randn(action.shape, device=self.device, dtype=torch.float32, generator=self.generator)
                        if epsilon_action is None
                        else epsilon_action[i]
                    )
                    if noise_a.dtype != torch.float32 or noise_a.shape != action.shape:
                        raise ValueError("epsilon_action must be float32 and match the clean payload shape")
                    noise_a = noise_a.to(self.device)
                current_actions.append((1 - sigma_a[:, None]) * action + sigma_a[:, None] * noise_a)
                action_times.append(sigma_a * 1000)
                target = noise_a - action if mode == "inverse_dynamics" else torch.zeros_like(action)
                targets["action"].append(target.masked_fill(~sample["action_mask"].to(self.device), 0))
                targets["action_times"].append(sigma_a * 1000)
            prepared.append(sample)
        inputs, spans, loss_masks = pack_model_inputs(
            prepared,
            current_videos,
            current_actions if has_action else None,
            timesteps=video_times,
            action_timesteps=action_times if has_action else None,
            video_condition_payloads=condition_payloads,
            video_condition_timesteps=condition_times,
            has_action=has_action,
            patch_size=self.patch_size,
            **self.position_options,
        )
        for key in ("video_tokens", "action_tokens", "state_tokens"):
            if key in inputs:
                inputs[key] = inputs[key].to(dtype=self.dtype)
        if has_action:
            inputs["action_tokens"] = inputs["action_tokens"].masked_fill(~inputs["action_mask"], 0)
        progress = float(getattr(self, "training_progress", 0.0))
        ramp = (progress - self.config.rollout_cache_warmup_fraction) / self.config.rollout_cache_ramp_fraction
        probability = min(1.0, max(0.0, ramp)) * self.config.rollout_cache_max_probability if has_action else 0.0
        inputs.update(
            task_modes=modes,
            triple_training=dict(
                progress=progress,
                rollout_cache_probability=probability,
                seed=int(torch.randint(2**31, (1,), generator=self.time_generator, device="cpu").item()),
                video_stop_sigma=self.config.video_stop_sigma,
                partial_video_steps=self.config.partial_video_steps,
                mix_seed=self.config.seed,
                optimizer_step=int(getattr(self, "training_step", 0)),
            ),
        )
        targets.update(spans=spans, loss_masks=loss_masks)
        return inputs, targets
