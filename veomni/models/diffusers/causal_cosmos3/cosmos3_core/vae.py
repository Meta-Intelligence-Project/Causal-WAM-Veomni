# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen Wan VAE encoding for already composed causal Cosmos video windows."""

from dataclasses import dataclass, field

import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
from torch import nn

from .vae_prefix import DecisionVAEPrefixCache


# 按最终拼接画布选择时间分块；保留逐层历史 cache。
ENCODE_CHUNK_FRAMES = {"256": 32, "480": 24, "720": 8, "768": 8}


def _encode_chunk_frames(height, width):
    if (height, width) in ((1024, 1024), (1184, 880), (880, 1184), (1360, 768), (768, 1360)):
        return ENCODE_CHUNK_FRAMES["768"]
    short_side = min(height, width)
    for limit, resolution in ((256, "256"), (640, "480"), (960, "720"), (2048, "768")):
        if short_side <= limit:
            return ENCODE_CHUNK_FRAMES[resolution]
    raise ValueError(f"Unsupported VAE resolution: {(height, width)}")


@dataclass
class StreamingEncodeState:
    """One finite observation window; convolution state belongs to the caller."""

    input_layout: tuple[int, int, int, int]
    input_dtype: torch.dtype
    device: torch.device
    owner_id: int
    model_id: int
    model_dtype: torch.dtype
    frame_count: int = 1
    closed: bool = False
    feat_cache: list = field(default_factory=list, repr=False)
    latent_chunks: list = field(default_factory=list, repr=False)

    @property
    def latents(self):
        if self.closed:
            raise ValueError("The streaming encode session is closed")
        return torch.cat(self.latent_chunks, dim=2)


class Cosmos3VideoVAE(nn.Module):
    """Encode uint8 [3,T,H,W] windows into normalized FP32 [C,Tz,Hz,Wz] latents.

    Construct once per training rank, after the DataLoader workers are created.
    Uses resolution-based temporal chunks with causal caches shared within a video.
    """

    def __init__(self, weights_path, *, device, dtype=torch.bfloat16, reset_prefix_cache_mb=0):
        super().__init__()
        if type(reset_prefix_cache_mb) is not int or reset_prefix_cache_mb < 0:
            raise ValueError("reset_prefix_cache_mb must be a non-negative integer")
        self.vae = AutoencoderKLWan.from_pretrained(weights_path, torch_dtype=dtype).to(device)
        self.vae.eval().requires_grad_(False)
        shape = (1, self.vae.config.z_dim, 1, 1, 1)
        self.register_buffer(
            "latent_mean",
            torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(shape),
            persistent=False,
        )
        self.reset_prefix_cache = (
            DecisionVAEPrefixCache(self, max_entries=8, max_bytes=reset_prefix_cache_mb * 1024**2)
            if reset_prefix_cache_mb
            else None
        )
        # 与原 causal 一致：在 VAE 精度下计算倒数并相乘，不能换成 FP32 除法。
        self.register_buffer(
            "latent_inv_std",
            1 / torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(shape),
            persistent=False,
        )

    @property
    def temporal_compression_factor(self):
        return self.vae.config.scale_factor_temporal

    def _apply(self, fn, recurse=True):
        if getattr(self, "reset_prefix_cache", None) is not None:
            self.reset_prefix_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _validate_stream_frames(self, frames):
        if frames.ndim != 5 or frames.shape[1] != 3 or min(frames.shape) < 1 or not frames.is_floating_point():
            raise ValueError("Streaming encode expects normalized floating RGB [B,3,T,H,W]")
        spatial = self.vae.config.scale_factor_spatial
        if frames.shape[3] % spatial or frames.shape[4] % spatial:
            raise ValueError("Streaming encode spatial dimensions must be VAE aligned")
        if frames.device != self.latent_mean.device or self.vae.training:
            raise ValueError("Streaming encode requires an eval-mode VAE on the input device")

    def _validate_stream_state(self, state):
        if not isinstance(state, StreamingEncodeState) or state.owner_id != id(self):
            raise ValueError("The streaming encode session belongs to another VAE")
        if state.closed:
            raise ValueError("The streaming encode session is closed")
        if state.model_id != id(self.vae) or state.model_dtype != self.vae.dtype:
            raise ValueError("The VAE model or dtype changed during streaming encode")

    def _stream_encode_chunk(self, state, frames):
        pixels = frames.to(dtype=self.vae.dtype)
        if self.vae.config.patch_size is not None:
            pixels = patchify(pixels, patch_size=self.vae.config.patch_size)
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            features = self.vae.encoder(pixels.contiguous(), feat_cache=state.feat_cache, feat_idx=[0])
            mean = self.vae.quant_conv(features).chunk(2, dim=1)[0]
            latent = ((mean - self.latent_mean) * self.latent_inv_std).to(state.input_dtype)
        state.latent_chunks.append(latent)
        return latent

    @torch.no_grad()
    def stream_encode_start(self, first_frame):
        """Prime a window with one normalized [B,3,1,H,W] real observation."""
        self._validate_stream_frames(first_frame)
        if first_frame.shape[2] != 1:
            raise ValueError("Streaming encode must start with exactly one frame")
        state = StreamingEncodeState(
            input_layout=(first_frame.shape[0], 3, *first_frame.shape[3:]),
            input_dtype=first_frame.dtype,
            device=first_frame.device,
            owner_id=id(self),
            model_id=id(self.vae),
            model_dtype=self.vae.dtype,
            feat_cache=[None] * self.vae._cached_conv_counts["encoder"],
        )
        try:
            self._stream_encode_chunk(state, first_frame)
        except Exception:
            self.stream_encode_abort(state)
            raise
        return state

    @torch.no_grad()
    def stream_encode_step(self, state, new_frames):
        """Append complete temporal groups; return only the new normalized latents."""
        self._validate_stream_state(state)
        self._validate_stream_frames(new_frames)
        layout = (new_frames.shape[0], 3, *new_frames.shape[3:])
        if layout != state.input_layout or new_frames.dtype != state.input_dtype or new_frames.device != state.device:
            raise ValueError("Streaming input layout, dtype and device must remain fixed")
        group = self.temporal_compression_factor
        if new_frames.shape[2] % group:
            raise ValueError("Streaming steps require complete temporal groups of real frames")
        chunks = []
        try:
            for start in range(0, new_frames.shape[2], group):
                chunks.append(self._stream_encode_chunk(state, new_frames[:, :, start : start + group]))
                state.frame_count += group
        except Exception:
            self.stream_encode_abort(state)
            raise
        return torch.cat(chunks, dim=2)

    @torch.no_grad()
    def stream_encode_finish(self, state):
        """Return the complete prefix and release this window's convolution caches."""
        self._validate_stream_state(state)
        try:
            return state.latents
        finally:
            self.stream_encode_abort(state)

    def stream_encode_abort(self, state):
        if not isinstance(state, StreamingEncodeState) or state.owner_id != id(self):
            raise ValueError("The streaming encode session belongs to another VAE")
        state.feat_cache.clear()
        state.latent_chunks.clear()
        state.closed = True

    def stream_encode_reset(self, state, first_frame):
        self.stream_encode_abort(state)
        return self.stream_encode_start(first_frame)

    def _encode_pixels(self, pixels):
        """首帧独立编码，后续按分辨率分块；每层历史特征在块间连续传递。"""
        frames, height, width = pixels.shape[2:]
        chunk_frames = _encode_chunk_frames(height, width)
        if self.vae.config.patch_size is not None:
            pixels = patchify(pixels, patch_size=self.vae.config.patch_size)
        self.vae.clear_cache()
        latents = []
        start = 0
        while start < frames:
            end = 1 if start == 0 else min(start + chunk_frames, frames)
            features = self.vae.encoder(
                pixels[:, :, start:end].contiguous(), feat_cache=self.vae._enc_feat_map, feat_idx=[0]
            )
            # 与旧 causal 一样逐块投影、归一化，只保留后验均值（等价于 mode）。
            mean = self.vae.quant_conv(features).chunk(2, dim=1)[0]
            latents.append((mean - self.latent_mean) * self.latent_inv_std)
            del features, mean
            start = end
        # 末块直接编码，不为凑满时间块额外 padding；数据管线已经保证 4n+1 帧。
        return torch.cat(latents, dim=2)

    @torch.no_grad()
    def forward(self, video):
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("Expected an already resized uint8 video [3,T,H,W]")
        _, frames, height, width = video.shape
        temporal, spatial = self.vae.config.scale_factor_temporal, self.vae.config.scale_factor_spatial
        if frames < 1 or (frames - 1) % temporal or height % spatial or width % spatial:
            raise ValueError("Video must have 1+k*temporal frames and VAE-aligned spatial dimensions")
        self.vae.eval()
        # 先 FP32 像素归一化，再转 VAE 精度；不做二次 resize，不改变原有 padding。
        pixels = video.to(device=self.latent_mean.device, dtype=torch.float32).unsqueeze(0)
        pixels = pixels.div_(127.5).sub_(1.0).to(self.vae.dtype)
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            try:
                # 默认使用 Diffusers 的首帧 + 4 帧分块。
                # latent = self.vae.encode(pixels).latent_dist.mode()
                # latent = (latent - self.latent_mean) * self.latent_inv_std
                # 预留旧 causal 分辨率分块；启用时替换上面两行（已包含归一化）。
                latent = self._encode_pixels(pixels)
            finally:
                # 成功和异常时都释放本样本的卷积状态，不跨视频保留。
                self.vae.clear_cache()
        return latent.squeeze(0).float()

    def encode_pack(self, samples):
        """Encode each variable-size sample separately; retain metadata, replace pixels."""
        return [
            {
                **{key: value for key, value in sample.items() if key != "video"},
                "video_latents": (
                    self.reset_prefix_cache.encode(sample)
                    if self.reset_prefix_cache is not None
                    else self.encode_decision_window(sample["video"])
                )
                if sample.get("layout_mode") == "decision_window"
                else self(sample["video"]),
            }
            for sample in samples
        ]

    def _decision_pixels(self, video):
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("Expected an already resized uint8 video [3,T,H,W]")
        group = self.temporal_compression_factor
        if video.shape[1] < 1 or (video.shape[1] - 1) % group:
            raise ValueError("Decision video must contain one first frame and complete VAE groups")
        self.vae.eval()
        return video.to(device=self.latent_mean.device, dtype=torch.float32).unsqueeze(0) / 127.5 - 1

    @torch.no_grad()
    def encode_decision_window(self, video):
        """Encode an independent window with original first-frame/group kernels."""
        pixels = self._decision_pixels(video)
        group = self.temporal_compression_factor
        state = self.stream_encode_start(pixels[:, :, :1])
        try:
            for start in range(1, video.shape[1], group):
                self.stream_encode_step(state, pixels[:, :, start : start + group])
            return self.stream_encode_finish(state).squeeze(0)
        finally:
            self.stream_encode_abort(state)

    @torch.no_grad()
    def decode(self, latents):
        """Invert latent normalization and return RGB uint8 [C,T,H,W] pixels."""
        if latents.ndim != 4 or latents.shape[0] != self.vae.config.z_dim:
            raise ValueError("Expected normalized [C,T,H,W] latents")
        self.vae.eval()
        value = latents.to(device=self.latent_mean.device, dtype=self.vae.dtype).unsqueeze(0)
        value = value / self.latent_inv_std + self.latent_mean
        with torch.autocast(device_type=value.device.type, enabled=False):
            try:
                decoded = self.vae.decode(value).sample
                if not torch.isfinite(decoded).all():
                    raise ValueError("VAE produced nonfinite pixels")
                return ((decoded.squeeze(0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
            finally:
                self.vae.clear_cache()
