# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen Wan VAE encoding for already composed causal Cosmos video windows."""

import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
from torch import nn


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


class Cosmos3VideoVAE(nn.Module):
    """Encode uint8 [3,T,H,W] windows into normalized FP32 [C,Tz,Hz,Wz] latents.

    Construct once per training rank, after the DataLoader workers are created.
    Uses resolution-based temporal chunks with causal caches shared within a video.
    """

    def __init__(self, weights_path, *, device, dtype=torch.bfloat16):
        super().__init__()
        self.vae = AutoencoderKLWan.from_pretrained(weights_path, torch_dtype=dtype).to(device)
        self.vae.eval().requires_grad_(False)
        shape = (1, self.vae.config.z_dim, 1, 1, 1)
        self.register_buffer(
            "latent_mean",
            torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(shape),
            persistent=False,
        )
        # 与原 causal 一致：在 VAE 精度下计算倒数并相乘，不能换成 FP32 除法。
        self.register_buffer(
            "latent_inv_std",
            1 / torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(shape),
            persistent=False,
        )

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
            {**{key: value for key, value in sample.items() if key != "video"}, "video_latents": self(sample["video"])}
            for sample in samples
        ]

    def stream_encoder(self):
        """Create an isolated, resettable encoder for a real-observation session."""
        return StreamingVideoEncoder(self)

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


class StreamingVideoEncoder:
    """Own causal convolution state independently of VAE decode/other videos.

    Encode the first RGB frame alone, then multiples of the temporal factor.
    Only actual observation pixels belong here; future placeholders and
    generated video must never be passed to this encoder.
    """

    def __init__(self, owner):
        self.owner = owner
        self.reset()

    def reset(self):
        self.features = [None] * self.owner.vae._cached_conv_counts["encoder"]
        self.shape = None

    @torch.no_grad()
    def encode(self, video):
        vae = self.owner.vae
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("Expected resized uint8 streaming video [3,T,H,W]")
        _, frames, height, width = video.shape
        temporal, spatial = vae.config.scale_factor_temporal, vae.config.scale_factor_spatial
        if height % spatial or width % spatial or frames < 1:
            raise ValueError("Streaming video must be spatially aligned and nonempty")
        if self.shape is None:
            if frames != 1:
                raise ValueError("A stream must begin with exactly one initial RGB frame")
        elif self.shape != (height, width) or frames % temporal:
            raise ValueError("Stream requires unchanged geometry and complete temporal chunks")
        pixels = video.to(device=self.owner.latent_mean.device, dtype=torch.float32).unsqueeze(0)
        pixels = pixels.div_(127.5).sub_(1.0).to(vae.dtype)
        if vae.config.patch_size is not None:
            pixels = patchify(pixels, patch_size=vae.config.patch_size)
        latents = []
        try:
            with torch.autocast(device_type=pixels.device.type, enabled=False):
                # Four-frame calls preserve the native Wan causal encoder schedule.
                for start in range(0, frames, temporal):
                    features = vae.encoder(
                        pixels[:, :, start : start + temporal].contiguous(), feat_cache=self.features, feat_idx=[0]
                    )
                    mean = vae.quant_conv(features).chunk(2, dim=1)[0]
                    latents.append((mean - self.owner.latent_mean) * self.owner.latent_inv_std)
                result = torch.cat(latents, dim=2).squeeze(0).float()
                if not torch.isfinite(result).all():
                    raise ValueError("VAE produced nonfinite streaming latents")
        except Exception:
            self.reset()
            raise
        self.shape = (height, width)
        return result
