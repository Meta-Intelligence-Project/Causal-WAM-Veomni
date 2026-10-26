# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen Wan VAE encoding for already composed causal Cosmos video windows."""

import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
from torch import nn


# 预留的旧 causal 分块配置（默认不启用）；按最终拼接画布选档。
ENCODE_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 8, "768": 8}


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
    Uses Diffusers' default 4-frame chunks; resolution-based chunking is reserved.
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
                latent = self.vae.encode(pixels).latent_dist.mode()
                latent = (latent - self.latent_mean) * self.latent_inv_std
                # 预留旧 causal 分辨率分块；启用时替换上面两行（已包含归一化）。
                # latent = self._encode_pixels(pixels)
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
