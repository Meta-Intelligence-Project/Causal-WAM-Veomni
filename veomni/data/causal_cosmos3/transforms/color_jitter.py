"""Shared, fixed-order photometric jitter on RGB uint8 video windows."""

import math
import random
from dataclasses import dataclass

import torch
from torchvision.transforms.v2 import functional as F


@dataclass(frozen=True)
class ColorJitterConfig:
    enabled: bool = False
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("color_jitter.enabled must be a boolean")
        for name in ("brightness", "contrast", "saturation", "hue"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"color_jitter.{name} must be a finite nonnegative number")
        if self.hue > 0.5:
            raise ValueError("color_jitter.hue must be in [0, 0.5]")

    @property
    def active(self):
        return self.enabled and any((self.brightness, self.contrast, self.saturation, self.hue))

    def sample_factors(self):
        """Draw once per sample; share the factors across all frames and cameras."""
        bcs = tuple(
            random.uniform(max(0, 1 - strength), 1 + strength) if strength else 1.0
            for strength in (self.brightness, self.contrast, self.saturation)
        )
        hue = random.uniform(-self.hue, self.hue) if self.hue else 0.0
        return (*bcs, hue)


def jitter_video(video, cameras, factors, *, chunk_size=8):
    """Return a jittered copy; contrast uses each camera/frame's grayscale mean.

    Only camera rectangles are modified, preserving any outside padding. Time
    chunks bound float32 temporaries without resampling the shared factors.
    """
    if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("Color jitter expects RGB uint8 video [3,T,H,W]")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Color jitter chunk_size must be a positive integer")
    brightness, contrast, saturation, hue = factors
    if factors == (1.0, 1.0, 1.0, 0.0):
        return video
    result = video.clone()
    for camera in cameras:
        top, left, height, width = camera.top, camera.left, camera.height, camera.width
        source = video[:, :, top : top + height, left : left + width]
        target = result[:, :, top : top + height, left : left + width]
        for start in range(0, video.shape[1], chunk_size):
            pixels = source[:, start : start + chunk_size].permute(1, 0, 2, 3).float().div_(255)
            if brightness != 1:
                pixels = F.adjust_brightness(pixels, brightness)
            if contrast != 1:
                pixels = F.adjust_contrast(pixels, contrast)
            if saturation != 1:
                pixels = F.adjust_saturation(pixels, saturation)
            if hue != 0:
                pixels = F.adjust_hue(pixels, hue)
            target[:, start : start + chunk_size] = pixels.mul_(255).round_().byte().permute(1, 0, 2, 3)
    return result
