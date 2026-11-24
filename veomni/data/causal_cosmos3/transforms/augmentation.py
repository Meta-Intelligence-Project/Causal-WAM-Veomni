"""Sample-wide color jitter on already composed uint8 videos."""

import math
from collections.abc import Mapping

import torch


class VideoColorJitter:
    """Share one ColorJitter draw across all frames of a concatenated video."""

    def __init__(self, *, p=1.0, brightness=0, contrast=0, saturation=0, hue=0):
        from torchvision.transforms.v2 import ColorJitter

        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("color_jitter.p must be a finite number in [0, 1]")
        self.p = p
        self.transform = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        for name in ("brightness", "contrast", "saturation", "hue"):
            bounds = getattr(self.transform, name)
            if bounds is not None and not all(math.isfinite(value) for value in bounds):
                raise ValueError(f"color_jitter.{name} must have finite bounds")
        self.active = any(
            getattr(self.transform, name) is not None for name in ("brightness", "contrast", "saturation", "hue")
        )

    def __call__(self, video):
        if self.p == 0 or not self.active:
            return video
        if self.p < 1 and torch.rand((), device="cpu").item() >= self.p:
            return video
        if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("Color jitter expects uint8 video [3,T,H,W]")
        # ColorJitter samples factors/order once per call; T is a leading batch dimension.
        return self.transform(video.permute(1, 0, 2, 3)).permute(1, 0, 2, 3).contiguous()


def build_video_augmentation(config, *, training=False):
    """Only training factories may construct the configured video augmentation."""
    if not training:
        return None
    options = config.get("augmentation", {})
    if not isinstance(options, Mapping) or set(options) - {"color_jitter"}:
        raise ValueError("augmentation must be a mapping containing only color_jitter")
    jitter = options.get("color_jitter", {})
    if not isinstance(jitter, Mapping):
        raise ValueError("augmentation.color_jitter must be a mapping")
    jitter = dict(jitter)
    enabled = jitter.pop("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("color_jitter.enabled must be a boolean")
    if not enabled:
        return None
    return VideoColorJitter(**jitter)
