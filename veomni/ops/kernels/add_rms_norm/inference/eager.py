"""PyTorch reference for inference residual RMSNorm."""

import math

import torch
from torch import Tensor


def validate(x: Tensor, residual: Tensor, weight: Tensor, eps: float) -> None:
    """Check the no-grad, matching-dtype last-dimension normalization contract."""
    if torch.is_grad_enabled():
        raise RuntimeError("add_rms_norm/inference requires disabled gradients")
    if x.ndim < 1 or x.shape != residual.shape or weight.shape != x.shape[-1:]:
        raise ValueError("add_rms_norm requires equal input shapes and a last-dimension weight")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("add_rms_norm supports float16, bfloat16 and float32")
    if x.dtype != residual.dtype or x.dtype != weight.dtype:
        raise ValueError("add_rms_norm inputs and weight must have the same dtype")
    if x.device != residual.device or x.device != weight.device:
        raise ValueError("add_rms_norm inputs and weight must be on the same device")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("add_rms_norm eps must be finite and nonnegative")


def wrapper(x: Tensor, residual: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, Tensor]:
    """Return normalized values and the rounded input-dtype residual sum.

    Normalization uses FP32 statistics and Llama-style casting before affine
    scaling. The vendor fusion can round differently from this two-op reference.
    """
    validate(x, residual, weight, eps)
    summed = x + residual
    values = summed.float()
    normalized = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return normalized * weight, summed
