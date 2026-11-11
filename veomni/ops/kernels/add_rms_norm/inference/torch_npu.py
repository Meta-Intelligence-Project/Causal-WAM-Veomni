"""Ascend fused addition and RMSNorm for no-grad inference."""

from torch import Tensor

from . import eager


def wrapper(x: Tensor, residual: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, Tensor]:
    """Adapt ``(y, rstd, sum)`` to ``(y, sum)`` without mutating either input.

    The vendor API does not support empty tensors. Empty inputs use the explicit
    eager reference; other unsupported inputs or vendor failures remain errors.
    """
    eager.validate(x, residual, weight, eps)
    if x.numel() == 0:
        return eager.wrapper(x, residual, weight, eps=eps)
    if x.ndim > 8:
        raise ValueError("torch_npu add_rms_norm supports at most eight dimensions")

    import torch_npu

    normalized, _, summed = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon=eps)
    return normalized, summed
