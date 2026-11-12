"""Noncausal packed inference attention with explicit cumulative group ends."""

from ....platform import NpuKernelRequirement
from ....registry import register_op
from . import eager, torch_npu_fia_v2


register_op(
    "attention",
    "tnd_inference",
    "eager",
    wrapper=eager.wrapper,
    description="No-grad grouped SDPA reference for packed TND inference",
)
register_op(
    "attention",
    "tnd_inference",
    "torch_npu_fia_v2",
    wrapper=torch_npu_fia_v2.wrapper,
    requirement=NpuKernelRequirement(),
    requires=("torch_npu",),
    description="Maskless BF16/FP16 D=128 TND inference using npu_fused_infer_attention_score_v2",
)
