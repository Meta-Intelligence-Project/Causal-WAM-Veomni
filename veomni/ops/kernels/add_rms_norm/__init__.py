"""Residual addition and RMSNorm for explicit no-grad inference.

The ``inference`` variant returns ``(normalized, residual_sum)``. It does not
provide a training backward; consumers keep their original training path.
"""

from ...platform import NpuKernelRequirement
from ...registry import register_op
from .inference import eager, torch_npu


register_op(
    "add_rms_norm",
    "inference",
    "eager",
    wrapper=eager.wrapper,
    description="PyTorch residual addition followed by RMSNorm for no-grad inference",
)
register_op(
    "add_rms_norm",
    "inference",
    "torch_npu",
    wrapper=torch_npu.wrapper,
    description="torch_npu.npu_add_rms_norm returning normalized values and the residual sum",
    requirement=NpuKernelRequirement(),
    requires=("torch_npu",),
)
