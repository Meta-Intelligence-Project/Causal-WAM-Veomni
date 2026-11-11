"""Residual RMSNorm contracts, vendor return ordering and native Ascend numerics."""

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp
from veomni.ops.kernels.add_rms_norm.inference import torch_npu as adapter
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 17])
def test_eager_residual_norm_matches_fp64_oracle(dtype, rows):
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(rows, 32, generator=generator).to(dtype)
    residual = torch.randn(rows, 32, generator=generator).to(dtype)
    weight = torch.randn(32, generator=generator).to(dtype)
    before = (x.clone(), residual.clone(), weight.clone())
    with torch.no_grad():
        actual, summed = VeomniOp("add_rms_norm", "inference", "eager")(x, residual, weight, eps=1e-5)
    rounded = (x + residual).double()
    expected = (rounded / (rounded.square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(dtype) * weight
    torch.testing.assert_close(actual, expected, atol=1e-6 if dtype == torch.float32 else 0.02, rtol=1e-5)
    torch.testing.assert_close(summed, x + residual, atol=0, rtol=0)
    for tensor, saved in zip((x, residual, weight), before, strict=True):
        torch.testing.assert_close(tensor, saved, atol=0, rtol=0)


def test_vendor_adapter_preserves_epsilon_return_order_and_errors(monkeypatch):
    x, residual, weight = torch.ones(3, 16), torch.full((3, 16), 2.0), torch.ones(16)
    normalized, summed = torch.full_like(x, 7), torch.full_like(x, 3)
    calls = []

    def fused(a, b, gamma, *, epsilon):
        calls.append((a, b, gamma, epsilon))
        return normalized, torch.zeros(3, 1), summed

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_add_rms_norm=fused))
    with torch.no_grad():
        output = adapter.wrapper(x, residual, weight, eps=0.0003)
    assert output[0] is normalized and output[1] is summed
    assert len(calls) == 1 and calls[0][3] == 0.0003
    with torch.no_grad():
        empty, empty_sum = adapter.wrapper(x[:0], residual[:0], weight, eps=1e-5)
    assert empty.shape == empty_sum.shape == (0, 16) and len(calls) == 1

    def fail(*args, **kwargs):
        raise RuntimeError("vendor failure")

    monkeypatch.setattr(sys.modules["torch_npu"], "npu_add_rms_norm", fail)
    with torch.no_grad(), pytest.raises(RuntimeError, match="vendor failure"):
        adapter.wrapper(x, residual, weight, eps=1e-5)


def test_residual_norm_rejects_gradients_and_invalid_inputs():
    op = VeomniOp("add_rms_norm", "inference", "eager")
    x, residual, weight = torch.ones(2, 16), torch.ones(2, 16), torch.ones(16)
    with pytest.raises(RuntimeError, match="disabled gradients"):
        op(x, residual, weight, eps=1e-5)
    with torch.no_grad():
        for a, b, gamma, eps in (
            (x, residual[:1], weight, 1e-5),
            (x, residual, weight[:8], 1e-5),
            (x, residual, weight.to(torch.bfloat16), 1e-5),
            (x, residual, weight, float("nan")),
        ):
            with pytest.raises(ValueError):
                op(a, b, gamma, eps=eps)


def test_residual_norm_registry_and_config_guards(monkeypatch):
    from veomni.arguments.arguments_types import OpsImplementationConfig

    rows = OP_REGISTRY.list_registered("add_rms_norm", "inference")
    assert {"eager", "torch_npu"} <= set(rows)
    assert OpsImplementationConfig().add_rms_norm_implementation == "eager"
    monkeypatch.setattr("veomni.utils.device.IS_NPU_AVAILABLE", False)
    with pytest.raises(ValueError, match="requires Ascend NPU"):
        OpsImplementationConfig(add_rms_norm_implementation="torch_npu")
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        VeomniOp("add_rms_norm", "inference", "missing_backend")


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2048, 4096, 5120])
def test_native_add_rms_norm_matches_separate_npu_ops(dtype, width):
    import torch_npu

    torch.manual_seed(71)
    # Compact target rows and cold-history rows, with deliberately strided inputs.
    x = torch.randn(2, 17, width, device="npu", dtype=dtype).transpose(0, 1)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device="npu", dtype=dtype)
    with torch.no_grad():
        expected_sum = x + residual
        expected = torch_npu.npu_rms_norm(expected_sum, weight, 1e-5)[0]
        actual, summed = VeomniOp("add_rms_norm", "inference", "torch_npu")(x, residual, weight, eps=1e-5)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(summed, expected_sum, atol=0, rtol=0)
    # Fused addition may feed unrounded values into normalization in BF16.
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.0625 if dtype == torch.bfloat16 else 2e-6,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
    )
