"""TND inference attention: independent numerics, FIA arguments and native parity."""

import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp
from veomni.ops.kernels.attention.tnd_inference import eager, torch_npu_fia_v2
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _oracle(q, k, v, qe, ke, scale):
    outputs, qs, ks = [], 0, 0
    repeats = q.shape[1] // k.shape[1]
    for endq, endk in zip(qe, ke, strict=True):
        query = q[qs:endq].double().transpose(0, 1)
        key = k[ks:endk].double().repeat_interleave(repeats, dim=1).transpose(0, 1)
        value = v[ks:endk].double().repeat_interleave(repeats, dim=1).transpose(0, 1)
        outputs.append(((query @ key.transpose(-1, -2) * scale).softmax(-1) @ value).transpose(0, 1))
        qs, ks = endq, endk
    return torch.cat(outputs)


def test_tnd_reference_matches_independent_unequal_gqa_groups_and_registry():
    torch.manual_seed(451)
    q, k, v = (
        torch.randn(8, 4, 8, dtype=torch.float64),
        torch.randn(19, 2, 8, dtype=torch.float64),
        torch.randn(19, 2, 8, dtype=torch.float64),
    )
    qe, ke, scale = (2, 3, 8), (4, 10, 19), 8**-0.5
    with torch.no_grad():
        actual = VeomniOp("attention", "tnd_inference", "eager")(q, k, v, query_ends=qe, kv_ends=ke, scale=scale)
    torch.testing.assert_close(actual, _oracle(q, k, v, qe, ke, scale), atol=1e-12, rtol=1e-12)
    assert {"eager", "torch_npu_fia_v2"} <= set(OP_REGISTRY.list_registered("attention", "tnd_inference"))
    with pytest.raises(RuntimeError, match="disabled gradients"):
        eager.wrapper(q, k, v, query_ends=qe, kv_ends=ke, scale=scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fia_v2_adapter_preserves_inputs_endpoints_gqa_and_no_mask(monkeypatch, dtype):
    q, k, v = (
        torch.randn(5, 16, 128, dtype=dtype),
        torch.randn(9, 8, 128, dtype=dtype),
        torch.randn(9, 8, 128, dtype=dtype),
    )
    calls = []

    def vendor(query, key, value, **kwargs):
        assert query is q and key is k and value is v
        calls.append(kwargs)
        return q.clone(), None

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score_v2=vendor))
    with torch.no_grad():
        output = torch_npu_fia_v2.wrapper(q, k, v, query_ends=(2, 5), kv_ends=(4, 9), scale=128**-0.5)
    torch.testing.assert_close(output, q, atol=0, rtol=0)
    assert calls == [
        dict(
            num_query_heads=16,
            num_key_value_heads=8,
            softmax_scale=128**-0.5,
            input_layout="TND",
            actual_seq_qlen=[2, 5],
            actual_seq_kvlen=[4, 9],
            sparse_mode=0,
            return_softmax_lse=False,
        )
    ]


@pytest.mark.parametrize("case", ["ends", "groups", "head_dim", "dtype", "strides", "missing_api"])
def test_fia_v2_rejects_unsupported_inputs_without_fallback(monkeypatch, case):
    q, k, v = (
        torch.randn(5, 4, 128, dtype=torch.bfloat16),
        torch.randn(9, 2, 128, dtype=torch.bfloat16),
        torch.randn(9, 2, 128, dtype=torch.bfloat16),
    )
    qe, ke = (2, 5), (4, 9)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    if case == "ends":
        qe = (2, 4)
    elif case == "groups":
        qe = (5,)
    elif case == "head_dim":
        q, k, v = q[..., :64], k[..., :64], v[..., :64]
    elif case == "dtype":
        q, k, v = q.float(), k.float(), v.float()
    elif case == "strides":
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
    with torch.no_grad(), pytest.raises(RuntimeError if case == "missing_api" else ValueError):
        torch_npu_fia_v2.wrapper(q, k, v, query_ends=qe, kv_ends=ke, scale=128**-0.5)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU and a FIA-v2-capable torch_npu/CANN pair")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_fia_v2_matches_fas_for_three_unequal_gqa_groups(dtype):
    import torch_npu

    if not hasattr(torch_npu, "npu_fused_infer_attention_score_v2"):
        pytest.skip("installed torch_npu lacks FIA v2")
    torch.manual_seed(452)
    q = torch.randn(113, 16, 128, device="npu", dtype=dtype)
    k = torch.randn(728, 8, 128, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    qe, ke, scale = (1, 33, 113), (16, 216, 728), 128**-0.5
    with torch.no_grad():
        expected = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=16,
            input_layout="TND",
            scale=scale,
            keep_prob=1.0,
            sparse_mode=0,
            actual_seq_qlen=qe,
            actual_seq_kvlen=ke,
        )[0]
        actual = VeomniOp("attention", "tnd_inference", "torch_npu_fia_v2")(
            q, k, v, query_ends=qe, kv_ends=ke, scale=scale
        )
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=1e-2)
