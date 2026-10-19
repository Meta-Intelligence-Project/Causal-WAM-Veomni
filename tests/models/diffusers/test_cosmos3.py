"""Packed causal attention and gradients, independent of a model or data pipeline."""

import pytest
import torch

from veomni.models.diffusers.cosmos3.attention import (
    CausalLayout,
    generation_attention,
    prepare_attention,
    understanding_attention,
)
from veomni.utils.device import IS_NPU_AVAILABLE


def fixed_inputs(task="policy", device="cpu", dtype=torch.float32):
    # One text prefix, three clean frames, three current frames, two actions,
    # and one immutable state per prediction block. Duplicate for packing tests.
    video_roles = (2, 2, 2) if task == "id" else (2, 1, 1)
    action_roles = (2, 2) if task == "fd" else (1, 1)
    roles = (-1, -1, 0, 0, 0, *video_roles, *action_roles, 3, 3)
    modalities = (0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3)
    blocks = (-1, -1, -1, 0, 1, -1, 0, 1, 0, 1, 0, 1)
    layout = CausalLayout((0,) * 12 + (1,) * 12, blocks * 2, roles * 2, modalities * 2, (1, 2))
    generator = torch.Generator().manual_seed(20261019)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    return dict(
        input_ids=torch.tensor([1, 2, 3, 4], device=device),
        video_tokens=rand(12, 8),
        video_timesteps=torch.tensor([0, 0, 0, 0, 400, 700] * 2, device=device),
        action_tokens=rand(4, 4),
        action_timesteps=torch.tensor([400, 700] * 2, device=device),
        action_mask=torch.tensor([[True, True, False, False]] * 4, device=device),
        state_tokens=rand(4, 4),
        state_mask=torch.tensor([[True, False, True, False]] * 4, device=device),
        position_ids=torch.tensor([blocks * 2, tuple(range(12)) * 2, (0,) * 24], device=device),
        layout=layout,
    )


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_grouped_attention_outputs_and_shared_kv_gradients(task):
    layout = fixed_inputs(task)["layout"]
    dense = prepare_attention(layout, "cpu", "dense")
    grouped = prepare_attention(layout, "cpu", "grouped", max_kv_tokens=13)
    torch.manual_seed(42)
    inputs = [
        torch.randn(n, heads, 8, dtype=torch.float64, requires_grad=True) for n, heads in ((20, 4), (24, 2), (24, 2))
    ]
    reference = generation_attention(*inputs, dense)
    actual = generation_attention(*inputs, grouped)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(reference, inputs, grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend UND TND validation")
@pytest.mark.parametrize("samples", [(0, 1, 0, 1, 1), (0,) * 7 + (1,) * 2053], ids=["interleaved", "long"])
def test_npu_und_packed_causal_forward_backward(samples, monkeypatch):
    import torch_npu

    torch.manual_seed(29)
    n = len(samples)
    layout = CausalLayout(samples, (-1,) * n, (-1,) * n, (0,) * n, (1, 1))
    inputs = [(torch.randn(n, heads, 128) * 0.2).to("npu", torch.bfloat16).requires_grad_() for heads in (4, 2, 2)]
    reference_inputs = [x.detach().float().cpu().requires_grad_() for x in inputs]
    # Independent dense mask: same sample and no later text position.
    ids, positions = torch.tensor(samples), torch.arange(n)
    visible = (ids[:, None] == ids[None, :]) & (positions[:, None] >= positions[None, :])
    q, k, v = reference_inputs
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(2, dim=1).transpose(0, 1),
        v.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=visible,
    ).transpose(0, 1)
    calls = []
    original = torch_npu.npu_fusion_attention

    def record(query, key, value, **kwargs):
        calls.append(
            (
                query.shape[1],
                key.shape[1],
                kwargs["input_layout"],
                kwargs["sparse_mode"],
            )
        )
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", record)
    actual = understanding_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd"))
    assert calls == [(4, 2, "TND", 2)]
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.float().cpu())
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float().cpu(), expected_grad, atol=0.02, rtol=0.04)
    # A query cannot read a future text token or another packed sample.
    key_grad, value_grad = torch.autograd.grad(
        understanding_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd"))[0].float().sum(),
        inputs[1:],
    )
    assert torch.count_nonzero(key_grad[1:]) == 0
    assert torch.count_nonzero(value_grad[1:]) == 0


def test_und_empty_text():
    layout = CausalLayout((0,), (-1,), (1,), (1,), (1,))
    query = torch.empty(0, 4, 8, requires_grad=True)
    key, value = (torch.empty(0, 2, 8) for _ in range(2))
    output = understanding_attention(query, key, value, prepare_attention(layout, "cpu"))
    assert output.shape == query.shape
    output.sum().backward()
    assert query.grad is not None


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend TND validation")
def test_npu_grouped_tnd_forward_backward():
    torch.manual_seed(19)
    layout = fixed_inputs()["layout"]
    inputs = [
        (torch.randn(n, heads, 128) * 0.2).to("npu", torch.bfloat16).requires_grad_()
        for n, heads in ((20, 4), (24, 2), (24, 2))
    ]
    oracle_inputs = [x.detach().float().cpu().requires_grad_() for x in inputs]
    expected = generation_attention(*oracle_inputs, prepare_attention(layout, "cpu", "dense"))
    actual = generation_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd", max_kv_tokens=13))
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, oracle_inputs, grad.float().cpu())
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float().cpu(), expected_grad, atol=0.02, rtol=0.04)
