"""Real Ascend fused-attention checks; skipped when no NPU is available."""

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.test_causal_cosmos3_triple import branch_layout, branch_visibility_reference
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import generation_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.triple_attention import prepare_branch_attention
from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type, get_torch_device


pytestmark = [
    pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator")),
    pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires a real Ascend NPU for triple-expert TND attention"),
]


@pytest.fixture
def npu_device():
    if get_device_type() != "npu" or not get_torch_device().is_available():
        pytest.skip("requires available Ascend NPU hardware; installed torch_npu alone is insufficient")
    return "npu"


@pytest.mark.parametrize("branch,modalities", [("video", (1,)), ("condition", (2, 3)), ("action", (2, 3))])
@pytest.mark.parametrize("gated", [False, True])
def test_real_npu_triple_branch_attention_forward_backward(npu_device, branch, modalities, gated):
    layout, modes = branch_layout()
    video_conditioning = (False, True, False) if gated else None
    text_conditioning = (True, False, True) if gated else None
    plan = prepare_branch_attention(
        layout,
        npu_device,
        "npu_tnd",
        modalities,
        modes,
        branch,
        video_conditioning=video_conditioning,
        text_conditioning=text_conditioning,
    )
    assert plan.backend == "npu_tnd" and plan.chunks
    # This independently constructed CPU pair mask also covers interleaved
    # samples and query order. The candidate uses the real fused TND kernel.
    visible = branch_visibility_reference(
        layout,
        plan.gen_indexes.cpu().tolist(),
        modes,
        branch,
        True,
        video_conditioning,
        text_conditioning,
    )
    generator = torch.Generator().manual_seed(89)
    tensors = [
        (0.2 * torch.randn(rows, heads, 128, generator=generator)).to(npu_device, torch.bfloat16).requires_grad_()
        for rows, heads in ((len(plan.gen_indexes), 4), (len(layout.roles), 2), (len(layout.roles), 2))
    ]
    reference_inputs = [tensor.detach().float().cpu().requires_grad_() for tensor in tensors]
    query, key, value = reference_inputs
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1),
        key.repeat_interleave(2, dim=1).transpose(0, 1),
        value.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=visible,
    ).transpose(0, 1)
    actual = generation_attention(*tensors, plan)
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
    probe = torch.randn(expected.shape, generator=generator).to(npu_device, torch.bfloat16)
    actual_gradients = torch.autograd.grad(actual, tensors, probe)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, probe.float().cpu())
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert bool(torch.isfinite(actual_gradient).all())
        torch.testing.assert_close(actual_gradient.float().cpu(), expected_gradient, atol=0.02, rtol=0.04)
