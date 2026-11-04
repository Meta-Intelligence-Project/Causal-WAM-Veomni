"""Real Ascend graph capture, full sampling parity and TND gradient checks."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.models.compare import ops_config_scope
from tests.models.diffusers.causal_cosmos3_serving.test_native import window
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.diffusers.causal_cosmos3.cosmos3_core import attention as attention_module
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import TNDChunk, generation_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.npu_replay_graph import clear_model_graph
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix
from veomni.models.diffusers.causal_cosmos3.serving.sampling import sample_decision
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU graph capture")


@pytest.fixture
def npu_model():
    import torch_npu  # noqa: F401

    config = OpsImplementationConfig(
        attn_implementation="eager", rms_norm_implementation="torch_npu", rotary_pos_emb_implementation="torch_npu"
    )
    with ops_config_scope(config):
        torch.manual_seed(17)
        net = Cosmos3Model(
            tiny_config(
                hidden_size=256,
                head_dim=128,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=512,
                mrope_section=[24, 20, 20],
                action_dim=55,
                state_dim=55,
            )
        ).to(device="npu", dtype=torch.bfloat16)
        net.eval().requires_grad_(False)
        try:
            yield net
        finally:
            clear_model_graph(net, close=True)


def npu_window(history, horizon=32):
    _, sample = window(history_steps=history, horizon=horizon)
    for name in ("video_latents", "action"):
        sample[name] = sample[name].to("npu")
    return sample


def compare(net, sample):
    expected = sample_decision(net, sample, steps=3, attention_backend="npu_tnd")
    actual = sample_decision(net, sample, steps=3, attention_backend="npu_tnd", npu_replay_graph=True)
    torch.npu.synchronize()
    torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
    torch.testing.assert_close(actual["video_latents"], expected["video_latents"], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in net.parameters())
    return actual["inference_cache"]["cond"]["npu_replay_graph_stats"]


@pytest.mark.parametrize("history,horizon", [(0, 16), (48, 32), (1024, 32)])
def test_npu_graph_reuses_capture_and_refreshes_changed_measured_conditions(npu_model, history, horizon):
    sample = npu_window(history, horizon)
    for index in range(3):
        sample = copy.deepcopy(sample)
        sample["video_latents"][:, : sample["observed_latents"]] += 0.05
        sample["states"] += 0.1
        sample["text_token_ids"] = torch.tensor([index + 3, index + 6])
        sample["conditioning_fps"] += 0.25
        sample["tick"] = index * 16
        if index == 2:
            sample["action_mask"][1] = False
        stats = compare(npu_model, sample)
        assert stats["captures"] == 1 and stats["refreshes"] == index
        assert stats["replays"] == 2 * (index + 1) and stats["failures"] == 0


@torch.no_grad()
def test_npu_graph_invalidates_after_weights_layout_and_reset(npu_model):
    sample = npu_window(48)
    assert compare(npu_model, sample)["captures"] == 1
    npu_model.action_proj_out.weight.add_(0.001)
    assert compare(npu_model, sample)["captures"] == 2
    sample["text_token_ids"] = torch.tensor([3, 4, 5])
    assert compare(npu_model, sample)["captures"] == 3
    clear_model_graph(npu_model)
    assert compare(npu_model, sample)["captures"] == 4


def test_npu_graph_replaces_buffers_when_inference_mode_changes(npu_model):
    sample = npu_window(48)
    for count, context in enumerate((torch.no_grad, torch.inference_mode, torch.no_grad), 1):
        with context():
            assert compare(npu_model, sample)["captures"] == count


@pytest.mark.parametrize("history", [48, 1024])
def test_npu_first_prefill_reuse_matches_the_complete_one_step_reference(npu_model, history):
    sample = npu_window(history)
    expected = sample_decision(npu_model, sample, steps=1, attention_backend="npu_tnd", cache_mode="off")
    actual = sample_decision(npu_model, sample, steps=1, attention_backend="npu_tnd")
    torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
    torch.testing.assert_close(actual["video_latents"], expected["video_latents"], rtol=0, atol=0)
    assert actual["inference_cache"]["cond"]["replay_calls"] == 0


def test_npu_rtc_span_path_matches_indexed_reference_with_deterministic_gradients(npu_model, monkeypatch):
    sample = npu_window(48)
    previous = torch.zeros(16, 55)
    previous[:, 0] = torch.linspace(0.05, 0.2, 16)
    rtc = RTCPrefix(previous, 8)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # Ascend FA backward is nondeterministic by default, including before
        # these optimizations. Compare the identical deterministic kernels.
        torch.use_deterministic_algorithms(True)
        actual = sample_decision(npu_model, sample, steps=3, attention_backend="npu_tnd", rtc=rtc)
        monkeypatch.setattr(
            attention_module, "select_tokens", lambda tensor, indexes, span: tensor.index_select(0, indexes)
        )
        expected = sample_decision(npu_model, sample, steps=3, attention_backend="npu_tnd", rtc=rtc)
        torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
        torch.testing.assert_close(actual["video_latents"], expected["video_latents"], rtol=0, atol=0)
        torch.testing.assert_close(
            actual["action"][:8], actual["prepared_prefix"].normalized_targets[:8], rtol=0, atol=0
        )
        assert actual["rtc_vjp_calls"] == 3
        assert actual["inference_cache"]["cond"]["replay_calls"] == 3
        assert all(parameter.grad is None for parameter in npu_model.parameters())
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("strided", [False, True])
def test_npu_tnd_span_fast_path_preserves_outputs_and_shared_kv_gradients(strided):
    import torch_npu  # noqa: F401

    torch.manual_seed(19)
    tensors = [
        torch.randn(count * (2 if strided else 1), heads, 128, device="npu", dtype=torch.bfloat16) * 0.2
        for count, heads in ((15, 4), (23, 2), (23, 2))
    ]
    tensors = [(tensor[::2] if strided else tensor).detach().requires_grad_() for tensor in tensors]
    chunks = (
        TNDChunk.from_lists(list(range(7)), list(range(16)), [7], [16], device="npu"),
        TNDChunk.from_lists(list(range(7, 15)), list(range(5, 23)), [8], [18], device="npu"),
    )
    indexed = tuple(replace(chunk, query_span=None, kv_span=None) for chunk in chunks)
    actual = generation_attention(*tensors, SimpleNamespace(backend="npu_tnd", chunks=chunks))
    expected = generation_attention(*tensors, SimpleNamespace(backend="npu_tnd", chunks=indexed))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, tensors, gradient)
    expected_grads = torch.autograd.grad(expected, tensors, gradient)
    for got, wanted in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, wanted, atol=0.002, rtol=0.02)
