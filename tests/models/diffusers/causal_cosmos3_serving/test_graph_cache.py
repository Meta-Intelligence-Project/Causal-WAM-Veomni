"""CPU checks for cache ownership, signature admission and request lifecycle."""

import copy
import gc
import weakref
from dataclasses import replace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.causal_cosmos3_serving.test_native import window
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import TNDChunk, select_tokens
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.npu_replay_graph import NPUReplayGraphCache
from veomni.models.diffusers.causal_cosmos3.serving.prefix_cache import PrefixCache
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix
from veomni.models.diffusers.causal_cosmos3.serving.sampling import sample_decision


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


class Replay:
    """Reexecute CPU math into stable outputs to test the graph controller only."""

    def __init__(self, forward, output):
        self.forward, self.output, self.fail = forward, output, False

    def replay(self):
        if self.fail:
            raise RuntimeError("replay failure")
        for target, value in zip(self.output, self.forward(), strict=True):
            target.copy_(value)


class Backend:
    def __init__(self):
        self.fail_capture = False

    def supports(self, tensor):
        return tensor.device.type == "cpu"

    def synchronize(self, device):
        pass

    def capture(self, forward, device):
        if self.fail_capture:
            raise RuntimeError("capture failure")
        output = tuple(value.clone() for value in forward())
        return Replay(forward, output), output, 0.01


def model():
    torch.manual_seed(42)
    return Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval().requires_grad_(False)


def inputs(sample):
    sample = dict(sample)
    sample["token_length"] = (
        len(sample["text_token_ids"]) + sample["video_latents"].shape[1] + len(sample["action"]) + 1
    )
    result, _, _ = pack_model_inputs(
        [sample],
        [sample["video_latents"]],
        [sample["action"]],
        timesteps=[torch.zeros(sample["video_latents"].shape[1])],
    )
    return result


def advance(value):
    result = dict(value)
    result["video_tokens"] = value["video_tokens"] + 0.2
    result["action_tokens"] = value["action_tokens"] + 0.15
    result["video_timesteps"] = value["video_timesteps"] + 300
    result["action_timesteps"] = value["action_timesteps"] + 300
    return result


def graph_entry(net, sample, cache):
    value = inputs(sample)
    entry = PrefixCache(net, value, "grouped", graph_cache=cache)
    entry(value)  # Ordinary prefill does not enter the graph.
    # CPU seam: advertise NPU admission while keeping the actual CPU attention
    # plan. This tests lifecycle math; real NPU capture is tested separately.
    entry.plan = replace(entry.plan, backend="npu_tnd")
    return entry, advance(value)


@torch.no_grad()
def oracle(net, sample):
    value = inputs(sample)
    entry = PrefixCache(net, value, "grouped")
    entry(value)
    result = entry(advance(value))
    return result.video.clone(), result.action.clone()


@pytest.mark.parametrize("indexes", [[0, 1, 2, 3, 4], [1, 2, 3], [4, 1, 4], []])
@pytest.mark.parametrize("strided", [False, True])
def test_contiguous_selection_matches_indexed_values_and_gradients(indexes, strided):
    source = torch.randn(5, 4, requires_grad=True)
    candidate = source[:, ::2] if strided else source
    reference = candidate.detach().clone().requires_grad_(True)
    chunk = TNDChunk.from_lists(indexes, indexes, [len(indexes)], [len(indexes)], device="cpu")
    actual = select_tokens(candidate, chunk.query_indexes, chunk.query_span)
    expected = reference.index_select(0, chunk.query_indexes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(source.grad[:, ::2] if strided else source.grad, reference.grad, rtol=0, atol=0)
    if chunk.query_span is not None and indexes and not strided:
        assert actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


@torch.no_grad()
def test_ordinary_prefill_is_first_result_and_kv_buffers_are_reused():
    net = model()
    _, sample = window()
    value = inputs(sample)
    entry = PrefixCache(net, value, "grouped")
    first = entry(value)
    assert first is entry.reference and entry.stats["replay_calls"] == 0
    pointers = [(layer["key"].data_ptr(), layer["value"].data_ptr()) for layer in entry.layers.values()]
    expected = oracle(net, sample)
    actual = entry(advance(value))
    torch.testing.assert_close(actual.video, expected[0])
    torch.testing.assert_close(actual.action, expected[1])
    assert pointers == [(layer["key"].data_ptr(), layer["value"].data_ptr()) for layer in entry.layers.values()]
    assert entry.stats["prefill_calls"] == entry.stats["replay_calls"] == 1


@torch.no_grad()
def test_compatible_requests_refresh_history_text_state_masks_and_rope():
    net, cache = model(), None
    cache = NPUReplayGraphCache(net, _backend=Backend())
    _, sample = window()
    previous = None
    for index in range(3):
        sample = copy.deepcopy(sample)
        sample["video_latents"] += 0.3
        sample["states"] += 0.2
        sample["text_token_ids"] = torch.tensor([index + 4, index + 5])
        sample["conditioning_fps"] += 0.5
        if index == 2:
            sample["action_mask"][1] = False
        expected = oracle(net, sample)
        entry, value = graph_entry(net, sample, cache)
        with cache.request():
            actual = entry(value)
            torch.testing.assert_close(actual.video, expected[0])
            torch.testing.assert_close(actual.action, expected[1])
        if previous is not None:
            assert not torch.equal(actual.action, previous)
        previous = actual.action.clone()
        owner = weakref.ref(entry)
        del entry
        gc.collect()
        assert owner() is None
    assert cache.stats["captures"] == 1
    assert cache.stats["refreshes"] == 2 and cache.stats["requests"] == 3
    cache.close()
    assert cache.stats["closed"] and not cache.stats["has_graph"]


@pytest.mark.parametrize("change", ["layout", "weight", "replacement", "scale"])
@torch.no_grad()
def test_layout_or_model_changes_invalidate_the_captured_graph(change):
    net = model()
    cache = NPUReplayGraphCache(net, _backend=Backend())
    _, sample = window()
    entry, value = graph_entry(net, sample, cache)
    with cache.request():
        entry(value)
    if change == "layout":
        sample["text_token_ids"] = torch.tensor([3, 4, 5])
    elif change == "weight":
        net.action_proj_out.weight.add_(0.1)
    elif change == "replacement":
        net.action_proj_out = copy.deepcopy(net.action_proj_out)
    else:
        net.config.timestep_scale *= 2
    expected = oracle(net, sample)
    entry, value = graph_entry(net, sample, cache)
    with cache.request():
        actual = entry(value)
        torch.testing.assert_close(actual.video, expected[0])
        torch.testing.assert_close(actual.action, expected[1])
    assert cache.stats["captures"] == 2 and cache.stats["invalidations"] == 1
    cache.close()


@pytest.mark.parametrize("failure", ["capture", "replay", "request"])
@torch.no_grad()
def test_graph_failures_drop_buffers_and_the_next_request_can_recover(failure):
    net, backend = model(), Backend()
    cache = NPUReplayGraphCache(net, _backend=backend)
    _, sample = window()
    entry, value = graph_entry(net, sample, cache)
    if failure == "capture":
        backend.fail_capture = True
    else:
        with cache.request():
            entry(value)
        if failure == "replay":
            cache._graph.fail = True
    with pytest.raises(RuntimeError, match="failure|cache cleared"):
        with cache.request():
            if failure == "request":
                raise RuntimeError("request failure")
            entry(value)
    assert not cache.stats["has_graph"] and cache._layers is None and cache._output is None
    backend.fail_capture = False
    entry, value = graph_entry(net, sample, cache)
    with cache.request():
        actual = entry(value)
        torch.testing.assert_close(actual.action, oracle(net, sample)[1])
    cache.close()


def test_request_lease_rejects_concurrent_reentry_reset_and_closed_reuse():
    cache = NPUReplayGraphCache(model(), _backend=Backend())
    with cache.request():
        with pytest.raises(RuntimeError, match="Concurrent"):
            with cache.request():
                pass
        with pytest.raises(RuntimeError, match="during sampling"):
            cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        with cache.request():
            pass


def test_inference_mode_boundaries_replace_inference_tensor_buffers():
    net = model()
    cache = NPUReplayGraphCache(net, _backend=Backend())
    _, sample = window()
    for count, mode in enumerate((torch.inference_mode, torch.no_grad, torch.inference_mode), 1):
        with mode():
            entry, value = graph_entry(net, sample, cache)
            with cache.request():
                actual = entry(value)
                torch.testing.assert_close(actual.action, oracle(net, sample)[1])
            assert cache.stats["captures"] == count
    cache.close()


@torch.no_grad()
def test_deterministic_kernel_mode_changes_invalidate_capture():
    net = model()
    cache = NPUReplayGraphCache(net, _backend=Backend())
    _, sample = window()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for count, mode in enumerate((False, True, False), 1):
            torch.use_deterministic_algorithms(mode)
            entry, value = graph_entry(net, sample, cache)
            with cache.request():
                actual = entry(value)
                torch.testing.assert_close(actual.action, oracle(net, sample)[1])
            assert cache.stats["captures"] == count
    finally:
        cache.close()
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(npu_replay_graph=1), "bool"),
        (dict(npu_replay_graph=True, rtc=RTCPrefix(torch.zeros(16, 55), 8)), "mutually exclusive"),
        (dict(npu_replay_graph=True, guidance=2), "guidance=1"),
        (dict(npu_replay_graph=True, cache_mode="plans"), "full cache"),
        (dict(npu_replay_graph=True, attention_backend="grouped"), "npu_tnd"),
        (dict(npu_replay_graph=True, attention_backend="npu_tnd"), "Ascend NPU"),
    ],
)
def test_graph_sampler_rejects_incompatible_modes_before_evaluation(options, match):
    _, sample = window()
    with pytest.raises(ValueError, match=match):
        sample_decision(model(), sample, **options)
