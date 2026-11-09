"""Packed episode streams against independent B1 forwards and complete gradients."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_segment import assert_gradients, gradients, segment_payload
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.segment import forward_segment
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.segment_batch import forward_segment_batch
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.utils.recompute_utils import apply_recompute_policy, build_policy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def model_for_batch():
    # All parameters participate in the oracle, including the text stream.
    return Cosmos3Model(tiny_config(enable_tbptt=True, tbptt_segment_steps=256, train_generation_only=False)).train()


def sample_for_stream(name, decisions=3, *, warm=False, backend="grouped", repeated=False):
    if repeated:
        prefix, clean, anchors = (), (), (4,)
    elif warm:
        anchors = tuple(range(6, 6 + 2 * decisions, 2))
        prefix, clean = (), tuple(range(5, anchors[-1] + 1))
    else:
        anchors = tuple(range(0, 2 * decisions, 2))
        prefix, clean = (-4, -3, -2, -1, 0), tuple(range(1, anchors[-1] + 1))
    sample = segment_payload(prefix=prefix, clean=clean, anchors=anchors, backend=backend)
    sample["context_id"] = name
    sample["input_ids"] = torch.tensor([1, 2, 3] if name == "a" else [9, 10, 11, 12, 13])
    sample["text_position_ids"] = torch.arange(len(sample["input_ids"]), dtype=torch.float32).expand(3, -1)
    if name != "a":
        sample["text_position_ids"] = sample["text_position_ids"] + 17
        sample["state_tokens"] = sample["state_tokens"] + 0.3
    return sample


def prepare_memories(model, warm, backend):
    with torch.no_grad():
        return tuple(
            forward_segment(model, sample_for_stream(name, backend=backend))["segment_memory"] if continued else None
            for name, continued in zip(("a", "b"), warm, strict=True)
        )


def memory_values(memory):
    if memory is None:
        return ()
    return (memory.position_ids, memory.input_ids, memory.text_position_ids, *memory.layer_inputs)


def loss_for_outputs(model, output, samples):
    total = sum(len(sample["decisions"]) for sample in samples)
    video_offset = action_offset = 0
    loss = 0.0
    for sample in samples:
        video_stop = video_offset + len(sample["future_video_tokens"])
        action_stop = action_offset + len(sample["action_tokens"])
        prediction = SimpleNamespace(
            video=output["video"][video_offset:video_stop], action=output["action"][action_offset:action_stop]
        )
        loss = loss + sum(model.loss(prediction, sample["targets"]).values()) * len(sample["decisions"]) / total
        video_offset, action_offset = video_stop, action_stop
    return loss


def serial_outputs(model, samples, memories):
    results = [forward_segment(model, sample, memory) for sample, memory in zip(samples, memories, strict=True)]
    return dict(
        video=torch.cat([result["video"] for result in results]),
        action=torch.cat([result["action"] for result in results]),
        segment_memories=tuple(result["segment_memory"] for result in results),
        segment_stats=tuple(result["segment_stats"] for result in results),
    )


@pytest.mark.parametrize("warm", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize(
    "backend,recompute", [("dense", "none"), ("grouped", "none"), ("grouped", "checkpoint"), ("grouped", "sac")]
)
def test_packed_streams_match_serial_outputs_and_all_parameter_gradients(warm, backend, recompute):
    torch.manual_seed(21)
    actual = model_for_batch()
    expected = copy.deepcopy(actual)
    memories = prepare_memories(actual, warm, backend)
    expected_memories = prepare_memories(expected, warm, backend)
    originals = tuple(tuple(value.clone() for value in memory_values(memory)) for memory in memories)
    if recompute != "none":
        actual.gradient_checkpointing_enable()
        if recompute == "sac":
            apply_recompute_policy(
                actual,
                build_policy(
                    enabled=True,
                    enable_reentrant=False,
                    early_stop=True,
                    recompute_layers=-1,
                    selective_recompute_layers=2,
                ),
            )
    samples = tuple(
        sample_for_stream(name, count, warm=continued, backend=backend)
        for name, count, continued in zip(("a", "b"), (8, 1), warm, strict=True)
    )
    calls = [0] * len(actual.layers)

    def record(index):
        def hook(_module, _args):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_pre_hook(record(i)) for i, layer in enumerate(actual.layers)]
    try:
        output = forward_segment_batch(actual, samples, memories)
        assert calls == [1] * len(actual.layers), "B2 must traverse each decoder layer only once"
        reference = serial_outputs(expected, samples, expected_memories)
        torch.testing.assert_close(output["video"], reference["video"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(output["action"], reference["action"], atol=2e-6, rtol=2e-5)
        assert output["segment_stats"] == reference["segment_stats"]
        loss = loss_for_outputs(actual, output, samples)
        reference_loss = loss_for_outputs(expected, reference, samples)
        torch.testing.assert_close(loss, reference_loss, atol=2e-6, rtol=2e-5)
        loss.backward()
        reference_loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert_gradients(gradients(actual), gradients(expected))
    for memory, saved in zip(memories, originals, strict=True):
        assert all(torch.equal(value, original) for value, original in zip(memory_values(memory), saved, strict=True))
    for left, right in zip(output["segment_memories"], reference["segment_memories"], strict=True):
        assert left.frame_ids == right.frame_ids and left.context_id == right.context_id
        for value, ref in zip(left.layer_inputs, right.layer_inputs, strict=True):
            assert not value.requires_grad and value.grad_fn is None
            torch.testing.assert_close(value, ref, atol=2e-6, rtol=2e-5)
    for left, right in zip(*[memory.layer_inputs for memory in output["segment_memories"]], strict=True):
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()


def test_packed_streams_isolate_text_observations_targets_and_cache():
    torch.manual_seed(22)
    model = model_for_batch().eval()
    samples = (sample_for_stream("a"), sample_for_stream("b", 1))
    with torch.no_grad():
        original = forward_segment_batch(model, samples, (None, None))
        changed = copy.deepcopy(samples[1])
        changed["input_ids"] += 1
        changed["text_position_ids"] += 50
        for key in (
            "prefix_video_tokens",
            "clean_video_tokens",
            "future_video_tokens",
            "action_tokens",
            "state_tokens",
        ):
            changed[key] += 100
        altered = forward_segment_batch(model, (samples[0], changed), (None, None))
    video_rows, action_rows = len(samples[0]["future_video_tokens"]), len(samples[0]["action_tokens"])
    assert torch.equal(original["video"][:video_rows], altered["video"][:video_rows])
    assert torch.equal(original["action"][:action_rows], altered["action"][:action_rows])
    assert not torch.equal(original["action"][action_rows:], altered["action"][action_rows:])
    for left, right in zip(original["segment_memories"][0].layer_inputs, altered["segment_memories"][0].layer_inputs):
        assert torch.equal(left, right)


def test_packed_stream_without_new_observations_preserves_memory_identity():
    torch.manual_seed(23)
    model = model_for_batch().eval()
    memories = prepare_memories(model, (True, False), "grouped")
    samples = (sample_for_stream("a", repeated=True), sample_for_stream("b", 1))
    with torch.no_grad():
        actual = forward_segment_batch(model, samples, memories)
        reference = serial_outputs(model, samples, memories)
    assert actual["segment_memories"][0] is memories[0]
    assert actual["segment_stats"][0]["new_clean_latents"] == 0
    torch.testing.assert_close(actual["video"], reference["video"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual["action"], reference["action"], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("fault", ["duplicate_context", "backend", "memory_count", "malformed_second"])
def test_packed_stream_preflight_rejects_invalid_batch_before_projection(fault):
    model = model_for_batch()
    samples = [sample_for_stream("a"), sample_for_stream("b", 1)]
    memories = (None, None)
    if fault == "duplicate_context":
        samples[1]["context_id"] = samples[0]["context_id"]
    elif fault == "backend":
        samples[1]["attention_backend"] = "dense"
    elif fault == "memory_count":
        memories = (None,)
    else:
        samples[1]["action_mask"] = samples[1]["action_mask"][:, :-1]

    def poison(_module, _args):
        raise AssertionError("Invalid packed stream reached projection before validation")

    handle = model.proj_in.register_forward_pre_hook(poison)
    try:
        with pytest.raises(ValueError):
            forward_segment_batch(model, tuple(samples), memories)
    finally:
        handle.remove()


@pytest.mark.parametrize("recompute", [False, True])
def test_public_packed_forward_preserves_weighted_native_loss_and_complete_gradients(recompute):
    torch.manual_seed(24)
    actual = model_for_batch()
    expected = copy.deepcopy(actual)
    memories = prepare_memories(actual, (True, False), "grouped")
    references = prepare_memories(expected, (True, False), "grouped")
    samples = (sample_for_stream("a", 3, warm=True), sample_for_stream("b", 1))
    if recompute:
        actual.gradient_checkpointing_enable()
    output = actual(stream_segments=samples, stream_memories=memories)
    reference = serial_outputs(expected, samples, references)
    reference_loss = loss_for_outputs(expected, reference, samples)
    torch.testing.assert_close(output.video, reference["video"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output.action, reference["action"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(sum(output.loss.values()), reference_loss, atol=2e-6, rtol=2e-5)
    assert len(output.segment_memories) == 2 and len(output.segment_stats) == 2
    sum(output.loss.values()).backward()
    reference_loss.backward()
    assert_gradients(gradients(actual), gradients(expected))
