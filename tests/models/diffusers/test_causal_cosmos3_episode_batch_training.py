"""Batched episode Trainer execution, decision weighting and atomic cache commits."""

import copy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers import test_causal_cosmos3_episode_training as fixtures
from veomni.models.diffusers.causal_cosmos3.cosmos3_core import segment as segment_core
from veomni.trainer import causal_cosmos3_episode as training
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops(monkeypatch):
    with ops_config_scope(eager_ops_config()):
        monkeypatch.setattr(training, "use_parallel_state", lambda *args: nullcontext())
        yield


@pytest.fixture
def roots(tmp_path, request):
    return fixtures.roots.__wrapped__(tmp_path, request)


class CountingRuntime(fixtures.Runtime):
    def __init__(self, model, owner):
        super().__init__(model, owner)
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return super().__call__(**kwargs)


def compare_gradients(actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        reference = expected_parameters[name].grad
        if parameter.grad is None or reference is None:
            assert parameter.grad is reference, name
        else:
            # Same tolerances as the existing raw-episode Trainer gradient oracle.
            torch.testing.assert_close(parameter.grad, reference, atol=1e-4, rtol=3e-5, msg=name)


@pytest.mark.parametrize("roots", [(117, 25)], indirect=True)
@pytest.mark.parametrize("decision_stride_steps,expected_counts", [(None, [2, 1]), (16, [4, 2])])
def test_real_raw_condition_model_batch_matches_two_accumulated_b1_gradients(
    roots, monkeypatch, decision_stride_steps, expected_counts
):
    _, first = fixtures.raw_segments(roots[0], decision_stride_steps=decision_stride_steps)
    _, second = fixtures.raw_segments(roots[1], decision_stride_steps=decision_stride_steps)
    raws = (first[0], {**second[0], "lane_id": 1})
    assert [len(raw["decisions"]) for raw in raws] == expected_counts
    # Read actual raw observations through the actual VAE/condition once, then
    # reuse these fixed noisy targets to isolate batching from RNG draw order.
    preparer = fixtures.condition()
    prepared = {}
    for raw in raws:
        encoded = preparer.get_condition(segment=raw)
        prepared[raw["lane_id"]] = preparer.process_condition(**encoded)

    def fixed_condition():
        owner = fixtures.condition()
        monkeypatch.setattr(owner, "process_condition", lambda *, episode: prepared[episode["lane_id"]])
        return owner

    torch.manual_seed(143)
    actual_model = fixtures.tiny_model()
    expected_model = copy.deepcopy(actual_model)
    actual_owner, expected_owner = fixed_condition(), fixed_condition()
    actual = fixtures.trainer_for(actual_model, actual_owner)
    expected = fixtures.trainer_for(expected_model, expected_owner)
    actual.base.model = CountingRuntime(actual_model, actual_owner)
    expected.base.model = CountingRuntime(expected_model, expected_owner)
    actual.base.num_micro_batches = 1
    expected.base.num_micro_batches = 2
    actual.on_step_begin(micro_batches=[{"segments": raws}])
    expected.on_step_begin(micro_batches=[{"segment": raw} for raw in raws])
    assert actual.episode_global_decisions == expected.episode_global_decisions == sum(expected_counts)

    backward_calls = []
    native_backward = torch.autograd.backward

    def observed_backward(*args, **kwargs):
        backward_calls.append(True)
        return native_backward(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", observed_backward)
    actual_loss, actual_losses = actual.forward_backward_step({"segments": raws})
    assert len(backward_calls) == actual.base.model.calls == 1
    expected_loss = 0
    for raw in raws:
        loss, _ = expected.forward_backward_step({"segment": raw})
        expected_loss = expected_loss + loss.detach()
    assert len(backward_calls) == 3 and expected.base.model.calls == 2
    torch.testing.assert_close(actual_loss.detach(), expected_loss, atol=1e-5, rtol=2e-5)
    assert set(actual_losses) == {"video_loss", "action_loss"}
    compare_gradients(actual_model, expected_model)
    assert isinstance(actual.last_episode_statistics, tuple)
    assert [value["actual_decisions"] for value in actual.last_episode_statistics] == expected_counts
    # The long stream continues; the short final stream is cleared independently.
    assert set(actual.segment_memories) == set(actual_owner.episode_streams) == {0}
    assert set(expected.segment_memories) == set(expected_owner.episode_streams) == {0}
    for left, right in zip(
        actual.segment_memories[0].layer_inputs, expected.segment_memories[0].layer_inputs, strict=True
    ):
        assert not left.requires_grad and left.grad_fn is None
        torch.testing.assert_close(left, right, atol=1e-5, rtol=2e-5)


def fake_raw(lane, count, *, reset=True, final=False):
    return dict(
        lane_id=lane,
        stream_id=("synthetic", lane),
        segment_id=0 if reset else 1,
        reset=reset,
        final=final,
        decisions=[{} for _ in range(count)],
    )


class FakeCondition:
    def __init__(self, fail_lane=None):
        self.episode_streams = {99: object()}
        self.calls = []
        self.cleared = []
        self.fail_lane = fail_lane

    def get_condition(self, *, segment):
        lane = segment["lane_id"]
        self.calls.append(lane)
        self.episode_streams[lane] = object()
        if lane == self.fail_lane:
            raise ValueError("second stream preparation failed")
        return dict(
            episode=dict(
                lane_id=lane,
                actual_decisions=len(segment["decisions"]),
                nominal_decisions=8,
            )
        )

    def process_condition(self, *, episode):
        return {"stream_segment": {"lane_id": episode["lane_id"], "history_latents": 4}}

    def clear_episode_streams(self, lane=None):
        self.cleared.append(lane)
        if lane is None:
            self.episode_streams.clear()
        else:
            self.episode_streams.pop(lane, None)


class TrackingMemories(dict):
    def __init__(self):
        super().__init__({99: object()})
        self.commits = []

    def __setitem__(self, lane, value):
        self.commits.append((lane, value))
        super().__setitem__(lane, value)


def fake_trainer(monkeypatch, raws, *, fail_lane=None, bad_candidate=None, dp=1, accum=1, global_decisions=None):
    def metadata(model, payload, memory):
        lane = payload["lane_id"]
        return dict(
            context=f"synthetic-{lane}",
            history=4,
            patches=1,
            frames=(0, 1, 2, 3),
            device=torch.device("cpu"),
            input_ids=torch.tensor([lane + 1]),
            text_positions=torch.zeros(3, 1),
        )

    monkeypatch.setattr(segment_core, "_prepare", metadata)
    owner = FakeCondition(fail_lane=fail_lane)
    parameter = torch.tensor(2.0, requires_grad=True)
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=2), layers=[object(), object()])
    candidates = [
        segment_core.SegmentMemory(
            frame_ids=(0, 1, 2, 3),
            position_ids=torch.zeros(3, 4),
            layer_inputs=(torch.ones(4, 2), torch.ones(4, 2)),
            input_ids=torch.tensor([raw["lane_id"] + 1]),
            text_position_ids=torch.zeros(3, 1),
            context_id=f"synthetic-{raw['lane_id']}",
            history_latents=4,
            patches_per_frame=1,
            model_id=id(model),
        )
        for raw in raws
    ]
    if bad_candidate == "attached":
        candidates[-1] = replace(candidates[-1], layer_inputs=(torch.ones(4, 2), torch.ones(4, 2, requires_grad=True)))
    elif bad_candidate == "other_context":
        candidates[-1] = replace(candidates[-1], context_id=candidates[0].context_id)
    elif bad_candidate == "missing":
        candidates[-1] = None
    elif bad_candidate == "short_tuple":
        candidates = candidates[:-1]

    class Runtime:
        def __init__(self):
            self.model = model
            self.condition_model = owner
            self.parallel_state = SimpleNamespace(dp_size=dp, dp_rank=0, dp_group=None)
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            assert set(kwargs) >= {"stream_segments", "stream_memories"}
            assert len(kwargs["stream_segments"]) == len(raws)
            return SimpleNamespace(
                loss={"video_loss": parameter * 3, "action_loss": parameter * 4},
                segment_memories=tuple(candidates),
                segment_stats=tuple({"retained_history_latents": 4} for _ in raws),
            )

    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.episode_training = True
    trainer.segment_memories = TrackingMemories()
    trainer.base = SimpleNamespace(
        model=Runtime(),
        num_micro_batches=accum,
        device=torch.device("cpu"),
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        on_step_begin=lambda **kwargs: None,
    )
    trainer.episode_global_decisions = global_decisions or sum(len(raw["decisions"]) for raw in raws)
    return trainer, owner, parameter, candidates


def test_batch_global_actual_decision_weighting_one_call_and_mixed_final_commit(monkeypatch):
    raws = (fake_raw(0, 2), fake_raw(1, 1, final=True))
    trainer, owner, parameter, candidates = fake_trainer(monkeypatch, raws, dp=2, accum=4, global_decisions=30)
    calls = []
    native_backward = torch.autograd.backward

    def observed_backward(*args, **kwargs):
        calls.append(True)
        return native_backward(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "backward", observed_backward)
    loss, losses = trainer.forward_backward_step({"segments": raws})
    expected_factor = 3 * 2 / 30
    torch.testing.assert_close(loss.detach(), torch.tensor(14 * expected_factor))
    torch.testing.assert_close(parameter.grad, torch.tensor(7 * expected_factor))
    assert trainer.base.model.calls == len(calls) == 1
    assert set(losses) == {"video_loss", "action_loss"}
    assert set(trainer.segment_memories) == set(owner.episode_streams) == {0, 99}
    assert trainer.segment_memories[0] is candidates[0]
    assert trainer.segment_memories.commits == [(0, candidates[0])]
    assert [item["actual_decisions"] for item in trainer.last_episode_statistics] == [2, 1]


@pytest.mark.parametrize(
    "bad",
    ["duplicate_lane", "missing_memory", "prepare_second", "attached", "other_context", "missing", "short_tuple"],
)
def test_batch_failure_clears_touched_lanes_without_partial_cache_commit(monkeypatch, bad):
    raws = (fake_raw(0, 2), fake_raw(1, 1))
    if bad == "duplicate_lane":
        raws = (raws[0], {**raws[1], "lane_id": 0})
    elif bad == "missing_memory":
        raws = (raws[0], {**raws[1], "reset": False, "segment_id": 1})
    trainer, owner, _, _ = fake_trainer(
        monkeypatch,
        raws,
        fail_lane=1 if bad == "prepare_second" else None,
        bad_candidate=bad,
    )
    for lane in (0, 1):
        owner.episode_streams[lane] = object()
    with pytest.raises((ValueError, RuntimeError), match="Episode|episode|memory|lane|preparation"):
        trainer.forward_backward_step({"segments": raws})
    touched = {raw["lane_id"] for raw in raws}
    assert touched.isdisjoint(trainer.segment_memories)
    assert touched.isdisjoint(owner.episode_streams)
    assert 99 in trainer.segment_memories and 99 in owner.episode_streams
    assert trainer.segment_memories.commits == []
    if bad in ("duplicate_lane", "missing_memory", "prepare_second"):
        assert trainer.base.model.calls == 0
    else:
        assert trainer.base.model.calls == 1


def test_on_step_begin_counts_all_streams_across_accumulated_microbatches(monkeypatch):
    raws = (fake_raw(0, 2), fake_raw(1, 1), fake_raw(2, 3))
    trainer, _, _, _ = fake_trainer(monkeypatch, raws)
    trainer.on_step_begin(micro_batches=[{"segment": raws[0]}, {"segments": raws[1:]}])
    assert trainer.episode_global_decisions == 6
