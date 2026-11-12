"""Real raw episode → persistent VAE → compact segment → model/trainer integration."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data import test_causal_cosmos3_episode as data_fixtures
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.causal_cosmos3_serving.test_episode_streaming import Tokenizer, image, reference
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.episode_streaming import EpisodeObservationEncoder
from veomni.trainer import causal_cosmos3_episode as training
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


@pytest.fixture
def roots(tmp_path, request):
    return data_fixtures.roots.__wrapped__(tmp_path, request)


def raw_segments(root, *, history=48, mode="policy", decision_stride_steps=None, length=4):
    options = {} if decision_stride_steps is None else dict(decision_stride_steps=decision_stride_steps)
    dataset = data_fixtures.action_dataset(root, history_steps=history, mode=mode, **options)
    result = []
    for raw in data_fixtures.stream(dataset, length=length):
        # video=False omits the union RGB read. Reconstruct its selected ticks
        # on the union clock, including overlapping target lookahead exactly once.
        past = raw["history_steps"] if raw["reset"] else 0
        read_end = raw.get("read_end_tick", raw["end_tick"])
        last = raw["decisions"][-1]["window"]
        real_end = last.start + last.real_actions
        ticks = [
            max(0, min(tick, real_end)) for tick in range(raw["start_tick"] - past, read_end + 1, raw["video_stride"])
        ]
        raw["video_frame_ticks"] = ticks
        raw["video"] = torch.stack([image(int(tick)) for tick in ticks], dim=1)
        result.append(raw)
    return dataset, result


def condition():
    result = CausalCosmos3ConditionModel(
        device="cpu",
        dtype=torch.float32,
        stage="action",
        seed=71,
        shift=1,
        history_encoding="episode_stream",
        attention_backend="grouped",
    )
    result.video_vae = Tokenizer()
    return result


def tiny_model():
    result = Cosmos3Model(
        tiny_config(
            latent_channels=1,
            vocab_size=256,
            action_dim=55,
            state_dim=55,
            enable_tbptt=True,
            tbptt_segment_steps=64,
            train_generation_only=True,
        )
    )
    result.freeze_model()
    return result.train()


def prepare_with_native_reference(owner, raw):
    encoded = owner.get_condition(segment=raw)
    rng = owner.rng_state_dict()
    compact = owner.process_condition(**encoded)["stream_segment"]
    owner.load_rng_state_dict(rng)
    native, targets = owner.prepare(encoded["episode"]["samples"])
    return encoded["episode"], compact, native, targets


def assert_compact_native_equivalent(model, compact, native, targets, output):
    video = torch.zeros_like(native["video_tokens"])
    for old, new in zip(targets["spans"], compact["targets"]["spans"], strict=True):
        count = new.current_video_rows.stop - new.current_video_rows.start
        section = slice(old.current_video_rows.stop - count, old.current_video_rows.stop)
        torch.testing.assert_close(
            compact["future_video_tokens"][new.current_video_rows], native["video_tokens"][section], atol=0, rtol=0
        )
        assert torch.equal(
            compact["future_video_timesteps"][new.current_video_rows], native["video_timesteps"][section]
        )
        assert torch.equal(
            compact["targets"]["loss_masks"]["video_loss_mask"][new.current_video_rows],
            targets["loss_masks"]["video_loss_mask"][section],
        )
        video[section] = output.video[new.current_video_rows]
    assert torch.equal(compact["action_tokens"], native["action_tokens"])
    assert torch.equal(compact["action_timesteps"], native["action_timesteps"])
    assert torch.equal(compact["action_mask"], native["action_mask"])
    assert torch.equal(compact["targets"]["loss_masks"]["action_loss_mask"], targets["loss_masks"]["action_loss_mask"])
    expanded = model.loss(SimpleNamespace(video=video, action=output.action), targets)
    for name in expanded:
        torch.testing.assert_close(output.loss[name], expanded[name], atol=1e-5, rtol=2e-5)


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
@pytest.mark.parametrize("history", [0, 48])
@pytest.mark.parametrize("mode", ["forward_dynamics", "inverse_dynamics", "policy"])
@pytest.mark.parametrize("decision_stride_steps", [None, 16])
def test_raw_segments_preserve_native_future_masks_targets_times_and_bounded_memory(
    roots, history, mode, decision_stride_steps
):
    _, raws = raw_segments(roots[0], history=history, mode=mode, decision_stride_steps=decision_stride_steps)
    owner, model, memory = condition(), tiny_model(), None
    prefix = 1 + history // 16
    actual = []
    for raw in raws:
        encoded, compact, native, targets = prepare_with_native_reference(owner, raw)
        assert compact["history_latents"] == prefix
        assert compact["prefix_frame_ids"] == (tuple(range(prefix)) if raw["reset"] else ())
        anchors = encoded["anchor_frame_ids"]
        expected_clean = (
            tuple(range(prefix, anchors[-1] + 1)) if raw["reset"] else tuple(range(actual[-1] + 1, anchors[-1] + 1))
        )
        assert compact["clean_frame_ids"] == expected_clean
        actual.append(anchors[-1])
        output = model(stream_segment=compact, stream_memory=memory)
        assert_compact_native_equivalent(model, compact, native, targets, output)
        assert len(output.segment_memory.frame_ids) == prefix
        assert output.segment_memory.frame_ids[-1] == anchors[-1]
        assert all(not value.requires_grad and value.grad_fn is None for value in output.segment_memory.layer_inputs)
        assert output.segment_stats["cold_prefix_latents"] == (prefix if raw["reset"] else 0)
        assert output.segment_stats["reused_history_latents"] == (0 if raw["reset"] else prefix)
        lane = owner.episode_streams[raw["lane_id"]]
        assert lane.encoder.state.latent_chunks == []
        assert lane.encoder.latents.shape[2] <= prefix + 4 + 2
        assert not lane.encoder.latents.requires_grad
        sum(output.loss.values()).backward()
        model.zero_grad(set_to_none=True)
        memory = output.segment_memory
    assert len(raws) == 2
    assert owner.video_vae.events.count(("start", 1)) == 1


@pytest.mark.parametrize("roots", [(600, 75)], indirect=True)
@pytest.mark.parametrize("segment_latents", [1, 4, 16])
def test_s16_h32_segments_reuse_lookahead_once_and_match_full_causal_vae(roots, segment_latents):
    _, raws = raw_segments(roots[0], decision_stride_steps=16, length=segment_latents)
    owner = condition()
    all_frames = []
    previous_anchor = None
    for raw in raws:
        overlap = 0 if raw["reset"] else 1 + (raw["action_horizon"] - raw["decision_stride_steps"]) // 4
        all_frames.append(raw["video"][:, overlap:])
        pixels = torch.cat(all_frames, dim=1).unsqueeze(0).float() / 127.5 - 1
        full = reference(pixels).squeeze(0)
        encoded = owner.get_condition(segment=raw)["episode"]
        assert encoded["nominal_decisions"] == segment_latents
        first_anchor = 3 + raw["start_tick"] // 16
        assert encoded["anchor_frame_ids"] == tuple(first_anchor + i for i in range(len(raw["decisions"])))
        for sample, anchor in zip(encoded["samples"], encoded["anchor_frame_ids"], strict=True):
            torch.testing.assert_close(sample["video_latents"], full[:, anchor - 3 : anchor + 3], atol=0, rtol=0)
        expected_clean = tuple(
            range(4 if previous_anchor is None else previous_anchor + 1, encoded["anchor_frame_ids"][-1] + 1)
        )
        assert encoded["clean_frame_ids"] == expected_clean
        # Target lookahead can be encoded for a loss, but clean model memory
        # never commits an observation beyond the last supervised anchor.
        if expected_clean:
            assert expected_clean[-1] < full.shape[1] - 1
        lane = owner.episode_streams[raw["lane_id"]]
        assert lane.end_tick == raw["end_tick"]
        assert lane.read_end_tick == raw["end_tick"] + 16
        assert lane.boundary_images.shape[1] == 5
        assert lane.encoder.frame_count == 1 + (48 + lane.read_end_tick) // 4
        assert lane.encoder.latent_count == full.shape[1]
        assert lane.encoder.latents.shape[2] <= 4 + segment_latents + 2
        previous_anchor = encoded["anchor_frame_ids"][-1]
    assert owner.video_vae.events.count(("start", 1)) == 1
    assert sum(count for kind, count in owner.video_vae.events if kind == "step") == pixels.shape[2] - 1


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
@pytest.mark.parametrize("shared_frame", [0, 2, 4])
def test_s16_h32_changed_shared_target_lookahead_discards_vae_lane(roots, shared_frame):
    _, raws = raw_segments(roots[0], decision_stride_steps=16)
    owner = condition()
    owner.get_condition(segment=raws[0])
    state = owner.episode_streams[0].encoder.state
    broken = copy.deepcopy(raws[1])
    broken["video"][:, shared_frame] ^= 1
    with pytest.raises(ValueError, match="boundary changed"):
        owner.get_condition(segment=broken)
    assert state.closed and owner.episode_streams == {}


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
def test_s16_h32_later_observations_do_not_change_an_earlier_decision(roots):
    _, raws = raw_segments(roots[0], decision_stride_steps=16)
    changed = copy.deepcopy(raws[0])
    later = torch.tensor(changed["video_frame_ticks"]) > 32
    assert later.any()
    changed["video"][:, later] = 255 - changed["video"][:, later]
    left_owner, right_owner = condition(), condition()
    left_encoded, left, _, _ = prepare_with_native_reference(left_owner, raws[0])
    right_encoded, right, _, _ = prepare_with_native_reference(right_owner, changed)
    torch.testing.assert_close(
        left_encoded["samples"][0]["video_latents"], right_encoded["samples"][0]["video_latents"], atol=0, rtol=0
    )
    assert not torch.equal(left["clean_video_tokens"], right["clean_video_tokens"])
    future_history = {**left, "clean_video_tokens": left["clean_video_tokens"] + 0.25}
    model = tiny_model()
    with torch.no_grad():
        a, b, c = (model(stream_segment=value) for value in (left, right, future_history))
    first = left["decisions"][0]
    for changed_output in (b, c):
        torch.testing.assert_close(
            a.video[: first["video_stop"]], changed_output.video[: first["video_stop"]], atol=0, rtol=0
        )
        torch.testing.assert_close(
            a.action[: first["action_stop"]], changed_output.action[: first["action_stop"]], atol=0, rtol=0
        )
    assert not torch.equal(a.video[first["video_stop"] :], c.video[first["video_stop"] :])


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
@pytest.mark.parametrize("decision_stride_steps", [None, 16])
def test_training_and_online_observations_share_identical_prefix_and_absolute_clock(roots, decision_stride_steps):
    _, raws = raw_segments(roots[0], decision_stride_steps=decision_stride_steps)
    owner = condition()
    online = EpisodeObservationEncoder(Tokenizer(), history_steps=48, action_horizon=32)
    latest = -1
    for raw in raws:
        encoded = owner.get_condition(segment=raw)["episode"]
        compact = owner.process_condition(episode=encoded)["stream_segment"]
        for index, (sample, anchor) in enumerate(zip(encoded["samples"], encoded["anchor_frame_ids"], strict=True)):
            tick = int(sample["window"].start)
            for now in range(latest + 1, tick + 1):
                online.add(now, image(now))
            observed, _ = online.finish(tick)
            latest = tick
            torch.testing.assert_close(observed, sample["video_latents"][:, :4], atol=0, rtol=0)
            assert online.prefix_global_ids == tuple(range(anchor - 3, anchor + 1))
            assert online.root_origin == -48
            if anchor in compact["prefix_frame_ids"]:
                frame = compact["prefix_frame_ids"].index(anchor)
                clock = compact["prefix_position_ids"][0, frame]
            else:
                frame = compact["clean_frame_ids"].index(anchor)
                clock = compact["clean_position_ids"][0, frame]
            torch.testing.assert_close(compact["state_position_ids"][0, index], clock, atol=0.002, rtol=0)
    assert sum(kind == "start" for kind, _ in online.tokenizer.events) == 1


@pytest.mark.parametrize("roots", [(149, 75)], indirect=True)
def test_three_segments_keep_one_vae_stream_through_partial_final_segment(roots):
    _, raws = raw_segments(roots[0])
    assert [len(raw["decisions"]) for raw in raws] == [2, 2, 1]
    owner, model, memory = condition(), tiny_model(), None
    anchors = []
    for raw in raws:
        encoded, compact, native, targets = prepare_with_native_reference(owner, raw)
        output = model(stream_segment=compact, stream_memory=memory)
        assert_compact_native_equivalent(model, compact, native, targets, output)
        anchors.extend(encoded["anchor_frame_ids"])
        memory = output.segment_memory
        assert memory.frame_ids == tuple(range(anchors[-1] - 3, anchors[-1] + 1))
        assert output.segment_stats["new_clean_latents"] == (2 if raw["reset"] or raw["final"] else 4)
    assert anchors == [3, 5, 7, 9, 11]
    assert owner.video_vae.events.count(("start", 1)) == 1


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
def test_condition_apply_releases_vae_lanes_and_resets_require_explicit_restart(roots):
    _, raws = raw_segments(roots[0])
    owner = condition()
    owner.get_condition(segment=raws[0])
    state = owner.episode_streams[0].encoder.state
    owner.to(device="cpu")
    assert owner.episode_streams == {} and state.closed
    with pytest.raises(ValueError, match="no matching"):
        owner.get_condition(segment=raws[1])
    assert owner.episode_streams == {}


class Runtime:
    def __init__(self, model, owner):
        self.model, self.condition_model = model, owner
        self.parallel_state = SimpleNamespace(dp_size=1, dp_rank=0, dp_group=None)

    def __call__(self, **kwargs):
        return self.model(**kwargs)


def trainer_for(model, owner):
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.episode_training = True
    trainer.segment_memories = {}
    trainer.base = SimpleNamespace(
        model=Runtime(model, owner),
        num_micro_batches=2,
        device=torch.device("cpu"),
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        on_step_begin=lambda **kwargs: None,
    )
    return trainer


@pytest.mark.parametrize("roots", [(75, 41)], indirect=True)
def test_trainer_weights_short_segments_by_actual_decisions_and_clears_final_state(roots, monkeypatch):
    _, raws = raw_segments(roots[0])
    assert [len(raw["decisions"]) for raw in raws] == [2, 1]
    torch.manual_seed(19)
    model, owner = tiny_model(), condition()
    expected_model, expected_condition = copy.deepcopy(model), condition()
    trainer = trainer_for(model, owner)
    monkeypatch.setattr(training, "use_parallel_state", lambda *args: nullcontext())
    trainer.on_step_begin(micro_batches=[{"segment": raw} for raw in raws])
    assert trainer.episode_global_decisions == 3
    actual_loss = 0
    for raw in raws:
        loss, _ = trainer.forward_backward_step({"segment": raw})
        actual_loss += loss.detach()
        if not raw["final"]:
            assert trainer.segment_memories[0].frame_ids[-1] == 5
    assert trainer.segment_memories == {} and owner.episode_streams == {}
    expected_loss, memory = 0, None
    for raw in raws:
        encoded = expected_condition.get_condition(segment=raw)
        prepared = expected_condition.process_condition(**encoded)["stream_segment"]
        output = expected_model(stream_segment=prepared, stream_memory=memory)
        expected_loss += sum(output.loss.values()) * len(raw["decisions"]) / 3
        memory = output.segment_memory
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss.detach(), atol=1e-5, rtol=2e-5)
    expected_parameters = dict(expected_model.named_parameters())
    for name, parameter in model.named_parameters():
        expected = expected_parameters[name].grad
        if parameter.grad is None or expected is None:
            assert parameter.grad is expected, name
        else:
            torch.testing.assert_close(parameter.grad, expected, atol=1e-4, rtol=3e-5, msg=name)


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
def test_trainer_gap_failure_discards_model_and_vae_state(roots, monkeypatch):
    _, raws = raw_segments(roots[0])
    owner, model = condition(), tiny_model()
    trainer = trainer_for(model, owner)
    monkeypatch.setattr(training, "use_parallel_state", lambda *args: nullcontext())
    trainer.episode_global_decisions = 4
    trainer.forward_backward_step({"segment": raws[0]})
    assert trainer.segment_memories and owner.episode_streams
    broken = copy.deepcopy(raws[1])
    broken["segment_id"] += 1
    with pytest.raises(Exception, match="order|gaps"):
        trainer.forward_backward_step({"segment": broken})
    assert trainer.segment_memories == {} and owner.episode_streams == {}


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
def test_trainer_missing_memory_failure_also_clears_existing_vae_lane(roots, monkeypatch):
    _, raws = raw_segments(roots[0])
    owner, model = condition(), tiny_model()
    trainer = trainer_for(model, owner)
    monkeypatch.setattr(training, "use_parallel_state", lambda *args: nullcontext())
    trainer.episode_global_decisions = 4
    trainer.forward_backward_step({"segment": raws[0]})
    state = owner.episode_streams[0].encoder.state
    trainer.segment_memories.clear()
    with pytest.raises(Exception, match="committed model memory"):
        trainer.forward_backward_step({"segment": raws[1]})
    assert trainer.segment_memories == {} and owner.episode_streams == {} and state.closed


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
@pytest.mark.parametrize("broken", [False, True])
def test_episode_validation_is_repeatable_preserves_rng_and_training_lane(roots, monkeypatch, broken):
    dataset, raws = raw_segments(roots[0])
    owner, model = condition(), tiny_model()
    trainer = trainer_for(model, owner)
    monkeypatch.setattr(training, "use_parallel_state", lambda *args: nullcontext())
    trainer.episode_global_decisions = 4
    trainer.forward_backward_step({"segment": raws[0]})
    training_streams = owner.episode_streams
    old_lane = training_streams[0]
    latent_before = old_lane.encoder.latents.clone()
    count_before = old_lane.encoder.frame_count
    memory = trainer.segment_memories[0]
    generator_before = {key: value.clone() for key, value in owner.rng_state_dict().items()}
    cpu_before = torch.get_rng_state().clone()
    flags_before = {id(module): module.training for module in model.modules()}
    trainer.base.validation_config = dict(stage="action", sample=dict(mode="joint"), validation=dict(seed=19))
    trainer.base.args = SimpleNamespace(
        model=SimpleNamespace(tokenizer_path="synthetic"),
        train=SimpleNamespace(
            global_rank=0,
            wandb=SimpleNamespace(enable=False),
            checkpoint=SimpleNamespace(output_dir=str(roots[0] / "validation-output")),
        ),
    )
    trainer.base.episode_split_manifest = {"split_id": "synthetic-episode-split"}
    validator = training.EpisodeValidator(trainer.base, dataset)
    published = []
    monkeypatch.setattr(validator, "_publish", lambda state, metrics, duration: published.append(dict(metrics)))

    def data_for_mode(mode, *, rank, world_size):
        assert rank == 0 and world_size == 1
        _, items = raw_segments(roots[0], mode=mode)
        if broken:
            items[0]["segment_id"] = 1
        return items

    monkeypatch.setattr(validator, "_dataset_for_mode", data_for_mode)
    state = SimpleNamespace(global_step=1, epoch=0)
    if broken:
        with pytest.raises(Exception, match="zero|preparation"):
            validator.evaluate(state)
        assert not published
    else:
        first, second = validator.evaluate(state), validator.evaluate(state)
        assert first == second and published == [first, second]
        for mode in ("fd", "id", "policy"):
            assert first[f"eval/{mode}/count"] == 4
            assert first[f"eval/{mode}/padding_count"] == 0
    assert owner.episode_streams is training_streams and owner.episode_streams[0] is old_lane
    assert old_lane.encoder.frame_count == count_before and not old_lane.encoder.state.closed
    assert torch.equal(old_lane.encoder.latents, latent_before)
    assert trainer.segment_memories[0] is memory
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert {id(module): module.training for module in model.modules()} == flags_before
    assert all(torch.equal(value, owner.rng_state_dict()[key]) for key, value in generator_before.items())
