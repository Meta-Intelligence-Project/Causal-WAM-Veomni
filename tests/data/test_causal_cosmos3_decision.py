"""Train/serve decision-window alignment on real reader and native model math."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data import test_causal_cosmos3_pipeline as pipeline_tests
from tests.data.test_causal_cosmos3_pipeline import statistics
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers import test_causal_cosmos3_vae as vae_tests
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.data.causal_cosmos3 import ActionColumns, ActionDataset, ActionProcessor, CandidateConfig
from veomni.data.causal_cosmos3.datasets.dataset import MODES
from veomni.data.causal_cosmos3.datasets.validation import iter_validation_batches
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import dense_generation_mask, prepare_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.rtc_batch import RTCObservedBatchBuilder
from veomni.models.diffusers.causal_cosmos3.serving.statistics import BlockStatistics, Quantiles


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def roots(tmp_path, request):
    return pipeline_tests.roots.__wrapped__(tmp_path, request)


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    return vae_tests.encoder.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("past,horizon", [(0, 16), (48, 32)])
def test_training_vae_matches_online_groups_and_future_targets_cannot_change_history(encoder, past, horizon):
    observed = 1 + past // 16
    pixels = torch.randint(0, 256, (3, 1 + (past + horizon) // 4, 16, 16), dtype=torch.uint8)
    original = dict(video=pixels, layout_mode="decision_window")
    encoded = encoder.encode_pack([original])[0]["video_latents"]
    observed_pixels = pixels[:, : 1 + past // 4].float().unsqueeze(0) / 127.5 - 1
    state = encoder.stream_encode_start(observed_pixels[:, :, :1])
    for start in range(1, observed_pixels.shape[2], 4):
        encoder.stream_encode_step(state, observed_pixels[:, :, start : start + 4])
    expected = encoder.stream_encode_finish(state).squeeze(0)
    torch.testing.assert_close(encoded[:, :observed], expected, rtol=0, atol=0)
    future_changed = pixels.clone()
    future_changed[:, 1 + past // 4 :] = 255 - future_changed[:, 1 + past // 4 :]
    changed = encoder.encode_pack([dict(original, video=future_changed)])[0]["video_latents"]
    torch.testing.assert_close(changed[:, :observed], expected, rtol=0, atol=0)
    assert not torch.equal(encoded[:, observed:], changed[:, observed:])
    assert "video" in original and encoder.vae.training is False


def decision_dataset(root, *, past=48, horizon=32, mode="policy", **options):
    return ActionDataset(
        root,
        source="fixture",
        processor=ActionProcessor(statistics()),
        tokenize=lambda _: [1, 2],
        cameras={"head": "cam"},
        viewpoint="ego_view",
        video=False,
        **dict(layout_mode="decision_window", mode=mode, history_steps=past, action_horizon=horizon, **options),
    )


def latent_sample(ds, anchor):
    sample = ds.make_sample(anchor)
    window = sample["window"]
    sample["video_latents"] = torch.randn(2, ds.num_video_latents(window), window.height // 16, window.width // 16)
    return sample


@pytest.mark.parametrize("weights", [(1, 1, 1), (1, 4, 2), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_joint_task_sampling_matches_main_weights_and_restores_rng(roots, weights):
    ds = decision_dataset(
        roots[0], mode="joint", mode_weights=weights, seed=73, rank=2, prompt_formatter=lambda s: s["mode"]
    )
    reference = np.random.default_rng(np.random.SeedSequence([73, 2, 0, 0]))
    expected = reference.choice(MODES, size=24, p=np.asarray(weights) / sum(weights)).tolist()
    samples = [ds.make_sample(i) for i in range(24)]
    assert [sample["mode"] for sample in samples] == expected
    assert all(sample["prompt"] == sample["mode"] for sample in samples)
    assert set(expected) == {mode for mode, weight in zip(MODES, weights) if weight}
    saved = copy.deepcopy(ds.state_dict())
    expected_next = [ds.make_sample(i)["mode"] for i in range(6)]
    restored = decision_dataset(roots[0], mode="joint", mode_weights=weights, seed=73, rank=2)
    restored.load_state_dict(saved)
    assert [restored.make_sample(i)["mode"] for i in range(6)] == expected_next


def test_explicit_tasks_and_failed_reads_do_not_consume_training_rng(roots, monkeypatch):
    from veomni.data.causal_cosmos3.datasets import dataset

    ds = decision_dataset(roots[0], mode="joint")
    for mode in MODES:
        assert ds.make_sample(0, mode=mode)["mode"] == mode
    assert ds.rng is None
    ds.make_sample(0)
    before = copy.deepcopy(ds.state_dict())
    for mode in MODES:
        ds.make_sample(0, mode=mode, history=4)
    assert ds.state_dict() == before
    with pytest.raises(ValueError, match="explicit task"):
        ds.make_sample(0, mode="joint")

    def fail(*args, **kwargs):
        raise OSError("unreadable observation")

    monkeypatch.setattr(dataset, "read_decision_window", fail)
    with pytest.raises(OSError, match="unreadable observation"):
        ds.make_sample(0)
    assert ds.state_dict() == before


@pytest.mark.parametrize("modes", [(mode,) for mode in MODES] + [MODES])
def test_tasks_restore_their_conditions_and_keep_mixed_pack_visibility(roots, modes):
    samples = [latent_sample(decision_dataset(roots[0], mode=mode), 0) for mode in modes]
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    inputs, targets = condition.prepare(
        samples,
        sigmas=[torch.tensor([0.4])] * len(samples),
        epsilon_video=[torch.full_like(s["video_latents"], 4) for s in samples],
        epsilon_action=[torch.full_like(s["action"], 4) for s in samples],
    )
    layout = inputs["layout"]
    plan = prepare_attention(layout, "cpu", "grouped")
    mask = dense_generation_mask(layout, "cpu")
    for i, (sample, span) in enumerate(zip(samples, targets["spans"], strict=True)):
        fd, inverse = sample["mode"] == "forward_dynamics", sample["mode"] == "inverse_dynamics"
        observed, frames = sample["observed_latents"], sample["video_latents"].shape[1]
        assert sample["condition_frame_indexes_vision"] == list(range(frames if inverse else observed))
        assert sample["condition_frame_indexes_action"] == (list(range(32)) if fd else [])
        expected_video = sample["video_latents"].clone()
        if not inverse:
            expected_video[:, observed:] = 0.6 * expected_video[:, observed:] + 1.6
        torch.testing.assert_close(inputs["video_tokens"][span.current_video_rows], patchify_video(expected_video, 2))
        expected_action = sample["action"] if fd else 0.6 * sample["action"] + 1.6
        torch.testing.assert_close(
            inputs["action_tokens"][span.action_rows], expected_action.masked_fill(~sample["action_mask"], 0)
        )
        video_mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows]
        patches = video_mask.numel() // frames
        assert not video_mask[: observed * patches].any()
        assert bool(video_mask[observed * patches :].all()) is not inverse
        action_mask = targets["loss_masks"]["action_loss_mask"][span.action_rows]
        assert torch.equal(action_mask, sample["action_valid_steps"][:, None] & sample["action_mask"] & (not fd))
        assert not inputs["video_timesteps"][span.current_video_rows][: observed * patches].any()
        assert bool(inputs["video_timesteps"][span.current_video_rows][observed * patches :].eq(0).all()) is inverse
        assert bool(inputs["action_timesteps"][span.action_rows].eq(0).all()) is fd
        for row, query in enumerate(plan.gen_indexes.tolist()):
            if layout.sample_ids[query] != i:
                continue
            keys = torch.where(mask[row])[0].tolist()
            assert all(layout.sample_ids[key] == i for key in keys)
            if layout.roles[query] == 0:
                assert all(layout.roles[key] in (-1, 0) for key in keys)
                assert all(layout.block_ids[key] <= layout.block_ids[query] for key in keys)
            elif layout.roles[query] == 2:
                assert all(layout.roles[key] != 1 for key in keys)
            elif layout.roles[query] == 1:
                assert all(
                    mask[row, key] for key in range(span.current_video.start + observed * patches, span.state.stop)
                )
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    grouped = model(**inputs, targets=targets, attention_backend="grouped")
    dense = model(**inputs, attention_backend="dense")
    torch.testing.assert_close(grouped.video, dense.video, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(grouped.action, dense.action, atol=2e-6, rtol=2e-5)
    if len(samples) > 1:
        separate = []
        for sample in samples:
            single_inputs, single_targets = condition.prepare(
                [sample],
                sigmas=[torch.tensor([0.4])],
                epsilon_video=[torch.full_like(sample["video_latents"], 4)],
                epsilon_action=[torch.full_like(sample["action"], 4)],
            )
            separate.append(model(**single_inputs, targets=single_targets, attention_backend="grouped"))
        for name in grouped.loss:
            torch.testing.assert_close(grouped.loss[name], torch.stack([o.loss[name] for o in separate]).mean())
    grouped.video.retain_grad()
    grouped.action.retain_grad()
    sum(grouped.loss.values()).backward()
    assert not grouped.video.grad[~targets["loss_masks"]["video_loss_mask"]].any()
    assert not grouped.action.grad[~targets["loss_masks"]["action_loss_mask"]].any()


@pytest.mark.parametrize("past,horizon", [(0, 16), (48, 32), (1024, 32)])
@pytest.mark.parametrize("anchor", [0, 5, 47, -1])
def test_decisions_cover_episode_boundaries_with_one_current_anchor(roots, past, horizon, anchor):
    ds = decision_dataset(roots[0], past=past, horizon=horizon)
    length = int(ds.episodes["dataset_to_index"][0] - ds.episodes["dataset_from_index"][0])
    anchor = length - 2 if anchor == -1 else anchor
    sample = ds.make_sample(anchor)
    assert len(ds) == length - 1
    assert ds.rng is None
    assert sample["anchors"].shape == sample["states"].shape == (1, 55)
    assert sample["anchors"][0, 0] == anchor
    assert sample["states"][0, 0] == pytest.approx(anchor / 100)
    expected_ticks = np.clip(anchor + np.arange(-past, horizon + 1, 4), 0, length - 1)
    assert sample["video_frame_ticks"] == expected_ticks.tolist()
    assert sample["action_valid_steps"].sum() == min(horizon, length - 1 - anchor)
    assert sample["observed_latents"] == 1 + past // 16
    assert sample["state_latent_indexes"] == [past // 16]
    assert sample["action_start_frame_offset"] == past + 1
    assert sample["condition_frame_indexes_action"] == []
    assert sample["condition_frame_indexes_vision"] == list(range(1 + past // 16))
    assert sample["token_layout"]["clean_video"] == 0
    decoded = ds.processor.decode(sample["action"], sample["anchors"], sample["action_mask"])
    expected = torch.arange(anchor, anchor + horizon).clamp_max(length - 2) + 1
    torch.testing.assert_close(decoded[:, 0], expected.float(), atol=2e-5, rtol=0)


@pytest.mark.parametrize("offset", [-2, 0, 1, 2])
def test_decision_target_offsets_use_current_state_once(roots, offset):
    ds = decision_dataset(roots[0], columns=ActionColumns(target="state_unified", target_offset=offset))
    sample = ds.make_sample(7)
    anchor = sample["window"].start
    assert anchor == max(0, -offset) + 7
    decoded = ds.processor.decode(sample["action"], sample["anchors"], sample["action_mask"])
    torch.testing.assert_close(
        decoded[:, 0], torch.arange(anchor + offset, anchor + offset + 32).float(), atol=2e-5, rtol=0
    )


def test_video_reader_samples_history_and_future_in_one_causal_window(roots, monkeypatch):
    from veomni.data.causal_cosmos3.readers import lerobot

    queries = []

    def decode(path, times, region, **kwargs):
        queries.append(times.copy())
        region[:] = np.rint(times * 30).astype(np.uint8)[None, :, None, None]

    monkeypatch.setattr(lerobot, "decode_into", decode)
    ds = decision_dataset(roots[0])
    ds.video = True
    for anchor in (0, 13, len(ds) - 2):
        sample = ds.make_sample(anchor)
        assert sample["video"].shape[1] == 21
        torch.testing.assert_close(
            sample["video"][0, :, 0, 0], torch.tensor(sample["video_frame_ticks"], dtype=torch.uint8)
        )
        assert np.allclose(queries[-1], np.asarray(sample["video_frame_ticks"]) / 30)


def test_training_and_online_packing_have_identical_visibility_positions_and_history(roots):
    ds = decision_dataset(roots[0])
    sample = latent_sample(ds, 32)
    observed = sample["observed_latents"]
    builder = RTCObservedBatchBuilder(
        template=ds.processor.template,
        statistics=BlockStatistics(
            *(
                Quantiles(torch.tensor(q["low"]), torch.tensor(q["high"]), torch.tensor(q["valid"]))
                for q in (statistics()[name] for name in ("state", "action"))
            ),
            provenance={},
        ),
        tokenize=lambda _: [1, 2],
        prompt="move",
        image_shape=(3, sample["window"].height, sample["window"].width),
        history_steps=48,
        controller_hz=sample["source_fps"],
    )
    # A prepared VAE prefix is already available. Real observation records are
    # still required, independently of whether pixels need to be encoded here.
    from veomni.models.diffusers.causal_cosmos3.serving.rtc_batch import RTCObservation

    image = torch.zeros(builder.image_shape, dtype=torch.uint8)
    online = builder(
        state=sample["anchors"][0],
        state_mask=sample["state_mask"],
        action_mask=sample["action_mask"],
        image=image,
        tick=32,
        first_image=image,
        history=tuple(RTCObservation(t, image) for t in range(0, 32, 4)),
        prepared_vision_prefix=sample["video_latents"][:, :observed],
    )
    online["token_length"] = sample["token_length"]
    inputs = [
        pack_model_inputs(
            [s], [s["video_latents"]], [s["action"]], timesteps=[torch.zeros(s["video_latents"].shape[1])]
        )[0]
        for s in (sample, online)
    ]
    assert inputs[0]["layout"] == inputs[1]["layout"]
    for name in ("position_ids", "state_tokens", "state_mask", "action_mask"):
        torch.testing.assert_close(inputs[0][name], inputs[1][name], rtol=0, atol=0)
    plan = prepare_attention(inputs[0]["layout"], "cpu", "grouped")
    mask = dense_generation_mask(inputs[0]["layout"], "cpu")
    layout = inputs[0]["layout"]
    for row, packed in enumerate(plan.gen_indexes.tolist()):
        if layout.roles[packed] == 0:
            keys = torch.where(mask[row])[0].tolist()
            assert all(layout.roles[key] in (-1, 0) for key in keys)
            assert all(layout.block_ids[key] <= layout.block_ids[packed] for key in keys)


@pytest.mark.parametrize("past", [0, 48, 1024])
def test_one_future_sigma_is_independent_of_clean_history_length_and_resumable(roots, past):
    ds = decision_dataset(roots[0], past=past)
    sample = latent_sample(ds, 0)
    model = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32, shift=1)
    reference = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32, shift=1)
    sigma = model.sample_sigmas([sample])[0]
    expected = reference.time_sampler(
        1, device="cpu", dtype=torch.float32, generator=reference.time_generator, shifts=torch.ones(1)
    )
    observed = sample["observed_latents"]
    assert not sigma[:observed].any()
    torch.testing.assert_close(sigma[observed:], expected.expand(sample["block_size"]), rtol=0, atol=0)
    torch.testing.assert_close(model.time_generator.get_state(), reference.time_generator.get_state(), rtol=0, atol=0)
    saved = copy.deepcopy(model.rng_state_dict())
    expected_next = model.sample_sigmas([sample])[0]
    model.load_rng_state_dict(saved)
    torch.testing.assert_close(model.sample_sigmas([sample])[0], expected_next, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tail", [False, True])
def test_decision_training_loss_has_no_history_or_padding_supervision_and_backpropagates(roots, tail, mode):
    ds = decision_dataset(roots[0], mode=mode)
    sample = latent_sample(ds, len(ds) - 2 if tail else 0)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    inputs, targets = condition.prepare([sample], sigmas=[torch.tensor([0.4])])
    span = targets["spans"][0]
    mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows]
    assert bool(mask.any()) == (not tail and mode != "inverse_dynamics")
    action_mask = targets["loss_masks"]["action_loss_mask"][span.action_rows]
    assert action_mask.sum() == (0 if mode == "forward_dynamics" else 2 if tail else 32)
    if tail:
        assert not action_mask[2:].any()
    assert not targets["video_times"][0][: sample["observed_latents"]].any()
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    output = model(**inputs, targets=targets, attention_backend="grouped")
    assert bool(output.loss["video_loss"] == 0) == (tail or mode == "inverse_dynamics")
    assert bool(output.loss["action_loss"] == 0) == (mode == "forward_dynamics")
    loss = sum(output.loss.values())
    assert torch.isfinite(loss)
    assert bool(loss > 0) == (not tail or mode != "forward_dynamics")
    output.video.retain_grad()
    output.action.retain_grad()
    loss.backward()
    assert not output.video.grad[~targets["loss_masks"]["video_loss_mask"]].any()
    assert not output.action.grad[~targets["loss_masks"]["action_loss_mask"]].any()
    for head, supervised in ((model.proj_out, mask.any()), (model.action_proj_out, action_mask.any())):
        assert torch.isfinite(head.weight.grad).all()
        assert bool(head.weight.grad.abs().sum() > 0) == bool(supervised)
    changed = {name: value.detach().clone() for name, value in (("video", output.video), ("action", output.action))}
    changed["video"][~targets["loss_masks"]["video_loss_mask"]] = 1e6
    changed["action"][~targets["loss_masks"]["action_loss_mask"]] = 1e6
    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Output

    ignored = model.loss(Cosmos3Output(**changed), targets)
    torch.testing.assert_close(ignored["action_loss"], output.loss["action_loss"], rtol=0, atol=0)
    torch.testing.assert_close(ignored["video_loss"], output.loss["video_loss"], rtol=0, atol=0)


def test_validation_records_the_training_context_and_does_not_touch_rng(roots):
    ds = decision_dataset(roots[0], past=1024)
    first = next(iter_validation_batches(ds))
    second = next(iter_validation_batches(ds))
    assert first["record"] == second["record"]
    assert first["record"]["identity"]["history_steps"] == 1024
    assert first["record"]["history_blocks"] == 65
    assert first["batch"]["samples"][0]["observed_latents"] == 65
    assert ds.rng is None


def test_joint_validation_evaluates_each_task_without_consuming_training_rng(roots):
    ds = decision_dataset(
        roots[0], mode="joint", past=1024, candidates=CandidateConfig(max_actions=32, start_stride=50)
    )
    for initialized in (False, True):
        if initialized:
            ds.make_sample(0)
        before = copy.deepcopy(ds.state_dict())
        first, second = list(iter_validation_batches(ds)), list(iter_validation_batches(ds))
        assert len(first) == 3 * len(ds)
        assert [item["record"] for item in first] == [item["record"] for item in second]
        assert [item["record"]["mode"] for item in first] == [mode for mode in MODES for _ in ds.windows]
        for item in first:
            assert item["weight"] == 1
            sample = item["batch"]["samples"][0]
            assert sample["mode"] == item["record"]["mode"]
            assert sample["observed_latents"] == item["record"]["history_blocks"] == 65
        assert ds.state_dict() == before


def test_tracked_recipe_reaches_the_native_dataset_factory(roots, monkeypatch):
    from veomni.data.causal_cosmos3.datasets import pipeline

    repo = Path(__file__).parents[2]
    recipe = yaml.safe_load((repo / "configs/dit/causal_cosmos3_edge_decision.yaml").read_text())
    config = yaml.safe_load((repo / recipe["data"]["train_path"]).read_text())
    config["sources"] = [
        dict(
            name="fixture",
            roots=[str(roots[0])],
            statistics=statistics(),
            cameras={"head": "cam"},
            viewpoint="ego_view",
            read=dict(video=False),
        )
    ]
    monkeypatch.setattr(pipeline, "get_parallel_state", lambda: SimpleNamespace(sp_size=1, dp_rank=0))
    ds = pipeline.build_dataset(config, lambda _: [1, 2])
    assert config["sample"]["mode"] == "joint"
    assert config["sample"]["mode_weights"] == [1, 1, 1]
    sample = ds[13]
    assert sample["layout_mode"] == "decision_window"
    assert sample["history_steps"] == 1024 and sample["observed_latents"] == 65
    assert sample["action"].shape == (32, 55) and sample["anchors"][0, 0] == 13


@pytest.mark.parametrize(
    "options",
    [
        dict(layout_mode="bad"),
        dict(history_steps=-1),
        dict(history_steps=17),
        dict(action_horizon=17),
        dict(mode="unknown"),
        dict(mode_weights=[0, 0, 0]),
        dict(full_history=False),
        dict(candidates=CandidateConfig(max_actions=64)),
    ],
)
def test_invalid_decision_recipes_are_rejected_before_iteration(roots, options):
    values = dict(layout_mode="decision_window", mode="policy", history_steps=48, action_horizon=32)
    values.update(options)
    with pytest.raises(ValueError):
        ActionDataset(
            roots[0],
            source="fixture",
            processor=ActionProcessor(statistics()),
            tokenize=lambda _: [1, 2],
            cameras={"head": "cam"},
            viewpoint="ego_view",
            video=False,
            **values,
        )
