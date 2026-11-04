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
from veomni.data.causal_cosmos3.datasets.validation import iter_validation_batches
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import dense_generation_mask, prepare_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
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


def decision_dataset(root, *, past=48, horizon=32, **options):
    return ActionDataset(
        root,
        source="fixture",
        processor=ActionProcessor(statistics()),
        tokenize=lambda _: [1, 2],
        cameras={"head": "cam"},
        viewpoint="ego_view",
        video=False,
        **dict(layout_mode="decision_window", mode="policy", history_steps=past, action_horizon=horizon, **options),
    )


def latent_sample(ds, anchor):
    sample = ds.make_sample(anchor)
    window = sample["window"]
    sample["video_latents"] = torch.randn(2, ds.num_video_latents(window), window.height // 16, window.width // 16)
    return sample


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


@pytest.mark.parametrize("tail", [False, True])
def test_decision_training_loss_has_no_history_or_padding_supervision_and_backpropagates(roots, tail):
    ds = decision_dataset(roots[0])
    sample = latent_sample(ds, len(ds) - 2 if tail else 0)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    inputs, targets = condition.prepare([sample], sigmas=[torch.tensor([0.4])])
    span = targets["spans"][0]
    mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows]
    assert bool(mask.any()) is not tail
    action_mask = targets["loss_masks"]["action_loss_mask"][span.action_rows]
    assert action_mask.sum() == (2 if tail else 32)
    if tail:
        assert not action_mask[2:].any()
    assert not targets["video_times"][0][: sample["observed_latents"]].any()
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    output = model(**inputs, targets=targets, attention_backend="grouped")
    assert bool(output.loss["video_loss"] == 0) is tail
    loss = sum(output.loss.values())
    assert torch.isfinite(loss) and loss > 0
    output.video.retain_grad()
    output.action.retain_grad()
    loss.backward()
    assert not output.video.grad[~targets["loss_masks"]["video_loss_mask"]].any()
    assert not output.action.grad[~targets["loss_masks"]["action_loss_mask"]].any()
    assert torch.isfinite(model.action_proj_out.weight.grad).all()
    assert model.action_proj_out.weight.grad.abs().sum() > 0
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
        dict(mode="joint"),
        dict(mode="inverse_dynamics"),
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
