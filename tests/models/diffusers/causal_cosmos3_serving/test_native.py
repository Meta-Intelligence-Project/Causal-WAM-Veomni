"""Native VAE/model integration, causal windows and gradient-safe RTC replay."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers import test_causal_cosmos3_vae
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import dense_generation_mask, prepare_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix, prefix_weights
from veomni.models.diffusers.causal_cosmos3.serving.rtc_batch import RTCObservation, RTCObservedBatchBuilder
from veomni.models.diffusers.causal_cosmos3.serving.sampling import sample_decision
from veomni.models.diffusers.causal_cosmos3.serving.serving_model import ServingModelPolicy
from veomni.models.diffusers.causal_cosmos3.serving.statistics import BlockStatistics, Quantiles
from veomni.models.diffusers.causal_cosmos3.serving.window_streaming import ObservationWindowEncoder


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
encoder = test_causal_cosmos3_vae.encoder


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def assets():
    template = ActionTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[:2] = True
    quantiles = Quantiles(torch.full((55,), -2.0), torch.full((55,), 2.0), mask)
    stats = BlockStatistics(
        quantiles,
        copy.deepcopy(quantiles),
        dict(geometry=dict(actions_per_block=32, video_stride=4, temporal_compression_factor=4)),
    )
    return template, mask, stats


def image(tick=0):
    return torch.full((3, 16, 16), tick % 255, dtype=torch.uint8)


def window(*, history_steps=32, horizon=32, tick=0):
    template, mask, stats = assets()
    builder = RTCObservedBatchBuilder(
        template=template,
        statistics=stats,
        tokenize=lambda _: [1, 2],
        prompt="move",
        image_shape=(3, 16, 16),
        history_steps=history_steps,
        action_horizon=horizon,
    )
    obs = 1 + history_steps // 16
    torch.manual_seed(73)
    prefix = torch.randn(2, obs, 1, 1)
    sample = builder(
        state=torch.zeros(55),
        state_mask=mask,
        action_mask=mask,
        image=image(tick),
        tick=tick,
        history=tuple(RTCObservation(index, image(index)) for index in range(tick)),
        first_image=image(),
        prepared_vision_prefix=prefix,
    )
    return builder, sample


def test_decision_layout_has_only_measured_visual_history_and_one_fresh_state():
    _, sample = window(history_steps=48, horizon=16, tick=16)
    sample["token_length"] = 2 + 5 + 16 + 1
    inputs, _, _ = pack_model_inputs(
        [sample], [sample["video_latents"]], [sample["action"]], timesteps=[torch.zeros(5)]
    )
    layout = inputs["layout"]
    assert layout.decision_windows == (4,)
    assert list(layout.modalities).count(1) == 5
    assert list(layout.modalities).count(2) == 16
    assert list(layout.modalities).count(3) == 1
    mask = dense_generation_mask(layout, "cpu")
    plan = prepare_attention(layout, "cpu", "grouped", max_kv_tokens=31)
    recovered = torch.zeros_like(mask)
    for chunk in plan.chunks:
        qs = ks = 0
        for qe, ke in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            recovered[chunk.query_indexes[qs:qe, None], chunk.kv_indexes[None, ks:ke]] = True
            qs, ks = qe, ke
    torch.testing.assert_close(recovered, mask)
    for row, packed in enumerate(plan.gen_indexes.tolist()):
        for key in torch.where(mask[row])[0].tolist():
            if layout.roles[packed] == 0:
                assert layout.roles[key] in (-1, 0)
                if layout.roles[key] == 0:
                    assert layout.frame_ids[key] <= layout.frame_ids[packed]
            if layout.roles[packed] == 3:
                assert layout.roles[key] in (-1, 3)
    # State time matches the last measured latent; action time starts at P+1.
    state = plan.modality_indexes[3]
    videos = plan.modality_indexes[1]
    assert inputs["position_ids"][0, state] == inputs["position_ids"][0, videos[3]]
    assert inputs["position_ids"].dtype == torch.float32


@pytest.mark.parametrize("cache_mode", ["off", "plans", "full"])
@pytest.mark.parametrize("guided", [False, True])
def test_native_sampler_cached_outputs_and_input_vjp_match(cache_mode, guided):
    _, sample = window()
    torch.manual_seed(42)
    model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval().requires_grad_(False)
    previous = torch.zeros(16, 55)
    previous[:, 0] = torch.linspace(0.05, 0.2, 16)
    rtc = RTCPrefix(previous, 8) if guided else None
    expected = sample_decision(model, sample, steps=3, cache_mode="off", rtc=rtc)
    actual = sample_decision(model, sample, steps=3, cache_mode=cache_mode, rtc=rtc)
    torch.testing.assert_close(actual["action"], expected["action"], atol=2e-6, rtol=3e-5)
    torch.testing.assert_close(actual["video_latents"], expected["video_latents"], atol=2e-6, rtol=3e-5)
    assert all(parameter.grad is None for parameter in model.parameters())
    if guided:
        assert actual["rtc_vjp_calls"] == 3
        torch.testing.assert_close(
            actual["action"][:8], actual["prepared_prefix"].normalized_targets[:8], rtol=0, atol=0
        )
    if cache_mode == "full":
        assert actual["inference_cache"]["cond"] == dict(prefill_calls=1, replay_calls=3, active_tokens=34)
    assert not actual["action"][:, 2:].any()
    torch.testing.assert_close(actual["video_latents"][:, :3], sample["video_latents"][:, :3], rtol=0, atol=0)


def test_rtc_keeps_joint_video_input_gradient_and_decaying_prefix():
    weights = prefix_weights(8, 16, 32)
    assert torch.equal(weights[:8], torch.ones(8))
    assert (weights[8:15] > weights[9:16]).all() and not weights[16:].any()
    from veomni.models.diffusers.causal_cosmos3.serving.rtc import guided_velocity

    x = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    weight = torch.tensor([0.0, 1.0])
    result = guided_velocity(x, lambda value: torch.stack((value[0], value.sum())), 0.5, target, weight, 5.0)
    ordinary = torch.stack((x[0], x.sum()))
    assert result[0] != ordinary[0]  # Action error propagates to the video input.


def test_window_rejects_missing_real_history_and_preserves_latest_anchor():
    builder, sample = window(tick=16)
    mask = sample["action_mask"]
    with pytest.raises(ValueError, match="missing real historical"):
        builder(
            state=torch.ones(55),
            state_mask=mask,
            action_mask=mask,
            image=image(16),
            tick=16,
            first_image=image(),
            prepared_vision_prefix=sample["video_latents"][:, :3],
        )
    torch.testing.assert_close(sample["anchors"], torch.zeros(1, 55))
    assert sample["states"].shape == (1, 55) and len(sample["action"]) == 32


def test_real_streaming_vae_isolates_interleaved_windows_and_offline_calls(encoder):
    torch.manual_seed(7)
    a = torch.rand(1, 3, 9, 16, 16) * 2 - 1
    b = torch.rand(1, 3, 5, 16, 16) * 2 - 1
    first, second = encoder.stream_encode_start(a[:, :, :1]), encoder.stream_encode_start(b[:, :, :1])
    encoder.stream_encode_step(first, a[:, :, 1:5])
    encoder(torch.randint(0, 256, (3, 5, 16, 16), dtype=torch.uint8))
    encoder.stream_encode_step(second, b[:, :, 1:])
    encoder.stream_encode_step(first, a[:, :, 5:])
    actual_a, actual_b = encoder.stream_encode_finish(first), encoder.stream_encode_finish(second)
    for value, actual in ((a, actual_a), (b, actual_b)):
        expected = encoder.vae.encode(value).latent_dist.mode()
        expected = (expected - encoder.latent_mean) * encoder.latent_inv_std
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    assert first.closed and second.closed and not first.feat_cache and not second.feat_cache
    with pytest.raises(ValueError, match="closed"):
        encoder.stream_encode_step(first, a[:, :, 1:5])


def test_streaming_vae_failure_aborts_only_its_own_state(encoder, monkeypatch):
    frames = torch.zeros(1, 3, 5, 16, 16)
    first, second = encoder.stream_encode_start(frames[:, :, :1]), encoder.stream_encode_start(frames[:, :, :1])
    original = encoder.vae.encoder.forward
    with monkeypatch.context() as patch:
        patch.setattr(
            encoder.vae.encoder,
            "forward",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("chunk failure")),
        )
        with pytest.raises(RuntimeError, match="chunk failure"):
            encoder.stream_encode_step(first, frames[:, :, 1:])
    assert first.closed and not second.closed
    monkeypatch.setattr(encoder.vae.encoder, "forward", original)
    encoder.stream_encode_step(second, frames[:, :, 1:])
    assert encoder.stream_encode_finish(second).shape[2] == 2


def test_long_repeated_first_window_preencodes_before_tick16_with_real_vae(encoder):
    stream = ObservationWindowEncoder(encoder, history_steps=1024, action_horizon=32, history_padding="repeat_first")
    stream.add(0, image())
    stream.prepare(16)
    assert stream.count == 253
    for tick in range(1, 17):
        stream.add(tick, image(tick))
    ticks = stream.window.observation_ticks
    pixels = torch.stack([image(index) for index in ticks], dim=1).unsqueeze(0).float() / 127.5 - 1
    expected = encoder.vae.encode(pixels).latent_dist.mode()
    expected = ((expected - encoder.latent_mean) * encoder.latent_inv_std).squeeze(0)
    actual, timing = stream.finish(16)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    assert actual.shape == (2, 65, 1, 1) and timing["sampled_frames_encoded_before_predict"] == 257


def test_native_policy_real_vae_and_rtc_preserve_physical_committed_prefix(encoder):
    template, mask, stats = assets()
    adapter = SimpleNamespace(
        state_mask=mask,
        action_mask=mask,
        dofs={"left": 6, "right": 6},
        eef_pose_frame="world",
        T_world_base=None,
        available_control_modes=["joint"],
    )
    torch.manual_seed(17)
    model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval().requires_grad_(False)
    policy = ServingModelPolicy(
        model,
        vae=encoder,
        tokenize=lambda _: [1, 2],
        statistics=stats,
        template=template,
        adapter=adapter,
        history_steps=32,
        enable_rtc=True,
        num_steps=2,
    )
    reset = policy.infer(
        dict(operation="reset", state=torch.zeros(55), image=image(), image_size=[16] * 4, prompt="move")
    )
    assert reset["actions"].shape == (32, 55)
    for tick in range(1, 17):
        policy.observe(dict(tick=tick, image=image(tick)))
    prefix = torch.from_numpy(reset["actions"][16:].copy())
    state = torch.zeros(55)
    state[0] = 0.7
    result = policy.infer(
        dict(
            operation="predict",
            tick=16,
            revision=0,
            state=state,
            image=image(16),
            history=[dict(tick=index, image=image(index)) for index in range(16)],
            rtc=dict(previous_actions=prefix, delay_steps=8),
        )
    )
    torch.testing.assert_close(torch.from_numpy(result["actions"][:8]), prefix[:8], atol=0, rtol=0)
    assert result["rtc_vjp_calls"] == 2
    policy.clear()
    assert policy.builder is None and not policy.window_encoder.frames


def test_local_model_planner_uses_native_streaming_builder_and_measured_anchor(encoder):
    from veomni.models.diffusers.causal_cosmos3.serving.rtc_session import make_model_planner

    template, mask, statistics = assets()
    builder = RTCObservedBatchBuilder(
        template=template,
        statistics=statistics,
        tokenize=lambda _: [1, 2],
        prompt="move",
        image_shape=(3, 16, 16),
        history_steps=32,
        vae=encoder,
    )
    model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval().requires_grad_(False)
    planner = make_model_planner(model, builder, template=template, horizon=32, num_steps=2)
    action = planner(
        state=torch.ones(55), state_mask=mask, action_mask=mask, image=image(), tick=0, history=(), rtc=None
    )
    assert action.shape == (32, 55) and torch.isfinite(action).all()
    assert not action[:, 2:].any()
