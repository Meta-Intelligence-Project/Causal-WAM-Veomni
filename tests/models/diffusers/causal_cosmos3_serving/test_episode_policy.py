"""Checkpoint history contracts and transactional online episode ownership."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers import test_causal_cosmos3_vae
from tests.models.diffusers.causal_cosmos3_serving.test_native import assets, image
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving import segment_cache
from veomni.models.diffusers.causal_cosmos3.serving.episode_streaming import EpisodeObservationEncoder
from veomni.models.diffusers.causal_cosmos3.serving.serving_model import ServingModelPolicy
from veomni.models.diffusers.causal_cosmos3.serving.window_streaming import ObservationWindowEncoder


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
encoder = test_causal_cosmos3_vae.encoder


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def policy(encoder, *, encoding="episode_stream", **options):
    template, mask, statistics = assets()
    adapter = SimpleNamespace(
        state_mask=mask,
        action_mask=mask,
        dofs={"left": 6, "right": 6},
        eef_pose_frame="world",
        T_world_base=None,
        available_control_modes=["joint"],
    )
    torch.manual_seed(193)
    model = Cosmos3Model(
        tiny_config(
            action_dim=55,
            state_dim=55,
            enable_tbptt=True,
            tbptt_segment_steps=64,
            history_encoding="window",
        )
    )
    # Exercise the policy boundary independently of the config constructor's
    # own validation, including externally supplied/corrupted checkpoint data.
    model.config.history_encoding = encoding
    return ServingModelPolicy(
        model,
        vae=encoder,
        tokenize=lambda text: [1, 2] if text else [3, 4],
        statistics=statistics,
        template=template,
        adapter=adapter,
        history_steps=32,
        num_steps=1,
        **options,
    )


def reset_request():
    return dict(operation="reset", state=torch.zeros(55), image=image(), image_size=[16] * 4, prompt="move")


def predict_request(tick=16):
    return dict(
        operation="predict",
        tick=tick,
        state=torch.zeros(55),
        image=image(tick),
        history=[dict(tick=index, image=image(index)) for index in range(tick)],
    )


def feed(owner, end=16):
    for tick in range(1, end + 1):
        owner.observe(dict(tick=tick, image=image(tick)))


def assert_cleared(owner, old_state=None):
    assert owner.builder is owner.first_image is None
    assert owner.episode_memories == {}
    assert not owner.window_encoder.frames
    assert owner.window_encoder.state is None
    assert owner.window_encoder.prefix_global_ids == ()
    if old_state is not None:
        assert old_state.closed and not old_state.feat_cache


@pytest.mark.parametrize("encoding", ["unknown", None, False])
def test_policy_rejects_invalid_checkpoint_history_encoding(encoder, encoding):
    with pytest.raises(ValueError, match="history_encoding must be"):
        policy(encoder, encoding=encoding)


@pytest.mark.parametrize(
    ("encoding", "encoder_class"),
    [("window", ObservationWindowEncoder), ("episode_stream", EpisodeObservationEncoder)],
)
def test_policy_selects_the_checkpoint_encoder_and_exports_its_contract(encoder, encoding, encoder_class):
    owner = policy(encoder, encoding=encoding)
    assert type(owner.window_encoder) is encoder_class
    assert owner.metadata["history_encoding"] == encoding
    assert owner.metadata["history_steps"] == 32
    assert owner.metadata["streaming_vae"] is True
    assert (owner.episode_prefix is None) == (encoding == "episode_stream")
    owner.close()


def test_episode_policy_rejects_graph_even_with_otherwise_valid_npu_options(encoder):
    # Reach the checkpoint-specific gate without requiring accelerator hardware.
    owner = policy(encoder)
    model = SimpleNamespace(
        config=owner.model.config,
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="npu"))]),
    )
    model.eval = lambda: model
    model.requires_grad_ = lambda _: model
    with pytest.raises(ValueError, match="Episode-stream checkpoints.*npu_replay_graph=False"):
        ServingModelPolicy(
            model,
            vae=encoder,
            tokenize=owner.tokenize,
            statistics=owner.statistics,
            template=owner.template,
            adapter=owner.adapter,
            history_steps=32,
            npu_replay_graph=True,
            attention_backend="npu_tnd",
        )
    owner.close()


def test_episode_policy_rejects_unaligned_execution_triggers(encoder):
    with pytest.raises(ValueError, match="execution triggers must align"):
        policy(encoder, execute_steps=8)


@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_episode_policy_real_vae_updates_only_observed_history_and_cfg_memories(encoder, guidance):
    owner = policy(encoder, guidance=guidance)
    try:
        cold = owner.infer(reset_request())
        serial, old_state = owner.episode_serial, owner.window_encoder.state
        assert cold["actions"].shape == (32, 55)
        assert owner.window_encoder.prefix_global_ids == (0, 1, 2)
        assert set(owner.episode_memories) == ({"cond"} if guidance == 1 else {"cond", "uncond"})
        feed(owner)
        owner.prepare_window(dict(anchor_tick=16))
        assert owner.window_encoder.encoded_tick == 16
        result = owner.infer(predict_request())
        assert result["actions"].shape == (32, 55)
        assert owner.episode_serial == serial and owner.window_encoder.state is old_state
        assert owner.window_encoder.prefix_global_ids == (1, 2, 3)
        assert owner.window_encoder.new_frame_ids == (3,)
        assert old_state.latent_chunks == []
        assert all(memory.frame_ids == (1, 2, 3) for memory in owner.episode_memories.values())
        assert result["streaming_timing"]["new_history_latents"] == 1
        assert result["inference_cache"]["cond"]["prefilled_history_latents"] == 1
    finally:
        owner.close()


def test_episode_policy_clear_invalidates_both_vae_and_model_contexts(encoder):
    owner = policy(encoder, guidance=2.0)
    owner.infer(reset_request())
    old_state, serial = owner.window_encoder.state, owner.episode_serial
    old_memory = owner.episode_memories["cond"]
    owner.clear()
    assert_cleared(owner, old_state)
    assert owner.episode_serial > serial
    with pytest.raises(ValueError, match="Reset with a fresh observation"):
        owner.infer(predict_request())
    owner.infer(reset_request())
    assert owner.episode_memories["cond"].context_id != old_memory.context_id
    owner.close()


@pytest.mark.parametrize("failure_stage", ["finish", "builder", "compact", "physical"])
def test_episode_infer_errors_after_vae_progress_clear_the_transaction_and_allow_reset(
    encoder, monkeypatch, failure_stage
):
    owner = policy(encoder)
    original = owner.infer(reset_request())
    feed(owner)
    old_state, serial = owner.window_encoder.state, owner.episode_serial

    def fail(*args, **kwargs):
        assert owner.window_encoder.encoded_tick == 16
        raise RuntimeError("injected post-VAE failure")

    with monkeypatch.context() as patch:
        if failure_stage == "finish":
            native_finish = owner.window_encoder.finish

            def failed_finish(anchor):
                native_finish(anchor)
                return fail()

            patch.setattr(owner.window_encoder, "finish", failed_finish)
        elif failure_stage == "builder":
            patch.setattr(type(owner.builder), "__call__", fail)
        elif failure_stage == "compact":
            patch.setattr(segment_cache, "compact_stream_payload", fail)
        else:
            patch.setattr(type(owner.template), "decode_action_delta", fail)
        with pytest.raises(RuntimeError, match="injected post-VAE failure"):
            owner.infer(predict_request())
    assert_cleared(owner, old_state)
    assert owner.episode_serial > serial
    with pytest.raises(ValueError, match="Reset with a fresh observation"):
        owner.infer(predict_request())
    recovered = owner.infer(reset_request())
    torch.testing.assert_close(torch.from_numpy(recovered["actions"]), torch.from_numpy(original["actions"]))
    owner.close()


@pytest.mark.parametrize("entry", ["observe", "prepare_window"])
def test_episode_background_vae_failure_clears_policy_and_requires_fresh_reset(encoder, monkeypatch, entry):
    owner = policy(encoder)
    owner.infer(reset_request())
    old_state = owner.window_encoder.state
    if entry == "observe":
        owner.prepare_window(dict(anchor_tick=16))
        feed(owner, end=15)
    else:
        feed(owner)

    def fail(*args, **kwargs):
        raise RuntimeError("injected native VAE failure")

    with monkeypatch.context() as patch:
        patch.setattr(encoder, "stream_encode_step", fail)
        with pytest.raises(RuntimeError, match="injected native VAE failure"):
            if entry == "observe":
                owner.observe(dict(tick=16, image=image(16)))
            else:
                owner.prepare_window(dict(anchor_tick=16))
    assert_cleared(owner, old_state)
    with pytest.raises(ValueError, match="Reset before sending observations"):
        owner.observe(dict(tick=16, image=image(16)))
    owner.infer(reset_request())
    owner.close()


@pytest.mark.parametrize("entry", ["observe", "prepare_window"])
def test_episode_background_bad_requests_fail_closed_before_next_inference(encoder, entry):
    owner = policy(encoder)
    owner.infer(reset_request())
    old_state = owner.window_encoder.state
    with pytest.raises(ValueError):
        if entry == "observe":
            owner.observe(dict(tick=0, image=image(7)))
        else:
            owner.prepare_window(dict(anchor_tick=1))
    assert_cleared(owner, old_state)
    owner.close()
