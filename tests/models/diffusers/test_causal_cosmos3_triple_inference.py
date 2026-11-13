"""Triple inference stages: independent clocks, matched KV and honest artifacts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3_inference import feedback_fixture
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample, video_payload_sample
from veomni.data.causal_cosmos3.transforms.processing import Geometry
from veomni.models.diffusers.causal_cosmos3.inference import (
    CausalCosmos3Pipeline,
    load_trained_model,
    rollout_sample,
    sample_block,
    teacher_forcing,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class StagedVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(expert_layout="triple", patch_size=2)
        self.calls = []

    def forward(self, *, expert_operation, **inputs):
        self.calls.append((expert_operation, inputs))
        output = SimpleNamespace(
            video=None, action=None, text_cache=None, video_cache=None, action_condition_cache=None
        )
        if expert_operation == "text":
            output.text_cache = tuple(inputs["input_ids"].tolist())
        elif expert_operation == "action_condition":
            output.action_condition_cache = object()
        elif expert_operation == "video":
            output.video = torch.ones_like(inputs["video_tokens"])
            output.video_cache = SimpleNamespace(
                tokens=inputs["video_tokens"].clone(),
                times=inputs["video_timesteps"].clone(),
                text=tuple(inputs["input_ids"].tolist()),
            )
        elif expert_operation == "action":
            assert inputs["video_cache"] is not None
            assert inputs["video_cache"].text == tuple(inputs["input_ids"].tolist())
            output.action = torch.ones_like(inputs["action_tokens"])
        return output


def action_sample(task):
    sample = payload_sample("fd" if task == "forward_dynamics" else "id", actions=32)
    sample["mode"] = task
    return sample


def test_idm_partial_video_cache_matches_final_sigma_and_action_has_its_own_clock():
    sample = action_sample("inverse_dynamics")
    model = StagedVelocity()
    noise = {"video": torch.full_like(sample["video_latents"], 3), "action": torch.full_like(sample["action"], 4)}
    rng = torch.get_rng_state()
    generated = sample_block(
        model, sample, 0, partial_video_steps=4, video_stop_sigma=0.9, action_steps=3, initial_noise=noise
    )
    torch.testing.assert_close(generated["video_latents"][:, 1:], noise["video"][:, 1:] - 0.1)
    torch.testing.assert_close(generated["action"][:, 0], noise["action"][:, 0] - 0.999, atol=2e-5, rtol=0)
    assert not generated["action"][:, 1].any()
    assert generated["video_sigma"] == 0.9 and not generated["video_is_clean"]
    assert generated["expert_calls"] == dict(text=1, action_condition=1, video=5, action=3)
    assert generated["solver_seconds"] >= sum(generated["stage_seconds"].values())
    assert generated["memory_profile"] is None
    names = [name for name, _ in model.calls]
    assert names == ["text", "action_condition"] + ["video"] * 5 + ["action"] * 3
    video_calls = [inputs for name, inputs in model.calls if name == "video"]
    assert [float(inputs["video_timesteps"].max()) for inputs in video_calls] == pytest.approx(
        [1000, 975, 950, 925, 900]
    )
    action_calls = [inputs for name, inputs in model.calls if name == "action"]
    cache = action_calls[0]["video_cache"]
    assert all(inputs["video_cache"] is cache for inputs in action_calls)
    torch.testing.assert_close(cache.tokens[-2:], torch.full_like(cache.tokens[-2:], 2.9))
    assert float(cache.times.max()) == 900
    assert float(action_calls[0]["action_timesteps"].max()) == 999
    assert torch.equal(torch.get_rng_state(), rng)


def test_idm_generation_is_invariant_to_measured_future_video_and_unknown_actions():
    sample = action_sample("inverse_dynamics")
    altered = copy.deepcopy(sample)
    altered["video_latents"][:, 1:] += 300
    altered["action"] += 1000
    left = sample_block(StagedVelocity(), sample, 0, action_steps=2)
    right = sample_block(StagedVelocity(), altered, 0, action_steps=2)
    torch.testing.assert_close(left["action"], right["action"], atol=0, rtol=0)
    torch.testing.assert_close(left["video_latents"], right["video_latents"], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["video", "forward_dynamics"])
def test_video_and_fdm_skip_action_denoising(mode):
    sample = video_payload_sample() if mode == "video" else action_sample(mode)
    model = StagedVelocity()
    generated = sample_block(model, sample, 0, video_steps=3)
    assert generated["expert_calls"] == dict(text=1, action_condition=int(mode != "video"), video=3, action=0)
    assert generated["video_is_clean"] and generated["video_sigma"] == 0
    if mode == "video":
        assert generated["action"] is None
        assert all("action_tokens" not in inputs for _, inputs in model.calls)
    else:
        torch.testing.assert_close(generated["action"], sample["action"].masked_fill(~sample["action_mask"], 0))


def test_cfg_uses_distinct_prompt_caches_at_each_stage():
    model = StagedVelocity()
    generated = sample_block(
        model,
        action_sample("inverse_dynamics"),
        0,
        partial_video_steps=1,
        action_steps=2,
        guidance=2,
        negative_text_ids=torch.tensor([17]),
    )
    assert generated["expert_calls"] == dict(text=2, action_condition=2, video=4, action=4)
    action_calls = [inputs for name, inputs in model.calls if name == "action"]
    assert action_calls[0]["video_cache"] is not action_calls[1]["video_cache"]


def test_idm_teacher_forcing_does_not_decode_its_noisy_future():
    initial, processor, source = feedback_fixture(horizon=32)
    geometry = Geometry()
    sample, _ = rollout_sample(
        initial,
        initial.state,
        initial.image,
        source=source,
        processor=processor,
        geometry=geometry,
        tokenize=lambda _: [1, 2],
        history_blocks=1,
        valid=32,
    )
    sample["mode"] = "inverse_dynamics"
    sample["ground_truth_actions"] = initial.state.expand(32, -1).clone()
    sample["timestamps"] = torch.tensor([0.0])
    sample["condition_frame_indexes_vision"] = [0, 1, 2]

    class VAE:
        def __init__(self):
            self.encode_calls = 0

        def encode_pack(self, samples):
            self.encode_calls += 1
            return [{**samples[0], "video_latents": torch.zeros(2, 3, 2, 2)}]

        def decode(self, _):
            raise AssertionError("IDM must not decode a noisy future")

    pipe = CausalCosmos3Pipeline(
        StagedVelocity(),
        VAE(),
        processor,
        lambda _: [1, 2],
        dict(source=source, sampling=dict(partial_video_steps=1, action_steps=1)),
    )
    result = teacher_forcing(pipe, sample)
    assert result.video_role == "observation" and result.reconstruction is None
    assert result.video.shape == initial.image.shape
    assert result.actions.shape == (32, 55)
    assert result.blocks[0]["video_sigma"] == 0.9
    assert pipe.vae.encode_calls == result.profile["vae_encode_calls"] == 1
    assert result.profile["vae_chunk_decode_calls"] == result.profile["vae_reconstruction_decode_calls"] == 0
    assert result.profile["vae_reconstruction_decode_pack_seconds"] is None
    assert result.profile["vae_encode_memory"] is None
    chunk = result.profile["chunks"][0]
    assert chunk["vae_decode_seconds"] is None and chunk["memory"] is None and chunk["chunk_memory"] is None
    assert chunk["chunk_seconds"] >= chunk["solver_seconds"] >= 0


def test_triple_rejects_policy_and_joint_distilled_schedule():
    sample = action_sample("inverse_dynamics")
    with pytest.raises(ValueError, match="policy"):
        sample_block(StagedVelocity(), {**sample, "mode": "policy"}, 0)
    with pytest.raises(ValueError, match="joint fixed-step"):
        sample_block(StagedVelocity(), sample, 0, fixed_step_sampler_config=dict(t_list=[1, 0.5]))


def test_pure_video_pipeline_removes_action_payloads_before_model_forward(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    initial, processor, source = feedback_fixture(horizon=32)
    sample = dict(
        video_payload_sample(),
        video=initial.image.expand(-1, 9, -1, -1).clone(),
        timestamps=torch.tensor([0.0]),
        real_frame_intervals=32,
        window=SimpleNamespace(video_stride=4),
    )
    monkeypatch.setattr(inference, "read_teacher_window", lambda *args, **kwargs: sample)

    class VAE:
        def encode_pack(self, samples):
            assert "action" not in samples[0] and "states" not in samples[0]
            return [{**samples[0], "video_latents": torch.zeros(2, 3, 2, 2)}]

        def decode(self, _):
            return torch.zeros(3, 9, 32, 32, dtype=torch.uint8)

    model = StagedVelocity()
    pipe = CausalCosmos3Pipeline(
        model,
        VAE(),
        None,
        lambda _: [1, 2],
        dict(source=source, sampling=dict(video_steps=1)),
    )
    result = pipe.teacher_forcing("unused", 0, 0, 32, task="video")
    assert result.action_role == "none" and result.actions.shape == (0, 55)
    assert result.video.shape[1] == 9
    assert [name for name, _ in model.calls] == ["text", "video"]


def test_video_reader_does_not_access_action_columns_or_processor(tmp_path, monkeypatch):
    import numpy as np

    from veomni.data.causal_cosmos3.readers import observation

    episode = dict(episode_index=np.array([0]), source_fps=np.array([30.0]))
    monkeypatch.setattr(observation, "read_metadata", lambda *args: (dict(fps=30.0), episode))
    monkeypatch.setattr(observation, "camera_layout", lambda *args: ())
    seen = {}

    def plan(root, episode_index, start, horizon, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            fps=30.0,
            video_stride=2,
            real_intervals=16,
            video_frames=9,
            height=32,
            width=32,
            columns=kwargs["columns"],
        )

    monkeypatch.setattr(observation, "plan_window", plan)
    monkeypatch.setattr(
        observation,
        "read_window",
        lambda window: dict(
            video=torch.zeros(3, 9, 32, 32, dtype=torch.uint8),
            timestamps=torch.arange(17) / 30,
            task_index=0,
            window=window,
        ),
    )
    sample = observation.read_teacher_window(
        tmp_path,
        0,
        0,
        16,
        source=dict(viewpoint="ego_view", cameras={"head": "camera"}, columns={"invalid_action_key": True}),
        processor=object(),
        geometry=Geometry(),
        tokenize=lambda _: [1, 2],
        task="video",
        history=2,
        prompt="move",
    )
    assert seen["columns"] is None and seen["video_stride"] == 2
    assert seen["block_size"] == 16
    assert sample["mode"] == "video" and sample["conditioning_fps"] == 15
    assert sample["token_length"] == 8
    assert not {"action", "states", "action_mask", "state_mask"} & sample.keys()


def test_pure_video_loading_needs_no_statistics(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    model = StagedVelocity()
    model.weight_loading_report = dict(mode="complete")
    monkeypatch.setattr(inference, "load_trained_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(inference, "Cosmos3VideoVAE", lambda *args, **kwargs: object())
    monkeypatch.setattr(inference, "build_prompt_tokenizer", lambda *args: object())
    monkeypatch.setattr(inference, "tokenize_prompt", lambda *args, **kwargs: [1, 2])

    def action_processor(*args, **kwargs):
        raise AssertionError("Pure video must not initialize action statistics")

    monkeypatch.setattr(inference, "ActionProcessor", action_processor)
    pipe = CausalCosmos3Pipeline.from_pretrained(
        "unused",
        dict(task="video", source={}, vae="unused", tokenizer="unused"),
    )
    assert pipe.processor is None


@pytest.fixture
def real_triple_model():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import (
        Cosmos3Config,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ops_config_scope(eager_ops_config()):
            config = Cosmos3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                mrope_section=[2, 1, 1],
                latent_channels=2,
                action_interface="shared",
                action_dim=4,
                state_dim=4,
                expert_layout="triple",
                action_hidden_size=16,
                action_intermediate_size=32,
                action_num_hidden_layers=1,
                action_layer_map=[2],
            )
            yield Cosmos3Model(config).eval().requires_grad_(False)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "mode,guidance",
    [
        ("video", 1),
        ("forward_dynamics", 1),
        ("inverse_dynamics", 1),
        ("inverse_dynamics", 2),
    ],
)
def test_real_triple_sampler_stages_never_recompute_video_during_action(real_triple_model, mode, guidance):
    model = real_triple_model
    sample = video_payload_sample() if mode == "video" else action_sample(mode)
    stage, video_calls, video_layers, action_layers = [None], [], [], []

    def operation_hook(module, args, kwargs):
        stage[0] = kwargs["expert_operation"]

    def video_hook(module, args):
        assert stage[0] != "action", "cached action solve re-entered the video expert"
        video_calls.append(stage[0])

    def video_layer_hook(module, args):
        if args[-1]["operation"] == "video":
            assert stage[0] != "action", "cached action solve executed a video decoder layer"
            video_layers.append(stage[0])

    handles = [
        model.register_forward_pre_hook(operation_hook, with_kwargs=True),
        model.proj_in.register_forward_pre_hook(video_hook),
    ]
    for layer in model.layers:
        handles.append(layer.register_forward_pre_hook(video_layer_hook))
    for layer in model.action_expert.layers:
        handles.append(layer.register_forward_pre_hook(lambda module, args: action_layers.append(stage[0])))
    try:
        generated = sample_block(
            model,
            sample,
            0,
            video_steps=2,
            action_steps=2,
            partial_video_steps=2,
            guidance=guidance,
            negative_text_ids=torch.tensor([17]),
        )
    finally:
        for handle in handles:
            handle.remove()
    multiplier = 2 if guidance != 1 else 1
    assert len(video_calls) == (3 if mode == "inverse_dynamics" else 2) * multiplier
    assert len(video_layers) == len(video_calls) * len(model.layers)
    assert torch.isfinite(generated["video_latents"]).all()
    if mode == "video":
        assert not action_layers and generated["action"] is None
    elif mode == "inverse_dynamics":
        assert action_layers.count("action") == 2 * multiplier
        assert generated["video_sigma"] == 0.9
        assert torch.isfinite(generated["action"]).all()
    else:
        assert action_layers == ["action_condition"]


@pytest.mark.parametrize("task", ["video", "fd", "id"])
def test_real_tiny_triple_validation_cli(task, monkeypatch, capsys):
    import json
    import sys

    from scripts.causal_cosmos3 import validate_network

    monkeypatch.setattr(sys, "argv", ["validate_network", "--tiny", "--expert-layout", "triple", "--task", task])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        validate_network.main()
    finally:
        torch.set_num_threads(threads)
    report = json.loads(capsys.readouterr().out)
    assert report["expert_layout"] == "triple" and report["task"] == task
    assert report["tiny"] and report["loss"] >= 0


def test_real_meta_initialization_replaces_uninitialized_action_parameters(real_triple_model):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model

    with torch.device("meta"):
        model = Cosmos3Model(copy.deepcopy(real_triple_model.config))
    model.to_empty(device="cpu")
    # Constructor initialization on meta cannot initialize allocated storage.
    # Poison every action parameter to make missing parent initialization fail
    # deterministically, including private embeddings and Linear biases.
    with torch.no_grad():
        for parameter in model.action_expert.parameters():
            parameter.fill_(float("nan"))
    model.apply(model._init_weights)
    for name, parameter in model.action_expert.named_parameters():
        assert torch.isfinite(parameter).all(), name
    generated = sample_block(
        model.eval(),
        action_sample("inverse_dynamics"),
        0,
        partial_video_steps=1,
        action_steps=1,
    )
    assert torch.isfinite(generated["video_latents"]).all()
    assert torch.isfinite(generated["action"]).all()


def test_real_video_only_checkpoint_loads_without_action_heads(real_triple_model, tmp_path):
    from safetensors.torch import save_file

    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model

    config = copy.deepcopy(real_triple_model.config)
    config.action_interface = "none"
    config.state_dim = 0
    source = Cosmos3Model(config)
    config.save_pretrained(tmp_path)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    restored = load_trained_model(tmp_path, task="video", dtype=torch.float32)
    assert not hasattr(restored, "action_expert")
    generated = sample_block(restored, video_payload_sample(), 0, video_steps=1)
    assert torch.isfinite(generated["video_latents"]).all()
    assert generated["expert_calls"]["action_condition"] == generated["expert_calls"]["action"] == 0
    with pytest.raises(ValueError, match="55D"):
        load_trained_model(tmp_path, dtype=torch.float32)


def test_video_profile_counts_pack_encoding_once_and_saves_run_profile(tmp_path, monkeypatch):
    import json

    import veomni.models.diffusers.causal_cosmos3.inference as inference

    sample = dict(
        video_payload_sample(frames=5),
        video=torch.zeros(3, 17, 32, 32, dtype=torch.uint8),
        timestamps=torch.tensor([0.0]),
        real_frame_intervals=32,
        window=SimpleNamespace(video_stride=2),
    )
    monkeypatch.setattr(inference, "read_teacher_window", lambda *args, **kwargs: sample)

    class VAE:
        def __init__(self):
            self.encode_calls = self.decode_calls = 0

        def encode_pack(self, samples):
            self.encode_calls += 1
            return [{**samples[0], "video_latents": torch.zeros(2, 5, 2, 2)}]

        def decode(self, _):
            self.decode_calls += 1
            return torch.zeros(3, 17, 32, 32, dtype=torch.uint8)

    pipe = CausalCosmos3Pipeline(
        StagedVelocity(),
        VAE(),
        None,
        lambda _: [1, 2],
        dict(source={}, sampling=dict(video_steps=1)),
    )
    result = pipe.teacher_forcing("unused", 0, 0, 32, task="video")
    profile = result.profile
    assert pipe.vae.encode_calls == profile["vae_encode_calls"] == 1
    assert pipe.vae.decode_calls == profile["vae_chunk_decode_calls"] + profile["vae_reconstruction_decode_calls"] == 3
    assert profile["vae_encode_scope"] == "entire_pack_once" and len(profile["chunks"]) == 2
    assert profile["vae_encode_memory"] is None
    assert profile["pipeline_seconds"] >= profile["vae_encode_pack_seconds"] + profile["chunk_seconds"]
    assert all(chunk["memory"] is None and chunk["chunk_memory"] is None for chunk in profile["chunks"])
    assert all(chunk["vae_decode_seconds"] is not None for chunk in profile["chunks"])
    monkeypatch.setattr(inference, "write_video", lambda path, *args: path.touch())
    output = inference.save_result(result, tmp_path / "result", config=pipe.config, checkpoint="unused", identity={})
    assert json.loads((output / "run.json").read_text())["profile"] == profile


def test_memory_profiling_can_be_disabled_without_changing_sampling():
    sample = action_sample("inverse_dynamics")
    enabled = sample_block(StagedVelocity(), sample, 0, partial_video_steps=1, action_steps=1, profile_memory=True)
    disabled = sample_block(StagedVelocity(), sample, 0, partial_video_steps=1, action_steps=1, profile_memory=False)
    torch.testing.assert_close(enabled["video_latents"], disabled["video_latents"], atol=0, rtol=0)
    torch.testing.assert_close(enabled["action"], disabled["action"], atol=0, rtol=0)
    assert enabled["memory_profile"] is disabled["memory_profile"] is None
