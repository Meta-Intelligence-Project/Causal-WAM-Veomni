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
    rollout,
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
            video=None,
            action=None,
            text_cache=inputs.get("text_cache"),
            video_cache=None,
            action_condition_cache=inputs.get("action_condition_cache"),
            prefill_seconds={},
        )
        assert expert_operation in ("video", "action")
        if output.text_cache is None:
            output.text_cache = tuple(inputs["input_ids"].tolist())
            output.prefill_seconds["text_prefill"] = 0.0
        if (
            expert_operation == "video"
            and inputs["video_conditioning"] == (True,)
            and output.action_condition_cache is None
        ):
            output.action_condition_cache = object()
            output.prefill_seconds["action_condition_prefill"] = 0.0
        if expert_operation == "video":
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
    assert generated["expert_calls"] == dict(text=1, action_condition=0, video=5, action=3)
    assert generated["solver_seconds"] >= sum(generated["stage_seconds"].values())
    assert generated["memory_profile"] is None
    names = [name for name, _ in model.calls]
    assert names == ["video"] * 5 + ["action"] * 3
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
    assert all(inputs["video_conditioning"] == (False,) for _, inputs in model.calls)


@pytest.mark.parametrize("denoising_mode", ["partial_denoised", "full_denoised"])
def test_idm_generation_is_invariant_to_measured_future_video_and_unknown_actions(denoising_mode):
    sample = action_sample("inverse_dynamics")
    altered = copy.deepcopy(sample)
    altered["video_latents"][:, 1:] += 300
    altered["action"] += 1000
    left = sample_block(StagedVelocity(), sample, 0, action_steps=2, idm_denoising_mode=denoising_mode)
    right = sample_block(StagedVelocity(), altered, 0, action_steps=2, idm_denoising_mode=denoising_mode)
    torch.testing.assert_close(left["action"], right["action"], atol=0, rtol=0)
    torch.testing.assert_close(left["video_latents"], right["video_latents"], atol=0, rtol=0)


@pytest.mark.parametrize("complete_video", [False, True])
@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_idm_full_video_finishes_before_action_and_never_needs_completion(complete_video, shift):
    sample = action_sample("inverse_dynamics")
    noise = {"video": torch.full_like(sample["video_latents"], 3), "action": torch.full_like(sample["action"], 4)}
    model = StagedVelocity()
    generated = sample_block(
        model,
        sample,
        0,
        idm_denoising_mode="full_denoised",
        full_video_steps=4,
        full_video_shift=shift,
        action_steps=3,
        complete_video=complete_video,
        initial_noise=noise,
    )
    initial_sigma = shift * 0.999 / (1 + (shift - 1) * 0.999)
    torch.testing.assert_close(
        generated["video_latents"][:, 1:], noise["video"][:, 1:] - initial_sigma, atol=2e-5, rtol=0
    )
    torch.testing.assert_close(generated["video_latents"][:, :1], sample["video_latents"][:, :1], atol=0, rtol=0)
    assert generated["video_is_clean"] and generated["video_sigma"] == 0
    assert generated["action_condition_video_sigma"] == 0
    assert generated["idm_denoising_mode"] == "full_denoised"
    assert generated["expert_calls"] == dict(text=1, action_condition=0, video=5, action=3)
    assert [name for name, _ in model.calls] == ["video"] * 5 + ["action"] * 3
    cache = model.calls[5][1]["video_cache"]
    assert not cache.times.any()
    assert all(inputs["video_cache"] is cache for _, inputs in model.calls[5:])
    assert "full_video" in generated["stage_seconds"] and "video_completion" not in generated["stage_seconds"]


def test_idm_default_is_exactly_explicit_partial_denoised():
    sample = action_sample("inverse_dynamics")
    default = sample_block(StagedVelocity(), sample, 0, action_steps=2)
    explicit = sample_block(StagedVelocity(), sample, 0, action_steps=2, idm_denoising_mode="partial_denoised")
    for name in ("video_latents", "action"):
        torch.testing.assert_close(default[name], explicit[name], atol=0, rtol=0)
    assert default["expert_calls"] == explicit["expert_calls"]


def test_pipeline_and_direct_sampling_inherit_checkpoint_full_video_schedule():
    model = StagedVelocity()
    model.config.idm_denoising_mode = "full_denoised"
    model.config.full_video_steps = 2
    model.config.full_video_shift = 3.0
    pipe = CausalCosmos3Pipeline(model, None, None, lambda _: [1], dict(source={}))
    assert pipe.sampling_options["idm_denoising_mode"] == "full_denoised"
    assert pipe.sampling_options["full_video_steps"] == 2
    assert pipe.sampling_options["full_video_shift"] == 3.0
    generated = sample_block(model, action_sample("inverse_dynamics"), 0, action_steps=1)
    assert generated["expert_calls"]["video"] == 3
    assert generated["action_condition_video_sigma"] == 0


@pytest.mark.parametrize(
    "options,match",
    [
        ({"idm_denoising_mode": "unknown"}, "idm_denoising_mode"),
        ({"idm_denoising_mode": ""}, "idm_denoising_mode"),
        ({"full_video_steps": 0}, "full_video_steps"),
        ({"full_video_steps": True}, "full_video_steps"),
        ({"full_video_shift": float("nan")}, "full_video_shift"),
    ],
)
def test_idm_denoising_options_are_validated(options, match):
    with pytest.raises(ValueError, match=match):
        sample_block(StagedVelocity(), action_sample("inverse_dynamics"), 0, **options)


@pytest.mark.parametrize("shift", [1.0, 3.0])
@pytest.mark.parametrize("stop,completion_steps", [(0.9, 1), (0.35, 4), (0.0005, 4)])
def test_idm_completion_integrates_same_latent_after_action_without_changing_action(shift, stop, completion_steps):
    sample = action_sample("inverse_dynamics")
    noise = {"video": torch.full_like(sample["video_latents"], 3), "action": torch.full_like(sample["action"], 4)}
    options = dict(initial_noise=noise, shift=shift, video_stop_sigma=stop, partial_video_steps=2, action_steps=3)
    partial = sample_block(StagedVelocity(), sample, 0, **options)
    model = StagedVelocity()
    full = sample_block(model, sample, 0, complete_video=True, video_completion_steps=completion_steps, **options)
    torch.testing.assert_close(full["action"], partial["action"], atol=0, rtol=0)
    torch.testing.assert_close(
        full["video_latents"][:, 1:], torch.full_like(noise["video"][:, 1:], 2), atol=2e-5, rtol=0
    )
    torch.testing.assert_close(full["video_latents"][:, :1], sample["video_latents"][:, :1], atol=0, rtol=0)
    assert full["video_is_clean"] and full["video_sigma"] == 0
    assert full["action_condition_video_sigma"] == stop
    names = [name for name, _ in model.calls]
    assert names == ["video"] * 3 + ["action"] * 3 + ["video"] * completion_steps
    completion = [inputs for name, inputs in model.calls[-completion_steps:] if name == "video"]
    assert float(completion[0]["video_timesteps"].max()) == pytest.approx(stop * 1000)
    cache = next(inputs["video_cache"] for name, inputs in model.calls if name == "action")
    assert float(cache.times.max()) == pytest.approx(stop * 1000)
    torch.testing.assert_close(cache.tokens[-2:], torch.full_like(cache.tokens[-2:], 2 + stop), atol=2e-5, rtol=0)
    assert "video_completion" in full["stage_seconds"]


class LatentFeedbackVAE:
    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode_pack(self, samples):
        assert len(samples) == 1 and samples[0]["video"].shape[1] == 1
        self.encoded.append(samples[0]["video"].clone())
        return [{"video_latents": torch.full((2, 1, 2, 2), 0.25)}]

    def decode(self, latent):
        self.decoded.append(latent.clone())
        return torch.full((3, 1 + (latent.shape[1] - 1) * 4, 32, 32), 117, dtype=torch.uint8)


@pytest.mark.parametrize("task", ["inverse_dynamics", "video"])
@pytest.mark.parametrize("history,horizon", [(1, 67), (3, 131), (4, 1)])
@pytest.mark.parametrize("denoising_mode", ["partial_denoised", "full_denoised"])
def test_triple_rollout_reuses_exact_clean_latents_and_only_initial_observation(
    monkeypatch, task, history, horizon, denoising_mode
):
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    initial, processor, source = feedback_fixture(horizon=horizon)
    if task == "video":
        initial.state = initial.state_mask = initial.action_mask = None
        initial.video_stride = 2
        processor = None
    model, vae = StagedVelocity(), LatentFeedbackVAE()
    pipe = CausalCosmos3Pipeline(
        model,
        vae,
        processor,
        lambda _: [1, 2],
        dict(
            source=source,
            sampling=dict(
                history=history,
                partial_video_steps=1,
                full_video_steps=2,
                idm_denoising_mode=denoising_mode,
                video_completion_steps=3,
                action_steps=2,
                video_steps=3,
            ),
        ),
    )
    samples, generated = [], []
    original = inference.sample_block

    def capture(model, sample, block, **kwargs):
        samples.append(copy.deepcopy(sample))
        output = original(model, sample, block, **kwargs)
        generated.append(output)
        return output

    monkeypatch.setattr(inference, "sample_block", capture)
    rng = torch.get_rng_state()
    result = rollout(pipe, initial, task=task)
    assert torch.equal(torch.get_rng_state(), rng)
    stride, block = (2, 16) if task == "video" else (4, 32)
    chunks = (horizon // stride + 7) // 8 if task == "video" else (horizon + block - 1) // block
    assert result.video.shape == (3, 1 + horizon // stride, 32, 32)
    assert len(vae.encoded) == result.profile["vae_encode_calls"] == 1
    torch.testing.assert_close(vae.encoded[0], initial.image)
    assert len(vae.decoded) == result.profile["vae_chunk_decode_calls"] == chunks
    assert result.profile["vae_reconstruction_decode_calls"] == 0
    assert result.profile["vae_encode_scope"] == "initial_observation_once"
    limit = 1 + 2 * ((history + 1) // 2)
    for previous, current in zip(generated, samples[1:], strict=False):
        expected = previous["video_latents"][:, -limit:]
        torch.testing.assert_close(current["video_latents"][:, : expected.shape[1]], expected, atol=0, rtol=0)
        assert not current["video_latents"][:, expected.shape[1] :].any()
    for record in result.blocks:
        assert record["video_sigma"] == 0 and record["video_is_clean"]
        assert record["history_latents"] <= limit
        assert "latent" in record["feedback"]
        assert record["expert_calls"]["action"] == (2 if task == "inverse_dynamics" else 0)
    assert result.ground_truth_video is result.ground_truth_actions is None
    if task == "video":
        assert result.actions.shape[0] == 0 and result.action_role == "none"
        assert all("action_tokens" not in inputs and "state_tokens" not in inputs for _, inputs in model.calls)
    else:
        assert result.actions.shape == (horizon, 55)
        torch.testing.assert_close(torch.tensor(result.blocks[-1]["feedback_state"]), result.actions[-1])
        for previous, current in zip(result.blocks, result.blocks[1:], strict=False):
            assert previous["feedback_state"] == current["anchor"]
        sigma = 0.0 if denoising_mode == "full_denoised" else 0.9
        assert all(r["action_condition_video_sigma"] == sigma for r in result.blocks)
        if denoising_mode == "full_denoised":
            assert all("video_completion" not in r["stage_seconds"] for r in result.blocks)


@pytest.mark.parametrize("task", ["inverse_dynamics", "video"])
def test_triple_rollout_pipeline_dispatches_task_to_initial_reader(monkeypatch, task):
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    initial, processor, source = feedback_fixture(horizon=1)
    if task == "video":
        initial.state = initial.state_mask = initial.action_mask = None
        initial.video_stride = 2
        processor = None
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return initial

    monkeypatch.setattr(inference, "read_initial_observation", read)
    pipe = CausalCosmos3Pipeline(
        StagedVelocity(),
        LatentFeedbackVAE(),
        processor,
        lambda _: [1, 2],
        dict(
            source=source,
            sampling=dict(video_steps=1, partial_video_steps=1, action_steps=1, video_completion_steps=1),
        ),
    )
    result = pipe.rollout(dataset="unused", horizon=1, task=task)
    assert calls[0]["task"] == task and calls[0]["processor"] is processor
    assert "latent_feedback" in result.semantics


@pytest.mark.parametrize("fps,stride", [(12.0, 1), (24.0, 2)])
def test_video_rollout_uses_native_source_grid_and_saves_honest_artifacts(monkeypatch, tmp_path, fps, stride):
    import json

    import veomni.models.diffusers.causal_cosmos3.inference as inference

    initial, _, source = feedback_fixture(horizon=17)
    initial.fps = fps
    initial.state = initial.state_mask = initial.action_mask = None
    pipe = CausalCosmos3Pipeline(
        StagedVelocity(),
        LatentFeedbackVAE(),
        None,
        lambda _: [1, 2],
        dict(source=source, sampling=dict(video_steps=2)),
    )
    result = rollout(pipe, initial, task="video")
    assert result.video.shape[1] == 1 + 17 // stride
    assert result.video_fps == fps / stride
    assert float(result.video_timestamps[-1]) == pytest.approx((17 // stride) * stride / fps)
    monkeypatch.setattr(inference, "write_video", lambda path, *args: path.touch())
    inference.save_result(result, tmp_path, config=pipe.config, checkpoint="unused", identity={})
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["evaluation_scope"].startswith("model-only open-loop")
    assert metadata["valid_actions"] == 0 and metadata["action_role"] == "none"
    assert metadata["profile"]["vae_encode_calls"] == 1
    assert (tmp_path / "generated.mp4").exists()
    assert not (tmp_path / "ground_truth.mp4").exists() and not (tmp_path / "actions.svg").exists()


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


def test_lazy_prefill_time_is_reported_once_and_excluded_from_video_phase(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    clock = [0.0]

    class ClockedVelocity(StagedVelocity):
        def forward(self, **inputs):
            output = super().forward(**inputs)
            output.prefill_seconds = {name: 0.2 if name == "text_prefill" else 0.3 for name in output.prefill_seconds}
            clock[0] += 1.0 + sum(output.prefill_seconds.values())
            return output

    monkeypatch.setattr(inference.time, "monotonic", lambda: clock[0])
    model = ClockedVelocity()
    output = sample_block(model, action_sample("forward_dynamics"), 0, video_steps=2)
    assert output["expert_calls"] == dict(text=1, action_condition=1, video=2, action=0)
    assert output["stage_seconds"] == pytest.approx(
        dict(text_prefill=0.2, action_condition_prefill=0.3, video_denoise=2.0)
    )
    assert output["solver_seconds"] == pytest.approx(2.5)
    assert [name for name, _ in model.calls] == ["video", "video"]
    first, second = (inputs for _, inputs in model.calls)
    assert first["text_cache"] is first["action_condition_cache"] is None
    assert second["text_cache"] is not None and second["action_condition_cache"] is not None


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
    assert generated["expert_calls"] == dict(text=2, action_condition=0, video=4, action=4)
    action_calls = [inputs for name, inputs in model.calls if name == "action"]
    assert action_calls[0]["video_cache"] is not action_calls[1]["video_cache"]


@pytest.mark.parametrize("denoising_mode", ["partial_denoised", "full_denoised"])
def test_idm_teacher_forcing_decodes_only_clean_future(denoising_mode):
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
            if denoising_mode == "partial_denoised":
                raise AssertionError("IDM must not decode a noisy future")
            return torch.zeros(3, 9, 32, 32, dtype=torch.uint8)

    pipe = CausalCosmos3Pipeline(
        StagedVelocity(),
        VAE(),
        processor,
        lambda _: [1, 2],
        dict(
            source=source,
            sampling=dict(
                partial_video_steps=1, full_video_steps=2, action_steps=1, idm_denoising_mode=denoising_mode
            ),
        ),
    )
    result = teacher_forcing(pipe, sample)
    partial = denoising_mode == "partial_denoised"
    assert result.video_role == ("observation" if partial else "generated")
    assert (result.reconstruction is None) == partial
    assert result.video.shape == (initial.image.shape if partial else (3, 9, 32, 32))
    assert result.actions.shape == (32, 55)
    assert result.blocks[0]["video_sigma"] == (0.9 if partial else 0)
    assert pipe.vae.encode_calls == result.profile["vae_encode_calls"] == 1
    assert (
        result.profile["vae_chunk_decode_calls"]
        == result.profile["vae_reconstruction_decode_calls"]
        == int(not partial)
    )
    assert (result.profile["vae_reconstruction_decode_pack_seconds"] is None) == partial
    assert result.profile["vae_encode_memory"] is None
    chunk = result.profile["chunks"][0]
    assert (chunk["vae_decode_seconds"] is None) == partial
    assert chunk["memory"] is None and chunk["chunk_memory"] is None
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
    assert [name for name, _ in model.calls] == ["video"]


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
        assert action_layers == ["video"]


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


@pytest.mark.parametrize(
    "conditioning,video_enabled,text_enabled", [("text", False, True), ("action", True, False), ("both", True, True)]
)
def test_fdm_sampler_exposes_three_condition_modes(conditioning, video_enabled, text_enabled):
    model = StagedVelocity()
    generated = sample_block(model, action_sample("forward_dynamics"), 0, video_steps=2, fdm_conditioning=conditioning)
    assert generated["fdm_conditioning"] == conditioning
    assert generated["expert_calls"]["action_condition"] == int(video_enabled)
    assert all(
        inputs["video_conditioning"] == (video_enabled,) and inputs["text_conditioning"] == (text_enabled,)
        for _, inputs in model.calls
    )
    with pytest.raises(ValueError, match="fdm_conditioning"):
        sample_block(model, action_sample("forward_dynamics"), 0, fdm_conditioning="invalid")


@pytest.mark.parametrize("denoising_mode", ["partial_denoised", "full_denoised"])
def test_real_idm_state_changes_only_action_not_video(real_triple_model, denoising_mode):
    sample = action_sample("inverse_dynamics")
    altered = copy.deepcopy(sample)
    altered["states"][:, 0] += 7
    options = dict(
        partial_video_steps=2, full_video_steps=2, idm_denoising_mode=denoising_mode, action_steps=2, seed=29
    )
    left = sample_block(real_triple_model, sample, 0, **options)
    right = sample_block(real_triple_model, altered, 0, **options)
    torch.testing.assert_close(left["video_latents"], right["video_latents"], atol=0, rtol=0)
    assert not torch.equal(left["action"], right["action"])


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_real_idm_video_completion_preserves_action_and_clean_history(real_triple_model, backend, guidance):
    sample = action_sample("inverse_dynamics")
    options = dict(
        partial_video_steps=2,
        action_steps=2,
        video_completion_steps=3,
        seed=137,
        guidance=guidance,
        negative_text_ids=torch.tensor([1]),
        attention_backend=backend,
    )
    partial = sample_block(real_triple_model, sample, 0, **options)
    complete = sample_block(real_triple_model, sample, 0, complete_video=True, **options)
    torch.testing.assert_close(complete["action"], partial["action"], atol=0, rtol=0)
    torch.testing.assert_close(complete["video_latents"][:, :1], sample["video_latents"][:, :1], atol=0, rtol=0)
    assert torch.isfinite(complete["video_latents"]).all()
    assert not torch.equal(complete["video_latents"][:, 1:], partial["video_latents"][:, 1:])
    branches = 1 if guidance == 1 else 2
    assert complete["expert_calls"] == dict(text=branches, action_condition=0, video=6 * branches, action=2 * branches)
    assert complete["video_sigma"] == 0 and complete["action_condition_video_sigma"] == 0.9


@pytest.mark.parametrize("completion_steps", [0, True, 1.5])
def test_video_completion_rejects_invalid_step_counts(completion_steps):
    with pytest.raises(ValueError, match="video_completion_steps"):
        sample_block(StagedVelocity(), action_sample("inverse_dynamics"), 0, video_completion_steps=completion_steps)


@pytest.mark.parametrize("conditioning", ["text", "action"])
def test_real_fdm_inference_blocks_disabled_condition(real_triple_model, conditioning):
    sample = action_sample("forward_dynamics")
    altered = copy.deepcopy(sample)
    if conditioning == "text":
        altered["action"][:, 0] += 7
        altered["states"][:, 0] += 9
    else:
        altered["text_token_ids"] = altered["text_token_ids"] + 7
    options = dict(video_steps=2, seed=31, fdm_conditioning=conditioning)
    left = sample_block(real_triple_model, sample, 0, **options)
    right = sample_block(real_triple_model, altered, 0, **options)
    torch.testing.assert_close(left["video_latents"], right["video_latents"], atol=0, rtol=0)
