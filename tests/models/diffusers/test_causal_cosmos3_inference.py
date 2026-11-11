"""Inference contracts: causality, frozen conditions, CFG and physical feedback."""

import copy
import math
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import fixed_inputs, tiny_config
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.data.causal_cosmos3.readers.observation import InitialObservation, make_sample, validate_feedback
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.fixed_step import FixedStepSampler
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.unipc import FlowUniPCMultistepScheduler
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import (
    CausalCosmos3Pipeline,
    load_trained_model,
    rollout,
    sample_block,
    teacher_forcing,
)
from veomni.models.diffusers.causal_cosmos3.rollout_profiler import RolloutProfiler
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class ConstantVelocity(torch.nn.Module):
    def __init__(self, cfg=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(patch_size=2)
        self.cfg = cfg
        self.times = []

    def forward(self, **inputs):
        self.times.append((inputs["video_timesteps"].clone(), inputs["action_timesteps"].clone()))
        value = float(inputs["input_ids"][0]) if self.cfg else 1.0
        return SimpleNamespace(
            video=torch.full_like(inputs["video_tokens"], value),
            action=torch.full_like(inputs["action_tokens"], value),
        )


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("fixed_step", [None, dict(t_list=[0.999, 0.5])])
def test_sampling_freezes_other_blocks_conditions_and_invalid_channels(task, steps, fixed_step):
    sample = payload_sample(task, actions=64)
    model = ConstantVelocity()
    noise = {"video": torch.ones_like(sample["video_latents"]) * 3, "action": torch.ones_like(sample["action"]) * 4}
    before = copy.deepcopy(sample)
    rng = torch.get_rng_state()
    generated = sample_block(model, sample, 0, steps=steps, initial_noise=noise, fixed_step_sampler_config=fixed_step)
    torch.testing.assert_close(generated["video_latents"][:, 0], sample["video_latents"][:, 0], atol=0, rtol=0)
    torch.testing.assert_close(generated["video_latents"][:, 3:], sample["video_latents"][:, 3:], atol=0, rtol=0)
    assert not generated["action"][:, 1].any()
    torch.testing.assert_close(
        generated["action"][32:], sample["action"][32:].masked_fill(~sample["action_mask"], 0), atol=0, rtol=0
    )
    if task == "id":
        torch.testing.assert_close(generated["video_latents"], sample["video_latents"], atol=0, rtol=0)
    elif fixed_step is None:
        # Reference UniPC schedule starts at sigma=0.999; constant flow integrates exactly.
        torch.testing.assert_close(
            generated["video_latents"][:, 1:3], noise["video"][:, 1:3] - 0.999, atol=2e-5, rtol=0
        )
    if task == "fd":
        torch.testing.assert_close(
            generated["action"], sample["action"].masked_fill(~sample["action_mask"], 0), atol=0, rtol=0
        )
    elif fixed_step is None:
        torch.testing.assert_close(generated["action"][:32, 0], noise["action"][:32, 0] - 0.999, atol=2e-5, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    for video_times, action_times in model.times:
        assert video_times.dtype == action_times.dtype == torch.float32
        # Both modalities use the same active block time, with zero for fixed positions.
        if task == "id":
            assert not video_times.any() and action_times.any()
        elif task == "fd":
            assert not action_times.any() and video_times.any()
        else:
            assert set(action_times.unique().tolist()) == set(video_times.unique().tolist())


@pytest.mark.parametrize("interval,expected", [(None, [800, 500, 200]), ([200, 800], [500]), ([900, 1001], [])])
def test_cfg_interval_uses_strict_model_time_bounds_and_skips_unconditional(interval, expected):
    sample = payload_sample("policy", actions=32)
    calls = []

    class RecordedVelocity(ConstantVelocity):
        def forward(self, **inputs):
            calls.append((int(inputs["input_ids"][0]), float(inputs["video_timesteps"].max())))
            return super().forward(**inputs)

    sample["text_token_ids"] = torch.tensor([2, 3])
    model = RecordedVelocity(cfg=True)
    sample_block(
        model,
        sample,
        0,
        guidance=3,
        guidance_interval=interval,
        negative_text_ids=torch.tensor([1]),
        fixed_step_sampler_config=dict(t_list=[0.8, 0.5, 0.2]),
    )
    assert [time for branch, time in calls if branch == 2] == [800, 500, 200]
    assert [time for branch, time in calls if branch == 1] == expected


@pytest.mark.parametrize("guidance", [0.5, 1.0, 3.0])
def test_cfg_norm_caps_joint_active_video_action_norm_without_amplification(guidance):
    sample = payload_sample("policy", actions=32)
    sample["text_token_ids"] = torch.tensor([2, 3])

    class ModalityVelocity(ConstantVelocity):
        def forward(self, **inputs):
            conditional = int(inputs["input_ids"][0]) == 2
            return SimpleNamespace(
                video=torch.full_like(inputs["video_tokens"], 2.0 if conditional else 1.0),
                action=torch.full_like(inputs["action_tokens"], 5.0 if conditional else 1.0),
            )

    noise = {"video": torch.zeros_like(sample["video_latents"]), "action": torch.zeros_like(sample["action"])}
    generated = sample_block(
        ModalityVelocity(),
        sample,
        0,
        guidance=guidance,
        normalize_cfg=True,
        negative_text_ids=torch.tensor([1]),
        initial_noise=noise,
        fixed_step_sampler_config=dict(t_list=[0.5]),
    )
    nv = sample["video_latents"][:, 1:].numel()
    na = len(sample["action"]) * int(sample["action_mask"].sum())
    v, a = 1 + guidance, 1 + guidance * 4
    scale = min(1, math.sqrt((nv * 2**2 + na * 5**2) / (nv * v**2 + na * a**2)))
    torch.testing.assert_close(
        generated["video_latents"][:, 1:], torch.full_like(noise["video"][:, 1:], -0.5 * v * scale)
    )
    torch.testing.assert_close(generated["action"][:, 0], torch.full_like(noise["action"][:, 0], -0.5 * a * scale))
    assert not generated["action"][:, 1].any()


def test_fixed_step_sde_matches_two_step_solution_without_consuming_global_rng():
    sigmas = [1.0, 0.5]
    sampler = FixedStepSampler(sigmas)
    sampler.set_timesteps(97, device="cpu", shift=7)
    assert sigmas == [1.0, 0.5]
    assert sampler.timesteps.tolist() == [1000, 500]
    initial = torch.tensor([10, 20], dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(123)
    fresh_noise = torch.randn(2, generator=torch.Generator().manual_seed(123))
    rng = torch.get_rng_state()
    state = sampler.step(torch.full_like(initial, 2), sampler.timesteps[0], initial, generator=generator)[0]
    generator_after_noise = generator.get_state().clone()
    state = sampler.step(
        torch.full_like(state, 2), sampler.timesteps[1], state, generator=generator, return_dict=True
    ).prev_sample
    torch.testing.assert_close(state, 0.5 * initial.float() - 2 + 0.5 * fresh_noise, atol=1e-6, rtol=0)
    assert state.dtype == torch.float32
    assert torch.equal(generator.get_state(), generator_after_noise)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("sigmas", [[], [0], [float("nan")], [1.1], [0.5, 0.8], [1, 0.5, 0.5]])
def test_fixed_step_rejects_invalid_sigma_schedule(sigmas):
    with pytest.raises(ValueError, match="t_list"):
        FixedStepSampler(sigmas)


@pytest.mark.parametrize("override", [False, True])
def test_pipeline_inherits_distilled_schedule_and_records_effective_guidance(override):
    _, processor, source = feedback_fixture()
    model = ConstantVelocity()
    model.config.fixed_step_sampler_config = dict(t_list=[0.999, 0.5], sample_type="sde")
    options = dict(guidance=3, guidance_interval=[200, 800], normalize_cfg=True, fixed_step_sampler_config=None)
    if override:
        options["fixed_step_sampler_config"] = dict(t_list=[0.75], sample_type="sde")
    config = dict(source=source, sampling=options)
    pipe = CausalCosmos3Pipeline(model, None, processor, lambda text: [1, 2], config)
    expected = options["fixed_step_sampler_config"] if override else model.config.fixed_step_sampler_config
    assert pipe.sampling_options["fixed_step_sampler_config"] == expected
    assert pipe.config["sampling"]["fixed_step_sampler_config"] == expected
    assert pipe.sampling_options["guidance"] == pipe.config["sampling"]["guidance"] == 1
    assert pipe.sampling_options["guidance_interval"] == [200, 800]
    assert pipe.sampling_options["normalize_cfg"]
    assert config["sampling"]["guidance"] == 3


@pytest.mark.parametrize("normalize", [False, True])
def test_cli_forwards_sampling_extensions_and_exports_resolved_config(tmp_path, monkeypatch, normalize):
    import tasks.infer.infer_causal_cosmos3 as cli
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    _, processor, source = feedback_fixture()
    path = tmp_path / "inference.yaml"
    path.write_text(
        yaml.safe_dump(dict(device="mock_device", source=source, sampling=dict(normalize_cfg=not normalize)))
    )
    saved = {}
    devices = []

    def build(checkpoint, config):
        pipe = CausalCosmos3Pipeline(ConstantVelocity(), None, processor, lambda text: [1, 2], config)
        pipe.last_input_identity = {}
        pipe.loading_report = {}
        pipe.rollout = lambda **kwargs: SimpleNamespace(
            semantics="test", actions=torch.zeros(3, 55), video=torch.zeros(3, 1, 32, 32)
        )
        return pipe

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=devices.append), raising=False)
    monkeypatch.setattr(inference.CausalCosmos3Pipeline, "from_pretrained", staticmethod(build))
    monkeypatch.setattr(inference, "save_result", lambda result, output, **kwargs: saved.update(kwargs["config"]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference",
            "--config",
            str(path),
            "--checkpoint",
            "unused",
            "--mode",
            "rollout",
            "--dataset",
            "unused",
            "--max-actions",
            "3",
            "--guidance",
            "3",
            "--guidance-interval",
            "200",
            "800",
            "--normalize-cfg" if normalize else "--no-normalize-cfg",
            "--fixed-step-sigmas",
            "0.999",
            "0.5",
            "--output",
            str(tmp_path / "result"),
        ],
    )
    cli.main()
    assert devices == ["mock_device"]
    assert saved["sampling"]["guidance"] == 1
    assert saved["sampling"]["guidance_interval"] == [200, 800]
    assert saved["sampling"]["normalize_cfg"] is normalize
    assert saved["sampling"]["fixed_step_sampler_config"] == dict(t_list=[0.999, 0.5], sample_type="sde")


def test_cfg_changes_only_active_prediction_and_repeats_with_fixed_noise():
    sample = payload_sample("policy", actions=32)
    sample["text_token_ids"] = torch.tensor([2, 3])
    sample["token_length"] = sum(
        (
            len(sample["text_token_ids"]),
            2 * len(patchify_video(sample["video_latents"], 2)),
            len(sample["action"]),
            len(sample["states"]),
        )
    )
    model = ConstantVelocity(cfg=True)
    noise = {"video": torch.zeros_like(sample["video_latents"]), "action": torch.zeros_like(sample["action"])}
    actual = sample_block(
        model, sample, 0, steps=4, guidance=3, negative_text_ids=torch.tensor([1]), initial_noise=noise
    )
    torch.testing.assert_close(
        actual["action"][:, 0], torch.full_like(actual["action"][:, 0], -4 * 0.999), atol=3e-5, rtol=0
    )
    first = sample_block(model, sample, 0, steps=3, seed=17)
    torch.randn(97)
    second = sample_block(model, sample, 0, steps=3, seed=17)
    torch.testing.assert_close(first["video_latents"], second["video_latents"], atol=0, rtol=0)
    torch.testing.assert_close(first["action"], second["action"], atol=0, rtol=0)


def test_real_model_cannot_read_current_clean_targets_or_future_ground_truth():
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    first = sample_block(model, sample, 0, steps=2, initial_noise=noise)
    changed = copy.deepcopy(sample)
    changed["video_latents"][:, 1:] += 500
    changed["action"] += 500
    changed["states"][1:] += 500
    second = sample_block(model, changed, 0, steps=2, initial_noise=noise)
    torch.testing.assert_close(first["video_latents"][:, 1:3], second["video_latents"][:, 1:3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first["action"][:32], second["action"][:32], atol=1e-6, rtol=1e-6)


def test_complete_checkpoint_loads_and_missing_action_or_state_fails(tmp_path):
    config = tiny_config(
        action_dim=55, state_dim=55, fixed_step_sampler_config=dict(t_list=[0.999, 0.5], sample_type="sde")
    )
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(config)
    config.save_pretrained(tmp_path)
    state = model.state_dict()
    save_file(state, tmp_path / "model.safetensors")
    restored = load_trained_model(tmp_path, dtype=torch.float32)
    assert restored.weight_loading_report["mode"] == "complete"
    assert restored.config.fixed_step_sampler_config == config.fixed_step_sampler_config
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    for key in ("state_proj_in.weight", "action_proj_out.weight"):
        broken = {name: value for name, value in state.items() if name != key}
        save_file(broken, tmp_path / "model.safetensors")
        with pytest.raises(ValueError, match="Missing Cosmos3 weights"):
            load_trained_model(tmp_path, dtype=torch.float32)


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_inference_preserves_training_ops_on_npu(tmp_path):
    root = Path(__file__).parents[3]
    training_ops = yaml.safe_load((root / "configs/dit/causal_cosmos3_edge.yaml").read_text())["model"][
        "ops_implementation"
    ]
    inference_ops = yaml.safe_load((root / "configs/infer/causal_cosmos3.yaml").read_text())["ops_implementation"]
    config = tiny_config(
        action_dim=55,
        state_dim=55,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        mrope_section=[16, 8, 8],
    )
    with ops_config_scope(OpsImplementationConfig(**training_ops)):
        trained = Cosmos3Model(config).to(device="npu", dtype=torch.bfloat16).train()
    config.save_pretrained(tmp_path)
    save_file(
        {key: value.cpu().contiguous() for key, value in trained.state_dict().items()}, tmp_path / "model.safetensors"
    )
    restored = load_trained_model(tmp_path, device="npu", dtype=torch.bfloat16, ops_implementation=inference_ops)
    inputs = fixed_inputs(device="npu", dtype=torch.bfloat16)
    for key in ("action_tokens", "state_tokens", "action_mask", "state_mask"):
        inputs[key] = torch.nn.functional.pad(inputs[key], (0, 51))
    with torch.no_grad(), torch.autocast("npu", dtype=torch.bfloat16):
        expected = trained(**inputs, attention_backend="npu_tnd")
        actual = restored(**inputs, attention_backend="npu_tnd")
    torch.testing.assert_close(actual.video, expected.video, atol=0, rtol=0)
    torch.testing.assert_close(actual.action, expected.action, atol=0, rtol=0)


def feedback_fixture(horizon=67):
    mask = torch.zeros(55, dtype=torch.bool)
    mask[0] = True
    stats = {name: dict(low=[-100.0] * 55, high=[100.0] * 55, valid=mask.tolist()) for name in ("state", "action")}
    processor = ActionProcessor(stats)
    source = dict(
        name="test",
        viewpoint="ego_view",
        position_unit="mm",
        gripper_unit="m",
        columns=dict(
            state="action_unified",
            target="action_unified",
            state_mask="mask_action",
            target_mask="mask_action",
            target_offset=1,
        ),
    )
    state = torch.zeros(55)
    state[0] = 2
    initial = InitialObservation(
        torch.ones(3, 1, 32, 32, dtype=torch.uint8) * 10, state, mask, mask.clone(), "move", 30.0, horizon, 0.0, {}
    )
    return initial, processor, source


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("mapped_source", [False, True])
@pytest.mark.parametrize("history,horizon", [(1, 67), (3, 131), (4, 131)])
@pytest.mark.parametrize("cache_history", [None, False, True])
def test_rollout_uses_decoded_pixels_and_final_valid_absolute_action(
    monkeypatch, mapped_source, history, horizon, profile, cache_history
):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    initial, processor, source = feedback_fixture()
    initial.horizon = horizon
    if mapped_source:
        processor.state_mapping = processor.action_mapping = [1, 0] + list(range(2, 55))
    seen = []

    class VAE:
        def encode_pack(self, samples):
            sample = samples[0]
            seen.append(copy.deepcopy(sample))
            return [{**sample, "video_latents": torch.zeros(2, 1 + (sample["video"].shape[1] - 1) // 4, 2, 2)}]

        def decode(self, latents):
            return torch.ones(3, 1 + (latents.shape[1] - 1) * 4, 32, 32, dtype=torch.uint8) * (20 + 30 * len(seen))

    caches = []

    def controlled(model, sample, block, **kwargs):
        caches.append(kwargs["history_cache"])
        action = torch.zeros_like(sample["action"])
        action[-32:, 0] = torch.arange(1, 33) / 100
        return dict(video_latents=sample["video_latents"], action=action, seeds={"video": 1, "action": 2})

    monkeypatch.setattr(inference_module, "sample_block", controlled)
    dummy = ConstantVelocity()
    config = dict(source=source, sampling=dict(history=history))
    if cache_history is not None:
        config["sampling"]["cache_history"] = cache_history
    pipe = CausalCosmos3Pipeline(dummy, VAE(), processor, lambda prompt: [1, 2], config)
    assert "cache_history" not in pipe.config["sampling"]
    assert config["sampling"].get("cache_history") is cache_history
    profiler = RolloutProfiler("cpu") if profile else None
    result = rollout(pipe, initial, profiler=profiler)
    assert caches and all(cache is caches[0] for cache in caches)
    assert isinstance(caches[0], inference_module.HistoryCache)
    if profile:
        summary = profiler.summary()
        assert summary["chunks"] == (horizon + 31) // 32
        assert summary["mean_chunk_seconds"] == pytest.approx(
            sum(record["seconds"] for record in result.blocks) / summary["chunks"]
        )
        assert set(summary["mean_stages_seconds"]) == {
            "prepare_sample",
            "vae_encode",
            "to_device",
            "sampling",
            "action_decode",
            "feedback_decode",
            "history_update",
        }
        assert sum(summary["mean_stages_seconds"].values()) <= summary["mean_chunk_seconds"]
        assert "peak_allocated_bytes" not in summary
    count, context_limit = (horizon + 31) // 32, (history + 1) // 2
    contexts = [min(i, context_limit) for i in range(count)]
    assert result.actions.shape == (horizon, 55)
    assert result.video.shape == (3, 1 + horizon // 4, 32, 32)
    assert [record["anchor"][0] for record in result.blocks] == pytest.approx([2 + 32 * i for i in range(count)])
    assert [record["feedback_state"][0] for record in result.blocks] == pytest.approx(
        [2 + min(32 * (i + 1), horizon) for i in range(count)]
    )
    assert [record["context_blocks"] for record in result.blocks] == contexts
    assert [record["history_blocks"] for record in result.blocks] == [min(history, 1 + 2 * c) for c in contexts]
    assert [record["history_pixel_frames"] for record in result.blocks] == [
        1 + 8 * min(i + 1, context_limit) for i in range(count)
    ]
    full_history = initial.image.clone()
    for index, sample in enumerate(seen):
        torch.testing.assert_close(sample["video"][:, :-8], full_history[:, -(1 + 8 * context_limit) :])
        assert sample["video"][:, -8:].eq(0).all()
        assert sample["history_blocks"] == history
        full_history = torch.cat(
            (full_history, torch.full_like(initial.image.expand(-1, 8, -1, -1), 50 + 30 * index)), dim=1
        )
    assert float(result.action_timestamps[-1]) == pytest.approx(horizon / 30)
    assert float(result.video_timestamps[-1]) == pytest.approx((horizon // 4) * 4 / 30)
    assert result.blocks[-1]["valid_steps"] == 3
    assert result.blocks[-1]["padded_action_steps"] == 29
    assert result.blocks[-1]["temporal_padding_frames"] == 8
    assert result.blocks[-1]["latent_shape"] == [2, 1 + 2 * (context_limit + 1), 2, 2]
    assert (processor.state_mapping is not None) == mapped_source
    # Reusing a pipeline for another episode or profiler warmup must not carry
    # its history into the next rollout.
    previous_cache = caches[0]
    caches.clear()
    seen.clear()
    rollout(pipe, initial)
    assert all(cache is caches[0] for cache in caches)
    assert caches[0] is not previous_cache


@pytest.mark.parametrize("task", ["forward_dynamics", "inverse_dynamics", "policy"])
@pytest.mark.parametrize("cache_history", [None, False, True])
def test_teacher_forcing_uses_measured_next_block_history_and_state(monkeypatch, task, cache_history):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    initial, processor, source = feedback_fixture()
    states = torch.zeros(65, 55)
    states[:32, 0], states[32:, 0] = 2, 70
    raw = dict(
        video=torch.full((3, 17, 32, 32), 10, dtype=torch.uint8),
        state=states,
        action=states[1:].clone(),
        state_mask=initial.state_mask,
        action_mask=initial.action_mask,
        real_action_steps=64,
        timestamps=torch.arange(65) / 30,
    )
    sample = make_sample(
        raw,
        source=source,
        processor=processor,
        geometry=Geometry(),
        tokenize=lambda prompt: [1],
        caption="move",
        fps=30,
        task=task,
        history=1,
    )
    sample["ground_truth_actions"] = raw["action"].clone()
    original = copy.deepcopy(sample)
    seen = []

    class VAE:
        def encode_pack(self, samples):
            return [{**samples[0], "video_latents": torch.full((2, 5, 2, 2), 0.5)}]

        def decode(self, latents):
            return torch.full((3, 17, 32, 32), 99, dtype=torch.uint8)

    caches = []

    def controlled(model, encoded, block, **kwargs):
        caches.append(kwargs["history_cache"])
        seen.append(block)
        torch.testing.assert_close(encoded["states"], original["states"], atol=0, rtol=0)
        assert encoded["video_latents"].eq(0.5).all()
        return dict(video_latents=encoded["video_latents"] + 1, action=torch.ones(64, 55), seeds={})

    monkeypatch.setattr(inference_module, "sample_block", controlled)
    config = dict(source=source, sampling=dict(history=1))
    if cache_history is not None:
        config["sampling"]["cache_history"] = cache_history
    pipe = CausalCosmos3Pipeline(ConstantVelocity(), VAE(), processor, lambda prompt: [1], config)
    assert "cache_history" not in pipe.config["sampling"]
    result = teacher_forcing(pipe, sample)
    assert len(caches) == 2 and caches[0] is caches[1]
    assert isinstance(caches[0], inference_module.HistoryCache)
    assert seen == [0, 1]
    assert [record["anchor"][0] for record in result.blocks] == [2, 70]
    assert all(record["padded_action_steps"] == record["temporal_padding_frames"] == 0 for record in result.blocks)
    assert result.actions[31, 0] == 102 and result.actions[32, 0] == 170
    torch.testing.assert_close(result.ground_truth_actions, raw["action"], atol=0, rtol=0)
    previous_cache = caches[0]
    caches.clear()
    teacher_forcing(pipe, sample)
    assert len(caches) == 2 and caches[0] is caches[1]
    assert caches[0] is not previous_cache


@pytest.mark.parametrize("units", [None, ("m", "source_native"), ("custom_position", "custom_gripper")])
def test_feedback_does_not_require_or_interpret_unit_labels(units):
    initial, processor, source = feedback_fixture()
    source.pop("position_unit")
    source.pop("gripper_unit")
    if units is not None:
        source["position_unit"], source["gripper_unit"] = units
    before = initial.state.clone()
    validate_feedback(source, processor, Geometry(), initial.state_mask, initial.action_mask)
    torch.testing.assert_close(initial.state, before, atol=0, rtol=0)


@pytest.mark.parametrize("change", ["offset", "field", "mask", "mapping"])
def test_invalid_feedback_contract_is_rejected(change):
    initial, processor, source = feedback_fixture()
    pipe = CausalCosmos3Pipeline(ConstantVelocity(), None, processor, lambda text: [1], dict(source=source))
    if change == "offset":
        source["columns"]["target_offset"] = 0
    if change == "field":
        source["columns"]["state"] = "state_unified"
    if change == "mask":
        initial.action_mask[1] = True
    if change == "mapping":
        processor.action_mapping = [1, 0] + list(range(2, 55))
    with pytest.raises(ValueError):
        validate_feedback(source, processor, pipe.geometry, initial.state_mask, initial.action_mask)


def test_unipc_preserves_constant_velocity_solution_and_schedule():
    for steps in (1, 2, 8):
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(steps, device="cpu", shift=1.0)
        state = torch.ones(1, 9)
        for time in scheduler.timesteps:
            state = scheduler.step(torch.ones(9), time, state, return_dict=False)[0]
        torch.testing.assert_close(state, torch.ones_like(state) * 0.001, atol=2e-5, rtol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_vae_decode_inverts_normalization_quantizes_and_clears_cache(fail):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae import Cosmos3VideoVAE

    class Decoder(torch.nn.Module):
        dtype = torch.float32
        config = SimpleNamespace(z_dim=2)
        cleared = 0

        def decode(self, value):
            torch.testing.assert_close(value, torch.full_like(value, 0.2))
            if fail:
                raise OSError("decode failed")
            return SimpleNamespace(sample=value[:, :1].expand(-1, 3, -1, -1, -1))

        def clear_cache(self):
            self.cleared += 1

    vae = Cosmos3VideoVAE.__new__(Cosmos3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.vae = Decoder()
    vae.register_buffer("latent_mean", torch.full((1, 2, 1, 1, 1), 0.2))
    vae.register_buffer("latent_inv_std", torch.full((1, 2, 1, 1, 1), 0.5))
    if fail:
        with pytest.raises(OSError, match="decode failed"):
            vae.decode(torch.zeros(2, 3, 2, 2))
    else:
        pixels = vae.decode(torch.zeros(2, 3, 2, 2))
        assert pixels.dtype == torch.uint8 and pixels.shape == (3, 3, 2, 2)
        assert pixels.eq(153).all()
    assert vae.vae.cleared == 1


def test_real_cfg_uses_distinct_text_without_changing_known_first_frame():
    sample = payload_sample("policy", actions=32)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    noise = dict(video=torch.randn_like(sample["video_latents"]), action=torch.randn_like(sample["action"]))
    plain = sample_block(model, sample, 0, steps=2, initial_noise=noise)
    guided = sample_block(
        model, sample, 0, steps=2, initial_noise=noise, guidance=2, negative_text_ids=torch.tensor([0])
    )
    assert torch.isfinite(guided["action"]).all()
    assert not torch.equal(plain["action"], guided["action"])
    torch.testing.assert_close(guided["video_latents"][:, 0], sample["video_latents"][:, 0], atol=0, rtol=0)


def test_one_frame_output_records_exact_fps_timestamps_and_physical_actions(tmp_path):
    import json

    import av
    import numpy as np

    from veomni.models.diffusers.causal_cosmos3.inference import InferenceResult, save_result

    initial, _, _ = feedback_fixture(horizon=3)
    actions = torch.zeros(3, 55)
    actions[:, 0] = torch.tensor([2.0, 3.0, 4.0])
    result = InferenceResult(
        torch.zeros(3, 1, 32, 32, dtype=torch.uint8),
        actions,
        initial.action_mask,
        12 + torch.arange(1, 4, dtype=torch.float64) / 25,
        torch.tensor([12.0], dtype=torch.float64),
        "model_only_open_loop",
        [dict(valid_steps=3, seconds=0.2)],
        video_fps=6.25,
        identity=dict(preprocessing=dict(canvas=[32, 32])),
    )
    output = save_result(
        result,
        tmp_path / "result",
        config=dict(source=dict(position_unit="mm")),
        checkpoint=tmp_path / "ckpt",
        identity={},
    )
    with av.open(str(output / "generated.mp4")) as container:
        assert float(container.streams.video[0].average_rate) == 6.25
        assert len(list(container.decode(video=0))) == 1
    with np.load(output / "actions.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["actions"], actions.numpy())
        np.testing.assert_array_equal(stored["action_timestamps"], result.action_timestamps.numpy())
        assert stored["actions"].shape[0] == 3
    run = json.loads((output / "run.json").read_text())
    assert run["video_fps"] == 6.25 and run["valid_actions"] == 3
    assert run["identity"]["preprocessing"]["canvas"] == [32, 32]
    assert (output / "actions.svg").stat().st_size > 0
    with pytest.raises(FileExistsError):
        save_result(result, output, config={}, checkpoint=tmp_path / "ckpt", identity={})


@pytest.mark.parametrize("compare_wall", [False, True])
def test_cli_profiles_twenty_chunks_after_separate_full_warmups(tmp_path, monkeypatch, compare_wall):
    import tasks.infer.infer_causal_cosmos3 as cli
    import veomni.models.diffusers.causal_cosmos3.inference as inference
    import veomni.models.diffusers.causal_cosmos3.rollout_profiler as profiling

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(device="cpu", geometry=dict(actions_per_block=32))))
    calls = []

    def fake_rollout(**kwargs):
        profiler = kwargs["profiler"]
        calls.append((kwargs["horizon"], profiler))
        if profiler is not None:
            for index in range(20):
                profiler.start_chunk()
                with profiler.stage("feedback_decode"):
                    pass
                profiler.finish_chunk(index)
        return SimpleNamespace(semantics="test", actions=torch.zeros(640, 55), video=torch.zeros(3, 161, 1, 1))

    pipe = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(packed_qkv_bytes=lambda: 1234),
        geometry=Geometry(),
        config={},
        last_input_identity={},
        loading_report={},
        rollout=fake_rollout,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda device: None, synchronize=lambda device: None, get_device_name=lambda device: "test"
        ),
        raising=False,
    )
    monkeypatch.setattr(inference.CausalCosmos3Pipeline, "from_pretrained", lambda *args: pipe)
    monkeypatch.setattr(inference, "save_result", lambda *args, **kwargs: None)

    class Trace:
        def __enter__(self):
            assert calls[-1][1] is None

        def __exit__(self, *args):
            pass

        def step(self):
            pass

    monkeypatch.setattr(profiling, "npu_trace", lambda *args, **kwargs: Trace())
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference",
            "--config",
            str(config_path),
            "--checkpoint",
            "unused",
            "--mode",
            "rollout",
            "--dataset",
            "unused",
            "--max-actions",
            "640",
            "--profile",
            "--profile-warmup-runs",
            "2",
            "--output",
            str(output),
        ]
        + (["--profile-npu-trace", "--compare-profile-wall"] if compare_wall else []),
    )
    cli.main()
    report = __import__("json").loads((output / "profile.json").read_text())
    assert [horizon for horizon, _ in calls] == [640] * (4 if compare_wall else 3)
    assert len({id(profiler) for _, profiler in calls if profiler is not None}) == 3
    assert report["chunks"] == len(report["chunks_detail"]) == 20
    assert report["packed_qkv_bytes"] == 1234
    assert len(report["warmup_runs"]) == 2
    assert report["mean_chunk_seconds"] == pytest.approx(calls[-1][1].summary()["mean_chunk_seconds"])
    assert report["trace_with_stack"] is compare_wall
    if compare_wall:
        comparison = __import__("json").loads((output / "wall_comparison.json").read_text())
        assert calls[2][1] is None and calls[3][1].trace is not None
        assert comparison == report["wall_comparison"]
        assert comparison["without_profiling_wall_seconds"] > 0
        assert comparison["with_profiling_context_wall_seconds"] >= comparison["with_profiling_rollout_wall_seconds"]
        assert comparison["action_max_abs_difference"] == 0 and comparison["video_identical"]


def test_rollout_profiler_synchronizes_and_resets_allocator_peaks(monkeypatch):
    events = []
    backend = SimpleNamespace(
        synchronize=lambda device: events.append("sync"),
        reset_peak_memory_stats=lambda device: events.append("reset"),
        max_memory_allocated=lambda device: 1234,
        max_memory_reserved=lambda device: 5678,
    )
    profiler = RolloutProfiler("cpu")
    profiler.backend = backend
    for index in range(2):
        profiler.start_chunk()
        with profiler.stage("sampling"):
            assert events[-1] == "sync"
            events.append("work")
        assert events[-1] == "sync"
        profiler.finish_chunk(index)
    assert events.count("reset") == 2
    assert profiler.summary()["peak_allocated_bytes"] == 1234
    assert profiler.summary()["peak_reserved_bytes"] == 5678


@pytest.mark.parametrize("fixed_schedule", [None, dict(t_list=[0.999, 0.5], sample_type="sde")])
def test_sample_block_profiles_three_regions_without_changing_outputs(fixed_schedule):
    sample = payload_sample("policy", actions=32)
    with torch.random.fork_rng(devices=[]), ops_config_scope(eager_ops_config()):
        torch.manual_seed(123)
        model = Cosmos3Model(tiny_config()).eval()
    options = dict(steps=2, seed=42, fixed_step_sampler_config=fixed_schedule)
    expected = sample_block(model, sample, 0, **options)
    profiler = RolloutProfiler("cpu")
    profiler.start_chunk()
    with profiler.stage("sampling"):
        actual = sample_block(model, sample, 0, profiler=profiler, **options)
    record = profiler.finish_chunk(0)
    torch.testing.assert_close(actual["video_latents"], expected["video_latents"], atol=0, rtol=0)
    torch.testing.assert_close(actual["action"], expected["action"], atol=0, rtol=0)
    assert actual["seeds"] == expected["seeds"]
    timings = record["sampling_stages_seconds"]
    assert set(timings) == {"noise_initialization", "denoising", "result_assembly"}
    assert all(value >= 0 for value in timings.values())
    assert sum(timings.values()) <= record["stages_seconds"]["sampling"] <= record["seconds"]
    assert profiler.summary()["mean_sampling_stages_seconds"] == timings
    profiler.start_chunk()
    assert profiler.sampling_stages == {}


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("block", [0, 1])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_reused_denoise_packing_matches_full_repacking(task, block, guidance):
    torch.manual_seed(123)
    sample = payload_sample(task, actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    options = dict(steps=3, guidance=guidance, negative_text_ids=torch.tensor([1]), initial_noise=noise)
    expected = sample_block(model, sample, block, reuse_denoise_packing=False, **options)
    actual = sample_block(model, sample, block, reuse_denoise_packing=True, **options)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mutation", ["positions", "inplace_positions", "training", "grad", "owner"])
def test_denoise_packing_rejects_stale_or_training_use(mutation):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    sample = payload_sample("policy", actions=32)
    with torch.no_grad():
        prepared = DenoisePacking(model, sample, 0, "grouped")
        inputs = prepared.update(sample["video_latents"], sample["action"], torch.tensor(500.0))
    if mutation == "positions":
        inputs["position_ids"] = inputs["position_ids"] + 1
    elif mutation == "inplace_positions":
        inputs["position_ids"].add_(1)
    elif mutation == "training":
        model.train()
    elif mutation == "owner":
        model = copy.deepcopy(model)
    with torch.set_grad_enabled(mutation == "grad"), pytest.raises(ValueError, match="Denoise packing"):
        model(**inputs)


def test_denoise_packing_times_remain_float32_with_float64_default():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    sample = payload_sample("policy", actions=32)
    model = ConstantVelocity()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        prepared = DenoisePacking(model, sample, 0, "grouped")
        inputs = prepared.update(sample["video_latents"], sample["action"], torch.tensor(500.0, dtype=torch.float32))
        assert inputs["video_timesteps"].dtype == inputs["action_timesteps"].dtype == torch.float32
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("context", [torch.inference_mode, lambda: torch.autocast("cpu", dtype=torch.bfloat16)])
def test_sampling_plan_falls_back_for_external_execution_contexts(context):
    sample = payload_sample("policy", actions=32)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    with context():
        expected = sample_block(model, sample, 0, steps=2, reuse_denoise_packing=False)
        actual = sample_block(model, sample, 0, steps=2, reuse_denoise_packing=True)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("backend", ["grouped", "dense"])
@pytest.mark.parametrize("history", [1, 4])
def test_pruned_queries_preserve_indirect_clean_history_and_targets(task, backend, history):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    torch.manual_seed(77)
    sample = payload_sample(task, actions=96)
    sample["history_blocks"] = history
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    with torch.no_grad():
        prepared = DenoisePacking(model, sample, 2, backend, cache_fixed_attention=False)
    plan = prepared.inputs["denoise_packing"].attention
    layout = prepared.inputs["layout"]
    selected = plan.gen_indexes[plan.query_selection].tolist()
    # Keep even C0: later retained clean frames can depend on it through layers.
    assert any(layout.roles[i] == 0 and layout.frame_ids[i] == 0 for i in selected)
    assert all(layout.roles[i] == 0 or layout.block_ids[i] == 2 for i in selected)
    options = dict(steps=3, attention_backend=backend, guidance=3, negative_text_ids=torch.tensor([1]))
    expected = sample_block(model, sample, 2, prune_attention=False, **options)
    actual = sample_block(model, sample, 2, prune_attention=True, **options)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)


def test_query_selection_handles_empty_output_and_rejects_partial_groups():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        generation_attention,
        select_attention_queries,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    sample = payload_sample("policy", actions=32)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    with torch.no_grad():
        prepared = DenoisePacking(model, sample, 0, "grouped", prune_attention=False)
        plan = prepared.inputs["denoise_packing"].attention
        empty = select_attention_queries(plan, [False] * len(plan.gen_indexes))
        q = torch.randn(len(plan.gen_indexes), 2, 4)
        kv = torch.randn(len(sample["text_token_ids"]) + len(plan.gen_indexes), 2, 4)
        assert torch.equal(generation_attention(q, kv, kv, empty), torch.zeros_like(q))
        partial = [False] * len(plan.gen_indexes)
        # Current target video and action share one query group.
        current = prepared.span.current_video.start + 1 - len(sample["text_token_ids"])
        partial[current] = True
        with pytest.raises(ValueError, match="complete attention groups"):
            select_attention_queries(plan, partial)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("backend", ["grouped", "dense"])
def test_fixed_attention_cache_is_branch_and_call_local(task, backend, monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers as layers

    torch.manual_seed(88)
    sample = payload_sample(task, actions=96)
    sample["history_blocks"] = 1
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    original = layers.understanding_attention
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(layers, "understanding_attention", counted)
    options = dict(steps=3, attention_backend=backend, guidance=3, negative_text_ids=torch.tensor([1]))
    for seed in (42, 43):
        calls.clear()
        expected = sample_block(model, sample, 2, seed=seed, cache_fixed_attention=False, **options)
        uncached_calls = len(calls)
        calls.clear()
        actual = sample_block(model, sample, 2, seed=seed, cache_fixed_attention=True, **options)
        assert len(calls) == 2 * len(model.layers)
        assert uncached_calls == 3 * len(calls)
        for name in ("video_latents", "action"):
            torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("block", [0, 2])
@pytest.mark.parametrize("history", [1, 4])
def test_pruned_group_dependencies_are_closed_across_layers(task, block, history):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import dense_generation_mask
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    torch.manual_seed(291)
    sample = payload_sample(task, actions=96)
    sample["history_blocks"] = history
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=4)).eval()
    with torch.no_grad():
        packing = DenoisePacking(model, sample, block, "dense", cache_fixed_attention=False)
        plan = packing.inputs["denoise_packing"].attention
        selected = plan.query_selection
        dropped = torch.ones(len(plan.gen_indexes), dtype=torch.bool)
        dropped[selected] = False
        mask = dense_generation_mask(packing.inputs["layout"], "cpu")
        # No retained query reads a discarded GEN row. With token-local FFN/norm,
        # this closure is sufficient for every layer, not just the first one.
        assert not mask[selected][:, plan.gen_indexes[dropped]].any()
        layout = packing.inputs["layout"]
        targets = torch.tensor(
            [role == 1 and bid == block for role, bid in zip(layout.roles, layout.block_ids) if role != -1]
        )
        fixed_rows = selected[~targets[selected]]
        # The cached rows are also closed against the changing noisy targets.
        assert not mask[fixed_rows][:, plan.gen_indexes[targets]].any()
        inputs = packing.update(sample["video_latents"], sample["action"], torch.tensor(500.0))
        inputs.pop("denoise_packing")  # Independent full dense attention oracle.
        expected = model(**inputs, output_hidden_states=True).last_hidden_state

        def corrupt_discarded_rows(module, args):
            und, gen, *rest = args
            gen = gen.clone()
            gen[dropped] = torch.randn_like(gen[dropped]) * 1000
            return (und, gen, *rest)

        hooks = [layer.register_forward_pre_hook(corrupt_discarded_rows) for layer in model.layers]
        try:
            actual = model(**inputs, output_hidden_states=True).last_hidden_state
        finally:
            for hook in hooks:
                hook.remove()
        retained = torch.cat((plan.modality_indexes[0], plan.gen_indexes[selected]))
        torch.testing.assert_close(actual[retained], expected[retained], atol=0, rtol=0)


def test_repacking_merges_groups_across_old_chunks_without_changing_kv():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        AttentionPlan,
        TNDChunk,
        select_attention_queries,
    )

    # Four old chunks, two groups each. Keep only the first group of each chunk:
    # the four survivors must fit into two new chunks (two 8-token KV groups each).
    chunks = tuple(
        TNDChunk(torch.arange(i * 4, i * 4 + 4), torch.arange(i * 16, i * 16 + 16), (2, 4), (8, 16)) for i in range(4)
    )
    plan = AttentionPlan(
        (), torch.arange(16), (), torch.empty(0, dtype=torch.long), (), chunks, None, "npu_tnd", max_kv_tokens=16
    )
    keep = [i % 4 < 2 for i in range(16)]
    selected = select_attention_queries(plan, keep)
    assert len(selected.chunks) == 2
    assert selected.query_selection.tolist() == [0, 1, 4, 5, 8, 9, 12, 13]
    for i, chunk in enumerate(selected.chunks):
        assert chunk.query_ends == (2, 4)
        assert chunk.kv_ends == (8, 16)
        torch.testing.assert_close(
            chunk.kv_indexes, torch.cat((chunks[2 * i].kv_indexes[:8], chunks[2 * i + 1].kv_indexes[:8]))
        )
    empty = select_attention_queries(plan, [False] * 16)
    assert empty.chunks == () and empty.query_selection.numel() == 0
    with pytest.raises(ValueError, match="complete attention groups"):
        select_attention_queries(plan, [True] + [False] * 15)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_npu_repacked_attention_matches_fp32_visibility_oracle():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        dense_generation_mask,
        generation_attention,
        prepare_attention,
        sdpa,
        select_attention_queries,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import build_packed_layout

    torch.manual_seed(293)
    layout, _, _ = build_packed_layout([payload_sample("policy", actions=96)])
    plan = prepare_attention(layout, "npu", "npu_tnd", max_kv_tokens=128)
    keep = [role == 1 and block == 2 for role, block in zip(layout.roles, layout.block_ids) if role != -1]
    selected = select_attention_queries(plan, keep)
    assert selected.query_selection.cpu().tolist() == [i for i, needed in enumerate(keep) if needed]
    assert sum(c.query_ends[-1] for c in selected.chunks) == sum(keep)
    q = torch.randn(len(plan.gen_indexes), 4, 64, dtype=torch.bfloat16)
    k = torch.randn(len(layout.roles), 2, 64, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with torch.no_grad():
        expected = sdpa(q.float(), k.float(), v.float(), dense_generation_mask(layout, "cpu"))
        actual = generation_attention(q.to("npu"), k.to("npu"), v.to("npu"), selected).cpu()
    rows = selected.query_selection.cpu()
    torch.testing.assert_close(actual[rows].float(), expected[rows], atol=3e-3, rtol=1e-2)
    assert torch.count_nonzero(actual[torch.tensor(keep).logical_not()]) == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
@pytest.mark.parametrize("max_kv_tokens", [32, 131072])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_npu_fixed_attention_cache_matches_sampling_and_reduces_query_rows(max_kv_tokens, guidance, monkeypatch):
    import torch_npu

    from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer import modeling_cosmos3_transformer as modeling

    torch.manual_seed(197)
    sample = payload_sample("policy", actions=96)
    sample["history_blocks"] = 1
    for name in ("video_latents", "action", "states", "action_mask", "state_mask"):
        sample[name] = sample[name].to("npu")
    with ops_config_scope(eager_ops_config()):
        model = (
            Cosmos3Model(tiny_config(hidden_size=256, head_dim=64, mrope_section=[16, 8, 8]))
            .eval()
            .to(device="npu", dtype=torch.bfloat16)
        )

    original_prepare = modeling.prepare_attention
    monkeypatch.setattr(
        modeling,
        "prepare_attention",
        lambda layout, device, backend: original_prepare(layout, device, backend, max_kv_tokens=max_kv_tokens),
    )
    original_fused = torch_npu.npu_fusion_attention
    calls, forwards, caches = [], [], []

    def counted_fused(q, k, v, *args, **kwargs):
        calls.append(
            (q.shape, k.shape, v.shape, kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"], kwargs["sparse_mode"])
        )
        return original_fused(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", counted_fused)
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        begin = len(calls)
        output = original_forward(*args, **kwargs)
        forwards.append(calls[begin:])
        caches.append(kwargs["denoise_packing"].fixed_attention)
        return output

    monkeypatch.setattr(model, "forward", counted_forward)
    options = dict(
        steps=3,
        attention_backend="npu_tnd",
        guidance=guidance,
        negative_text_ids=torch.tensor([1]),
        seed=42,
    )
    expected = sample_block(model, sample, 2, cache_fixed_attention=False, **options)
    reference = list(forwards)
    assert all(cache is None for cache in caches)
    calls.clear()
    forwards.clear()
    caches.clear()
    actual = sample_block(model, sample, 2, cache_fixed_attention=True, **options)
    for name in ("video_latents", "action"):
        # BF16 row compaction changes GEMM rounding; CFG amplifies it.
        # Exact mask/row semantics are covered separately by the FP32 layer oracle.
        torch.testing.assert_close(actual[name], expected[name], atol=5e-3, rtol=1e-2)

    branches = 1 if guidance == 1 else 2
    assert len(forwards) == len(reference) == 3 * branches
    assert all(cache is not None for cache in caches)
    assert len({id(cache) for cache in caches}) == branches
    assert all(len(cache.layers) == len(model.layers) for cache in caches)
    # Cold passes retain the dependency closure; warm passes only compute targets.
    for actual_calls, full_calls in zip(forwards[branches:], reference[branches:], strict=True):
        assert all(call[-1] == 0 for call in actual_calls)  # No repeated UND attention.
        actual_gen = sum(call[0][0] for call in actual_calls)
        full_gen = sum(call[0][0] for call in full_calls if call[-1] == 0)
        assert actual_gen < full_gen
        assert len(actual_calls) <= len(full_calls)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_npu_query_pruning_keeps_only_selected_groups():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        generation_attention,
        prepare_attention,
        select_attention_queries,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs

    sample = payload_sample("policy", actions=96)
    inputs, _, _ = pack_model_inputs(
        [sample], [sample["video_latents"]], [sample["action"]], timesteps=[torch.zeros(1)], patch_size=2
    )
    layout = inputs["layout"]
    plan = prepare_attention(layout, "npu", "npu_tnd", max_kv_tokens=128)
    full = select_attention_queries(plan, [True] * len(plan.gen_indexes))
    assert len(full.query_selection) == len(plan.gen_indexes)
    keep = tuple(role == 1 and block == 2 for role, block in zip(layout.roles, layout.block_ids) if role != -1)
    selected = select_attention_queries(plan, keep)
    assert 0 < len(selected.chunks) < len(plan.chunks)
    assert selected.query_selection.tolist() == [i for i, retained in enumerate(keep) if retained]
    torch.manual_seed(91)
    query = torch.randn(len(plan.gen_indexes), 4, 64, device="npu", dtype=torch.bfloat16)
    key = torch.randn(len(layout.roles), 2, 64, device="npu", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    with torch.no_grad():
        expected = generation_attention(query, key, value, plan)
        actual = generation_attention(query, key, value, selected)
    rows = selected.query_selection
    torch.testing.assert_close(actual[rows], expected[rows], atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("block", [0, 2])
@pytest.mark.parametrize("history", [1, 4])
def test_block_cache_restores_retained_rows_and_compacts_cold_kv(task, backend, block, history, monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers as layers
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        dense_generation_mask,
        select_attention_queries,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking

    torch.manual_seed(321)
    sample = payload_sample(task, actions=96, history=history)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    with torch.no_grad():
        packing = DenoisePacking(model, sample, block, backend)
        prepared = packing.inputs["denoise_packing"]
        cache = prepared.fixed_attention
        plan = prepared.attention
        fixed = torch.ones(len(plan.gen_indexes), dtype=torch.bool)
        fixed[cache.dynamic] = False
        mask = dense_generation_mask(prepared.layout, "cpu")
        assert not mask[fixed][:, plan.gen_indexes[cache.dynamic]].any()
        torch.testing.assert_close(cache.kv_global[cache.dynamic_kv], plan.gen_indexes[cache.dynamic])
        keep = [
            prepared.layout.roles[i] == 1 and prepared.layout.block_ids[i] == block for i in plan.gen_indexes.tolist()
        ]
        target = select_attention_queries(plan, keep)
        for table, full in zip(cache.dynamic_tables, prepared.tables[1], strict=True):
            torch.testing.assert_close(table, full[cache.dynamic], atol=0, rtol=0)
        cold = cache.cold
        assert cold is not None
        discarded = torch.ones(len(plan.gen_indexes), dtype=torch.bool)
        discarded[cold.rows] = False
        # Use the independent dense oracle, including the transitive history edge.
        assert not mask[cold.rows][:, plan.gen_indexes[discarded]].any()
        torch.testing.assert_close(cold.kv_global, mask[cold.rows].any(dim=0).nonzero().flatten())
        torch.testing.assert_close(cold.kv_global[cold.gen_kv], plan.gen_indexes[cold.rows])
        torch.testing.assert_close(cold.kv_global[cold.text_kv], plan.modality_indexes[0][cold.text_rows])
        torch.testing.assert_close(cold.kv_global[cold.warm_kv], cache.kv_global)
        assert len(cold.kv_global) < len(mask[0])
        if block == 2 and history == 1:
            assert len(cold.kv_global) > len(cache.kv_global)

        def ordered_visibility(attention, query_rows, key_rows):
            result = {}
            for chunk in attention.chunks:
                qs = ks = 0
                for qe, ke in zip(chunk.query_ends, chunk.kv_ends, strict=True):
                    keys = key_rows[chunk.kv_indexes[ks:ke]].tolist()
                    for query in query_rows[chunk.query_indexes[qs:qe]].tolist():
                        result[query] = keys
                    qs, ks = qe, ke
            return result

        global_keys = torch.arange(len(mask[0]))
        gen_rows = torch.arange(len(plan.gen_indexes))
        expected_visibility = ordered_visibility(plan, gen_rows, global_keys)
        for attention, query_rows, keys in (
            (cache.compact_plan, cache.dynamic, cache.kv_global),
            (cold.plan, cold.rows, cold.kv_global),
        ):
            actual_visibility = ordered_visibility(attention, query_rows, keys)
            assert actual_visibility == {q: expected_visibility[q] for q in query_rows.tolist()}
        assert sum(len(c.query_ends) for c in cache.compact_plan.chunks) <= sum(
            len(c.query_ends) for c in target.chunks
        )

    captures = []
    attention_shapes = []
    original_attention = layers.selected_generation_attention

    def capture_attention(q, k, v, plan):
        attention_shapes.append((len(q), len(k), len(v)))
        return original_attention(q, k, v, plan)

    monkeypatch.setattr(layers, "selected_generation_attention", capture_attention)
    buffers = {}

    def capture(layer, args, outputs):
        captures.append(tuple(t.clone() for t in outputs))
        active_cache = args[4]
        if active_cache is None:
            return
        entry = active_cache.layers[layer]
        fixed = torch.ones(len(entry.key), dtype=torch.bool)
        fixed[active_cache.dynamic_kv] = False
        identity = (id(layer), id(active_cache))
        if identity in buffers:
            pointers, key, value = buffers[identity]
            assert pointers == (entry.key.data_ptr(), entry.value.data_ptr())
            torch.testing.assert_close(entry.key[fixed], key, atol=0, rtol=0)
            torch.testing.assert_close(entry.value[fixed], value, atol=0, rtol=0)
        else:
            buffers[identity] = (entry.key.data_ptr(), entry.value.data_ptr()), entry.key[fixed], entry.value[fixed]

    handles = [layer.register_forward_hook(capture) for layer in model.layers]
    options = dict(steps=3, attention_backend=backend, guidance=3, negative_text_ids=torch.tensor([1]))
    try:
        sample_block(model, sample, block, reuse_denoise_packing=False, **options)
        reference = list(captures)
        captures.clear()
        attention_shapes.clear()
        projections = []
        hook = model.layers[0].self_attn.add_q_proj.register_forward_pre_hook(
            lambda m, a: projections.append(len(a[0]))
        )
        try:
            sample_block(model, sample, block, **options)
        finally:
            hook.remove()
        assert projections == [len(cold.rows)] * 2 + [len(cache.dynamic)] * 4
        expected_shapes = []
        for step in range(3):
            queries = len(cold.rows) if step == 0 else len(cache.dynamic)
            keys = len(cold.kv_global) if step == 0 else len(cache.kv_global)
            # Positive and negative CFG prompts have different text lengths.
            for text_delta in (0, len(sample["text_token_ids"]) - len(options["negative_text_ids"])):
                expected_shapes.extend([(queries, keys - text_delta, keys - text_delta)] * len(model.layers))
        assert attention_shapes == expected_shapes
        assert len(captures) == len(reference) == 18
        # Compare every retained intermediate row, not only final sampled targets.
        for (text, gen), (expected_text, expected_gen) in zip(captures, reference, strict=True):
            torch.testing.assert_close(text, expected_text, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(gen[cold.rows], expected_gen[cold.rows], atol=2e-6, rtol=2e-5)
            assert not gen[discarded].count_nonzero()
    finally:
        for hook in handles:
            hook.remove()


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_history_cache_preserves_full_layers_with_immutable_history(task, backend):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache

    torch.manual_seed(971)
    sample = payload_sample(task, actions=96)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    cache = HistoryCache()
    captured = []
    retained = []

    def capture(module, args, out):
        captured.append(tuple(t.clone() for t in out))
        fixed = args[4]
        if fixed is not None:
            retained.append(torch.cat((fixed.cold.rows, fixed.history.reused_rows)))

    hooks = [layer.register_forward_hook(capture) for layer in model.layers]
    options = dict(steps=3, guidance=3, negative_text_ids=torch.tensor([1]), attention_backend=backend)
    try:
        for block in range(3):
            captured.clear()
            expected = sample_block(model, sample, block, prune_attention=False, **options)
            reference = list(captured)
            captured.clear()
            retained.clear()
            actual = sample_block(model, sample, block, history_cache=cache, **options)
            assert len(captured) == len(reference) == 18
            for (text, gen), (ref_text, ref_gen), rows in zip(captured, reference, retained, strict=True):
                torch.testing.assert_close(text, ref_text, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(gen[rows], ref_gen[rows], atol=3e-6, rtol=3e-5)
                discarded = torch.ones(len(gen), dtype=torch.bool)
                discarded[rows] = False
                assert not gen[discarded].count_nonzero()
            for key in ("video_latents", "action"):
                torch.testing.assert_close(actual[key], expected[key], atol=3e-6, rtol=3e-5)
            assert set(cache.branches) == {True, False}
            # Only observed clean frames are persisted, never noisy targets,
            # conditions, action/state or future clean placeholders.
            for _, step, identities, layers in cache.branches.values():
                assert step == block
                assert max(frame for frame, patch in identities) == block * sample["block_size"]
                assert len(layers) == len(model.layers)
                assert all(len(layer.hidden) == len(identities) for layer in layers.values())
    finally:
        for hook in hooks:
            hook.remove()


def test_history_cache_identity_reset_and_window_eviction():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking, HistoryCache

    sample = payload_sample("policy", actions=96)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    cache = HistoryCache()

    def bind(block, sequence, branch=True):
        return (
            DenoisePacking(
                model, sample, block, "grouped", history_cache=cache, sequence_block=sequence, branch=branch
            )
            .inputs["denoise_packing"]
            .fixed_attention.history
        )

    with torch.no_grad():
        default = DenoisePacking(model, sample, 0, "grouped", history_cache=cache)
        assert default.inputs["denoise_packing"].fixed_attention.history is not None
        first = bind(1, 1)
        first.following.update({layer: object() for layer in model.layers})
        second = bind(2, 2)
        assert second.previous and len(second.reused_rows) == len(first.clean_rows)
        second.following.update({layer: object() for layer in model.layers})
        # A cropped context keeps absolute frame identities and forgets evicted
        # frames; the caller explicitly accepts changed window semantics here.
        shifted = bind(1, 3)
        assert all(frame >= 4 for frame, patch in cache.branches[True][2])
        assert len(shifted.reused_rows) > 0
        assert not bind(1, 4, branch=False).previous
        assert not bind(1, 8).previous  # Nonsequential call resets this branch.
        old = bind(1, 9)
        old.following.update({layer: object() for layer in model.layers})
        sample["conditioning_fps"] *= 2
        assert not bind(1, 10).previous
        old = bind(1, 11)
        old.following.update({layer: object() for layer in model.layers})
        sample["text_token_ids"] = sample["text_token_ids"].flip(0)
        assert not bind(1, 12).previous
        partial = bind(1, 13)
        partial.following[model.layers[0]] = object()
        incomplete = bind(1, 14)
        assert not incomplete.previous and not len(incomplete.reused_rows)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("history", [1, 4])
def test_history_cache_compacts_fresh_dependency_closure_and_owns_snapshots(task, backend, history, monkeypatch):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import layers
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import dense_generation_mask
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import DenoisePacking, HistoryCache

    torch.manual_seed(619)
    sample = payload_sample(task, actions=96, history=history, height=4, width=6)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    history_cache = HistoryCache()
    shapes = []
    original = layers.selected_generation_attention

    def attention(q, k, v, plan):
        shapes.append((len(q), len(k), len(v)))
        return original(q, k, v, plan)

    monkeypatch.setattr(layers, "selected_generation_attention", attention)
    with torch.no_grad():
        # Includes a local-coordinate shift with continuous global identities.
        for sequence, block in enumerate((0, 1, 2, 1)):
            packing = DenoisePacking(
                model, sample, block, backend, history_cache=history_cache, sequence_block=sequence
            )
            prepared = packing.inputs["denoise_packing"]
            cache, plan = prepared.fixed_attention, prepared.attention
            cold, binding = cache.cold, cache.history
            mask = dense_generation_mask(prepared.layout, "cpu")
            cached = torch.zeros(len(plan.gen_indexes), dtype=torch.bool)
            cached[binding.reused_rows] = True
            needed = torch.zeros_like(cached)
            needed[cache.dynamic] = True
            needed[binding.clean_rows] = True
            fresh = needed & ~cached
            while True:
                expanded = (fresh | mask[fresh][:, plan.gen_indexes].any(0)) & ~cached
                if torch.equal(expanded, fresh):
                    break
                fresh = expanded
            torch.testing.assert_close(cold.rows, fresh.nonzero().flatten())
            assert not cached[cold.rows].any()
            visible = mask[fresh].any(0) | mask[cache.dynamic].any(0)
            torch.testing.assert_close(cold.kv_global, visible.nonzero().flatten())
            assert len(cold.kv_global) < mask.shape[1]
            assert len(cold.rows) < len(plan.gen_indexes)
            assert len(binding.fresh_clean) + len(binding.reused_clean) == len(binding.clean_rows)
            assert len(torch.cat((binding.fresh_clean, binding.reused_clean)).unique()) == len(binding.clean_rows)
            for slot, previous in zip(binding.reused_kv, binding.previous_kv, strict=True):
                row = binding.reused_rows[(binding.previous_rows == previous).nonzero()[0, 0]]
                assert cold.kv_global[slot] == plan.gen_indexes[row]

            old = {
                layer: (entry.hidden.clone(), entry.key.clone(), entry.value.clone())
                for layer, entry in binding.previous.items()
            }
            previous = dict(binding.previous)
            shapes.clear()
            inputs = packing.update(sample["video_latents"], sample["action"], torch.tensor(900.0))
            model(**inputs)
            assert not binding.previous
            assert shapes == [(len(cold.rows), len(cold.kv_global), len(cold.kv_global))] * len(model.layers)
            saved = {
                layer: (entry.hidden.clone(), entry.key.clone(), entry.value.clone())
                for layer, entry in binding.following.items()
            }
            for layer, entry in binding.following.items():
                for tensor in (entry.hidden, entry.key, entry.value):
                    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                torch.testing.assert_close(
                    entry.hidden, cache.layers[layer].hidden[binding.clean_rows], atol=0, rtol=0
                )
                if layer in old:
                    for new_tensor, old_tensor in zip((entry.hidden, entry.key, entry.value), old[layer], strict=True):
                        torch.testing.assert_close(
                            new_tensor[binding.reused_clean], old_tensor[binding.previous_rows], atol=0, rtol=0
                        )
            shapes.clear()
            model(**packing.update(sample["video_latents"], sample["action"], torch.tensor(500.0)))
            assert shapes == [(len(cache.dynamic), len(cache.kv_global), len(cache.kv_global))] * len(model.layers)
            for snapshots, reference in ((previous, old), (binding.following, saved)):
                for layer, entry in snapshots.items():
                    for tensor, before in zip((entry.hidden, entry.key, entry.value), reference[layer], strict=True):
                        torch.testing.assert_close(tensor, before, atol=0, rtol=0)


def test_history_releases_consumed_snapshots_and_resets_after_interrupted_forward():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache

    sample = payload_sample("policy", actions=96)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    cache = HistoryCache()
    sample_block(model, sample, 0, steps=2, history_cache=cache)
    previous = {
        layer: [weakref.ref(t) for t in (entry.hidden, entry.key, entry.value)]
        for layer, entry in cache.branches[True][3].items()
    }

    def interrupt(layer, args, output):
        # Once a layer has copied its next snapshot, no cache retains its old
        # clean tensors. Text tensors can deliberately remain shared.
        assert all(ref() is None for ref in previous[layer])
        if layer is model.layers[1]:
            raise RuntimeError("interrupted cold forward")

    hooks = [layer.register_forward_hook(interrupt) for layer in model.layers]
    try:
        with pytest.raises(RuntimeError, match="interrupted cold forward"):
            sample_block(model, sample, 1, steps=2, history_cache=cache)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(cache.branches[True][3]) == 2
    expected = sample_block(model, sample, 2, steps=2, history_cache=HistoryCache())
    actual = sample_block(model, sample, 2, steps=2, history_cache=cache)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_partial_query_selection_preserves_each_ordered_kv_group(backend):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
        generation_attention,
        prepare_attention,
        select_attention_queries,
    )
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import build_packed_layout

    layout, _, _ = build_packed_layout([payload_sample("policy", actions=96)])
    plan = prepare_attention(layout, "cpu", backend)
    keep = [i % 3 != 0 for i in range(len(plan.gen_indexes))]
    selected = select_attention_queries(plan, keep, allow_partial=True)
    q = torch.randn(len(plan.gen_indexes), 4, 8)
    k = torch.randn(len(layout.roles), 2, 8)
    v = torch.randn_like(k)
    with torch.no_grad():
        full = generation_attention(q, k, v, plan)
        actual = generation_attention(q, k, v, selected)
    torch.testing.assert_close(actual[keep], full[keep], atol=3e-7, rtol=1e-5)
    assert not actual[torch.tensor(keep).logical_not()].any()


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_npu_history_cache_matches_immutable_history_and_skips_clean_projections():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache

    torch.manual_seed(613)
    sample = payload_sample("policy", actions=96)
    for name in ("video_latents", "action", "states", "action_mask", "state_mask"):
        sample[name] = sample[name].to("npu")
    with ops_config_scope(eager_ops_config()):
        model = (
            Cosmos3Model(tiny_config(hidden_size=256, head_dim=64, mrope_section=[16, 8, 8]))
            .eval()
            .to(device="npu", dtype=torch.bfloat16)
        )
    cache = HistoryCache()
    options = dict(steps=3, attention_backend="npu_tnd", guidance=3, negative_text_ids=torch.tensor([1]))
    for block in range(3):
        # A fresh cache uses compact queries but has no cross-block hits.
        reference_rows = []
        hook = model.layers[0].self_attn.add_q_proj.register_forward_pre_hook(
            lambda module, args, counts=reference_rows: counts.append(len(args[0]))
        )
        try:
            expected = sample_block(model, sample, block, history_cache=HistoryCache(), **options)
        finally:
            hook.remove()
        projected = []
        hook = model.layers[0].self_attn.add_q_proj.register_forward_pre_hook(
            lambda module, args, counts=projected: counts.append(len(args[0]))
        )
        try:
            actual = sample_block(model, sample, block, history_cache=cache, **options)
        finally:
            hook.remove()
        for key in ("video_latents", "action"):
            torch.testing.assert_close(actual[key], expected[key], atol=5e-3, rtol=1e-2)
        if block:
            # Both CFG branches reuse the previous clean rows on their first step.
            assert projected[0] == projected[1] < reference_rows[0]
        else:
            assert projected == reference_rows
        assert projected[2] < projected[0]


@pytest.mark.parametrize("budget", [2, 4, 6, 8, 20])
@pytest.mark.parametrize("fixed_kv", [False, True])
def test_residual_reuse_counts_isolation_and_full_refresh_equivalence(budget, fixed_kv):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import residual_refresh_steps

    torch.manual_seed(123)
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    calls = []
    embeddings, text_norms = [], []
    handle = model.layers[0].register_forward_hook(lambda *args: calls.append(1))
    embedding_hook = model.embed_tokens.register_forward_hook(lambda *args: embeddings.append(1))
    norm_hook = model.norm.register_forward_hook(lambda *args: text_norms.append(1))
    first = None
    options = dict(steps=20, cache_fixed_attention=fixed_kv, attention_backend="dense")
    try:
        for block in (0, 1, 0):
            calls.clear()
            embeddings.clear()
            text_norms.clear()
            actual = sample_block(model, sample, block, residual_reuse_full_steps=budget, **options)
            assert len(calls) == budget
            assert len(embeddings) == budget
            assert not text_norms
            assert actual["denoising"] == dict(
                solver_steps=20,
                full_backbone_steps=list(residual_refresh_steps(budget, 20)),
                full_backbone_evals=budget,
                reuse_evals=20 - budget,
            )
            assert torch.isfinite(actual["action"]).all() and torch.isfinite(actual["video_latents"]).all()
            assert not actual["action"][:, 1].any()
            torch.testing.assert_close(actual["video_latents"][:, :1], sample["video_latents"][:, :1], atol=0, rtol=0)
            if block == 0:
                if first is not None:
                    torch.testing.assert_close(actual["action"], first, atol=0, rtol=0)
                first = actual["action"].clone()
            if budget == 20:
                expected = sample_block(model, sample, block, **options)
                for name in ("action", "video_latents"):
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    finally:
        handle.remove()
        embedding_hook.remove()
        norm_hook.remove()


@torch.no_grad()
def test_residual_reuse_pre_norm_formula_and_guards():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import ResidualCache

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    inputs = fixed_inputs()
    cache = ResidualCache()
    before, after = [], []
    h1 = model.layers[0].register_forward_pre_hook(lambda module, args: before.append(args[1].clone()))
    h2 = model.layers[-1].register_forward_hook(lambda module, args, out: after.append(out[1].clone()))
    model(**inputs, residual_cache=cache)
    h1.remove()
    h2.remove()
    torch.testing.assert_close(cache.residual, after[0].float() - before[0].float(), atol=0, rtol=0)
    inputs["video_tokens"] = inputs["video_tokens"] + 0.25
    captured = []
    hook = model.norm_moe_gen.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    model(**inputs, residual_cache=cache, residual_refresh=False)
    hook.remove()
    refreshed_input = []
    hook = model.layers[0].register_forward_pre_hook(lambda module, args: refreshed_input.append(args[1].clone()))
    model(**inputs)
    hook.remove()
    torch.testing.assert_close(captured[0], refreshed_input[0] + cache.residual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="empty"):
        model(**inputs, residual_cache=ResidualCache(), residual_refresh=False)
    with pytest.raises(ValueError, match="no-grad"):
        model(**inputs, residual_cache=cache, output_text_logits=True)
    with pytest.raises(ValueError, match="no-grad"):
        model.train()(**inputs, residual_cache=cache)


@pytest.mark.parametrize(
    "options",
    [
        {"residual_reuse_full_steps": 3},
        {"residual_reuse_full_steps": 4, "steps": 10},
        {"residual_reuse_full_steps": 4, "guidance": 2},
        {"residual_reuse_full_steps": 4, "fixed_step_sampler_config": {"t_list": [0.9, 0.5]}},
    ],
)
def test_residual_reuse_invalid_sampler_options(options):
    with pytest.raises(ValueError, match="Residual reuse"):
        sample_block(ConstantVelocity(), payload_sample("policy"), 0, **options)


@pytest.mark.parametrize("reference_packing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_history", [False, True])
def test_residual_reuse_matches_without_fixed_kv_and_with_history(reference_packing, dtype, use_history):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache

    torch.manual_seed(123)
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).to(dtype=dtype).eval()
    cache = HistoryCache() if use_history else None
    options = dict(steps=20, residual_reuse_full_steps=6, attention_backend="dense")
    for block in (0, 1):
        expected = sample_block(model, sample, block, reuse_denoise_packing=False, **options)
        actual = sample_block(
            model, sample, block, history_cache=cache, reuse_denoise_packing=not reference_packing, **options
        )
        for key in ("action", "video_latents"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("interval,expected", [(None, 6), ([0, 1], 3)])
def test_denoising_metadata_counts_cfg_evaluations(interval, expected):
    result = sample_block(
        ConstantVelocity(cfg=True),
        payload_sample("policy"),
        0,
        steps=3,
        guidance=2,
        negative_text_ids=torch.tensor([1]),
        guidance_interval=interval,
    )
    assert result["denoising"]["solver_steps"] == 3
    assert result["denoising"]["full_backbone_evals"] == expected


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_qkv_preserves_gqa_layout_and_reuses_packed_weights(variant, dtype, bias):
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(variant, attention_bias=bias)).to(dtype=dtype).eval()
    attention = model.layers[0].self_attn
    before = {name: value.clone() for name, value in model.state_dict().items()}
    hidden = torch.randn(13, model.config.hidden_size, dtype=dtype)
    with torch.no_grad():
        expected = [attention._project_qkv(hidden, branch) for branch in ("und", "gen")]
        model.set_fused_qkv(True)
        for branch, reference in zip(("und", "gen"), expected, strict=True):
            actual = attention._project_qkv(hidden, branch)
            packed = getattr(attention, f"_{branch}_qkv_weight")
            attention._project_qkv(hidden[:3], branch)
            assert getattr(attention, f"_{branch}_qkv_weight") is packed
            for result, target in zip(actual, reference, strict=True):
                assert result.is_contiguous()
                torch.testing.assert_close(result, target, atol=0.02 if dtype == torch.bfloat16 else 1e-6, rtol=1e-5)
            empty = attention._project_qkv(hidden[:0], branch)
            assert [part.shape for part in empty] == [(0, part.shape[-1]) for part in actual]
    assert model.packed_qkv_bytes() > 0
    assert model.state_dict().keys() == before.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    model.set_fused_qkv(False)
    assert model.packed_qkv_bytes() == 0


def test_fused_qkv_rebuilds_after_weight_reload_replacement_and_dtype_change():
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    model.set_fused_qkv(True)
    attention = model.layers[0].self_attn
    hidden = torch.randn(9, 32)
    with torch.no_grad():
        attention._project_qkv(hidden, "gen")
        old = attention._gen_qkv_weight
        state = model.state_dict()
        state["layers.0.self_attn.add_q_proj.weight"] = torch.ones_like(attention.add_q_proj.weight)
        model.load_state_dict(state, strict=True)
        changed = attention._project_qkv(hidden, "gen")
        assert attention._gen_qkv_weight is not old
        torch.testing.assert_close(changed[0], attention.add_q_proj(hidden), atol=1e-6, rtol=1e-5)
        old = attention._gen_qkv_weight
        attention.add_k_proj.weight = torch.nn.Parameter(torch.full_like(attention.add_k_proj.weight, 0.3))
        changed = attention._project_qkv(hidden, "gen")
        assert attention._gen_qkv_weight is not old
        torch.testing.assert_close(changed[1], attention.add_k_proj(hidden), atol=1e-6, rtol=1e-5)
        old = attention._gen_qkv_weight
        attention.add_v_proj.weight.data = torch.full_like(attention.add_v_proj.weight, 0.4)
        changed = attention._project_qkv(hidden, "gen")
        assert attention._gen_qkv_weight is not old
        torch.testing.assert_close(changed[2], attention.add_v_proj(hidden), atol=1e-6, rtol=1e-5)
        model.to(dtype=torch.bfloat16)
        attention._project_qkv(hidden.to(torch.bfloat16), "gen")
        assert attention._gen_qkv_weight.dtype == torch.bfloat16


@pytest.mark.parametrize("training,grad_enabled", [(True, True), (False, True), (True, False)])
def test_fused_qkv_keeps_original_projections_and_gradients_outside_inference(training, grad_enabled):
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).train(training)
    candidate = copy.deepcopy(model)
    candidate.set_fused_qkv(True)
    hidden = torch.randn(7, 32, requires_grad=grad_enabled)
    reference_hidden = hidden.detach().clone().requires_grad_(grad_enabled)
    with torch.set_grad_enabled(grad_enabled):
        expected = model.layers[0].self_attn._project_qkv(reference_hidden, "gen")
        actual = candidate.layers[0].self_attn._project_qkv(hidden, "gen")
        for output, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
        if grad_enabled:
            sum(t.square().sum() for t in actual).backward()
            sum(t.square().sum() for t in expected).backward()
            torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=0, rtol=0)
            for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
                torch.testing.assert_close(
                    getattr(candidate.layers[0].self_attn, name).weight.grad,
                    getattr(model.layers[0].self_attn, name).weight.grad,
                    atol=0,
                    rtol=0,
                )
    assert candidate.packed_qkv_bytes() == 0


def test_fused_qkv_keeps_external_execution_modes_and_rejects_projection_hooks():
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    model.set_fused_qkv(True)
    attention = model.layers[0].self_attn
    hidden = torch.randn(7, 32)
    with torch.inference_mode():
        attention._project_qkv(hidden, "gen")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        attention._project_qkv(hidden, "gen")
    assert model.packed_qkv_bytes() == 0
    handle = attention.add_q_proj.register_forward_hook(lambda _module, _args, output: output + 1)
    try:
        with torch.no_grad(), pytest.raises(ValueError, match="without forward hooks"):
            attention._project_qkv(hidden, "gen")
    finally:
        handle.remove()


@pytest.mark.parametrize("training,grad_enabled", [(True, True), (False, True), (True, False)])
def test_residual_norm_fusion_retains_the_original_training_path(training, grad_enabled):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers import RMSNorm

    with ops_config_scope(eager_ops_config()):
        norm = RMSNorm(32, tiny_config()).train(training)
    # This deliberately has no __call__: the inference-only handle must not run.
    norm.add_op = SimpleNamespace(impl="torch_npu")
    hidden, residual = torch.randn(7, 32), torch.randn(7, 32)
    with torch.set_grad_enabled(grad_enabled):
        normalized, summed = norm.add_residual(hidden, residual)
        torch.testing.assert_close(summed, hidden + residual, atol=0, rtol=0)
        torch.testing.assert_close(normalized, norm(hidden + residual), atol=0, rtol=0)


@pytest.mark.parametrize("task", ["policy", "fd", "id"])
@pytest.mark.parametrize("residual_steps", [None, 8])
@pytest.mark.parametrize("cache_mode", ["none", "block", "history"])
def test_fused_inference_preserves_sampling_cache_boundaries(task, residual_steps, cache_mode):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers import RMSNorm

    torch.manual_seed(123)
    with ops_config_scope(eager_ops_config()):
        reference = Cosmos3Model(tiny_config()).eval()
    candidate = copy.deepcopy(reference)
    candidate.set_fused_qkv(True)
    calls = []

    class AddNormReference:
        impl = "torch_npu"

        def __init__(self, norm):
            self.norm = norm

        def __call__(self, x, residual, weight, *, eps):
            calls.append(x.shape)
            assert weight is self.norm.weight and eps == self.norm.eps
            summed = x + residual
            return self.norm(summed), summed

    for module in candidate.modules():
        if isinstance(module, RMSNorm):
            module.add_op = AddNormReference(module)
    sample = payload_sample(task, actions=64)
    options = dict(steps=20, residual_reuse_full_steps=residual_steps, cache_fixed_attention=cache_mode != "none")
    expected_cache = HistoryCache() if cache_mode == "history" else None
    actual_cache = HistoryCache() if cache_mode == "history" else None
    before = copy.deepcopy(sample)
    for block in (0, 1):
        expected = sample_block(reference, sample, block, history_cache=expected_cache, **options)
        actual = sample_block(candidate, sample, block, history_cache=actual_cache, **options)
        for key in ("action", "video_latents"):
            torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=1e-5)
        assert actual["denoising"] == expected["denoising"]
    assert calls and candidate.packed_qkv_bytes() > 0
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_pipeline_qkv_option_is_explicit_validated_and_does_not_mutate_config():
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    config = dict(source={}, inference_optimizations=dict(fuse_qkv=True))
    pipe = CausalCosmos3Pipeline(model, None, None, lambda text: [1, 2], config)
    assert pipe.model.layers[0].self_attn.fuse_qkv
    assert config == dict(source={}, inference_optimizations=dict(fuse_qkv=True))
    CausalCosmos3Pipeline(model, None, None, lambda text: [1, 2], dict(source={}))
    assert not model.layers[0].self_attn.fuse_qkv
    for invalid in (dict(fuse_qkv="false"), dict(unknown=True), None):
        with pytest.raises(ValueError, match="inference_optimizations"):
            CausalCosmos3Pipeline(
                model, None, None, lambda text: [1, 2], dict(source={}, inference_optimizations=invalid)
            )


@pytest.mark.parametrize(
    "entry", ["infer_causal_cosmos3.sh", "profile_causal_cosmos3.sh", "profile_causal_cosmos3_denoise.sh"]
)
def test_shell_entrypoint_forwards_fusion_ablations(tmp_path, entry):
    import json
    import os
    import subprocess

    python = tmp_path / "record_python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    print('/mock/torch/lib')\n"
        "else:\n"
        "    print(json.dumps({'args': sys.argv[1:], 'card': os.environ['ASCEND_RT_VISIBLE_DEVICES']}))\n"
    )
    python.chmod(0o755)
    root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/inference" / entry),
            "--fuse-qkv",
            "--add-rms-norm-implementation",
            "torch_npu",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(python),
            "ASCEND_RT_VISIBLE_DEVICES": "3",
            "CHECKPOINT": "/model",
            "DATASET": "/dataset",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    recorded = json.loads(result.stdout)
    assert recorded["card"] == "3"
    assert recorded["args"][-3:] == ["--fuse-qkv", "--add-rms-norm-implementation", "torch_npu"]
    assert recorded["args"][0] == "tasks/infer/infer_causal_cosmos3.py"
    assert recorded["args"][recorded["args"].index("--checkpoint") + 1] == "/model"
    assert recorded["args"][recorded["args"].index("--dataset") + 1] == "/dataset"
    assert ("--compare-profile-wall" in recorded["args"]) is (entry != "infer_causal_cosmos3.sh")
    assert ("--denoise-only" in recorded["args"]) is (entry == "profile_causal_cosmos3_denoise.sh")


def test_npu_trace_records_code_stacks_and_schedule(monkeypatch, tmp_path):
    from veomni.models.diffusers.causal_cosmos3.rollout_profiler import npu_trace

    captured = {}
    exports = []

    def profile(**kwargs):
        captured.update(kwargs)
        return "trace"

    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=profile,
        schedule=lambda **kwargs: kwargs,
        tensorboard_trace_handler=lambda path: lambda _profiler: exports.append(("trace", path)),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    assert npu_trace(tmp_path, start=3, active=2) == "trace"
    assert captured["activities"] == ["cpu", "npu"]
    assert captured["schedule"] == dict(wait=2, warmup=1, active=2, repeat=1)
    assert captured["with_stack"] and captured["record_shapes"] and captured["profile_memory"]
    captured["on_trace_ready"](
        SimpleNamespace(export_stacks=lambda path, **kwargs: exports.append(("stacks", path, kwargs)))
    )
    assert exports == [
        ("trace", str(tmp_path)),
        ("stacks", str(tmp_path / "stacks_cpu.txt"), dict(metric="self_cpu_time_total")),
    ]


def test_unprofiled_wall_helper_has_only_boundary_synchronization(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.rollout_profiler as profiling

    events = []
    device = SimpleNamespace(type="wall_backend")
    backend = SimpleNamespace(synchronize=lambda selected: events.append(("sync", selected)))
    monkeypatch.setattr(torch, "wall_backend", backend, raising=False)
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))

    def rollout(**kwargs):
        events.append(("rollout", kwargs))
        return "result"

    pipe = SimpleNamespace(device=device, rollout=rollout)
    options = dict(horizon=640, episode=2)
    result, elapsed = profiling.timed_rollout(pipe, options)
    assert result == "result" and elapsed == 2.5
    assert events == [("sync", device), ("rollout", dict(horizon=640, episode=2, profiler=None)), ("sync", device)]
    assert options == dict(horizon=640, episode=2)


def test_cli_fusion_options_reach_model_loading_and_export(tmp_path, monkeypatch):
    import tasks.infer.infer_causal_cosmos3 as cli
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(device="cpu", ops_implementation=dict(rms_norm_implementation="torch_npu")))
    )
    loaded, exported = [], []

    def load(_checkpoint, config):
        loaded.append(copy.deepcopy(config))
        result = SimpleNamespace(semantics="test", actions=torch.zeros(32, 55), video=torch.zeros(3, 9, 1, 1))
        return SimpleNamespace(
            config=config, rollout=lambda **kwargs: result, last_input_identity={}, loading_report={}
        )

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None), raising=False)
    monkeypatch.setattr(inference.CausalCosmos3Pipeline, "from_pretrained", load)
    monkeypatch.setattr(inference, "save_result", lambda *args, **kwargs: exported.append(kwargs["config"]))
    argv = [
        "inference",
        "--config",
        str(config_path),
        "--checkpoint",
        "unused",
        "--dataset",
        "unused",
        "--mode",
        "rollout",
        "--output",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--fuse-qkv", "--add-rms-norm-implementation", "torch_npu"])
    cli.main()
    assert loaded[0]["inference_optimizations"] == dict(fuse_qkv=True)
    assert exported[0]["ops_implementation"]["add_rms_norm_implementation"] == "torch_npu"
    monkeypatch.setattr(sys, "argv", argv + ["--compare-profile-wall"])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend NPU")
@pytest.mark.parametrize("fuse_qkv,fuse_add_norm", [(True, False), (False, True), (True, True)])
def test_native_fusions_preserve_cached_bf16_sampling(fuse_qkv, fuse_add_norm):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import HistoryCache

    ops = dict(
        attn_implementation="eager", rms_norm_implementation="torch_npu", rotary_pos_emb_implementation="torch_npu"
    )
    config = tiny_config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        mrope_section=[16, 8, 8],
    )
    with ops_config_scope(OpsImplementationConfig(**ops)):
        reference = Cosmos3Model(config).to(device="npu", dtype=torch.bfloat16).eval()
    with ops_config_scope(
        OpsImplementationConfig(**ops, add_rms_norm_implementation="torch_npu" if fuse_add_norm else "eager")
    ):
        candidate = Cosmos3Model(config).to(device="npu", dtype=torch.bfloat16).eval()
    candidate.load_state_dict(reference.state_dict())
    candidate.set_fused_qkv(fuse_qkv)
    sample = payload_sample("policy", actions=64)
    sample = {key: value.to("npu") if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    expected_cache, actual_cache = HistoryCache(), HistoryCache()
    for block in (0, 1):
        options = dict(steps=20, seed=42, attention_backend="npu_tnd")
        expected = sample_block(reference, sample, block, history_cache=expected_cache, **options)
        actual = sample_block(candidate, sample, block, history_cache=actual_cache, **options)
        for key in ("video_latents", "action"):
            assert torch.isfinite(actual[key]).all()
            # A small-model numerical regression, not a physical-action quality threshold.
            torch.testing.assert_close(actual[key], expected[key], atol=0.05, rtol=0.02)


@pytest.mark.parametrize("traced", [False, True])
def test_denoise_timer_excludes_noise_assembly_and_trace_context(monkeypatch, traced):
    import veomni.models.diffusers.causal_cosmos3.rollout_profiler as profiling

    events = []

    class Trace:
        def __enter__(self):
            events.append("trace_enter")

        def __exit__(self, *args):
            events.append("trace_exit")

    clock = iter([1.0, 3.0, 5.0, 9.0] if traced else [3.0, 5.0])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))
    timer = profiling.DenoiseTimer("cpu", trace_factory=(lambda: Trace()) if traced else None)
    timer.backend = SimpleNamespace(synchronize=lambda device: events.append("sync"))
    with timer.stage("noise_initialization", sampling=True):
        events.append("noise")
    with timer.stage("denoising", sampling=True):
        events.append("denoise")
    with timer.stage("result_assembly", sampling=True):
        events.append("assembly")
    assert timer.seconds == 2.0 and timer.context_seconds == (8.0 if traced else 2.0)
    assert events[0] == "noise" and events[-1] == "assembly"
    assert events.count("sync") == (4 if traced else 2)
    if traced:
        assert events.index("trace_enter") < events.index("denoise") < events.index("trace_exit")
    with pytest.raises(ValueError, match="one sample_block"):
        with timer.stage("denoising", sampling=True):
            pass


def test_denoise_trace_has_stacks_and_no_rollout_schedule(monkeypatch, tmp_path):
    from veomni.models.diffusers.causal_cosmos3.rollout_profiler import denoise_trace

    captured = {}
    vendor = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=lambda **kwargs: captured.update(kwargs),
        tensorboard_trace_handler=lambda path: lambda profiler: None,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=vendor))
    denoise_trace(tmp_path)
    assert "schedule" not in captured
    assert captured["with_stack"] and captured["record_shapes"] and captured["profile_memory"]


def test_prepare_denoise_sample_encodes_once_and_crops_history(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.denoise_benchmark as benchmark

    reads, encodes = [], []

    def read(*args, **kwargs):
        reads.append((args, kwargs))
        return dict(real_action_steps=args[3])

    def encode(samples):
        encodes.append(samples)
        return [
            dict(
                video_latents=torch.zeros(2, 35, 2, 2),
                action=torch.zeros(544, 4),
                states=torch.zeros(17, 4),
                action_mask=torch.ones(4).bool(),
                state_mask=torch.ones(4).bool(),
            )
        ]

    monkeypatch.setattr(benchmark, "read_teacher_window", read)
    pipe = SimpleNamespace(
        geometry=Geometry(),
        history=32,
        source={},
        processor=None,
        tokenize=None,
        vae=SimpleNamespace(encode_pack=encode),
        device=torch.device("cpu"),
    )
    sample, block, identity = benchmark.prepare_denoise_sample(pipe, "dataset", 2, 7, 17, prompt="move")
    assert reads[0][0] == ("dataset", 2, 39, 544)
    assert reads[0][1]["task"] == "policy" and reads[0][1]["history"] == 32
    assert len(encodes) == 1 and block == 16
    assert identity["global_block"] == 17 and identity["latent_shape"] == list(sample["video_latents"].shape)


@pytest.mark.parametrize("reuse_history,residual", [(False, None), (True, None), (True, 8)])
def test_denoise_benchmark_repeats_fixed_noise_and_independent_history(monkeypatch, tmp_path, reuse_history, residual):
    from contextlib import nullcontext

    import veomni.models.diffusers.causal_cosmos3.denoise_benchmark as benchmark

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    sample = payload_sample("policy", actions=64)
    before = copy.deepcopy(sample)
    calls = []
    native_sampler = benchmark.sample_block

    def sample_block(*args, **kwargs):
        result = native_sampler(*args, **kwargs)
        calls.append(
            dict(
                block=args[2],
                cache=kwargs.get("history_cache"),
                noise=kwargs["initial_noise"],
                timer=kwargs.get("profiler"),
            )
        )
        return result

    monkeypatch.setattr(benchmark, "sample_block", sample_block)
    monkeypatch.setattr(benchmark, "denoise_trace", lambda path: nullcontext())
    pipe = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        history=32,
        config={},
        sampling_options=dict(steps=20, residual_reuse_full_steps=residual, seed=42),
    )
    report = benchmark.benchmark_denoising(
        pipe, sample, 1, runs=2, warmup_runs=1, reuse_history=reuse_history, trace_directory=tmp_path
    )
    target_calls = [call for call in calls if call["block"] == 1]
    assert len(target_calls) == 4  # one warmup, two unprofiled, one profiled
    assert len({id(call["noise"]) for call in calls}) == 1
    if reuse_history:
        assert calls[0]["block"] == 0 and calls[0]["timer"] is None
        assert len({id(call["cache"]) for call in target_calls}) == 4
        assert next(iter(calls[0]["cache"].branches.values()))[1] == 0
        assert all(next(iter(call["cache"].branches.values()))[1] == 1 for call in target_calls)
    else:
        assert all(call["cache"] is None for call in calls)
    assert report["without_profiling"]["runs"] == 2 and report["with_profiling"]["runs"] == 1
    assert report["without_profiling"]["median_ms"] > 0
    assert report["denoising"]["full_backbone_evals"] == (8 if residual else 20)
    assert report["profiled_output_max_abs_difference"] == dict(video_latents=0, action=0)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_cli_denoise_mode_bypasses_rollout_and_video_export(tmp_path, monkeypatch):
    import json

    import tasks.infer.infer_causal_cosmos3 as cli
    import veomni.models.diffusers.causal_cosmos3.denoise_benchmark as benchmark
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")
    calls = []
    pipe = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(inference.CausalCosmos3Pipeline, "from_pretrained", lambda *args: pipe)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(set_device=lambda device: None, synchronize=lambda device: None), raising=False
    )
    monkeypatch.setattr(
        benchmark, "prepare_denoise_sample", lambda *args, **kwargs: ("encoded", 16, dict(global_block=17))
    )

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(without_profiling=dict(median_ms=7), with_profiling=dict(mean_ms=9))

    monkeypatch.setattr(benchmark, "benchmark_denoising", run)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference",
            "--config",
            str(config),
            "--checkpoint",
            "unused",
            "--dataset",
            "unused",
            "--mode",
            "rollout",
            "--output",
            str(output),
            "--profile",
            "--profile-npu-trace",
            "--compare-profile-wall",
            "--denoise-only",
            "--denoise-block",
            "17",
            "--denoise-runs",
            "2",
            "--max-actions",
            "32",
        ],
    )
    cli.main()
    assert calls[0][0] == (pipe, "encoded", 16)
    assert calls[0][1]["seed_block"] == 17 and calls[0][1]["runs"] == 2
    assert json.loads((output / "denoise_profile.json").read_text())["identity"] == dict(global_block=17)
    assert list(output.iterdir()) == [output / "denoise_profile.json"]
