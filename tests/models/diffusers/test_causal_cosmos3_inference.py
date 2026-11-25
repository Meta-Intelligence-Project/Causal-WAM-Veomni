"""Inference contracts: causality, frozen conditions, CFG and physical feedback."""

import copy
import json
import math
import sys
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
@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
def test_cli_forwards_sampling_extensions_and_exports_resolved_config(tmp_path, monkeypatch, normalize, task):
    import tasks.infer.infer_causal_cosmos3 as cli
    import veomni.models.diffusers.causal_cosmos3.inference as inference

    _, processor, source = feedback_fixture()
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(dict(device="cpu", source=source, sampling=dict(normalize_cfg=not normalize))))
    saved = {}
    devices = []

    def build(checkpoint, config):
        pipe = CausalCosmos3Pipeline(ConstantVelocity(), None, processor, lambda text: [1, 2], config)
        pipe.last_input_identity = {}
        pipe.loading_report = {}

        def generate(**kwargs):
            assert kwargs["task"] == task
            return SimpleNamespace(semantics="test", actions=torch.zeros(3, 55), video=torch.zeros(3, 1, 32, 32))

        pipe.rollout = generate
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
            "--task",
            task,
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
    assert devices == []
    assert saved["task"] == task
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


def test_id_sampler_uses_current_observations_and_state_without_future_leakage():
    torch.manual_seed(37)
    sample = payload_sample("id", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    baseline = sample_block(model, sample, 0, steps=2, initial_noise=noise)
    torch.testing.assert_close(baseline["video_latents"], sample["video_latents"], atol=0, rtol=0)
    changed = copy.deepcopy(sample)
    changed["video_latents"][:, 3:] += 100
    changed["action"] += 100
    changed["states"][1:] += 100
    result = sample_block(model, changed, 0, steps=2, initial_noise=noise)
    torch.testing.assert_close(result["action"][:32], baseline["action"][:32], atol=1e-6, rtol=1e-6)
    for key, rows in (("video_latents", (slice(None), slice(1, 3))), ("states", slice(0, 1))):
        changed = copy.deepcopy(sample)
        changed[key][rows] += torch.randn_like(changed[key][rows]) * 10
        result = sample_block(model, changed, 0, steps=2, initial_noise=noise)
        assert not torch.allclose(result["action"][:32], baseline["action"][:32])


def test_id_joint_sampler_generates_video_with_observed_and_target_streams_isolated():
    torch.manual_seed(37)
    sample = payload_sample("id", actions=64)
    sample["target_frame_indexes_vision"] = list(range(1, 5))
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    baseline = sample_block(model, sample, 0, steps=2, initial_noise=noise)
    assert not torch.allclose(baseline["video_latents"][:, 1:3], sample["video_latents"][:, 1:3])
    torch.testing.assert_close(baseline["video_latents"][:, :1], sample["video_latents"][:, :1], atol=0, rtol=0)
    torch.testing.assert_close(baseline["video_latents"][:, 3:], sample["video_latents"][:, 3:], atol=0, rtol=0)
    for field, rows in (
        ("video_latents", (slice(None), slice(1, 3))),
        ("states", slice(0, 1)),
    ):
        changed = copy.deepcopy(sample)
        changed[field][rows] += torch.randn_like(changed[field][rows]) * 10
        result = sample_block(model, changed, 0, steps=2, initial_noise=noise)
        torch.testing.assert_close(
            result["video_latents"][:, 1:3], baseline["video_latents"][:, 1:3], atol=1e-6, rtol=1e-6
        )
        assert not torch.allclose(result["action"][:32], baseline["action"][:32])
    changed = copy.deepcopy(sample)
    changed["video_latents"][:, 3:] += 100
    changed["states"][1:] += 100
    changed["action"] += 100
    result = sample_block(model, changed, 0, steps=2, initial_noise=noise)
    torch.testing.assert_close(
        result["video_latents"][:, 1:3], baseline["video_latents"][:, 1:3], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(result["action"][:32], baseline["action"][:32], atol=1e-6, rtol=1e-6)


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


def test_native_dcp_checkpoint_loads_saved_training_config_and_requires_all_weights(tmp_path):
    import torch.distributed.checkpoint as dcp

    config = tiny_config(action_dim=55, state_dim=55)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(config)
    run = tmp_path / "run"
    config.save_pretrained(run / "model_assets")
    state = model.state_dict()
    step = run / "checkpoints/global_step_4000"
    dcp.save({"model": state}, checkpoint_id=step / "model/ckpt")
    for checkpoint in (step, step / "model/ckpt"):
        restored = load_trained_model(checkpoint, dtype=torch.float32)
        assert restored.weight_loading_report["mode"] == "complete"
        assert restored.weight_loading_report["format"] == "veomni_dcp"
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    for index, key in enumerate(("state_proj_in.weight", "action_proj_out.weight")):
        broken_step = run / f"checkpoints/global_step_{index}"
        dcp.save(
            {"model": {name: value for name, value in state.items() if name != key}},
            checkpoint_id=broken_step / "model/ckpt",
        )
        with pytest.raises(ValueError, match="Missing Cosmos3 weights"):
            load_trained_model(broken_step, dtype=torch.float32)


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_inference_preserves_training_ops_on_npu(tmp_path):
    root = Path(__file__).parents[3]
    training_ops = yaml.safe_load((root / "configs/dit/causal_cosmos3_edge_action.yaml").read_text())["model"][
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


@pytest.mark.parametrize("mapped_source", [False, True])
@pytest.mark.parametrize("history,horizon", [(1, 67), (3, 131), (4, 131)])
@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
@pytest.mark.parametrize("structured_prompt", [None, True, False], ids=["default", "structured", "plain"])
def test_rollout_uses_decoded_pixels_and_final_valid_absolute_action(
    monkeypatch, mapped_source, history, horizon, task, structured_prompt
):
    initial, processor, source = feedback_fixture()
    initial.horizon = horizon
    if structured_prompt is not None:
        source["structured_prompt"] = structured_prompt
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

    contexts_seen = []

    def controlled(model, context, stage, **kwargs):
        if stage in ("video", "policy"):
            contexts_seen.append(context)
        if stage == "action":
            assert kwargs["conditioned_video"].eq(3).all()
        action = torch.zeros(context.action_shape)
        action[:, 0] = torch.arange(1, 33) / 100
        return dict(video_latents=torch.full(context.video_shape, 3.0), action=action, seeds={})

    import veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation as generation
    import veomni.models.diffusers.causal_cosmos3.observation_session as observation_module

    monkeypatch.setattr(generation, "generate_block", controlled)
    monkeypatch.setattr(observation_module, "generate_block", controlled)
    dummy = ConstantVelocity()
    pipe = CausalCosmos3Pipeline(
        dummy,
        VAE(),
        processor,
        lambda prompt: list(prompt.encode()),
        dict(source=source, sampling=dict(history=history)),
    )
    result = rollout(pipe, initial, task=task)
    count, context_limit = (horizon + 31) // 32, (history + 1) // 2
    contexts = [min(i, context_limit) for i in range(count)]
    assert result.actions.shape == (horizon, 55)
    assert result.video.shape == (3, 1 + horizon // 4, 32, 32)
    assert all(
        record["sampling"] == ("video_then_inverse_action" if task == "inverse_dynamics" else "joint_policy")
        for record in result.blocks
    )
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
    for index, (sample, context) in enumerate(zip(seen, contexts_seen, strict=True)):
        prompt = bytes(context.text_ids.tolist()).decode()
        if structured_prompt is False:
            assert prompt == initial.caption
        else:
            assert json.loads(prompt)["actions"][0]["description"] == initial.caption + "."
        # VAE receives only history, without future black frames or fabricated rows.
        assert set(sample) == {"video"}
        torch.testing.assert_close(sample["video"], full_history[:, -(1 + 8 * context_limit) :])
        assert context.history_frames == history
        assert context.history_video.shape[1] == 1 + 2 * contexts[index]
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


@pytest.mark.parametrize("task", ["forward_dynamics", "inverse_dynamics", "policy"])
@pytest.mark.parametrize("structured_prompt", [None, True, False], ids=["default", "structured", "plain"])
def test_teacher_forcing_uses_measured_next_block_history_and_state(monkeypatch, task, structured_prompt):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    initial, processor, source = feedback_fixture()
    if structured_prompt is not None:
        source["structured_prompt"] = structured_prompt
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
        tokenize=lambda prompt: list(prompt.encode()),
        caption="move",
        fps=30,
        task=task,
        history=1,
    )
    if structured_prompt is False:
        assert sample["prompt"] == "move"
    else:
        assert json.loads(sample["prompt"])["actions"][0]["description"] == "move."
    assert sample["text_token_ids"].tolist() == list(sample["prompt"].encode())
    sample["ground_truth_actions"] = raw["action"].clone()
    if task == "inverse_dynamics":
        assert sample["target_frame_indexes_vision"] == [1, 2, 3, 4]
    original = copy.deepcopy(sample)
    seen = []

    class VAE:
        def encode_pack(self, samples):
            return [{**samples[0], "video_latents": torch.full((2, 5, 2, 2), 0.5)}]

        def decode(self, latents):
            return torch.full((3, 17, 32, 32), 99, dtype=torch.uint8)

    def controlled(pipe, encoded, block):
        seen.append(block)
        torch.testing.assert_close(encoded["states"], original["states"], atol=0, rtol=0)
        assert encoded["video_latents"].eq(0.5).all()
        return dict(
            video_latents=encoded["video_latents"][:, : 1 + (block + 1) * 2] + 1, action=torch.ones(32, 55), seeds={}
        )

    monkeypatch.setattr(inference_module, "generate_teacher_block", controlled)
    pipe = CausalCosmos3Pipeline(
        ConstantVelocity(), VAE(), processor, lambda prompt: [1], dict(source=source, sampling=dict(history=1))
    )
    result = teacher_forcing(pipe, sample)
    assert seen == [0, 1]
    assert [record["anchor"][0] for record in result.blocks] == [2, 70]
    assert all(record["padded_action_steps"] == record["temporal_padding_frames"] == 0 for record in result.blocks)
    assert result.actions[31, 0] == 102 and result.actions[32, 0] == 170
    torch.testing.assert_close(result.ground_truth_actions, raw["action"], atol=0, rtol=0)


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


def test_robotwin_feedback_requires_explicit_action_as_state_and_matching_channels():
    initial, processor, source = feedback_fixture()
    source.pop("columns")
    with pytest.raises(ValueError, match="target_offset=1"):
        validate_feedback(source, processor, Geometry(), initial.state_mask, initial.action_mask)
    source["rollout_state"] = "action_as_state"
    validate_feedback(source, processor, Geometry(), initial.state_mask, initial.action_mask)
    initial.action_mask[1] = True
    with pytest.raises(ValueError, match="matching state/action masks"):
        validate_feedback(source, processor, Geometry(), initial.state_mask, initial.action_mask)
    source["rollout_state"] = "unknown"
    with pytest.raises(ValueError, match="rollout_state"):
        validate_feedback(source, processor, Geometry(), initial.state_mask, initial.action_mask)


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


def direct_context(sample, block=1):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_inputs import InferenceContext

    return InferenceContext(
        text_ids=sample["text_token_ids"],
        history_video=sample["video_latents"][:, : 1 + block * sample["block_size"]],
        state=sample["states"][block],
        state_mask=sample["state_mask"],
        action_mask=sample["action_mask"],
        block_size=sample["block_size"],
        actions_per_block=32,
        history_frames=sample["history_blocks"],
        video_fps=sample["conditioning_fps"],
        action_fps=sample["conditioning_fps_action"],
    )


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("stage", ["video", "action", "policy", "forward_dynamics"])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_direct_generation_matches_full_window_reference(backend, stage, guidance):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation import generate_block
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_inputs import InferenceInputs
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs

    torch.manual_seed(71)
    sample = payload_sample(
        {"video": "id", "action": "id", "forward_dynamics": "fd", "policy": "policy"}[stage],
        actions=96,
        height=3,
        width=5,
        history=2,
    )
    if stage == "video":
        sample["target_frame_indexes_vision"] = list(range(1, 7))
        sample["condition_frame_indexes_action"] = list(range(96))
    context = direct_context(sample)
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    conditions = {}
    target_noise = {}
    if stage == "action":
        conditions["conditioned_video"] = sample["video_latents"][:, 3:5]
    else:
        target_noise["video"] = noise["video"][:, 3:5]
    if stage == "forward_dynamics":
        conditions["conditioned_action"] = sample["action"][32:64]
    if stage in ("action", "policy"):
        target_noise["action"] = noise["action"][32:64]
    plan = InferenceInputs(context, stage, patch_size=2, **conditions)
    direct = plan.model_inputs(
        target_noise.get("video"), target_noise.get("action"), torch.tensor(500.0), torch.float32
    )
    full, _, _ = pack_model_inputs([sample], [noise["video"]], [noise["action"]], timesteps=[torch.tensor([500.0])])
    # Every direct token has exactly its original role and physical coordinates.
    keep = []
    for i, (m, r, b) in enumerate(zip(full["layout"].modalities, full["layout"].roles, full["layout"].block_ids)):
        if m == 0 or (m == 1 and r == 0 and b < 1):
            keep.append(i)
        elif b == 1:
            if m == 1 and ((stage == "action" and r == 0) or (stage != "action" and r == 1)):
                keep.append(i)
            elif stage != "video" and m in (2, 3):
                keep.append(i)
    assert len(plan.layout.roles) == len(keep) < len(full["layout"].roles)
    for name in ("roles", "block_ids", "frame_ids", "modalities"):
        assert getattr(plan.layout, name) == tuple(getattr(full["layout"], name)[i] for i in keep)
    torch.testing.assert_close(direct["position_ids"], full["position_ids"][:, keep], atol=0, rtol=0)
    if stage == "action":
        assert all(r == 0 for m, r in zip(plan.layout.modalities, plan.layout.roles) if m == 1)
    if stage == "video":
        assert set(plan.layout.modalities) == {0, 1}
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    options = dict(steps=3, guidance=guidance, negative_text_ids=torch.tensor([1]), attention_backend=backend)
    reference = sample_block(model, sample, 1, initial_noise=noise, **options)
    result = generate_block(model, context, stage, initial_noise=target_noise, **conditions, **options)
    if stage != "action":
        torch.testing.assert_close(result["video_latents"], reference["video_latents"][:, 3:5], atol=2e-6, rtol=2e-5)
    if stage in ("action", "policy"):
        torch.testing.assert_close(result["action"], reference["action"][32:64], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("fixed_step", [None, dict(t_list=[0.999, 0.5], sample_type="sde")])
def test_direct_solver_has_only_targets_and_preserves_conditions(fixed_step):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation import (
        generate_action_block,
        generate_video_block,
    )

    sample = payload_sample("id", actions=64)
    context = direct_context(sample)
    conditioned_video = sample["video_latents"][:, 3:5].clone()
    before = copy.deepcopy(context)

    class ActionVelocity(ConstantVelocity):
        def forward(self, **inputs):
            assert len(inputs["action_tokens"]) == 32
            assert len(inputs["state_tokens"]) == 1
            assert {r for m, r in zip(inputs["layout"].modalities, inputs["layout"].roles) if m == 1} == {0}
            assert not inputs["video_timesteps"].any()
            return super().forward(**inputs)

    model = ActionVelocity()
    options = dict(steps=2, seed=7, fixed_step_sampler_config=fixed_step)
    first = generate_action_block(model, context, conditioned_video, **options)
    second = generate_action_block(model, context, conditioned_video, **options)
    if fixed_step is not None:

        class VideoVelocity(ConstantVelocity):
            def forward(self, **inputs):
                assert "action_tokens" not in inputs and "state_tokens" not in inputs
                return SimpleNamespace(video=torch.ones_like(inputs["video_tokens"]))

        video = generate_video_block(VideoVelocity(), context, **options)
        assert first["seeds"]["sampler"] != video["seeds"]["sampler"]
    torch.testing.assert_close(first["action"], second["action"], atol=0, rtol=0)
    assert not first["action"][:, 1].any()
    torch.testing.assert_close(conditioned_video, sample["video_latents"][:, 3:5], atol=0, rtol=0)
    torch.testing.assert_close(context.history_video, before.history_video, atol=0, rtol=0)
    torch.testing.assert_close(context.state, before.state, atol=0, rtol=0)


def test_real_session_waits_for_feedback_and_never_commits_predictions(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.observation_session as observations

    initial, processor, source = feedback_fixture()
    pipe = CausalCosmos3Pipeline(ConstantVelocity(), None, processor, lambda text: [1], dict(source=source))
    session = pipe.session(caption="move", fps=30, state_mask=initial.state_mask, action_mask=initial.action_mask)
    calls = []

    def prediction(pipe, images, state, **options):
        calls.append((images.clone(), state.clone(), options["index"]))
        return dict(actions=torch.full((32, 55), 999.0), video_latents=torch.full((2, 3, 2, 2), 999.0))

    monkeypatch.setattr(observations, "predict_observation", prediction)
    with pytest.raises(ValueError, match="real observation"):
        session.predict()
    session.observe(initial.image, initial.state, timestamp=0.0)
    first = session.predict()
    assert first["action_timestamps"][-1] == pytest.approx(32 / 30)
    first["actions"].fill_(888)
    with pytest.raises(ValueError, match="real observation"):
        session.predict()
    feedback = torch.full((3, 8, 32, 32), 37, dtype=torch.uint8)
    state = initial.state + 5
    with pytest.raises(ValueError, match="timestamp"):
        session.observe(feedback, state, timestamp=0.0)
    session.observe(feedback, state, timestamp=32 / 30)
    feedback.fill_(99)
    state.fill_(99)
    session.predict()
    assert calls[1][0][:, 1:].eq(37).all()
    torch.testing.assert_close(calls[1][1], initial.state + 5)
    assert [call[2] for call in calls] == [0, 1]
    session.reset()
    with pytest.raises(ValueError, match="real observation"):
        session.predict()


def test_observed_prediction_uses_generated_video_only_in_action_stage(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.observation_session as observations

    initial, processor, source = feedback_fixture()
    events = []

    class HistoryVAE:
        def encode_pack(self, samples):
            assert set(samples[0]) == {"video"}
            torch.testing.assert_close(samples[0]["video"], initial.image)
            events.append("encode_history")
            return [dict(video_latents=torch.ones(2, 1, 2, 2))]

    def video(model, context, **options):
        events.append("video")
        return dict(video_latents=torch.full(context.video_shape, 123.0), seeds={"video": 1})

    def action(model, context, conditioned_video, **options):
        events.append("action")
        assert conditioned_video.eq(123).all()
        assert context.history_video.eq(1).all()
        return dict(video_latents=conditioned_video, action=torch.zeros(context.action_shape), seeds={"action": 2})

    monkeypatch.setattr(observations, "generate_video_block", video)
    monkeypatch.setattr(observations, "generate_action_block", action)
    pipe = CausalCosmos3Pipeline(ConstantVelocity(), HistoryVAE(), processor, lambda text: [1], dict(source=source))
    session = pipe.session(caption="move", fps=30, state_mask=initial.state_mask, action_mask=initial.action_mask)
    session.observe(initial.image, initial.state, timestamp=0.0)
    result = session.predict()
    assert events == ["encode_history", "video", "action"]
    assert result["video_latents"][:, 0].eq(1).all()
    assert result["video_latents"][:, 1:].eq(123).all()


@pytest.mark.parametrize("task", ["inverse_dynamics", "policy"])
def test_real_model_session_runs_two_observations_without_training_packing(monkeypatch, task):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import packing

    def forbidden(*args, **kwargs):
        raise AssertionError("Observed-input inference must not construct training samples")

    monkeypatch.setattr(packing, "pack_model_inputs", forbidden)
    monkeypatch.setattr(inference_module, "pack_model_inputs", forbidden)
    initial, processor, source = feedback_fixture()
    encoded = []

    class VAE:
        def encode_pack(self, samples):
            assert set(samples[0]) == {"video"}
            images = samples[0]["video"]
            encoded.append(images.clone())
            return [dict(video_latents=torch.ones(2, 1 + (images.shape[1] - 1) // 4, 2, 2))]

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval()
    pipe = CausalCosmos3Pipeline(
        model, VAE(), processor, lambda text: [1, 2], dict(source=source, sampling=dict(steps=2))
    )
    session = pipe.session(
        caption="move", fps=30.0, state_mask=initial.state_mask, action_mask=initial.action_mask, task=task
    )
    session.observe(initial.image, initial.state, timestamp=0.0)
    first = session.predict()
    state = initial.state.clone()
    state[0] = 30
    session.observe(torch.full((3, 8, 32, 32), 42, dtype=torch.uint8), state, timestamp=32 / 30)
    second = session.predict()
    assert [v.shape[1] for v in encoded] == [1, 9]
    assert encoded[1][:, 1:].eq(42).all()
    assert first["video_latents"].shape[1] == 3 and second["video_latents"].shape[1] == 5
    assert second["anchor"][0] == 30
    for result in (first, second):
        assert result["actions"].shape == (32, 55)
        assert torch.isfinite(result["actions"]).all()
        assert not result["actions"][:, ~initial.action_mask].any()


def test_teacher_idm_actions_condition_on_measured_not_predicted_video(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    sample = payload_sample("id", actions=64)
    sample["target_frame_indexes_vision"] = list(range(1, 5))
    measured = sample["video_latents"][:, 3:5].clone()
    predicted = measured + 100

    def video(model, context, **options):
        return dict(video_latents=predicted, seeds={})

    def action(model, context, conditioned_video, **options):
        torch.testing.assert_close(conditioned_video, measured, atol=0, rtol=0)
        assert context.history_video.shape[1] == 3
        torch.testing.assert_close(context.state, sample["states"][1])
        return dict(action=torch.zeros(32, 4), video_latents=conditioned_video, seeds={})

    monkeypatch.setattr(inference_module, "generate_video_block", video)
    monkeypatch.setattr(inference_module, "generate_action_block", action)
    pipe = SimpleNamespace(model=None, geometry=Geometry(), sampling_options={})
    result = inference_module.generate_teacher_block(pipe, sample, 1)
    torch.testing.assert_close(result["video_latents"][:, 3:], predicted, atol=0, rtol=0)


@pytest.mark.parametrize("entry", ["session", "rollout"])
@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
def test_observed_prediction_rejects_missing_action_statistics_before_encoding(entry, task):
    initial, processor, source = feedback_fixture()
    processor.statistics["action"][2][0] = False

    class UnusedVAE:
        def encode_pack(self, samples):
            pytest.fail("Missing action statistics must fail before encoding or generation")

    pipe = CausalCosmos3Pipeline(ConstantVelocity(), UnusedVAE(), processor, lambda text: [1], dict(source=source))
    with pytest.raises(ValueError, match="action statistics do not cover active source channels"):
        if entry == "rollout":
            rollout(pipe, initial, task=task)
        else:
            session = pipe.session(
                caption="move", fps=30, state_mask=initial.state_mask, action_mask=initial.action_mask, task=task
            )
            session.observe(initial.image, initial.state, timestamp=0.0)
            session.predict()


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("stage", ["video", "action", "policy", "forward_dynamics"])
@pytest.mark.parametrize("history,guidance", [(1, 1.0), (3, 2.0)])
def test_static_kv_matches_native_targets_after_history_eviction(variant, backend, stage, history, guidance):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation import generate_block

    torch.manual_seed(148)
    sample = payload_sample("policy", actions=160, height=3, width=5, history=history)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(variant, num_hidden_layers=3)).eval()
    for block in (0, 1, 3):
        context = direct_context(sample, block)
        conditions = {}
        if stage == "action":
            conditions["conditioned_video"] = sample["video_latents"][:, 1 + 2 * block : 3 + 2 * block]
        if stage == "forward_dynamics":
            conditions["conditioned_action"] = sample["action"][32 * block : 32 * (block + 1)]
        options = dict(
            steps=3, guidance=guidance, negative_text_ids=torch.tensor([3, 4, 5]), attention_backend=backend
        )
        rng = torch.get_rng_state()
        expected = generate_block(model, context, stage, **conditions, **options)
        actual = generate_block(model, context, stage, kv_cache=True, **conditions, **options)
        torch.testing.assert_close(actual["video_latents"], expected["video_latents"], atol=2e-6, rtol=3e-5)
        if stage in ("action", "policy"):
            torch.testing.assert_close(actual["action"], expected["action"], atol=2e-6, rtol=3e-5)
            assert not actual["action"][:, 1].any()
        assert actual["seeds"] == expected["seeds"]
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("fixed_step", [False, True])
def test_static_kv_reuses_prefix_between_idm_stages_but_recomputes_state(fixed_step):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation import (
        generate_action_block,
        generate_video_block,
    )

    sample = payload_sample("id", actions=96, height=3, width=5, history=1)
    context = direct_context(sample)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    text_calls, state_calls, projected_rows = [], [], []
    handles = [
        model.embed_tokens.register_forward_pre_hook(lambda module, args: text_calls.append(len(args[0]))),
        model.state_proj_in.register_forward_pre_hook(lambda module, args: state_calls.append(len(args[0]))),
        model.layers[0].self_attn.add_k_proj.register_forward_pre_hook(
            lambda module, args: projected_rows.append(len(args[0]))
        ),
    ]
    caches = {}
    options = dict(kv_cache=True, observation_caches=caches, steps=3, guidance=2, negative_text_ids=torch.tensor([3]))
    if fixed_step:
        options["fixed_step_sampler_config"] = dict(t_list=[0.9, 0.5, 0.2], sample_type="sde")
    try:
        video = generate_video_block(model, context, **options)
        assert len(text_calls) == 2 and not state_calls
        snapshots = [
            (entry.video_k.clone(), entry.video_v.clone()) for cache in caches.values() for entry in cache.layers
        ]
        text_calls.clear()
        projected_rows.clear()
        cached = generate_action_block(model, context, video["video_latents"], **options)
        assert not text_calls and len(state_calls) == 6
        assert projected_rows == [12 + 32 + 1] * 6  # No historical token is projected during the action loop.
        for entry, (k, v) in zip([e for c in caches.values() for e in c.layers], snapshots, strict=True):
            torch.testing.assert_close(entry.video_k, k, atol=0, rtol=0)
            torch.testing.assert_close(entry.video_v, v, atol=0, rtol=0)
        reference_options = {
            key: value for key, value in options.items() if key not in ("kv_cache", "observation_caches")
        }
        expected = generate_action_block(model, context, video["video_latents"], **reference_options)
        torch.testing.assert_close(cached["action"], expected["action"], atol=2e-6, rtol=3e-5)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("change", ["text", "history", "positions", "budget", "model", "backend"])
@torch.no_grad()
def test_static_kv_rejects_changed_prefix_instead_of_reusing_stale_values(change):
    from dataclasses import replace

    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import ObservationKVCache
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_inputs import InferenceInputs

    sample = payload_sample("id", actions=96)
    context = direct_context(sample)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    initial = InferenceInputs(context, "video", patch_size=2)
    cache = ObservationKVCache(model, initial, attention_backend="dense")
    backend = "dense"
    if change == "text":
        context = replace(context, text_ids=context.text_ids + 1)
    elif change == "history":
        context = replace(context, history_video=context.history_video + 1)
    elif change == "budget":
        context = replace(context, history_frames=context.history_frames + 1)
    elif change == "model":
        model = copy.deepcopy(model)
    elif change == "backend":
        backend = "grouped"
    inputs = InferenceInputs(context, "video", patch_size=2)
    if change == "positions":
        inputs.positions = inputs.positions + 1
    with pytest.raises(ValueError, match="same model, text, observed window and positions"):
        cache.bind(inputs, model=model, attention_backend=backend)


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_static_kv_npu_tnd_matches_native_idm_and_policy(dtype):
    from dataclasses import replace

    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.generation import generate_block

    torch.manual_seed(18)
    sample = payload_sample("policy", actions=96, height=3, width=5, history=3)
    context = direct_context(sample, block=2)
    context = replace(
        context,
        **{key: getattr(context, key).to("npu") for key in ("history_video", "state", "state_mask", "action_mask")},
    )
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval().to(device="npu", dtype=dtype)
    for stage in ("video", "action", "policy", "forward_dynamics"):
        conditions = {}
        if stage == "action":
            conditions["conditioned_video"] = sample["video_latents"][:, 5:7].to("npu")
        elif stage == "forward_dynamics":
            conditions["conditioned_action"] = sample["action"][64:96].to("npu")
        options = dict(steps=3, guidance=2, negative_text_ids=torch.tensor([3]), attention_backend="npu_tnd")
        expected = generate_block(model, context, stage, **conditions, **options)
        actual = generate_block(model, context, stage, kv_cache=True, **conditions, **options)
        atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (0.02, 0.02)
        torch.testing.assert_close(actual["video_latents"], expected["video_latents"], atol=atol, rtol=rtol)
        if stage in ("action", "policy"):
            torch.testing.assert_close(actual["action"], expected["action"], atol=atol, rtol=rtol)
