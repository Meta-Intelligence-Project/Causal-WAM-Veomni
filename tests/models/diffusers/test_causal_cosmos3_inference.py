"""Inference contracts: causality, frozen conditions, CFG and physical feedback."""

import copy
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
    path.write_text(
        yaml.safe_dump(
            dict(device="cpu", source=source, sampling=dict(normalize_cfg=not normalize, use_kv_cache=not normalize))
        )
    )
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
            "--use-kv-cache" if normalize else "--no-use-kv-cache",
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
    assert saved["sampling"]["use_kv_cache"] is normalize
    assert saved["sampling"]["fixed_step_sampler_config"] == dict(t_list=[0.999, 0.5], sample_type="sde")


@pytest.mark.parametrize("task", ["policy", "fd", "id_joint", "id_video", "id_action"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("cfg", [False, True])
def test_kv_cache_matches_uncached_sampling_and_skips_fixed_compute(task, backend, cfg):
    """Real multi-layer network, two blocks, CFG and independent IDM stages."""
    torch.manual_seed(173)
    sample = payload_sample("id" if task.startswith("id_") else task, actions=64, height=3, width=5)
    if task in ("id_joint", "id_video"):
        sample["target_frame_indexes_vision"] = list(range(1, sample["video_latents"].shape[1]))
    if task == "id_video":
        sample["condition_frame_indexes_action"] = list(range(len(sample["action"])))
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    options = dict(
        steps=3,
        attention_backend=backend,
        guidance=2.5 if cfg else 1.0,
        negative_text_ids=torch.tensor([3, 1, 2]),
        normalize_cfg=cfg,
        seed=71,
    )
    und_calls, gen_rows = [], []
    handles = [
        model.layers[0].mlp.register_forward_pre_hook(lambda module, args: und_calls.append(len(args[0]))),
        model.layers[0].mlp_moe_gen.register_forward_pre_hook(lambda module, args: gen_rows.append(len(args[0]))),
    ]
    try:
        for block in (0, 1):
            expected = sample_block(model, sample, block, use_kv_cache=False, **options)
            und_calls.clear()
            gen_rows.clear()
            rng = torch.get_rng_state().clone()
            actual = sample_block(model, sample, block, use_kv_cache=True, **options)
            assert torch.equal(torch.get_rng_state(), rng)
            for name in ("video_latents", "action"):
                torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
            assert actual["seeds"] == expected["seeds"]
            branches = 2 if cfg else 1
            assert len(und_calls) == branches
            assert len(gen_rows) == options["steps"] * branches
            assert max(gen_rows[branches:]) < gen_rows[0]
            # Next block has changed fixed conditions: reuse must not escape
            # the preceding sample_block call, even when shapes stay identical.
            sample["video_latents"] += 0.2
            sample["states"] += 0.3
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("reuse_window,task", [(True, "id"), (True, "fd"), (True, "policy"), (False, "id")])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("cfg", [False, True])
def test_request_cache_reuses_text_and_fixed_window_without_stale_outputs(reuse_window, task, backend, cfg):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import InferenceKVCache

    torch.manual_seed(173)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    session = InferenceKVCache(reuse_window=reuse_window)
    sample = payload_sample(task, actions=64, height=3, width=5)
    if task == "id":
        sample["target_frame_indexes_vision"] = [1, 2, 3, 4]
    options = dict(
        steps=3, attention_backend=backend, guidance=2.5 if cfg else 1, negative_text_ids=torch.tensor([3, 2])
    )
    text_calls, gen_rows = [], []
    handles = [
        model.layers[0].mlp.register_forward_pre_hook(lambda module, args: text_calls.append(len(args[0]))),
        model.layers[0].mlp_moe_gen.register_forward_pre_hook(lambda module, args: gen_rows.append(len(args[0]))),
    ]
    try:
        for index in range(4):
            if not reuse_window:
                # Two rollout blocks, each with video then action. Layout grows
                # and fixed video/state change; only text may survive.
                sample = payload_sample("id", actions=32 if index < 2 else 64, height=3, width=5)
                sample["video_latents"] += index
                sample["states"] += index
                sample["target_frame_indexes_vision"] = (
                    list(range(1, sample["video_latents"].shape[1])) if index % 2 == 0 else []
                )
                if index % 2 == 0:
                    sample["condition_frame_indexes_action"] = list(range(len(sample["action"])))
            block = index % 2 if reuse_window else index // 2
            expected = sample_block(model, sample, block, **options)
            text_calls.clear()
            gen_rows.clear()
            actual = sample_block(model, sample, block, use_kv_cache=True, _cache_session=session, **options)
            for name in ("video_latents", "action"):
                torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
            branches = 2 if cfg else 1
            assert len(text_calls) == (branches if index == 0 else 0)
            if reuse_window and index:
                assert len(set(gen_rows)) == 1  # No fixed-row prefill on later blocks.
        # New text in a new sample must invalidate text, even at identical length.
        sample = copy.deepcopy(sample)
        sample["text_token_ids"] = sample["text_token_ids"] + 1
        expected = sample_block(model, sample, 0, **options)
        text_calls.clear()
        actual = sample_block(model, sample, 0, use_kv_cache=True, _cache_session=session, **options)
        assert len(text_calls) == 1  # The unchanged negative CFG branch remains reusable.
        torch.testing.assert_close(actual["action"], expected["action"], atol=2e-5, rtol=2e-5)
        if cfg:
            sample = copy.deepcopy(sample)
            options["negative_text_ids"] = torch.tensor([4, 3])
            expected = sample_block(model, sample, 0, **options)
            text_calls.clear()
            actual = sample_block(model, sample, 0, use_kv_cache=True, _cache_session=session, **options)
            assert len(text_calls) == 1  # Only the changed negative branch rebuilds.
            torch.testing.assert_close(actual["action"], expected["action"], atol=2e-5, rtol=2e-5)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("task", ["policy", "id"])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("cfg", [False, True])
def test_rollout_history_cache_reuses_growing_prefix_and_idm_stages(task, backend, cfg):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import InferenceKVCache

    torch.manual_seed(173)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    session = InferenceKVCache()
    options = dict(
        steps=3, attention_backend=backend, guidance=2.5 if cfg else 1, negative_text_ids=torch.tensor([3, 2])
    )
    projected = []
    hook = model.layers[0].self_attn.add_k_proj.register_forward_pre_hook(
        lambda module, args: projected.append(len(args[0]))
    )
    history = None
    try:
        for block in range(3):
            sample = payload_sample(task, actions=32 * (block + 1), height=3, width=5)
            start = 1 + block * sample["block_size"]
            if history is not None:
                sample["video_latents"][:, :start] = history
            sample["states"] += block  # State must be recomputed, even on visual hits.
            if task == "id":
                sample["target_frame_indexes_vision"] = list(range(1, sample["video_latents"].shape[1]))
                sample["condition_frame_indexes_action"] = list(range(len(sample["action"])))
            for stage in range(2 if task == "id" else 1):
                expected = sample_block(model, sample, block, **options)
                projected.clear()
                before = session.history_stats()["reused_tokens"]
                actual = sample_block(model, sample, block, use_kv_cache=True, _cache_session=session, **options)
                for name in ("video_latents", "action"):
                    torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
                hit = session.history_stats()["reused_tokens"] - before
                branches = 2 if cfg else 1
                patches = 6  # ceil(3 / 2) * ceil(5 / 2)
                frames = start if stage else (0 if block == 0 else start - sample["block_size"])
                assert hit == frames * patches * branches
                full_gen = sample["token_length"] - len(sample["text_token_ids"])
                assert projected[0] == full_gen - hit // branches
                if task == "id" and stage == 0:
                    sample = dict(
                        sample,
                        video_latents=actual["video_latents"],
                        target_frame_indexes_vision=[],
                        condition_frame_indexes_action=[],
                    )
            history = actual["video_latents"].clone()
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "mutation,expected_frames",
    [
        ("later_frame", 1),
        ("initial_frame", 0),
        ("text", 0),
        ("window", 0),
        ("history_limit", 0),
        ("positions", 1),
    ],
)
def test_rollout_history_invalidates_changed_inputs_and_causal_dependents(mutation, expected_frames):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import InferenceKVCache

    torch.manual_seed(173)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    session = InferenceKVCache()
    sample = payload_sample("id", actions=96, height=3, width=5)
    sample["target_frame_indexes_vision"] = list(range(1, sample["video_latents"].shape[1]))
    options = dict(steps=2, seed_block=1, attention_backend="grouped")
    sample_block(model, sample, 1, use_kv_cache=True, _cache_session=session, **options)
    sample = copy.deepcopy(sample)
    if mutation == "later_frame":
        # Changing frame 2 invalidates sibling frame 1 as well as later blocks.
        sample["video_latents"][:, 2] += 0.1
    elif mutation == "initial_frame":
        sample["video_latents"][:, 0] += 0.1
    elif mutation == "text":
        sample["text_token_ids"] += 1
    elif mutation == "window":
        options["seed_block"] = 2
    elif mutation == "history_limit":
        sample["history_blocks"] += 1
    else:
        sample["conditioning_fps"] *= 2
    expected = sample_block(model, sample, 1, **options)
    actual = sample_block(model, sample, 1, use_kv_cache=True, _cache_session=session, **options)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
    assert session.history_stats()["reused_tokens"] == expected_frames * 6


def test_rollout_history_does_not_publish_partial_prefill():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import InferenceKVCache

    torch.manual_seed(173)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    session = InferenceKVCache()
    sample = payload_sample("policy", actions=96)
    options = dict(steps=2, attention_backend="grouped")
    sample_block(model, sample, 1, use_kv_cache=True, _cache_session=session, **options)

    def fail(module, args, output):
        raise RuntimeError("interrupted prefill")

    hook = model.layers[1].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="interrupted prefill"):
            sample_block(model, sample, 2, use_kv_cache=True, _cache_session=session, **options)
    finally:
        hook.remove()
    assert session.history["conditional"].snapshot is None
    before = session.history_stats()["reused_tokens"]
    expected = sample_block(model, sample, 2, **options)
    actual = sample_block(model, sample, 2, use_kv_cache=True, _cache_session=session, **options)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)
    assert session.history_stats()["reused_tokens"] == before


def test_text_cache_invalidates_changed_position_coordinates():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import BlockKVCache, TextKVCache

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    inputs = fixed_inputs()
    inputs["position_ids"] = inputs["position_ids"].float()
    text = TextKVCache()
    with torch.no_grad():
        model(**inputs, _inference_cache=BlockKVCache(text=text))
        previous = text.layers[0]
        inputs["position_ids"] = inputs["position_ids"].clone()
        rows = [i for i, role in enumerate(inputs["layout"].roles) if role == -1]
        inputs["position_ids"][:, rows] += 0.5
        expected = model(**inputs)
        actual = model(**inputs, _inference_cache=BlockKVCache(text=text))
        assert text.layers[0] is not previous
        torch.testing.assert_close(actual.video, expected.video, atol=2e-5, rtol=2e-5)


def test_kv_cache_fixed_step_sampler_and_default_off():
    sample = payload_sample("id")
    sample["target_frame_indexes_vision"] = [1, 2]
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    options = dict(fixed_step_sampler_config=dict(t_list=[0.999, 0.5, 0.2]), seed=18)
    implicit = sample_block(model, sample, 0, **options)
    disabled = sample_block(model, sample, 0, use_kv_cache=False, **options)
    enabled = sample_block(model, sample, 0, use_kv_cache=True, **options)
    for name in ("video_latents", "action"):
        torch.testing.assert_close(implicit[name], disabled[name], atol=0, rtol=0)
        torch.testing.assert_close(enabled[name], disabled[name], atol=2e-5, rtol=2e-5)


def test_kv_cache_rejects_training_and_autograd():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import BlockKVCache

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    inputs = fixed_inputs()
    with pytest.raises(ValueError, match="disabled gradients"):
        model(**inputs, _inference_cache=BlockKVCache())
    model.train()
    with torch.no_grad(), pytest.raises(ValueError, match="model.eval"):
        model(**inputs, _inference_cache=BlockKVCache())


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_kv_cache_npu_tnd_matches_uncached_idm_sampling():
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import InferenceKVCache

    torch.manual_seed(173)
    sample = payload_sample("id", actions=64, height=3, width=5)
    sample["target_frame_indexes_vision"] = list(range(1, sample["video_latents"].shape[1]))
    sample["video_latents"] = sample["video_latents"].to("npu")
    config = tiny_config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        mrope_section=[16, 8, 8],
    )
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        rotary_pos_emb_implementation="torch_npu",
        rms_norm_implementation="torch_npu",
    )
    with ops_config_scope(ops):
        model = Cosmos3Model(config).to(device="npu", dtype=torch.bfloat16).eval()
    session = InferenceKVCache()
    for block in (0, 1):
        expected = sample_block(model, sample, block, steps=3, attention_backend="npu_tnd")
        actual = sample_block(
            model, sample, block, steps=3, attention_backend="npu_tnd", use_kv_cache=True, _cache_session=session
        )
        for name in ("video_latents", "action"):
            torch.testing.assert_close(actual[name], expected[name], atol=5e-3, rtol=5e-3)
    assert session.history_stats()["reused_tokens"] > 0


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_kv_cache_rejects_non_boolean_config(value):
    _, processor, source = feedback_fixture()
    with pytest.raises(ValueError, match="boolean"):
        CausalCosmos3Pipeline(
            ConstantVelocity(),
            None,
            processor,
            lambda text: [1, 2],
            dict(source=source, sampling=dict(use_kv_cache=value)),
        )


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
def test_rollout_uses_decoded_pixels_and_final_valid_absolute_action(
    monkeypatch, mapped_source, history, horizon, task
):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    initial, processor, source = feedback_fixture()
    initial.horizon = horizon
    if mapped_source:
        processor.state_mapping = processor.action_mapping = [1, 0] + list(range(2, 55))
    seen = []
    sessions = []

    class VAE:
        def encode_pack(self, samples):
            sample = samples[0]
            seen.append(copy.deepcopy(sample))
            return [{**sample, "video_latents": torch.zeros(2, 1 + (sample["video"].shape[1] - 1) // 4, 2, 2)}]

        def decode(self, latents):
            return torch.ones(3, 1 + (latents.shape[1] - 1) * 4, 32, 32, dtype=torch.uint8) * (20 + 30 * len(seen))

    def controlled(model, sample, block, **kwargs):
        sessions.append(kwargs.get("_cache_session"))
        assert sample["mode"] == task
        if task == "inverse_dynamics":
            if sample["condition_frame_indexes_action"]:
                assert sample["target_frame_indexes_vision"] == list(range(1, sample["video_latents"].shape[1]))
                return dict(video_latents=sample["video_latents"] + 3, action=sample["action"], seeds={})
            assert sample["target_frame_indexes_vision"] == []
            assert sample["video_latents"].eq(3).all()
        action = torch.zeros_like(sample["action"])
        action[-32:, 0] = torch.arange(1, 33) / 100
        return dict(video_latents=sample["video_latents"], action=action, seeds={"video": 1, "action": 2})

    monkeypatch.setattr(inference_module, "sample_block", controlled)
    dummy = ConstantVelocity()
    pipe = CausalCosmos3Pipeline(
        dummy,
        VAE(),
        processor,
        lambda prompt: [1, 2],
        dict(source=source, sampling=dict(history=history, use_kv_cache=True)),
    )
    result = rollout(pipe, initial, task=task)
    assert sessions and all(session is sessions[0] for session in sessions)
    assert sessions[0] is not None and not sessions[0].reuse_window
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


@pytest.mark.parametrize("task", ["forward_dynamics", "inverse_dynamics", "policy"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_teacher_forcing_uses_measured_next_block_history_and_state(monkeypatch, task, use_cache):
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
    if task == "inverse_dynamics":
        assert sample["target_frame_indexes_vision"] == [1, 2, 3, 4]
    original = copy.deepcopy(sample)
    seen = []
    sessions = []

    class VAE:
        def encode_pack(self, samples):
            return [{**samples[0], "video_latents": torch.full((2, 5, 2, 2), 0.5)}]

        def decode(self, latents):
            return torch.full((3, 17, 32, 32), 99, dtype=torch.uint8)

    def controlled(model, encoded, block, **kwargs):
        seen.append(block)
        sessions.append(kwargs.get("_cache_session"))
        torch.testing.assert_close(encoded["states"], original["states"], atol=0, rtol=0)
        assert encoded["video_latents"].eq(0.5).all()
        return dict(video_latents=encoded["video_latents"] + 1, action=torch.ones(64, 55), seeds={})

    monkeypatch.setattr(inference_module, "sample_block", controlled)
    pipe = CausalCosmos3Pipeline(
        ConstantVelocity(),
        VAE(),
        processor,
        lambda prompt: [1],
        dict(source=source, sampling=dict(history=1, use_kv_cache=use_cache)),
    )
    pipe.sampling_options["use_kv_cache"] = use_cache
    result = teacher_forcing(pipe, sample)
    assert len(sessions) == 2
    assert sessions[0] is sessions[1]
    assert (sessions[0] is not None) == use_cache
    if use_cache:
        assert sessions[0].reuse_window
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
