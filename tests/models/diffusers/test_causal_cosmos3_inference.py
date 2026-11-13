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
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.inference_cache import OnlinePolicyKVCache
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.unipc import FlowUniPCMultistepScheduler
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import (
    CausalCosmos3Pipeline,
    create_policy_caches,
    load_trained_model,
    rollout,
    sample_block,
    teacher_forcing,
)
from veomni.models.diffusers.causal_cosmos3.inference import (
    compact_policy_sample as build_compact_policy_sample,
)
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def compact_policy_sample(sample, revision):
    """Replay measured latent blocks through the production compact adapter."""
    size = sample["block_size"]
    start = 0 if revision == 0 else 1 + (revision - 1) * size
    measured = sample["video_latents"][:, start : 1 + revision * size]
    return build_compact_policy_sample(
        dict(sample, mode="policy"),
        measured,
        revision,
        actions_per_block=len(sample["action"]) // len(sample["states"]),
        patch_size=2,
    )


def new_cache(model, sample, *, text_ids=None, **options):
    return OnlinePolicyKVCache(
        model,
        sample["video_latents"][:, :1],
        text_ids=sample["text_token_ids"] if text_ids is None else text_ids,
        block_size=sample["block_size"],
        actions_per_block=len(sample["action"]),
        history=sample["history_blocks"],
        conditioning_fps=sample["conditioning_fps"],
        conditioning_fps_action=sample["conditioning_fps_action"],
        **options,
    )


def branch_caches(model, sample, **options):
    return create_policy_caches(
        SimpleNamespace(
            model=model,
            geometry=Geometry(actions_per_block=len(sample["action"])),
            history=sample["history_blocks"],
            sampling_options=dict(
                guidance=options.get("guidance", 1),
                negative_text_ids=options.get("negative_text_ids"),
                attention_backend=options.get("attention_backend", "grouped"),
                position_options=options.get("position_options"),
            ),
        ),
        sample["video_latents"][:, :1],
        text_ids=sample["text_token_ids"],
        conditioning_fps=sample["conditioning_fps"],
        conditioning_fps_action=sample["conditioning_fps_action"],
    )


@torch.no_grad()
def test_cache_initializes_from_one_real_latent_and_predicts_with_explicit_state():
    torch.manual_seed(314)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    initial = torch.randn(2, 1, 3, 5)
    cache = OnlinePolicyKVCache(
        model,
        initial,
        text_ids=torch.tensor([1, 2]),
        block_size=2,
        actions_per_block=32,
        history=3,
        conditioning_fps=7.5,
        conditioning_fps_action=30,
    )
    assert cache.frame_ids == (0,) and cache.next_frame == 1
    assert not hasattr(cache, "sample")
    noisy_video, noisy_action = torch.randn(2, 2, 3, 5), torch.randn(32, 4)
    state = torch.zeros(1, 4)
    masks = dict(action_mask=torch.tensor([True, False, True, True]), state_mask=torch.ones(4, dtype=torch.bool))
    snapshots = [(entry.video_k.clone(), entry.video_v.clone()) for entry in cache.layers]
    tables = cache.prediction_tables
    first = cache.predict(noisy_video, noisy_action, state, torch.tensor(500.0), revision=0, **masks)
    changed = cache.predict(noisy_video, noisy_action, state + 1, torch.tensor(500.0), revision=0, **masks)
    assert first[0].shape == noisy_video.shape and first[1].shape == noisy_action.shape
    assert not first[1][:, 1].any() and not changed[1][:, 1].any()
    assert not torch.allclose(first[1], changed[1], atol=1e-7, rtol=1e-7)
    assert cache.prediction_tables is tables and cache.revision == 0 and cache.next_frame == 1
    for entry, (key, value) in zip(cache.layers, snapshots, strict=True):
        torch.testing.assert_close(entry.video_k, key, atol=0, rtol=0)
        torch.testing.assert_close(entry.video_v, value, atol=0, rtol=0)
    # Updating observations needs neither state nor target placeholders.
    cache.update(torch.randn(2, 2, 3, 5), revision=1)
    with pytest.raises(ValueError, match="current revision"):
        cache.predict(noisy_video, noisy_action, state, torch.tensor(500.0), revision=0, **masks)
    # A different prediction state is accepted at the new observation boundary.
    cache.predict(noisy_video, noisy_action, state + 2, torch.tensor(500.0), revision=1, **masks)


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
@pytest.mark.parametrize("history,guidance", [(1, 1), (3, 2), (32, 1)])
@pytest.mark.parametrize("actions_per_block", [16, 32])
@torch.no_grad()
def test_online_cache_matches_full_measured_episode_after_multiple_evictions(
    variant, history, guidance, actions_per_block
):
    torch.manual_seed(148)
    sample = payload_sample(
        "policy",
        actions=5 * actions_per_block,
        height=3,
        width=5,
        history=history,
        geometry=Geometry(actions_per_block=actions_per_block),
    )
    size = sample["block_size"]
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(variant)).eval()
    caches = {}
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    negative = torch.tensor([3, 4, 5])  # Different text length changes each branch's visual offset.
    for revision in range(5):
        compact = compact_policy_sample(sample, revision)
        if revision == 0:
            caches["cond"] = new_cache(model, compact)
            if guidance != 1:
                caches["uncond"] = new_cache(model, compact, text_ids=negative)
        else:
            observed = sample["video_latents"][:, 1 + (revision - 1) * size : 1 + revision * size]
            for cache in caches.values():
                cache.update(observed, revision=revision)
        snapshots = [
            (entry.video_k.clone(), entry.video_v.clone()) for cache in caches.values() for entry in cache.layers
        ]
        options = dict(steps=2, guidance=guidance, negative_text_ids=negative)
        expected = sample_block(model, sample, revision, initial_noise=noise, **options)
        actual = sample_block(
            model,
            compact,
            0,
            kv_cache=caches,
            initial_noise=dict(
                video=noise["video"][:, size * revision : size * revision + 1 + size],
                action=noise["action"][actions_per_block * revision : actions_per_block * (revision + 1)],
            ),
            **options,
        )
        torch.testing.assert_close(
            actual["video_latents"][:, 1:],
            expected["video_latents"][:, 1 + size * revision : 1 + size * (revision + 1)],
            atol=3e-6,
            rtol=3e-6,
        )
        torch.testing.assert_close(
            actual["action"],
            expected["action"][actions_per_block * revision : actions_per_block * (revision + 1)],
            atol=3e-6,
            rtol=3e-6,
        )
        for cache in caches.values():
            start = 1 + size * revision
            assert cache.frame_ids == tuple(range(max(0, start - history), start))
            assert not hasattr(cache, "sample") and cache.next_frame == start
            assert all(len(entry.video_k) <= history * cache.patches for entry in cache.layers)
        entries = [entry for cache in caches.values() for entry in cache.layers]
        for entry, (k, v) in zip(entries, snapshots, strict=True):
            torch.testing.assert_close(entry.video_k, k, atol=0, rtol=0)
            torch.testing.assert_close(entry.video_v, v, atol=0, rtol=0)


@torch.no_grad()
def test_online_cache_rejects_duplicate_partial_and_changed_feedback():
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    cache = new_cache(model, compact_policy_sample(sample, 0))
    observed = sample["video_latents"][:, 1:3]
    with pytest.raises(ValueError, match="advance by one"):
        cache.update(observed, revision=0)
    with pytest.raises(ValueError, match="complete measured"):
        cache.update(observed[:, :1], revision=1)
    assert cache.revision == 0 and cache.next_frame == 1
    cache.update(observed, revision=1)
    with pytest.raises(ValueError, match="advance by one"):
        cache.update(observed, revision=1)


def test_policy_cache_joint_prefill_matches_full_layer_kv_and_predict_is_readonly(monkeypatch):
    torch.manual_seed(91)
    sample = payload_sample("policy", actions=96, height=3, width=5, history=3)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(num_hidden_layers=3)).eval()
    inputs, _, _ = pack_model_inputs(
        [sample], [sample["video_latents"]], [sample["action"]], timesteps=[torch.zeros(7)]
    )
    expected, row_counts = {}, []
    for index, layer in enumerate(model.layers):
        original = layer.self_attn.project_generation

        def project(gen, tables, original=original, index=index):
            row_counts.append(len(gen))
            result = original(gen, tables)
            expected[index] = tuple(t.detach().clone() for t in result[1:])
            return result

        monkeypatch.setattr(layer.self_attn, "project_generation", project)
    with torch.no_grad():
        model(**inputs)
        full_kv = dict(expected)
        row_counts.clear()
        compact = compact_policy_sample(sample, 0)
        cache = new_cache(model, compact)
        for revision in (1, 2):
            compact = compact_policy_sample(sample, revision)
            cache.update(sample["video_latents"][:, 1 + (revision - 1) * 2 : 1 + revision * 2], revision=revision)
        # 6 patches/frame; block 2 sees frames 2,3,4 after frame-level eviction.
        for i, entry in enumerate(cache.layers):
            torch.testing.assert_close(entry.video_k, full_kv[i][0][12:30], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(entry.video_v, full_kv[i][1][12:30], atol=1e-6, rtol=1e-6)
        snapshots = [(e.video_k.clone(), e.video_v.clone()) for e in cache.layers]
        for time in (900.0, 200.0):
            cache.predict(
                compact["video_latents"][:, 1:],
                compact["action"],
                compact["states"],
                torch.tensor(time),
                action_mask=compact["action_mask"],
                state_mask=compact["state_mask"],
                revision=cache.revision,
            )
        assert max(row_counts) == 2 * 6 + 32 + 1  # Current block only, no old/future token projections.
        for entry, (k, v) in zip(cache.layers, snapshots, strict=True):
            torch.testing.assert_close(entry.video_k, k, atol=0, rtol=0)
            torch.testing.assert_close(entry.video_v, v, atol=0, rtol=0)


def test_policy_cache_failed_prefill_does_not_commit_partial_layers(monkeypatch):
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    with torch.no_grad():
        initial = compact_policy_sample(sample, 0)
        cache = new_cache(model, initial)
        before, tables = cache.layers, tuple(value.clone() for value in cache.prediction_tables)
        observed = sample["video_latents"][:, 1:3]
        original = model.layers[1].self_attn.project_generation

        def fail(*args):
            raise RuntimeError("prefill failed")

        monkeypatch.setattr(model.layers[1].self_attn, "project_generation", fail)
        with pytest.raises(RuntimeError, match="prefill failed"):
            cache.update(observed, revision=1)
        assert cache.layers is before and cache.next_frame == 1 and cache.frame_ids == (0,)
        assert cache.revision == 0 and not hasattr(cache, "sample")
        for actual, expected in zip(cache.prediction_tables, tables, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        monkeypatch.setattr(model.layers[1].self_attn, "project_generation", original)
        cache.update(observed, revision=1)
        assert cache.next_frame == 3 and cache.revision == 1


def test_policy_cache_rejects_training_and_conditioned_tasks():
    sample = compact_policy_sample(payload_sample("policy"), 0)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    with pytest.raises(RuntimeError, match="eval.*no_grad"):
        new_cache(model, sample)
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match="eval.*no_grad"):
            new_cache(model, sample)
        model.eval()
        caches = branch_caches(model, sample)
        for conditions in (
            dict(condition_frame_indexes_action=[0]),
            dict(condition_frame_indexes_vision=[0, 1]),
        ):
            with pytest.raises(ValueError, match="Policy conditions"):
                sample_block(model, dict(sample, **conditions), 0, steps=1, kv_cache=caches)
        with pytest.raises(ValueError, match="exactly one real latent"):
            OnlinePolicyKVCache(
                model,
                sample["video_latents"],
                text_ids=sample["text_token_ids"],
                block_size=2,
                actions_per_block=32,
                history=1,
                conditioning_fps=7.5,
                conditioning_fps_action=30,
            )
        with pytest.raises(ValueError, match="Initialize and update"):
            sample_block(model, sample, 0, steps=1, kv_cache={})
        with pytest.raises(ValueError, match="compact block zero"):
            sample_block(model, sample, 1, steps=1, kv_cache={})


@pytest.mark.parametrize("change", ["negative_text", "position", "backend", "revision", "geometry", "action_frames"])
def test_policy_cache_rejects_changed_branch_inputs(change):
    sample = compact_policy_sample(payload_sample("policy", actions=64), 0)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    options = dict(steps=1, guidance=2, negative_text_ids=torch.tensor([3]))
    with torch.no_grad():
        caches = branch_caches(model, sample, **options)
    sample_block(model, sample, 0, kv_cache=caches, **options)
    if change == "negative_text":
        options["negative_text_ids"] = torch.tensor([4, 5])
    elif change == "position":
        options["position_options"] = dict(base_fps=12.0)
    elif change == "backend":
        options["attention_backend"] = "dense"
    elif change == "revision":
        sample = dict(sample, cache_revision=1)
    elif change == "geometry":
        sample = dict(sample, conditioning_fps=60)
    else:
        sample = dict(sample, action_frame_ids=[1] * 32)
    with pytest.raises(ValueError, match="Policy KV cache"):
        sample_block(model, sample, 0, kv_cache=caches, **options)


@pytest.mark.launch_plan(provider=LaunchSpec())
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")
def test_policy_cache_npu_fp32_parity_after_eviction():
    torch.manual_seed(18)
    sample = payload_sample("policy", actions=96, height=3, width=5, history=3)
    for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
        sample[key] = sample[key].to("npu")
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval().to("npu")
    caches = {}
    noise = dict(video=torch.randn_like(sample["video_latents"]), action=torch.randn_like(sample["action"]))
    options = dict(steps=3, attention_backend="npu_tnd", guidance=2, negative_text_ids=torch.tensor([3]))
    with torch.no_grad():
        for block in range(3):
            compact = compact_policy_sample(sample, block)
            if block == 0:
                caches = branch_caches(model, compact, **options)
            else:
                for cache in caches.values():
                    cache.update(sample["video_latents"][:, 1 + (block - 1) * 2 : 1 + block * 2], revision=block)
            expected = sample_block(model, sample, block, initial_noise=noise, **options)
            actual = sample_block(
                model,
                compact,
                0,
                kv_cache=caches,
                initial_noise=dict(
                    video=noise["video"][:, 2 * block : 2 * block + 3],
                    action=noise["action"][32 * block : 32 * block + 32],
                ),
                **options,
            )
            torch.testing.assert_close(
                actual["video_latents"][:, 1:],
                expected["video_latents"][:, 1 + 2 * block : 3 + 2 * block],
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                actual["action"], expected["action"][32 * block : 32 * block + 32], atol=1e-5, rtol=1e-5
            )


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
    assert pipe.kv_cache is True and pipe.config["sampling"]["kv_cache"] is True
    expected = options["fixed_step_sampler_config"] if override else model.config.fixed_step_sampler_config
    assert pipe.sampling_options["fixed_step_sampler_config"] == expected
    assert pipe.config["sampling"]["fixed_step_sampler_config"] == expected
    assert pipe.sampling_options["guidance"] == pipe.config["sampling"]["guidance"] == 1
    assert pipe.sampling_options["guidance_interval"] == [200, 800]
    assert pipe.sampling_options["normalize_cfg"]
    assert config["sampling"]["guidance"] == 3


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode", ["teacher_forcing", "rollout"])
def test_cli_forwards_sampling_extensions_and_exports_resolved_config(tmp_path, monkeypatch, normalize, mode):
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
        pipe.rollout = lambda *args, **kwargs: SimpleNamespace(
            semantics="test", actions=torch.zeros(3, 55), video=torch.zeros(3, 1, 32, 32)
        )
        pipe.teacher_forcing = pipe.rollout
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
            mode,
            *(["--no-kv-cache"] if mode == "rollout" else []),
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
    assert saved["sampling"]["kv_cache"] is (mode == "teacher_forcing")
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


@pytest.mark.parametrize("use_cache", [False, True])
def test_real_model_cannot_read_current_clean_targets_or_future_ground_truth(use_cache):
    sample = payload_sample("policy", actions=64)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).eval()
    if use_cache:
        sample = compact_policy_sample(sample, 0)
    noise = {"video": torch.randn_like(sample["video_latents"]), "action": torch.randn_like(sample["action"])}
    with torch.no_grad():
        caches = branch_caches(model, sample) if use_cache else None
    first = sample_block(model, sample, 0, steps=2, initial_noise=noise, kv_cache=caches)
    changed = copy.deepcopy(sample)
    changed["video_latents"][:, 1:] += 500
    changed["action"] += 500
    changed["states"][1:] += 500
    with torch.no_grad():
        caches = branch_caches(model, changed) if use_cache else None
    second = sample_block(model, changed, 0, steps=2, initial_noise=noise, kv_cache=caches)
    torch.testing.assert_close(first["video_latents"][:, 1:3], second["video_latents"][:, 1:3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first["action"][:32], second["action"][:32], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("format_name", ["safetensors", "bin", "dcp"])
def test_complete_checkpoint_loads_and_missing_action_or_state_fails(tmp_path, format_name):
    import torch.distributed.checkpoint as dcp

    config = tiny_config(
        action_dim=55, state_dim=55, fixed_step_sampler_config=dict(t_list=[0.999, 0.5], sample_type="sde")
    )
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(config)
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    config.save_pretrained(checkpoint)

    def save_weights(tensors):
        if format_name == "dcp":
            dcp.save({"model": tensors}, checkpoint_id=checkpoint, no_dist=True)
        elif format_name == "bin":
            torch.save(tensors, checkpoint / "pytorch_model.bin")
        else:
            save_file(tensors, checkpoint / "model.safetensors")

    state = model.state_dict()
    save_weights(state)
    restored = load_trained_model(checkpoint, dtype=torch.float32)
    assert restored.weight_loading_report["mode"] == "complete"
    assert restored.config.fixed_step_sampler_config == config.fixed_step_sampler_config
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    inputs = fixed_inputs()
    for key in ("action_tokens", "state_tokens", "action_mask", "state_mask"):
        inputs[key] = torch.nn.functional.pad(inputs[key], (0, 51))
    with torch.no_grad():
        expected = model(**inputs, attention_backend="dense")
        actual = restored(**inputs, attention_backend="dense")
    torch.testing.assert_close(actual.video, expected.video, atol=0, rtol=0)
    torch.testing.assert_close(actual.action, expected.action, atol=0, rtol=0)
    for key in ("state_proj_in.weight", "action_proj_out.weight"):
        broken = {name: value for name, value in state.items() if name != key}
        save_weights(broken)
        with pytest.raises(ValueError, match="Missing Cosmos3 weights"):
            load_trained_model(checkpoint, dtype=torch.float32)


@pytest.mark.parametrize("damage", ["smaller", "larger", "unexpected"])
def test_dcp_rejects_incompatible_weights_before_reading_tensors(tmp_path, monkeypatch, damage):
    import torch.distributed.checkpoint as dcp

    config = tiny_config(action_dim=55, state_dim=55)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(config)
    config.save_pretrained(tmp_path)
    state = model.state_dict()
    if damage == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        value = state["action_proj_out.weight"]
        state["action_proj_out.weight"] = value[:-1] if damage == "smaller" else torch.cat((value, value[:1]))
    dcp.save({"model": state}, checkpoint_id=tmp_path, no_dist=True)

    def fail_load(**kwargs):
        pytest.fail("Incompatible DCP tensors must not be loaded")

    monkeypatch.setattr(dcp, "load", fail_load)
    with pytest.raises(ValueError, match="Unexpected Cosmos3 weights|Shape mismatch"):
        load_trained_model(tmp_path, dtype=torch.float32)


def test_dcp_preserves_nonpersistent_buffers_and_requires_checkpoint_config(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    from veomni.models.diffusers.causal_cosmos3 import inference

    config = tiny_config(action_dim=55, state_dim=55)
    assets = tmp_path / "assets"
    config.save_pretrained(assets)
    checkpoint = tmp_path / "model" / "ckpt"
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(config)
    dcp.save({"model": model.state_dict()}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(FileNotFoundError, match="model_assets/config.json") as error:
        load_trained_model(checkpoint)
    assert str(checkpoint / "config.json") in str(error.value)
    (checkpoint / "config.json").write_bytes((assets / "config.json").read_bytes())
    original_builder = inference._empty_model
    buffer = torch.arange(8, dtype=torch.float32)

    def build_with_buffer(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        result.register_buffer("derived_buffer", buffer.clone(), persistent=False)
        return result

    monkeypatch.setattr(inference, "_empty_model", build_with_buffer)
    restored = load_trained_model(checkpoint, dtype=torch.bfloat16)
    torch.testing.assert_close(restored.derived_buffer, buffer, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        actual = restored.state_dict()[key]
        torch.testing.assert_close(actual, value.to(actual.dtype), atol=0, rtol=0)
    assert restored.weight_loading_report["format"] == "dcp"
    assert not torch.distributed.is_initialized()


def test_pipeline_rejects_external_model_config(tmp_path):
    with pytest.raises(ValueError, match="model_config is no longer supported"):
        CausalCosmos3Pipeline.from_pretrained(tmp_path, {"model_config": "/another/run/config.json"})


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
def test_rollout_uses_decoded_pixels_and_final_valid_absolute_action(monkeypatch, mapped_source, history, horizon):
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

    def controlled(model, sample, block, **kwargs):
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
        dict(source=source, sampling=dict(history=history, kv_cache=False)),
    )
    result = rollout(pipe, initial)
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


@pytest.mark.parametrize("task", ["forward_dynamics", "inverse_dynamics", "policy"])
def test_teacher_forcing_uses_measured_next_block_history_and_state(monkeypatch, task):
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

    def controlled(model, encoded, block, **kwargs):
        seen.append(block)
        torch.testing.assert_close(encoded["states"], original["states"], atol=0, rtol=0)
        assert encoded["video_latents"].eq(0.5).all()
        return dict(video_latents=encoded["video_latents"] + 1, action=torch.ones(64, 55), seeds={})

    monkeypatch.setattr(inference_module, "sample_block", controlled)
    pipe = CausalCosmos3Pipeline(
        ConstantVelocity(),
        VAE(),
        processor,
        lambda prompt: [1],
        dict(source=source, sampling=dict(history=1, kv_cache=False)),
    )
    result = teacher_forcing(pipe, sample)
    assert seen == [0, 1]
    assert [record["anchor"][0] for record in result.blocks] == [2, 70]
    assert all(record["padded_action_steps"] == record["temporal_padding_frames"] == 0 for record in result.blocks)
    assert result.actions[31, 0] == 102 and result.actions[32, 0] == 170
    torch.testing.assert_close(result.ground_truth_actions, raw["action"], atol=0, rtol=0)


@pytest.mark.parametrize("horizon", [3, 64, 67])
@pytest.mark.parametrize("guidance", [1, 2])
def test_cached_teacher_forcing_replays_only_arrived_rgb_and_measured_state(monkeypatch, horizon, guidance):
    import veomni.models.diffusers.causal_cosmos3.inference as inference_module

    initial, processor, source = feedback_fixture(horizon)
    padded = ((horizon + 31) // 32) * 32
    states = torch.zeros(padded + 1, 55)
    states[:, 0] = 2 + torch.arange(padded + 1)
    pixels = torch.arange(1 + padded // 4, dtype=torch.uint8)[None, :, None, None].expand(3, -1, 32, 32).clone()
    raw = dict(
        video=pixels,
        state=states,
        action=states[1:].clone(),
        state_mask=initial.state_mask,
        action_mask=initial.action_mask,
        real_action_steps=horizon,
        timestamps=torch.arange(padded + 1) / 30,
    )
    sample = make_sample(
        raw,
        source=source,
        processor=processor,
        geometry=Geometry(),
        tokenize=lambda prompt: [1, 2],
        caption="move",
        fps=30,
        task="policy",
        history=3,
    )
    sample["ground_truth_actions"] = raw["action"][:horizon].clone()
    seen, predictions, decoded, identities = [], [], [], []
    original_sampler = inference_module.sample_block

    def observe_prediction(model, compact, block, **options):
        k = len(predictions)
        assert block == 0 and options["seed_block"] == k
        assert "video" not in compact and "anchors" not in compact and "ground_truth_actions" not in compact
        assert compact["video_latents"][:, 1:].eq(0).all() and compact["action"].eq(0).all()
        torch.testing.assert_close(compact["states"], sample["states"][k : k + 1], atol=0, rtol=0)
        caches = options["kv_cache"]
        identities.append({name: id(cache) for name, cache in caches.items()})
        assert set(caches) == ({"cond"} if guidance == 1 else {"cond", "uncond"})
        for cache in caches.values():
            assert cache.revision == k and cache.next_frame == 1 + 2 * k
        output = original_sampler(model, compact, block, **options)
        predictions.append(output)
        return output

    class VAE:
        def stream_encoder(self):
            class Stream:
                def encode(self, video):
                    seen.append(video.clone())
                    selected = video if len(seen) == 1 else video[:, 3::4]
                    return selected[:2, :, ::16, ::16].float() / 255

            return Stream()

        def encode_pack(self, samples):
            # Future GT encoding is permitted only after all predictions, for reconstruction export.
            assert len(predictions) == padded // 32
            return [dict(video_latents=samples[0]["video"][:2, ::4, ::16, ::16].float() / 255)]

        def decode(self, latents):
            decoded.append(latents.clone())
            return torch.zeros(3, 1 + (latents.shape[1] - 1) * 4, 32, 32, dtype=torch.uint8)

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval()
    pipe = CausalCosmos3Pipeline(
        model,
        VAE(),
        processor,
        lambda prompt: [1, 2],
        dict(source=source, sampling=dict(kv_cache=True, history=3, steps=2, guidance=guidance)),
    )
    monkeypatch.setattr(inference_module, "sample_block", observe_prediction)
    result = teacher_forcing(pipe, sample)
    count = padded // 32
    assert [video.shape[1] for video in seen] == [1] + [8] * (count - 1)
    assert all(branches == identities[0] for branches in identities)
    torch.testing.assert_close(torch.cat(seen, dim=1), pixels[:, : 1 + 8 * (count - 1)], atol=0, rtol=0)
    for k, latent in enumerate(decoded[:-1]):
        torch.testing.assert_close(latent[:, : 1 + 2 * k], pixels[:2, : 1 + 8 * k : 4, ::16, ::16].float() / 255)
        torch.testing.assert_close(latent[:, 1 + 2 * k :], predictions[k]["video_latents"][:, 1:])
    assert result.actions.shape == (horizon, 55) and torch.isfinite(result.actions).all()
    assert result.video.shape == result.reconstruction.shape == (3, 1 + horizon // 4, 32, 32)
    assert [record["anchor"][0] for record in result.blocks] == [2 + 32 * k for k in range(count)]
    assert result.blocks[-1]["padded_action_steps"] == padded - horizon
    assert result.blocks[-1]["valid_steps"] == horizon - 32 * (count - 1)
    assert all(record["latent_shape"] == [2, 3, 2, 2] for record in result.blocks)
    torch.testing.assert_close(result.ground_truth_actions, raw["action"][:horizon], atol=0, rtol=0)


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
