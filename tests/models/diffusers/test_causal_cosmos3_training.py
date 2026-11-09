"""逐 block 时间、独立 RNG、FP32 加噪/target 和既有 packing 的连接。"""

import copy
import io
import json
import random
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.train_causal_cosmos3 import configure_gradient_accumulation
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import action_weight_loading, tiny_config
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample, video_payload_sample
from veomni.data.causal_cosmos3 import collate_action_pack
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.configuration_cosmos3_condition import (
    CausalCosmos3ConditionConfig,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core import packing
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.flow_matching import (
    TrainTimeSampler,
    build_teacher_forcing_frame_block_ids,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import SampleSpans, pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.trainer.base import BaseTrainer
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTModelArguments,
    DiTModelRuntime,
    DiTTrainer,
    DiTTrainingArguments,
    VeOmniDiTArguments,
)
from veomni.utils.device import IS_NPU_AVAILABLE


def sigma_sample(frames, block_size=2, resolution="256"):
    # payload 放在 meta 上，确保时间采样只依赖形状，不读取或搬运视频数据。
    return dict(
        video_latents=torch.empty(2, frames, 1, 1, device="meta"), block_size=block_size, resolution=resolution
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend decision-window training")
@pytest.mark.parametrize("history,horizon", [(0, 16), (48, 32), (1024, 32)])
@pytest.mark.parametrize("mode", ["forward_dynamics", "inverse_dynamics", "policy"])
@pytest.mark.parametrize("enable_tbptt", [False, True])
def test_npu_decision_condition_and_checkpointed_training_gradients(history, horizon, mode, enable_tbptt):
    from tests.models.diffusers.causal_cosmos3_serving.test_native import window
    from veomni.arguments.arguments_types import OpsImplementationConfig

    _, sample = window(history_steps=history, horizon=horizon)
    sample["mode"] = mode
    if mode == "forward_dynamics":
        sample["condition_frame_indexes_action"] = list(range(horizon))
    elif mode == "inverse_dynamics":
        sample["condition_frame_indexes_vision"] = list(range(sample["video_latents"].shape[1]))
    sample["resolution"] = "256"
    # The serving sampler normally fills this metadata. Training datasets
    # already supply it; this tiny fixture has one patch per latent frame.
    sample["token_length"] = len(sample["text_token_ids"]) + sample["video_latents"].shape[1] + horizon + 1
    ops = OpsImplementationConfig(
        attn_implementation="eager", rms_norm_implementation="torch_npu", rotary_pos_emb_implementation="torch_npu"
    )
    with ops_config_scope(ops):
        net = Cosmos3Model(
            tiny_config(
                hidden_size=256,
                intermediate_size=512,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                mrope_section=[24, 20, 20],
                action_dim=55,
                state_dim=55,
                train_generation_only=True,
                enable_tbptt=enable_tbptt,
                tbptt_segment_steps=256,
            )
        ).to(device="npu", dtype=torch.bfloat16)
    net.freeze_model()
    net.gradient_checkpointing_enable()
    condition = CausalCosmos3ConditionModel(device="npu", dtype=torch.bfloat16, shift=1)
    inputs, targets = condition.prepare([sample])
    inputs["video_tokens"].requires_grad_()
    output = net(**inputs, targets=targets, attention_backend="npu_tnd")
    output.video.retain_grad()
    output.action.retain_grad()
    loss = sum(output.loss.values())
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert not output.video.grad[~targets["loss_masks"]["video_loss_mask"]].any()
    assert not output.action.grad[~targets["loss_masks"]["action_loss_mask"]].any()
    assert not targets["video_times"][0][: sample["observed_latents"]].any()
    assert bool(net.proj_out.weight.grad.abs().sum() > 0) == (mode != "inverse_dynamics")
    assert bool(net.action_proj_out.weight.grad.abs().sum() > 0) == (mode != "forward_dynamics")
    if enable_tbptt:
        old = torch.tensor(
            [
                role == 0 and frame < max(0, sample["observed_latents"] - 16)
                for role, frame, modality in zip(
                    inputs["layout"].roles, inputs["layout"].frame_ids, inputs["layout"].modalities, strict=True
                )
                if modality == 1
            ],
            device="npu",
        )
        assert not inputs["video_tokens"].grad[old].any()
        assert (
            inputs["video_tokens"].grad[: sample["observed_latents"]][~old[: sample["observed_latents"]]].abs().sum()
            > 0
        )
    for parameter in net.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
        if not parameter.requires_grad:
            assert parameter.grad is None


@pytest.mark.parametrize(
    "frames,block_size,expected",
    [
        (1, 2, [0]),
        (7, 2, [0, 1, 1, 2, 2, 3, 3]),
        (6, 2, [0, 1, 1, 2, 2, 3]),
        (4, 1, [0, 1, 2, 3]),
        (8, 3, [0, 1, 1, 1, 2, 2, 2, 3]),
    ],
)
def test_teacher_forcing_sampling_indexes(frames, block_size, expected):
    actual = build_teacher_forcing_frame_block_ids(frames, block_size)
    assert actual.device.type == "cpu" and actual.dtype == torch.long
    assert actual.tolist() == expected


@pytest.mark.parametrize("distribution", ["uniform", "logitnormal", "waver", "ltx2"])
def test_variable_pack_uses_one_draw_per_block(distribution):
    samples = [sigma_sample(7), sigma_sample(4, block_size=3, resolution="480"), sigma_sample(1)]
    condition = CausalCosmos3ConditionModel(device="cpu", seed=7, train_time_distribution=distribution)
    actual = condition.sample_sigmas(samples)
    generator = torch.Generator(device="cpu").manual_seed(7)
    draws = TrainTimeSampler(distribution)(
        7, generator=generator, shifts=torch.tensor([3, 3, 3, 3, 5, 5, 3], dtype=torch.float32)
    )
    for value, indexes in zip(actual, ([0, 1, 1, 2, 2, 3, 3], [4, 5, 5, 5], [6]), strict=True):
        assert value.dtype == torch.float32 and not value.requires_grad
        assert ((value >= 0) & (value <= 1)).all()
        torch.testing.assert_close(value, draws[indexes], atol=0, rtol=0)
    # 除了返回值，还核对 RNG 消耗量，防止按 latent 多抽或跳过首 latent。
    assert torch.equal(condition.time_generator.get_state(), generator.get_state())


def test_resolution_shift_and_scalar_shift():
    samples = [sigma_sample(7), sigma_sample(4, resolution="480")]
    raw = CausalCosmos3ConditionModel(device="cpu", seed=19, shift=1).sample_sigmas(samples)
    shifted = CausalCosmos3ConditionModel(device="cpu", seed=19).sample_sigmas(samples)
    for source, actual, shift in zip(raw, shifted, (3, 5), strict=True):
        expected = shift * source / (1 + (shift - 1) * source)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("distribution", ["uniform", "logitnormal", "waver", "ltx2"])
@pytest.mark.parametrize("default_dtype", [torch.bfloat16, torch.float64])
def test_sampling_keeps_fp32_before_any_dtype_conversion(distribution, default_dtype):
    samples = [sigma_sample(7), sigma_sample(4, resolution="480")]
    expected = CausalCosmos3ConditionModel(device="cpu", seed=13, train_time_distribution=distribution).sample_sigmas(
        samples
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        actual = CausalCosmos3ConditionModel(
            device="cpu", seed=13, train_time_distribution=distribution
        ).sample_sigmas(samples)
    finally:
        torch.set_default_dtype(previous)
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


def test_sampling_metadata_stays_on_cpu_under_another_default_device():
    samples = [sigma_sample(7), sigma_sample(4, resolution="480")]
    expected = CausalCosmos3ConditionModel(device="cpu", seed=11).sample_sigmas(samples)
    with torch.device("meta"):
        actual = CausalCosmos3ConditionModel(device="cpu", seed=11).sample_sigmas(samples)
    for value, reference in zip(actual, expected, strict=True):
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend RNG"))]
)
def test_official_runtime_restores_time_and_noise_streams(device):
    samples = [sigma_sample(7), sigma_sample(4, resolution="480")]
    condition = CausalCosmos3ConditionModel(device=device, seed=5)
    condition.sample_sigmas(samples)
    torch.randn((17,), device=device, dtype=torch.float32, generator=condition.generator)
    # 直接调用官方 runtime 钩子并序列化，不为测试重新实现保存/恢复逻辑。
    runtime = SimpleNamespace(condition_model=condition)
    buffer = io.BytesIO()
    torch.save(DiTModelRuntime.extra_state(runtime), buffer)
    expected_time = condition.sample_sigmas(samples)
    expected_noise = torch.randn((17,), device=device, dtype=torch.float32, generator=condition.generator)
    buffer.seek(0)
    restored = CausalCosmos3ConditionModel(device=device, seed=999)
    DiTModelRuntime.load_extra_state(
        SimpleNamespace(condition_model=restored), torch.load(buffer, map_location=device, weights_only=True)
    )
    for actual, expected in zip(restored.sample_sigmas(samples), expected_time, strict=True):
        assert actual.device.type == device and actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_noise = torch.randn((17,), device=device, dtype=torch.float32, generator=restored.generator)
    torch.testing.assert_close(actual_noise, expected_noise, atol=0, rtol=0)


def test_condition_standalone_keeps_base_seed_without_parallel_state(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as training

    monkeypatch.setattr(training, "is_parallel_state_initialized", lambda: False)

    def unexpected_parallel_state():
        raise AssertionError("Standalone condition must not construct a default parallel state")

    monkeypatch.setattr(training, "get_parallel_state", unexpected_parallel_state)
    condition = CausalCosmos3ConditionModel(device="cpu", seed=42)
    assert condition.time_generator.initial_seed() == 42
    assert condition.generator.initial_seed() == 42 + 32768


def test_condition_rng_draws_are_independent_across_dp_ranks_and_restore(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as training

    monkeypatch.setattr(training, "is_parallel_state_initialized", lambda: True)
    conditions = []
    for rank in (0, 1):
        monkeypatch.setattr(training, "get_parallel_state", lambda rank=rank: SimpleNamespace(dp_rank=rank))
        conditions.append(CausalCosmos3ConditionModel(device="cpu", seed=42))
    first, second = conditions
    samples = [sigma_sample(7)]
    assert first.time_generator.initial_seed() == 42
    assert second.time_generator.initial_seed() == 43
    assert second.generator.initial_seed() == 43 + 32768
    assert not torch.equal(first.sample_sigmas(samples)[0], second.sample_sigmas(samples)[0])
    assert not torch.equal(torch.randn(17, generator=first.generator), torch.randn(17, generator=second.generator))
    saved = second.rng_state_dict()
    expected_time = second.sample_sigmas(samples)[0]
    expected_noise = torch.randn(17, generator=second.generator)
    second.load_rng_state_dict(saved)
    torch.testing.assert_close(second.sample_sigmas(samples)[0], expected_time, atol=0, rtol=0)
    torch.testing.assert_close(torch.randn(17, generator=second.generator), expected_noise, atol=0, rtol=0)


def test_time_noise_and_global_rng_are_independent():
    global_state = torch.get_rng_state()
    samples = [sigma_sample(7)]
    first = CausalCosmos3ConditionModel(device="cpu", seed=3)
    second = CausalCosmos3ConditionModel(device="cpu", seed=3)
    torch.randn((17,), generator=first.generator)
    torch.testing.assert_close(first.sample_sigmas(samples)[0], second.sample_sigmas(samples)[0], atol=0, rtol=0)
    second.sample_sigmas(samples)
    torch.randn((17,), generator=second.generator)
    torch.testing.assert_close(
        torch.randn((17,), generator=first.generator), torch.randn((17,), generator=second.generator), atol=0, rtol=0
    )
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize("frames,block_size", [(0, 2), (3, 0), (3, -1)])
def test_invalid_geometry_does_not_advance_time_rng(frames, block_size):
    condition = CausalCosmos3ConditionModel(device="cpu")
    before = condition.time_generator.get_state()
    with pytest.raises(ValueError, match="must be >= 1"):
        condition.sample_sigmas([sigma_sample(frames, block_size)])
    assert torch.equal(condition.time_generator.get_state(), before)


def test_empty_sigma_pack():
    with pytest.raises(ValueError, match="nonempty"):
        CausalCosmos3ConditionModel(device="cpu").sample_sigmas([])


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_sampled_sigmas_feed_existing_packing_without_resampling(task):
    sample = payload_sample(task, actions=64)
    sample["resolution"] = "256"
    condition = CausalCosmos3ConditionModel(device="cpu")
    (sigma,) = condition.sample_sigmas([sample])
    before = condition.time_generator.get_state()
    inputs, _, _ = pack_model_inputs([sample], [sample["video_latents"]], [sample["action"]], timesteps=[sigma * 1000])
    frames = len(sigma)
    expected_video = sigma * 1000
    expected_video[sample["condition_frame_indexes_vision"]] = 0
    expected_action = (sigma * 1000)[sample["action_frame_ids"]]
    expected_action[sample["condition_frame_indexes_action"]] = 0
    assert inputs["video_timesteps"].dtype == inputs["action_timesteps"].dtype == torch.float32
    torch.testing.assert_close(inputs["video_timesteps"][:frames], torch.zeros(frames), atol=0, rtol=0)
    torch.testing.assert_close(inputs["video_timesteps"][frames:], expected_video, atol=0, rtol=0)
    torch.testing.assert_close(inputs["action_timesteps"], expected_action, atol=0, rtol=0)
    assert torch.equal(condition.time_generator.get_state(), before)


def training_sample(task="policy", **geometry):
    return {**payload_sample(task, **geometry), "resolution": "256"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("explicit", [False, True])
def test_video_prepare_without_action_fields_at_packing_boundary(monkeypatch, dtype, explicit):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as module

    samples = [
        dict(
            mode="video",
            video_latents=torch.arange(8 * frames).reshape(2, frames, 2, 2).to(dtype),
            block_size=2,
            history_blocks=1,
            resolution="256",
            condition_frame_indexes_vision=[0],
        )
        for frames in (3, 5)
    ]
    originals = copy.deepcopy(samples)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=dtype, stage="video", seed=73)
    reference = CausalCosmos3ConditionModel(device="cpu", dtype=dtype, stage="video", seed=73)
    sigmas = [torch.full((s["video_latents"].shape[1],), 0.25) for s in samples] if explicit else None
    noise = [torch.full_like(s["video_latents"], 0.5, dtype=torch.float32) for s in samples] if explicit else None
    expected_sigmas = reference.sample_sigmas(samples) if sigmas is None else sigmas
    expected_noise = (
        [torch.randn(s["video_latents"].shape, generator=reference.generator) for s in samples]
        if noise is None
        else noise
    )
    calls = []

    # Isolate the noise/RNG boundary; the integration test below uses real Packing.
    def packing_boundary(prepared, videos, actions, *, timesteps, **kwargs):
        assert actions is None
        assert kwargs["has_action"] is False
        calls.append(prepared)
        for i, sample in enumerate(prepared):
            assert sample is not samples[i] and sample.keys() == samples[i].keys()
            assert sample["video_latents"].dtype == torch.float32
            clean, sigma = samples[i]["video_latents"].float(), expected_sigmas[i][None, :, None, None]
            torch.testing.assert_close(videos[i], expected_noise[i] * sigma + clean * (1 - sigma), atol=0, rtol=0)
            torch.testing.assert_close(timesteps[i], expected_sigmas[i] * 1000, atol=0, rtol=0)
        return (
            dict(video_tokens=torch.cat([v.flatten() for v in videos]), video_timesteps=torch.cat(timesteps)),
            (),
            {"video_loss_mask": torch.ones(1, dtype=torch.bool)},
        )

    monkeypatch.setattr(module, "pack_model_inputs", packing_boundary)
    inputs, targets = condition.prepare(samples, sigmas=sigmas, epsilon_video=noise)
    assert len(calls) == 1
    assert set(inputs) == {"video_tokens", "video_timesteps"}
    assert set(targets) == {"video", "video_times", "spans", "loss_masks"}
    assert inputs["video_tokens"].dtype == dtype and inputs["video_timesteps"].dtype == torch.float32
    for i, sample in enumerate(samples):
        torch.testing.assert_close(
            targets["video"][i], expected_noise[i] - sample["video_latents"].float(), atol=0, rtol=0
        )
        torch.testing.assert_close(sample["video_latents"], originals[i]["video_latents"], atol=0, rtol=0)
        assert sample.keys() == originals[i].keys()
    for key, state in condition.rng_state_dict().items():
        assert torch.equal(state, reference.rng_state_dict()[key])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_video_prepare_packing_and_model_forward_backward(dtype, backend):
    samples = [
        {**video_payload_sample(frames=frames, height=height, width=width, history=2), "resolution": "256"}
        for frames, height, width in ((1, 2, 2), (4, 3, 5), (7, 2, 2))
    ]
    for sample in samples:
        sample["video_latents"] = sample["video_latents"].to(dtype)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=dtype, stage="video", seed=73)
    sigmas = [torch.full((s["video_latents"].shape[1],), 0.25) for s in samples]
    noises = [torch.full_like(s["video_latents"], 0.5, dtype=torch.float32) for s in samples]
    inputs, targets = condition.prepare(samples, sigmas=sigmas, epsilon_video=noises)
    assert set(targets) == {"video", "video_times", "spans", "loss_masks"}
    assert set(targets["loss_masks"]) == {"video_loss_mask"}
    assert all(span.action_rows is None for span in targets["spans"])
    model = training_model(action_interface="none", state_dim=0).to(dtype=dtype).eval()
    with torch.no_grad():
        output = model(**inputs, attention_backend=backend)
        assert output.action is None and output.loss is None
        assert torch.isfinite(output.video).all()
        for i, (sample, span) in enumerate(zip(samples, targets["spans"], strict=True)):
            single_inputs, single_targets = condition.prepare([sample], sigmas=[sigmas[i]], epsilon_video=[noises[i]])
            single = model(**single_inputs, attention_backend=backend)
            torch.testing.assert_close(
                output.video[span.current_video_rows],
                single.video[single_targets["spans"][0].current_video_rows],
                atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
            )
            torch.testing.assert_close(targets["video"][i], noises[i] - sample["video_latents"].float())

    model.train()
    output = model(**inputs, targets=targets, attention_backend=backend)
    assert output.action is None and set(output.loss) == {"video_loss"}
    loss = output.loss["video_loss"]
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.proj_in.weight.grad).all()
    assert model.proj_in.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("frames", [(1,), (1, 4, 5)])
def test_video_loss_masks_normalization_and_zero_supervision(dtype, frames):
    samples = [{**video_payload_sample(frames=f, height=3, width=5), "resolution": "256"} for f in frames]
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=dtype, stage="video")
    inputs, targets = condition.prepare(samples)
    targets["video"] = [torch.zeros_like(v) for v in targets["video"]]
    # No action attribute: pure-video loss must not read action predictions.
    output = SimpleNamespace(video=torch.full_like(inputs["video_tokens"], 2, requires_grad=True))
    model = training_model(action_interface="none", state_dim=0, video_loss_weight=3)
    losses = model.loss(output, targets)
    assert set(losses) == {"video_loss"}
    expected_loss = 0.0
    expected_gradient = torch.zeros_like(output.video)
    for sample, span in zip(samples, targets["spans"], strict=True):
        valid = torch.ones_like(sample["video_latents"], dtype=torch.bool)
        valid[:, sample["condition_frame_indexes_vision"]] = False
        valid[:, sample["num_valid_video_latents"] :] = False
        count = int(valid.sum())
        if count:
            expected_loss += 4 * 3 / len(samples)
        gradient = valid.float() * (4 * 3 / (max(count, 1) * len(samples)))
        expected_gradient[span.current_video_rows] = patchify_video(gradient, 2).to(dtype)
    torch.testing.assert_close(losses["video_loss"], torch.tensor(expected_loss), atol=0, rtol=0)
    losses["video_loss"].backward()
    torch.testing.assert_close(
        output.video.grad, expected_gradient, atol=1e-7, rtol=1e-2 if dtype == torch.bfloat16 else 1e-5
    )
    # Perturb all elements excluded from direct supervision, including spatial patch padding.
    with torch.no_grad():
        output.video[expected_gradient == 0] += 100
    torch.testing.assert_close(model.loss(output, targets)["video_loss"], losses["video_loss"], atol=0, rtol=0)


def test_action_loss_requires_predictions_when_targets_exist():
    condition = CausalCosmos3ConditionModel(device="cpu")
    inputs, targets = condition.prepare([training_sample()])
    output = SimpleNamespace(video=inputs["video_tokens"], action=None)
    with pytest.raises(ValueError, match="Action targets require action predictions"):
        training_model().loss(output, targets)


@pytest.mark.parametrize(
    "stage,kind",
    [
        ("action", "video"),
        ("action", "missing_action"),
        ("video", "action"),
        ("video", "action_noise"),
        ("video", "mixed"),
    ],
)
def test_prepare_rejects_stage_mismatch_before_rng_or_packing(monkeypatch, stage, kind):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as module

    action = training_sample()
    video = dict(mode="video", video_latents=action["video_latents"], block_size=2, resolution="256")
    samples = [action] if kind in ("action", "missing_action") else [video]
    if kind == "missing_action":
        action.pop("action")
    if kind == "mixed":
        samples.append(action)
    condition = CausalCosmos3ConditionModel(device="cpu", stage=stage)
    before = condition.rng_state_dict()

    def fail(*args, **kwargs):
        raise AssertionError("Invalid samples must fail before Packing")

    monkeypatch.setattr(module, "pack_model_inputs", fail)
    options = dict(epsilon_action=[torch.zeros(1)]) if kind == "action_noise" else {}
    with pytest.raises(ValueError, match="stage"):
        condition.prepare(samples, **options)
    for key, state in before.items():
        assert torch.equal(condition.rng_state_dict()[key], state)


@pytest.mark.parametrize("stage", ["Video", "unknown", None])
def test_condition_rejects_unknown_stage(stage):
    with pytest.raises(ValueError, match="stage must be action or video"):
        CausalCosmos3ConditionConfig(stage=stage)


@pytest.mark.parametrize("payload_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("training_task", ["online_training", "offline_training"])
def test_official_runtime_builds_registered_condition(tmp_path, monkeypatch, payload_dtype, training_task):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as training
    import veomni.trainer.dit_trainer as dit
    from veomni.models import build_config

    monkeypatch.setattr(training, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(dit, "get_device_type", lambda: "cpu")
    config = CausalCosmos3ConditionConfig(vae_path=tmp_path / "vae", payload_dtype=payload_dtype, shift=2)
    config.save_pretrained(tmp_path)
    assert build_config(str(tmp_path)).to_dict() == config.to_dict()
    condition_type = tiny_config().condition_model_type
    assert condition_type == config.model_type
    runtime = SimpleNamespace(
        args=SimpleNamespace(condition_model_path=str(tmp_path), condition_model_cfg={"shift": 3}),
        train_args=SimpleNamespace(seed=73, training_task=training_task),
    )
    DiTModelRuntime._build_condition_model(runtime, condition_type)
    condition = runtime.condition_model
    assert isinstance(condition, CausalCosmos3ConditionModel)
    assert condition.config.shift == 3
    assert condition.device == torch.device("cpu")
    assert condition.dtype == getattr(torch, payload_dtype)
    assert condition.video_vae is None
    assert not condition.state_dict()
    assert condition.time_generator.initial_seed() == 73
    assert condition.generator.initial_seed() == 73 + 32768


@pytest.fixture
def runtime_assets(tmp_path, monkeypatch):
    """使用真实模型、配置和权重加载器，仅替换分布式包装和设备相关操作。"""
    import veomni.distributed.torch_parallelize as parallelize
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as training
    import veomni.models.model_runtime as runtime_module
    import veomni.optim as optim
    import veomni.trainer.dit_trainer as dit

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(runtime_module.VeOmniModelRuntime, "setup", lambda self: None)
    monkeypatch.setattr(runtime_module, "use_parallel_state", lambda _: nullcontext())
    monkeypatch.setattr(runtime_module, "get_parallel_state_by_name", lambda _: SimpleNamespace(sp_size=1))
    monkeypatch.setattr(runtime_module.VeOmniModelRuntime, "_freeze_model_module", lambda self: None)
    monkeypatch.setattr(training, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(dit, "get_device_type", lambda: "cpu")
    events = []

    def cpu_parallelize(model, *, weights_path, should_skip_hf_weight_load, **kwargs):
        assert all(parameter.is_meta for parameter in model.parameters())
        events.append(("load", should_skip_hf_weight_load))
        if should_skip_hf_weight_load:
            model.to_empty(device="cpu")
        else:
            load_model_weights(model, weights_path, init_device="cpu")
        return model

    def cpu_optimizer(model, *, lr, **kwargs):
        assert not any(parameter.is_meta for parameter in model.parameters())
        events.append(("optimizer",))
        return torch.optim.SGD(model.parameters(), lr=lr, foreach=False)

    monkeypatch.setattr(parallelize, "build_parallelize_model", cpu_parallelize)
    monkeypatch.setattr(optim, "build_optimizer", cpu_optimizer)
    with ops_config_scope(eager_ops_config()):
        source = Cosmos3Model(tiny_config(action_interface="domain", action_dim=64, state_dim=0))
        weights = tmp_path / "transformer"
        weights.mkdir()
        # 原始 checkpoint 中的 audio 分支也需要由已有 converter 显式跳过。
        state = source.state_dict()
        state["audio_proj_in.weight"] = torch.ones(1)
        state["audio_proj_out.weight"] = torch.ones(1)
        state["audio_modality_embed"] = torch.ones(1)
        save_file(state, weights / "model.safetensors")
        config_path, condition_path = tmp_path / "config", tmp_path / "condition"
        tiny_config(action_dim=55, state_dim=55).save_pretrained(config_path)
        CausalCosmos3ConditionConfig(payload_dtype="float32").save_pretrained(condition_path)
        args = DiTModelArguments(
            model_path=str(weights),
            config_path=str(config_path),
            condition_model_path=str(condition_path),
            ops_implementation=eager_ops_config(),
        )
        train = DiTTrainingArguments(max_steps=1)
        yield args, train, source, events


def test_native_runtime_uses_config_and_skips_tensor_reads(runtime_assets, monkeypatch):
    import veomni.models.checkpoint.weights as weights

    args, train, source, events = runtime_assets
    policy = action_weight_loading()
    args.model_config["weight_loading"] = policy
    read_names = []
    safe_open = weights.safe_open

    @contextmanager
    def audited_open(*positional, **kwargs):
        with safe_open(*positional, **kwargs) as reader:

            def read_tensor(name):
                assert not name.startswith(("action_", "audio_"))
                read_names.append(name)
                return reader.get_tensor(name)

            yield SimpleNamespace(keys=reader.keys, get_tensor=read_tensor)

    monkeypatch.setattr(weights, "safe_open", audited_open)
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.base = SimpleNamespace(args=SimpleNamespace(model=args, train=train))
    runtime = trainer._build_model_runtime()
    assert type(runtime) is DiTModelRuntime
    assert CausalCosmos3Trainer._build_model_runtime is DiTTrainer._build_model_runtime
    assert events == [("load", False), ("optimizer",)]
    report = runtime.model.weight_loading_report
    assert report["mode"] == "partial"
    assert report["loaded_count"] == len(read_names)
    assert set(report["initialized"]) == {
        "action_proj_in.weight",
        "action_proj_in.bias",
        "action_proj_out.weight",
        "action_proj_out.bias",
        "state_proj_in.weight",
        "state_proj_in.bias",
        "action_modality_embed",
        "state_modality_embed",
    }
    assert {"audio_proj_in.weight", "audio_proj_out.weight", "audio_modality_embed"} <= set(report["skipped"])
    for name, parameter in source.named_parameters():
        if name not in report["skipped"]:
            torch.testing.assert_close(runtime.model.get_parameter(name), parameter, atol=0, rtol=0)
    assert runtime.model.action_proj_in.in_features == runtime.model.state_proj_in.in_features == 55
    assert runtime.model.action_proj_out.out_features == 55
    assert all(torch.isfinite(parameter).all() for parameter in runtime.model.parameters())
    assert runtime.condition_model.video_vae is None
    assert runtime.checkpoint.runtime is runtime
    assert runtime.model.config.weight_loading == {}
    assert "weight_loading" not in runtime.model.config.to_dict()
    assert policy == action_weight_loading()


def test_runtime_complete_reload_preserves_all_trained_heads(runtime_assets):
    args, train, _, events = runtime_assets
    trained = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if name.startswith(("action_", "state_")):
                parameter.fill_(1.25)
    save_file(trained.state_dict(), args.model_path + "/model.safetensors")
    runtime = DiTModelRuntime(args, train=train)
    assert events == [("load", False), ("optimizer",)]
    assert runtime.model.weight_loading_report["mode"] == "complete"
    assert runtime.model.weight_loading_report["initialized"] == []
    for name, parameter in trained.named_parameters():
        torch.testing.assert_close(runtime.model.get_parameter(name), parameter, atol=0, rtol=0)


@pytest.mark.parametrize("name", ["action_proj_in.weight", "layers.0.self_attn.to_k.weight"])
def test_runtime_complete_reload_rejects_missing_weights(runtime_assets, name):
    args, train, _, _ = runtime_assets
    args.model_config["weight_loading"] = {"initialize_missing_patterns": []}
    trained = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    state = trained.state_dict()
    del state[name]
    save_file(state, args.model_path + "/model.safetensors")
    with pytest.raises(ValueError, match="Missing Cosmos3 weights") as error:
        DiTModelRuntime(args, train=train)
    assert name in str(error.value)


@pytest.mark.parametrize("broken", ["missing", "shape"])
def test_configured_partial_load_keeps_backbone_checks(runtime_assets, broken):
    args, train, _, _ = runtime_assets
    args.model_config["weight_loading"] = action_weight_loading()
    path = args.model_path + "/model.safetensors"
    state = load_file(path)
    name = "layers.0.self_attn.to_k.weight"
    if broken == "missing":
        del state[name]
    else:
        state[name] = torch.zeros(1, 1)
    save_file(state, path)
    with pytest.raises(ValueError, match="Missing Cosmos3 weights|Shape mismatch") as error:
        DiTModelRuntime(args, train=train)
    assert name in str(error.value)


@pytest.mark.parametrize("partial_loading", [False, True])
def test_runtime_dcp_skips_original_weights_and_delegates_restore(runtime_assets, monkeypatch, partial_loading):
    import veomni.models.checkpoint.manager as manager

    args, train, _, events = runtime_assets
    if partial_loading:
        args.model_config["weight_loading"] = action_weight_loading()
    trained = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    with torch.no_grad():
        trained.action_modality_embed.fill_(2)
        trained.state_modality_embed.fill_(3)
    args.model_path += "/does-not-exist"
    train.checkpoint.load_path = "full-dcp"
    runtime = DiTModelRuntime(args, train=train)
    assert events == [("load", True), ("optimizer",)]
    assert runtime.model.weight_loading_report is None
    noise_state = runtime.condition_model.rng_state_dict()
    runtime.condition_model.generator.manual_seed(0)

    # 模拟 DCP 的 I/O 边界，实际恢复调用仍经过父类 runtime 和 checkpoint manager。
    def restore(path, state, **kwargs):
        assert path == train.checkpoint.load_path
        state["model"].load_state_dict(trained.state_dict())
        state["extra_state"]["condition_model_rng_state"] = noise_state

    runtime.checkpoint.checkpointer = SimpleNamespace(wait_for_pending_save=lambda: None, load=restore)
    monkeypatch.setattr(manager.dist, "barrier", lambda: None)
    runtime.load()
    for name, parameter in trained.named_parameters():
        torch.testing.assert_close(runtime.model.get_parameter(name), parameter, atol=0, rtol=0)
    for name, state in noise_state.items():
        assert torch.equal(runtime.condition_model.rng_state_dict()[name], state)
    assert "weight_loading" not in runtime.model.config.to_dict()


def test_configured_partial_load_also_works_with_eager_build(runtime_assets, monkeypatch):
    import veomni.models.auto as auto

    args, _, source, _ = runtime_assets
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(global_rank=0))
    model = auto.build_foundation_model(
        args.config_path,
        weights_path=args.model_path,
        torch_dtype="float32",
        init_device="cpu",
        ops_implementation=args.ops_implementation,
        config_kwargs={"weight_loading": action_weight_loading()},
    )
    assert model.weight_loading_report["mode"] == "partial"
    assert len(model.weight_loading_report["initialized"]) == 8
    torch.testing.assert_close(model.layers[0].self_attn.to_k.weight, source.layers[0].self_attn.to_k.weight)


@pytest.fixture
def raw_condition_pack(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae as vae_module

    constructed = []

    class FrozenVAE(torch.nn.Module):
        def __init__(self, path, *, device, dtype, reset_prefix_cache_mb=0):
            super().__init__()
            constructed.append((path, device, dtype))
            self.cache_mb = reset_prefix_cache_mb
            self.weight = torch.nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        def encode_pack(self, samples):
            assert not self.training and not self.weight.requires_grad
            assert not torch.is_grad_enabled()
            return [
                {
                    **{key: value for key, value in sample.items() if key != "video"},
                    "video_latents": sample["video"][:2].float() / 255,
                }
                for sample in samples
            ]

    monkeypatch.setattr(vae_module, "Cosmos3VideoVAE", FrozenVAE)
    samples = [training_sample("policy"), training_sample("id", actions=64)]
    for sample in samples:
        _, frames, height, width = sample.pop("video_latents").shape
        sample["video"] = torch.ones(3, frames, height, width, dtype=torch.uint8)
        for key in ("action_frame_ids", "state_latent_indexes"):
            sample[key] = sample[key].tolist()
    config = CausalCosmos3ConditionConfig(vae_path="vae-fixture", payload_dtype="float32", attention_backend="grouped")
    return CausalCosmos3ConditionModel(config, device="cpu"), samples, constructed


def test_standard_condition_calls_reuse_prepare_and_preserve_metadata(raw_condition_pack):
    condition, samples, constructed = raw_condition_pack
    before = condition.rng_state_dict()
    encoded = condition.get_condition(samples)
    assert len(constructed) == 1
    assert constructed[0] == ("vae-fixture", torch.device("cpu"), torch.float32)
    condition.get_condition(samples)
    assert len(constructed) == 1
    for source, result in zip(samples, encoded["samples"], strict=True):
        assert "video" in source and "video_latents" not in source
        assert "video" not in result
        assert result["action_frame_ids"] is source["action_frame_ids"]
        assert result["state_latent_indexes"] is source["state_latent_indexes"]
        assert isinstance(result["action_frame_ids"], list)
    for key, value in condition.rng_state_dict().items():
        assert torch.equal(value, before[key])
    tensor_samples = [
        {
            **sample,
            "action_frame_ids": torch.tensor(sample["action_frame_ids"], dtype=torch.long),
            "state_latent_indexes": torch.tensor(sample["state_latent_indexes"], dtype=torch.long),
        }
        for sample in encoded["samples"]
    ]
    expected = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32).prepare(tensor_samples)
    kwargs = condition.process_condition(**encoded)
    assert kwargs.pop("attention_backend") == "grouped"
    targets = kwargs.pop("targets")
    assert_prepared_equal((kwargs, targets), expected)
    assert kwargs["position_ids"].dtype == kwargs["video_timesteps"].dtype == torch.float32
    assert not targets["video"][0].requires_grad


def test_dit_preforward_preserves_list_indexes_and_moves_payloads(raw_condition_pack):
    _, samples, _ = raw_condition_pack
    microbatch = collate_action_pack(samples)
    trainer = DiTTrainer.__new__(DiTTrainer)
    # A different device verifies transfer boundaries without an accelerator.
    trainer.base = SimpleNamespace(device="meta", LOG_SAMPLE=False)
    moved = trainer.preforward(microbatch)

    assert microbatch["samples"] is samples
    for actual, original in zip(moved["samples"], samples, strict=True):
        for key in ("action_frame_ids", "state_latent_indexes"):
            assert isinstance(actual[key], list) and all(type(index) is int for index in actual[key])
            assert actual[key] == original[key]
        assert actual["video"].device.type == "meta"
        assert actual["action"].device.type == "meta"
        assert all(value.device.type == "meta" for value in actual.values() if isinstance(value, torch.Tensor))
        assert all(value.device.type == "cpu" for value in original.values() if isinstance(value, torch.Tensor))


def test_condition_forwards_reset_cache_budget_without_consuming_rng(raw_condition_pack):
    condition, samples, _ = raw_condition_pack
    condition.config.vae_reset_prefix_cache_mb = 4096
    before = condition.rng_state_dict()
    condition.get_condition(samples)
    assert condition.video_vae.cache_mb == 4096
    for key, value in condition.rng_state_dict().items():
        assert torch.equal(value, before[key])


def test_native_dit_forward_backward_uses_standard_condition_calls(raw_condition_pack, monkeypatch):
    import veomni.trainer.dit_trainer as dit

    condition, samples, _ = raw_condition_pack
    model = training_model()
    before = condition.rng_state_dict()
    encoded = condition.get_condition(samples)
    tensor_samples = [
        {
            **sample,
            "action_frame_ids": torch.tensor(sample["action_frame_ids"], dtype=torch.long),
            "state_latent_indexes": torch.tensor(sample["state_latent_indexes"], dtype=torch.long),
        }
        for sample in encoded["samples"]
    ]
    prepared = condition.process_condition(tensor_samples)
    expected = model(**prepared).loss
    expected_gradient = torch.autograd.grad(sum(expected.values()) / 2, model.proj_in.weight)[0]
    condition.load_rng_state_dict(before)

    class Runtime:
        condition_model = condition
        parallel_state = None

        def __call__(self, **kwargs):
            return model(**kwargs)

    monkeypatch.setattr(dit, "use_parallel_state", lambda state: nullcontext())
    trainer = DiTTrainer.__new__(DiTTrainer)
    trainer.training_task = "online_training"
    trainer.base = SimpleNamespace(
        device="cpu",
        LOG_SAMPLE=False,
        model=Runtime(),
        num_micro_batches=2,
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
    )
    loss, losses = trainer.forward_backward_step(collate_action_pack(samples))
    for key in losses:
        torch.testing.assert_close(losses[key], expected[key] / 2, atol=0, rtol=0)
    torch.testing.assert_close(loss, sum(expected.values()) / 2, atol=0, rtol=0)
    torch.testing.assert_close(model.proj_in.weight.grad, expected_gradient, atol=0, rtol=0)
    assert model.proj_in.weight.grad.abs().sum() > 0
    assert condition.video_vae.weight.grad is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("chunk", [3, 16])
def test_video_trainer_validates_tbptt_before_runtime_setup(tmp_path, monkeypatch, chunk):
    sources = tmp_path / "sources.yaml"
    sources.write_text("stage: video\n")
    tiny_config(action_interface="none", state_dim=0).save_pretrained(tmp_path / "model")
    args = VeOmniDiTArguments(
        model=DiTModelArguments(
            config_path=str(tmp_path / "model"),
            model_config={"enable_tbptt": True, "tbptt_segment_steps": 16 * chunk},
        ),
        data=DiTDataArguments(train_path=str(sources)),
        train=DiTTrainingArguments(micro_batch_size=1, max_steps=1),
    )
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.tokenize = lambda prompt: [1]
    trainer.dataset_config = {"stage": "video"}
    trainer.base = SimpleNamespace(args=args)

    def reached_runtime(_self):
        raise RuntimeError("validated Video TBPTT reached runtime setup")

    monkeypatch.setattr(DiTTrainer, "_setup", reached_runtime)
    if chunk == 3:
        with pytest.raises(ValueError, match="tbptt_segment_steps.*block"):
            trainer._setup()
    else:
        with pytest.raises(RuntimeError, match="validated Video TBPTT reached runtime setup"):
            trainer._setup()
        assert args.model.condition_model_cfg["stage"] == "video"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "enabled,geometry,horizon,steps,error",
    [
        (True, {"video_stride": 2}, 32, 256, "Action TBPTT requires 16 action steps per latent"),
        (True, {"video_stride": 8}, 32, 256, "Action TBPTT requires 16 action steps per latent"),
        (False, {"video_stride": 2}, 32, 256, None),
        (True, {}, 0, 256, "Action TBPTT requires a positive VAE-aligned action_horizon"),
        (True, {}, 3, 256, "Action TBPTT requires a positive VAE-aligned action_horizon"),
        (True, {}, 32, 16, "tbptt_segment_steps must be divisible by action_horizon"),
        (True, {}, 32, 48, "tbptt_segment_steps must be divisible by action_horizon"),
        (True, {}, 32, 32, None),
        (True, {}, 32, 256, None),
        (True, {}, 16, 16, None),
    ],
)
def test_action_trainer_validates_step_units_before_runtime_setup(
    tmp_path, monkeypatch, enabled, geometry, horizon, steps, error
):
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump(dict(stage="action", geometry=geometry, sample=dict(action_horizon=horizon))))
    tiny_config(action_interface="shared", action_dim=55, state_dim=55).save_pretrained(tmp_path / "model")
    args = VeOmniDiTArguments(
        model=DiTModelArguments(
            config_path=str(tmp_path / "model"),
            model_config={"enable_tbptt": enabled, "tbptt_segment_steps": steps},
        ),
        data=DiTDataArguments(train_path=str(sources)),
        train=DiTTrainingArguments(micro_batch_size=1, max_steps=1),
    )
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.tokenize = lambda prompt: [1]
    trainer.dataset_config = dict(stage="action", geometry=geometry, sample=dict(action_horizon=horizon))
    trainer.base = SimpleNamespace(args=args)

    def reached_runtime(_self):
        raise RuntimeError("validated Action step units reached runtime setup")

    monkeypatch.setattr(DiTTrainer, "_setup", reached_runtime)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            trainer._setup()
    else:
        with pytest.raises(RuntimeError, match="validated Action step units reached runtime setup"):
            trainer._setup()


@pytest.mark.parametrize("stage", ["action", "video"])
def test_native_dit_train_step_accumulates_variable_loader_packs(raw_condition_pack, monkeypatch, tmp_path, stage):
    import veomni.data.data_loader as data_loader
    import veomni.trainer.causal_cosmos3 as causal
    import veomni.trainer.dit_trainer as dit

    condition, samples, _ = raw_condition_pack
    if stage == "video":
        samples = [{**video_payload_sample(frames=frames), "resolution": "256"} for frames in (5, 9)]
        for sample in samples:
            _, frames, height, width = sample.pop("video_latents").shape
            sample["video"] = torch.ones(3, frames, height, width, dtype=torch.uint8)
    parallel = SimpleNamespace(dp_rank=0, dp_size=1, sp_size=1, sp_enabled=False)
    monkeypatch.setattr(data_loader, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(dit, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(causal, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(dit, "use_parallel_state", lambda state: nullcontext())
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(causal, "build_dataset", lambda *args, **kwargs: [samples[0], samples[0], samples[1]])
    source_config = tmp_path / "sources.yaml"
    source_config.write_text("sources: []\n")
    model_config = tiny_config(**({} if stage == "action" else dict(action_interface="none", state_dim=0)))
    model_config.save_pretrained(tmp_path / "model")
    budget = 2 * samples[0]["token_length"]
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path=str(tmp_path / "model")),
        data=DiTDataArguments(
            train_path=str(source_config),
            max_seq_len=budget,
            dyn_bsz_buffer_size=3,
        ),
        train=DiTTrainingArguments(micro_batch_size=1, max_steps=1),
    )
    configure_gradient_accumulation(args, 2)
    args.data.dataloader.num_workers = 0
    args.data.dataloader.prefetch_factor = None
    args.data.dataloader.pin_memory = False
    args.data.dataloader.shuffle = False
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.tokenize = lambda prompt: [1]
    trainer.dataset_config = {"stage": stage}
    args.train.checkpoint.output_dir = str(tmp_path / "output")
    trainer.base = BaseTrainer.__new__(BaseTrainer)
    trainer.base.args = args
    trainer.base._setup = lambda _: torch.device("cpu")
    trainer._setup()
    assert args.model.condition_model_cfg["stage"] == stage
    condition = CausalCosmos3ConditionModel(
        CausalCosmos3ConditionConfig(**{**condition.config.to_dict(), **args.model.condition_model_cfg}), device="cpu"
    )
    assert not args.data.enable_multisource
    trainer._build_data_transform()
    trainer._build_dataset()
    trainer._build_dataloader()
    loader = trainer.base.train_dataloader
    assert isinstance(loader, causal.CausalCosmos3RestartDataLoader)
    assert trainer.base.train_steps == len(loader) == 1
    microbatches = next(iter(loader))
    assert [len(batch["samples"]) for batch in microbatches] == [2, 1]

    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(model_config)
    before = condition.rng_state_dict()
    reference = [
        model(**condition.process_condition(**condition.get_condition(**batch))).loss for batch in microbatches
    ]
    expected_losses = {key: torch.stack([losses[key] for losses in reference]).mean() for key in reference[0]}
    assert set(expected_losses) == ({"video_loss"} if stage == "video" else {"video_loss", "action_loss"})
    expected_gradient = torch.autograd.grad(sum(expected_losses.values()), model.proj_in.weight)[0]
    assert expected_gradient.abs().sum() > 0
    expected_weight = model.proj_in.weight.detach() - 0.01 * expected_gradient
    expected_rng = condition.rng_state_dict()
    condition.load_rng_state_dict(before)

    class Runtime:
        condition_model = condition
        parallel_state = parallel
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)

        def __call__(self, **kwargs):
            return model(**kwargs)

        def clip_grad_norm(self):
            return torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), foreach=False)

    metrics, reshard_calls = {}, []
    trainer.base = SimpleNamespace(
        device="cpu",
        LOG_SAMPLE=False,
        state=SimpleNamespace(global_step=0),
        model=Runtime(),
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        _reset_async_activation_offload_if_enabled=lambda runtime: None,
        sync_before_train_step=lambda: None,
        on_step_begin=lambda **kwargs: None,
        on_step_end=lambda **kwargs: metrics.update(kwargs),
        model_reshard=lambda step, count: reshard_calls.append((step, count)),
        _configure_hsdp_allreduce=lambda step, count: None,
    )
    trainer.train_step(iter([microbatches]))

    assert trainer.base.num_micro_batches == 2
    assert trainer.base.state.global_step == 1
    assert reshard_calls == [(0, 2), (1, 2)]
    assert trainer.base.model.lr_scheduler.last_epoch == 1
    assert metrics["loss"] == pytest.approx(sum(expected_losses.values()).item())
    for key, expected in expected_losses.items():
        assert metrics["loss_dict"][key] == pytest.approx(expected.item())
    torch.testing.assert_close(model.proj_in.weight, expected_weight, atol=1e-6, rtol=1e-5)
    for key, expected in expected_rng.items():
        assert torch.equal(condition.rng_state_dict()[key], expected)
    assert condition.video_vae.weight.grad is None


def test_get_condition_validates_raw_pack_without_loading_vae():
    condition = CausalCosmos3ConditionModel(device="cpu")
    with pytest.raises(ValueError, match="nonempty"):
        condition.get_condition([])
    with pytest.raises(ValueError, match="vae_path"):
        condition.get_condition([{"video": torch.empty(3, 1, 16, 16, dtype=torch.uint8)}])
    assert condition.video_vae is None


def modality_rows(inputs, section, modality):
    # 独立按 layout 计算测试所需的模态行号，不假设 clean/current 各占一半。
    global_indexes = [i for i, value in enumerate(inputs["layout"].modalities) if value == modality]
    return torch.tensor(
        [row for row, index in enumerate(global_indexes) if section.start <= index < section.stop],
        dtype=torch.long,
        device=inputs["video_tokens"].device,
    )


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend noising"))]
)
def test_fixed_noise_targets_conditions_and_precision(task, dtype, device):
    sample = training_sample(task, actions=64, height=3, width=5)
    sample["action_frame_ids"] = [1] * 8 + [2] * 8 + [3] * 24 + [4] * 24
    sample["state_latent_indexes"] = sample["state_latent_indexes"].tolist()
    originals = {key: value.clone() for key, value in sample.items() if isinstance(value, torch.Tensor)}
    ev = torch.full_like(sample["video_latents"], 0.5, requires_grad=True)
    ea = torch.full_like(sample["action"], -0.25, requires_grad=True)
    sigma = torch.tensor([0.125, 0.25, 0.25, 0.75, 0.75], dtype=torch.float32)
    condition = CausalCosmos3ConditionModel(device=device, dtype=dtype)
    before = condition.rng_state_dict()
    inputs, targets = condition.prepare([sample], sigmas=[sigma], epsilon_video=[ev], epsilon_action=[ea])
    (span,) = targets["spans"]
    video_rows = modality_rows(inputs, span.current_video, 1)
    clean_rows = modality_rows(inputs, span.clean_video, 1)
    video = sample["video_latents"]
    action = sample["action"]
    expected_video = video * (1 - sigma[None, :, None, None]) + 0.5 * sigma[None, :, None, None]
    expected_video[:, sample["condition_frame_indexes_vision"]] = video[:, sample["condition_frame_indexes_vision"]]
    action_sigma = sigma[sample["action_frame_ids"]]
    expected_action = action * (1 - action_sigma[:, None]) - 0.25 * action_sigma[:, None]
    expected_action[sample["condition_frame_indexes_action"]] = action[sample["condition_frame_indexes_action"]]
    expected_action = expected_action.masked_fill(~sample["action_mask"], 0)
    torch.testing.assert_close(
        inputs["video_tokens"][video_rows].cpu(), patchify_video(expected_video, 2).to(dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        inputs["video_tokens"][clean_rows].cpu(), patchify_video(video, 2).to(dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(inputs["action_tokens"].cpu(), expected_action.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(targets["video"][0].cpu(), 0.5 - video, atol=0, rtol=0)
    expected_target = torch.zeros_like(action) if task == "fd" else -0.25 - action
    torch.testing.assert_close(
        targets["action"][0].cpu(), expected_target.masked_fill(~sample["action_mask"], 0), atol=0, rtol=0
    )
    for key in ("video_tokens", "action_tokens", "state_tokens"):
        assert inputs[key].device.type == device and inputs[key].dtype == dtype
        assert not inputs[key].requires_grad
    for key in ("video_timesteps", "action_timesteps", "position_ids"):
        assert inputs[key].dtype == torch.float32
    for key in ("video", "action", "video_times", "action_times"):
        assert targets[key][0].dtype == torch.float32 and not targets[key][0].requires_grad
    torch.testing.assert_close(targets["video_times"][0].cpu(), sigma * 1000, atol=0, rtol=0)
    torch.testing.assert_close(targets["action_times"][0].cpu(), action_sigma * 1000, atol=0, rtol=0)
    # padding 只影响监督 mask；纯时间 padding 的 current 仍保留正常插值结果。
    current_mask = targets["loss_masks"]["video_loss_mask"][video_rows]
    assert not current_mask[-1]
    assert not targets["loss_masks"]["action_loss_mask"][-7:].any()
    if task == "fd":
        assert not targets["loss_masks"]["action_loss_mask"].any()
    if task == "id":
        assert not targets["loss_masks"]["video_loss_mask"].any()
    for key, value in originals.items():
        torch.testing.assert_close(sample[key], value, atol=0, rtol=0)
    for key, state in before.items():
        assert torch.equal(condition.rng_state_dict()[key], state)


@pytest.mark.parametrize("tasks", [("fd",), ("id",), ("policy",), ("fd", "policy"), ("fd", "fd")])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend noise RNG"))]
)
def test_generated_noise_preserves_draw_order_and_all_fd_skip(tasks, device):
    samples = [training_sample(task, actions=32 * (i + 1)) for i, task in enumerate(tasks)]
    condition = CausalCosmos3ConditionModel(device=device, dtype=torch.float32, seed=23)
    generator = torch.Generator(device=device).manual_seed(23 + 32768)
    videos = [s["video_latents"].to(device) for s in samples]
    actions = [s["action"].to(device) for s in samples]
    ev = [torch.randn(v.shape, dtype=torch.float32, device=device, generator=generator) for v in videos]
    ea = [
        torch.zeros_like(a)
        if all(task == "fd" for task in tasks)
        else torch.randn(a.shape, dtype=torch.float32, device=device, generator=generator)
        for a in actions
    ]
    time_state = condition.time_generator.get_state()
    _, targets = condition.prepare(
        samples, sigmas=[torch.full((v.shape[1],), 0.5, dtype=torch.float32) for v in videos]
    )
    for i, sample in enumerate(samples):
        torch.testing.assert_close(targets["video"][i], ev[i] - videos[i], atol=0, rtol=0)
        target = torch.zeros_like(actions[i]) if all(task == "fd" for task in tasks) else ea[i] - actions[i]
        torch.testing.assert_close(
            targets["action"][i], target.masked_fill(~sample["action_mask"].to(device), 0), atol=0, rtol=0
        )
    assert torch.equal(condition.generator.get_state(), generator.get_state())
    assert torch.equal(condition.time_generator.get_state(), time_state)


def assert_prepared_equal(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        assert left.keys() == right.keys()
        for key, value in left.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right[key], atol=0, rtol=0)
            elif isinstance(value, list):
                for item, reference in zip(value, right[key], strict=True):
                    torch.testing.assert_close(item, reference, atol=0, rtol=0)
            elif isinstance(value, dict):
                for name, item in value.items():
                    torch.testing.assert_close(item, right[key][name], atol=0, rtol=0)
            else:
                assert value == right[key]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend replay"))]
)
def test_prepare_replays_both_streams_and_partial_noise_overrides(device):
    samples = [training_sample("fd"), training_sample("id", actions=64), training_sample("policy")]
    condition = CausalCosmos3ConditionModel(device=device, seed=17)
    condition.prepare(samples)
    state = condition.rng_state_dict()
    first = condition.prepare(samples)
    condition.load_rng_state_dict(state)
    assert_prepared_equal(condition.prepare(samples), first)
    for override in ("epsilon_video", "epsilon_action"):
        reference = CausalCosmos3ConditionModel(device=device, seed=17)
        kwargs = {
            override: [
                torch.zeros_like(s["video_latents"] if override == "epsilon_video" else s["action"]) for s in samples
            ]
        }
        supplied = reference.prepare(samples, **kwargs)
        reference.load_rng_state_dict(CausalCosmos3ConditionModel(device=device, seed=17).rng_state_dict())
        assert_prepared_equal(reference.prepare(samples, **kwargs), supplied)


@pytest.mark.parametrize("default_dtype", [torch.bfloat16, torch.float64])
def test_prepare_keeps_fp32_noise_and_interpolation_under_default_dtype(default_dtype):
    samples = [training_sample("fd"), training_sample("policy", actions=64)]
    expected = CausalCosmos3ConditionModel(device="cpu", seed=7).prepare(samples)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = CausalCosmos3ConditionModel(device="cpu", seed=7).prepare(samples)
    finally:
        torch.set_default_dtype(previous)
    assert_prepared_equal(actual, expected)


def test_uniform_sigma_matches_explicit_per_latent_values():
    samples = [training_sample("policy", actions=64)]
    condition = CausalCosmos3ConditionModel(device="cpu", seed=2)
    state = condition.rng_state_dict()
    uniform = condition.prepare(samples, sigmas=[torch.tensor([0.375], dtype=torch.float32)])
    condition.load_rng_state_dict(state)
    expanded = condition.prepare(samples, sigmas=[torch.full((5,), 0.375, dtype=torch.float32)])
    assert_prepared_equal(uniform, expanded)


@pytest.mark.parametrize(
    "invalid", ["sigma_dtype", "sigma_shape", "video_dtype", "video_shape", "action_dtype", "action_shape", "count"]
)
def test_prepare_rejects_invalid_explicit_noise_and_sigma(invalid):
    sample = training_sample()
    kwargs = dict(
        sigmas=[torch.tensor([0.5], dtype=torch.float32)],
        epsilon_video=[torch.zeros_like(sample["video_latents"])],
        epsilon_action=[torch.zeros_like(sample["action"])],
    )
    if invalid == "count":
        kwargs["epsilon_video"] = []
    elif invalid.startswith("sigma"):
        kwargs["sigmas"] = [
            torch.ones(2, dtype=torch.float32) if invalid.endswith("shape") else kwargs["sigmas"][0].bfloat16()
        ]
    else:
        key = "epsilon_video" if invalid.startswith("video") else "epsilon_action"
        kwargs[key] = [kwargs[key][0][:-1] if invalid.endswith("shape") else kwargs[key][0].bfloat16()]
    with pytest.raises(ValueError, match="float32|per sample"):
        CausalCosmos3ConditionModel(device="cpu").prepare([sample], **kwargs)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend prepared inputs"))],
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_prepared_inputs_keep_clean_history_network_gradients(device, checkpointing):
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    samples = [training_sample("policy", actions=64), training_sample("id", actions=64)]
    condition = CausalCosmos3ConditionModel(device=device, dtype=dtype)
    inputs, targets = condition.prepare(samples)
    inputs["video_tokens"].requires_grad_()
    model = training_model(device).to(dtype=dtype)
    model.gradient_checkpointing = checkpointing
    backend = "grouped" if device == "cpu" else "npu_tnd"
    output = model(**inputs, targets=targets, attention_backend=backend)
    losses = output.loss
    assert output["loss"] is losses
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    clean = modality_rows(inputs, targets["spans"][0].clean_video, 1)
    assert inputs["video_tokens"].grad[clean].abs().sum() > 0
    assert model.proj_in.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.proj_in.weight.grad).all()


def training_model(device="cpu", **config):
    with ops_config_scope(eager_ops_config()):
        return Cosmos3Model(tiny_config(**config)).to(device)


def constant_predictions(inputs, targets, errors):
    targets = {
        **targets,
        "video": [torch.zeros_like(value) for value in targets["video"]],
    }
    output = SimpleNamespace(video=torch.full_like(inputs["video_tokens"], 1000, requires_grad=True))
    if "action" in targets:
        targets["action"] = [torch.zeros_like(value) for value in targets["action"]]
        output.action = torch.full_like(inputs["action_tokens"], 1000, requires_grad=True)
    with torch.no_grad():
        for span, error in zip(targets["spans"], errors, strict=True):
            output.video[span.current_video_rows] = error
            if "action" in targets:
                output.action[span.action_rows] = error
    return output, targets


@pytest.mark.parametrize("tasks", [("fd",), ("id",), ("policy",), ("fd", "id", "policy"), ("fd", "fd")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend loss"))]
)
def test_loss_normalization_masks_weights_and_prediction_gradients(tasks, dtype, device):
    samples = [training_sample(task, actions=32 * (i + 1), height=3, width=5) for i, task in enumerate(tasks)]
    condition = CausalCosmos3ConditionModel(device=device, dtype=dtype)
    model = training_model(device, video_loss_weight=3, action_loss_weight=5)
    inputs, targets = condition.prepare(samples)
    output, targets = constant_predictions(inputs, targets, range(1, len(samples) + 1))
    losses = model.loss(output, targets)
    expected = {"video_loss": 0.0, "action_loss": 0.0}
    video_grad = torch.zeros_like(output.video.cpu())
    action_grad = torch.zeros_like(output.action.cpu())
    for i, (sample, span) in enumerate(zip(samples, targets["spans"], strict=True)):
        error = i + 1
        vm = torch.ones_like(sample["video_latents"], dtype=torch.bool)
        vm[:, sample["condition_frame_indexes_vision"]] = False
        vm[:, sample["num_valid_video_latents"] :] = False
        am = sample["action_mask"].expand_as(sample["action"]).clone()
        am[~sample["action_valid_steps"]] = False
        am[sample["condition_frame_indexes_action"]] = False
        for name, mask, weight in (("video", vm, 3), ("action", am, 5)):
            count = int(mask.sum())
            if count:
                expected[name + "_loss"] += error**2 * weight / len(samples)
            gradient = mask.float() * (2 * error * weight / (max(count, 1) * len(samples)))
            if name == "video":
                video_grad[span.current_video_rows] = patchify_video(gradient, 2).to(dtype)
            else:
                action_grad[span.action_rows] = gradient.to(dtype)
    for name, value in losses.items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(
            value.cpu(), torch.tensor(expected[name], dtype=torch.float32), atol=1e-6, rtol=2e-6
        )
    sum(losses.values()).backward()
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(output.video.grad.cpu(), video_grad, atol=1e-7, rtol=tolerance)
    torch.testing.assert_close(output.action.grad.cpu(), action_grad, atol=1e-7, rtol=tolerance)
    # 修改条件、clean、时间 padding、无效维度等不监督的位置，不应改变直接 loss。
    with torch.no_grad():
        output.video[~targets["loss_masks"]["video_loss_mask"]] += 100
        output.action[~targets["loss_masks"]["action_loss_mask"]] += 100
    changed = model.loss(output, targets)
    for name in losses:
        torch.testing.assert_close(changed[name], losses[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "height,width,patch_size", [(24, 20, 2), (45, 40, 2), (33, 40, 2), (3, 5, 2), (3, 5, 3), (3, 5, 1)]
)
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend spatial loss"))],
)
def test_loss_crops_patch_padding_before_normalization(height, width, patch_size, device):
    sample = training_sample(height=height, width=width)
    _, frames, h, w = sample["video_latents"].shape
    patches = ((h + patch_size - 1) // patch_size) * ((w + patch_size - 1) // patch_size)
    sample["token_length"] = (
        len(sample["text_token_ids"]) + 2 * frames * patches + len(sample["action"]) + len(sample["states"])
    )
    condition = CausalCosmos3ConditionModel(device=device, dtype=torch.float32, patch_size=patch_size)
    model = training_model(device, patch_size=patch_size)
    inputs, targets = condition.prepare([sample])
    output, targets = constant_predictions(inputs, targets, [1])
    span = targets["spans"][0]
    real_pixels = patchify_video(torch.ones_like(sample["video_latents"]), patch_size).bool().to(device)
    with torch.no_grad():
        current = output.video[span.current_video_rows]
        current.masked_fill_(~real_pixels, 1000)
    losses = model.loss(output, targets)
    torch.testing.assert_close(losses["video_loss"].cpu(), torch.tensor(10.0), atol=0, rtol=0)
    sum(losses.values()).backward()
    assert torch.count_nonzero(output.video.grad[span.current_video_rows][~real_pixels]) == 0


def test_loss_honors_individual_spatial_patch_masks():
    sample = training_sample(height=3, width=5)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    model = training_model(video_loss_weight=1)
    inputs, targets = condition.prepare([sample])
    output, targets = constant_predictions(inputs, targets, [2])
    span = targets["spans"][0]
    mask = targets["loss_masks"]["video_loss_mask"][span.current_video_rows].reshape(3, 6)
    mask[1, [0, 2]] = False
    loss = model.loss(output, targets)["video_loss"]
    torch.testing.assert_close(loss, torch.tensor(4.0), atol=0, rtol=0)
    loss.backward()
    # 剩余 9 个空间点、2 个通道，有效元素共 18，梯度为 2*error/18。
    gradient = output.video.grad[span.current_video_rows].reshape(3, 6, 8)
    assert torch.count_nonzero(gradient[1, [0, 2]]) == 0
    torch.testing.assert_close(gradient[1, 1], torch.full((8,), 4 / 18), atol=1e-7, rtol=1e-6)


def test_fully_conditioned_modalities_keep_zero_losses_connected():
    sample = training_sample("fd")
    sample["condition_frame_indexes_vision"] = list(range(sample["video_latents"].shape[1]))
    model = training_model()
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    inputs, targets = condition.prepare([sample])
    output, targets = constant_predictions(inputs, targets, [2])
    losses = model.loss(output, targets)
    assert all(value.item() == 0 and value.requires_grad for value in losses.values())
    sum(losses.values()).backward()
    assert output.video.grad is not None and torch.count_nonzero(output.video.grad) == 0
    assert output.action.grad is not None and torch.count_nonzero(output.action.grad) == 0


def test_loss_follows_reordered_samples_and_clean_current_spans(monkeypatch):
    samples = [training_sample(), training_sample("id", actions=64, height=3, width=5)]
    model = training_model()
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    state = condition.rng_state_dict()
    inputs, targets = condition.prepare(samples)
    generator = torch.Generator().manual_seed(13)
    output = SimpleNamespace(
        video=torch.randn(inputs["video_tokens"].shape, generator=generator, requires_grad=True),
        action=torch.randn(inputs["action_tokens"].shape, generator=generator, requires_grad=True),
    )
    expected = model.loss(output, targets)
    expected_grad = torch.autograd.grad(sum(expected.values()), (output.video, output.action))
    original_layout, _, length = packing.build_packed_layout(samples)
    sections, permutation, cursor = [{}, {}], [], 0
    for sample_id in (1, 0):
        for name in ("state", "action", "current_video", "text", "clean_video"):
            section = getattr(targets["spans"][sample_id], name)
            count = section.stop - section.start
            sections[sample_id][name] = slice(cursor, cursor + count)
            permutation.extend(range(section.start, section.stop))
            cursor += count
    spans = tuple(SampleSpans(**section) for section in sections)
    layout = packing.CausalLayout(
        **{
            key: tuple(getattr(original_layout, key)[i] for i in permutation)
            for key in ("sample_ids", "block_ids", "roles", "modalities", "frame_ids")
        },
        history_blocks=original_layout.history_blocks,
        block_sizes=original_layout.block_sizes,
    )
    monkeypatch.setattr(packing, "build_packed_layout", lambda *args, **kwargs: (layout, spans, length))
    condition.load_rng_state_dict(state)
    _, reordered_targets = condition.prepare(samples)
    rows = {}
    for name, modality in (("video", 1), ("action", 2)):
        old_rows = {
            i: row
            for row, i in enumerate(i for i, value in enumerate(original_layout.modalities) if value == modality)
        }
        rows[name] = [old_rows[i] for i in permutation if original_layout.modalities[i] == modality]
    reordered = SimpleNamespace(
        video=output.video.detach()[rows["video"]].requires_grad_(),
        action=output.action.detach()[rows["action"]].requires_grad_(),
    )
    actual = model.loss(reordered, reordered_targets)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    gradients = torch.autograd.grad(sum(actual.values()), (reordered.video, reordered.action))
    for name, gradient, reference in zip(("video", "action"), gradients, expected_grad, strict=True):
        torch.testing.assert_close(gradient, reference[rows[name]], atol=0, rtol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("stage", ["action", "video"])
def test_validation_inputs_repeat_across_steps_orders_and_restore_condition_rng(stage):
    from veomni.data.causal_cosmos3.datasets.validation import validation_seed
    from veomni.trainer.causal_cosmos3 import fixed_validation_inputs

    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32, stage=stage)
    samples = [training_sample(), training_sample("fd", actions=64), training_sample("id")]
    if stage == "video":
        samples = [{**video_payload_sample(frames=f), "resolution": "256"} for f in (3, 5, 7)]
    streams = ("time", "video", "action") if stage == "action" else ("time", "video")
    seeds = [{name: validation_seed(73, {"window": i}, name) for name in streams} for i in range(3)]
    before = condition.rng_state_dict()
    expected = [fixed_validation_inputs(condition, [sample], seed) for sample, seed in zip(samples, seeds)]
    for i in (2, 0, 1, 2):
        # Training advances both of its generators, without changing validation.
        condition.sample_sigmas([samples[i]])
        torch.randn(17, generator=condition.generator)
        actual = fixed_validation_inputs(condition, [samples[i]], seeds[i])
        for name in streams[1:]:
            for key in (name + "_tokens", name + "_timesteps"):
                torch.testing.assert_close(actual[key], expected[i][key], atol=0, rtol=0)
            for key in (name, name + "_times"):
                torch.testing.assert_close(actual["targets"][key][0], expected[i]["targets"][key][0], atol=0, rtol=0)
        if stage == "video":
            assert "action_tokens" not in actual and "action" not in actual["targets"]
    condition.load_rng_state_dict(before)
    fixed_validation_inputs(condition, [samples[0]], seeds[0])
    for key, state in before.items():
        assert torch.equal(condition.rng_state_dict()[key], state)
    # Increasing video geometry does not advance the independent action stream.
    changed = {**samples[0], "video_latents": torch.zeros(2, 3, 4, 4)}
    changed["token_length"] += 18
    changed_inputs = fixed_validation_inputs(condition, [changed], seeds[0])
    if stage == "action":
        torch.testing.assert_close(
            changed_inputs["targets"]["action"][0], expected[0]["targets"]["action"][0], atol=0, rtol=0
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("fail", [False, True])
def test_validation_state_restores_mixed_flags_and_every_rng(monkeypatch, fail):
    import veomni.trainer.causal_cosmos3 as validation

    monkeypatch.setattr(validation, "get_device_rng_state", torch.get_rng_state)
    monkeypatch.setattr(validation, "set_device_rng_state", torch.set_rng_state)
    model = training_model()
    model.train()
    model.layers[0].eval()
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32)
    runtime = SimpleNamespace(model=model, condition_model=condition)
    flags = [module.training for root in (model, condition) for module in root.modules()]
    np_state = np.random.get_state()
    before = (
        random.getstate(),
        np_state,
        torch.get_rng_state(),
        condition.rng_state_dict(),
    )
    try:
        with validation.validation_state(runtime):
            assert all(not module.training for root in (model, condition) for module in root.modules())
            random.random(), np.random.rand(), torch.randn(7)
            condition.sample_sigmas([sigma_sample(7)])
            torch.randn(7, generator=condition.generator)
            condition.video_vae = torch.nn.Linear(1, 1).eval()
            if fail:
                raise OSError("preparation failed")
    except OSError:
        assert fail
    assert [
        module.training
        for root in (model, condition)
        for module in root.modules()
        if module is not condition.video_vae
    ] == flags
    assert not condition.video_vae.training
    assert random.getstate() == before[0]
    assert np.random.get_state()[0] == np_state[0]
    assert (np.random.get_state()[1] == np_state[1]).all()
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(before[2], torch.get_rng_state())
    assert all(torch.equal(value, condition.rng_state_dict()[key]) for key, value in before[3].items())


@pytest.fixture
def validation_loss_harness(tmp_path, monkeypatch, request):
    import veomni.trainer.causal_cosmos3 as validation
    from veomni.data.causal_cosmos3.datasets.validation import validation_seed

    monkeypatch.setattr(validation, "use_parallel_state", lambda state: nullcontext())
    monkeypatch.setattr(validation, "get_device_rng_state", torch.get_rng_state)
    monkeypatch.setattr(validation, "set_device_rng_state", torch.set_rng_state)
    stage = getattr(request, "param", "action")
    condition = CausalCosmos3ConditionModel(
        device="cpu", dtype=torch.float32, attention_backend="grouped", stage=stage
    )
    monkeypatch.setattr(condition, "get_condition", lambda samples: {"samples": samples})
    model_options = {} if stage == "action" else {"action_interface": "none", "state_dim": 0}
    model = training_model(video_loss_weight=3, action_loss_weight=5, **model_options)
    items = []
    tasks = (
        (("forward_dynamics", "fd"), ("inverse_dynamics", "id"), ("policy", "policy"))
        if stage == "action"
        else (("video", None),)
    )
    streams = ("time", "video", "action") if stage == "action" else ("time", "video")
    for mode, task in tasks:
        for index, (error, weight, supervised) in enumerate(
            ((1, 1, True), (2, 1, True), (100, 0, True), (0, 1, False))
        ):
            sample = training_sample(task) if stage == "action" else {**video_payload_sample(), "resolution": "256"}
            if not supervised:
                sample["condition_frame_indexes_vision"] = list(range(sample["video_latents"].shape[1]))
                if stage == "action":
                    sample["condition_frame_indexes_action"] = list(range(len(sample["action"])))
            record = dict(
                mode=mode,
                seeds={name: validation_seed(7, [mode, index], name) for name in streams},
            )
            items.append(dict(batch={"samples": [sample]}, record=record, weight=weight, error=error))

    class Runtime:
        parallel_state = SimpleNamespace(dp_rank=0, dp_size=1, dp_group=None)
        calls = 0

        def __call__(self, **inputs):
            assert not torch.is_grad_enabled() and not self.model.training
            item = items[self.calls % len(items)]
            self.calls += 1
            output, targets = constant_predictions(inputs, inputs["targets"], [item["error"]])
            return SimpleNamespace(loss=model.loss(output, targets))

    runtime = Runtime()
    runtime.model, runtime.condition_model = model, condition
    dataset = list(range(4))
    monkeypatch.setattr(validation, "iter_validation_batches", lambda dataset, rank, size, seed: iter(items))
    validation_config = {"sample": {"mode": "joint"}, "validation": {"seed": 7}}
    if stage == "video":
        validation_config = {"stage": "video", "validation": {"seed": 7}}
    trainer = SimpleNamespace(
        model=runtime,
        device="cpu",
        model_fwd_context=nullcontext(),
        validation_config=validation_config,
        args=SimpleNamespace(
            model=SimpleNamespace(tokenizer_path=None),
            train=SimpleNamespace(
                global_rank=0,
                wandb=SimpleNamespace(enable=False),
                checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
            ),
        ),
    )
    return validation.CausalCosmos3Validator(trainer, dataset), items


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validation_mode_means_masks_padding_and_logging(validation_loss_harness):
    validator, items = validation_loss_harness
    runtime = validator.runtime
    parameters = {name: value.detach().clone() for name, value in runtime.model.named_parameters()}
    state = SimpleNamespace(global_step=5, epoch=0)
    actual = validator.evaluate(state)
    assert runtime.calls == len(items)
    expected_losses = {"video": 7.5} if validator.modes == ("video",) else {"fd": 7.5, "id": 12.5, "policy": 20}
    assert {key for key in actual if key.endswith("/loss")} == {f"eval/{tag}/loss" for tag in expected_losses}
    for mode, expected in expected_losses.items():
        assert actual[f"eval/{mode}/loss"] == pytest.approx(expected)
        assert actual[f"eval/{mode}/count"] == 2
        assert actual[f"eval/{mode}/unsupervised_count"] == 1
        assert actual[f"eval/{mode}/padding_count"] == 1
    assert validator.evaluate(SimpleNamespace(global_step=900, epoch=1)) == actual
    logs = [json.loads(line) for line in (validator.output_dir / "validation.jsonl").read_text().splitlines()]
    assert [entry["global_step"] for entry in logs] == [5, 900]
    assert len({entry["validation_id"] for entry in logs}) == 1
    for name, value in runtime.model.named_parameters():
        torch.testing.assert_close(value, parameters[name], atol=0, rtol=0)
        assert value.grad is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("stage", [None, "unknown", "video"])
def test_validator_rejects_invalid_or_mismatched_stage(validation_loss_harness, stage):
    from veomni.trainer.causal_cosmos3 import CausalCosmos3Validator

    validator, _ = validation_loss_harness
    validator.trainer.validation_config["stage"] = stage
    with pytest.raises(ValueError, match="stage"):
        CausalCosmos3Validator(validator.trainer, validator.dataset)
    assert validator.runtime.calls == 0


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validator_rejects_iterator_mode_mismatch_before_forward(validation_loss_harness):
    validator, items = validation_loss_harness
    items[0]["record"]["mode"] = "unexpected"
    before = validator.runtime.condition_model.rng_state_dict()
    with pytest.raises(ValueError, match="iterator mode"):
        validator.evaluate(SimpleNamespace(global_step=1, epoch=0))
    assert validator.runtime.calls == 0
    for name, state in before.items():
        assert torch.equal(state, validator.runtime.condition_model.rng_state_dict()[name])


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validator_requires_all_selected_loss_fields(validation_loss_harness, monkeypatch):
    validator, _ = validation_loss_harness
    missing = validator.loss_names[-1]
    monkeypatch.setattr(
        type(validator.runtime),
        "__call__",
        lambda self, **inputs: SimpleNamespace(loss={name: torch.tensor(0.0) for name in validator.loss_names[:-1]}),
    )
    with pytest.raises(KeyError, match=missing):
        validator.evaluate(SimpleNamespace(global_step=1, epoch=0))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validation_no_supervision_and_prepare_failure_restore_state(validation_loss_harness, monkeypatch):
    validator, items = validation_loss_harness
    for item in items:
        sample = item["batch"]["samples"][0]
        sample["condition_frame_indexes_vision"] = list(range(sample["video_latents"].shape[1]))
        if "action" in sample:
            sample["condition_frame_indexes_action"] = list(range(len(sample["action"])))
    with pytest.raises(ValueError, match="no supervised windows"):
        validator.evaluate(SimpleNamespace(global_step=1, epoch=0))
    before = validator.runtime.condition_model.rng_state_dict()

    def failed_prepare(*args, **kwargs):
        torch.randn(7, generator=validator.runtime.condition_model.generator)
        raise OSError("VAE failure")

    monkeypatch.setattr(validator.runtime.condition_model, "get_condition", failed_prepare)
    with pytest.raises(OSError, match="VAE failure"):
        validator.evaluate(SimpleNamespace(global_step=2, epoch=0))
    assert validator.runtime.model.training
    assert all(
        torch.equal(value, validator.runtime.condition_model.rng_state_dict()[key]) for key, value in before.items()
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_validation_callback_step_epoch_deduplication(monkeypatch):
    import veomni.trainer.callbacks.base as callbacks
    from veomni.trainer.causal_cosmos3 import CausalCosmos3EvaluateCallback

    monkeypatch.setattr(callbacks, "get_parallel_state", lambda: SimpleNamespace())
    calls = []
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(eval_steps=2, eval_epochs=1)))
    callback = CausalCosmos3EvaluateCallback(
        trainer, SimpleNamespace(evaluate=lambda state: calls.append(state.global_step))
    )
    for step in (1, 2, 3, 4):
        callback.on_step_end(SimpleNamespace(global_step=step, epoch=0))
    callback.on_epoch_end(SimpleNamespace(global_step=4, epoch=0))
    callback.on_epoch_end(SimpleNamespace(global_step=5, epoch=1))
    assert calls == [2, 4, 5]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_causal_trainer_connects_validation_dataset_to_callback(validation_loss_harness, monkeypatch, enabled):
    import veomni.trainer.callbacks.base as callbacks
    from veomni.trainer.causal_cosmos3 import CausalCosmos3EvaluateCallback

    validator, items = validation_loss_harness
    base = validator.trainer
    previous = object()
    base.global_state_callback = object()
    base.evaluate_callback, base._callbacks = previous, [base.global_state_callback, previous]
    checkpoint = base.args.train.checkpoint
    checkpoint.save_steps, checkpoint.save_epochs = 5, 1
    checkpoint.save_hf_weights = False
    checkpoint.hf_save_steps, checkpoint.hf_save_epochs = 5, 1
    base.validation_dataset = validator.dataset if enabled else None
    base.args.train.eval_steps, base.args.train.eval_epochs = 2, 1
    monkeypatch.setattr(DiTTrainer, "__init__", lambda self, args: setattr(self, "base", base))
    monkeypatch.setattr(callbacks, "get_parallel_state", lambda: validator.parallel)
    trainer = CausalCosmos3Trainer(base.args, dataset_config=base.validation_config or {}, tokenize=lambda prompt: [1])
    if not enabled:
        assert base.evaluate_callback is previous and base._callbacks == [base.global_state_callback, previous]
        assert not hasattr(trainer, "validator")
        return
    callback = base.evaluate_callback
    assert isinstance(callback, CausalCosmos3EvaluateCallback)
    assert callback.trainer is base and callback.validator is trainer.validator
    assert base._callbacks == [base.global_state_callback, callback]
    callback.on_step_end(SimpleNamespace(global_step=1, epoch=0))
    assert validator.runtime.calls == 0
    callback.on_step_end(SimpleNamespace(global_step=2, epoch=0))
    callback.on_epoch_end(SimpleNamespace(global_step=2, epoch=0))
    assert validator.runtime.calls == len(items)
    key, expected = ("eval/video/loss", 7.5) if validator.modes == ("video",) else ("eval/policy/loss", 20)
    assert trainer.validator.last_metrics[key] == pytest.approx(expected)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validation_keeps_next_training_input_noise_loss_and_update(validation_loss_harness, monkeypatch):
    import copy

    validator, items = validation_loss_harness
    runtime = validator.runtime
    # Use the real tiny Cosmos forward here; the arithmetic fixture above uses
    # controlled errors only for the independently calculable means.
    monkeypatch.setattr(type(runtime), "__call__", lambda self, **inputs: self.model(**inputs))
    sample = items[-4]["batch"]["samples"][0]
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def next_step():
        next_input = torch.randn(13)
        inputs = runtime.condition_model.process_condition([sample])
        optimizer.zero_grad()
        outputs = runtime.model(**inputs)
        loss = sum(outputs.loss.values())
        loss.backward()
        optimizer.step()
        scheduler.step()
        return (
            next_input,
            inputs["video_tokens"].clone(),
            inputs["action_tokens"].clone() if "action_tokens" in inputs else torch.empty(0),
            loss.detach().clone(),
            {name: value.detach().clone() for name, value in runtime.model.named_parameters()},
        )

    # Populate Adam state before testing evaluation's isolation.
    next_step()
    model_state = copy.deepcopy(runtime.model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    condition_state = runtime.condition_model.rng_state_dict()
    global_rng = torch.get_rng_state()
    reference = next_step()
    runtime.model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    runtime.condition_model.load_rng_state_dict(condition_state)
    torch.set_rng_state(global_rng)
    validator.evaluate(SimpleNamespace(global_step=1, epoch=0))
    assert scheduler.state_dict() == scheduler_state
    for key, state in optimizer_state["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(optimizer.state_dict()["state"][key][name], value, atol=0, rtol=0)
    for name, value in runtime.model.state_dict().items():
        torch.testing.assert_close(value, model_state[name], atol=0, rtol=0)
    actual = next_step()
    for value, expected in zip(actual[:4], reference[:4]):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    for name, value in actual[4].items():
        torch.testing.assert_close(value, reference[4][name], atol=0, rtol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("mode_id,tag,expected", [(0, "fd", 7.5), (1, "id", 12.5), (2, "policy", 20.0)])
def test_validation_single_mode_executes_only_its_windows(validation_loss_harness, mode_id, tag, expected):
    validator, items = validation_loss_harness
    validator.modes = (validator.modes[mode_id],)
    items[:] = items[mode_id * 4 : (mode_id + 1) * 4]
    result = validator.evaluate(SimpleNamespace(global_step=1, epoch=0))
    assert validator.runtime.calls == 4
    assert {key for key in result if key.endswith("/loss")} == {f"eval/{tag}/loss"}
    assert result[f"eval/{tag}/loss"] == pytest.approx(expected)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("reshard_error", [False, True])
def test_validation_reshards_no_backward_parameters_and_restores_rng_on_cleanup_error(monkeypatch, reshard_error):
    import veomni.trainer.causal_cosmos3 as validation

    class Reshardable(torch.nn.Linear):
        calls = 0

        def reshard(self):
            self.calls += 1
            if reshard_error:
                raise RuntimeError("reshard failed")

    monkeypatch.setattr(validation, "FSDPModule", Reshardable)
    monkeypatch.setattr(validation, "get_device_rng_state", torch.get_rng_state)
    monkeypatch.setattr(validation, "set_device_rng_state", torch.set_rng_state)
    model = Reshardable(1, 1)
    condition = CausalCosmos3ConditionModel(device="cpu")
    before = torch.get_rng_state()
    try:
        with validation.validation_state(SimpleNamespace(model=model, condition_model=condition)):
            torch.randn(13)
    except RuntimeError:
        assert reshard_error
    assert model.calls == 1 and model.training and condition.training
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend fixed validation noise")
def test_validation_npu_noise_is_independent_of_training_rank_and_stream_position(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition as condition_module
    from veomni.trainer.causal_cosmos3 import fixed_validation_inputs

    monkeypatch.setattr(condition_module, "is_parallel_state_initialized", lambda: True)
    seeds = {"time": 71, "video": 72, "action": 73}
    sample = training_sample()
    reference = None
    for rank in (0, 7, 2):
        monkeypatch.setattr(condition_module, "get_parallel_state", lambda rank=rank: SimpleNamespace(dp_rank=rank))
        condition = CausalCosmos3ConditionModel(device="npu")
        condition.sample_sigmas([sample])
        torch.randn(31 + rank, device=condition.device, generator=condition.generator)
        before = condition.rng_state_dict()
        actual = fixed_validation_inputs(condition, [sample], seeds)
        signature = [actual["targets"][name][0].cpu() for name in ("video", "action", "video_times", "action_times")]
        if reference is None:
            reference = signature
        for value, expected in zip(signature, reference):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        assert all(torch.equal(value, condition.rng_state_dict()[key]) for key, value in before.items())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validation_wandb_preserves_monotonic_training_history(validation_loss_harness, monkeypatch):
    import sys

    validator, _ = validation_loss_harness
    validator.trainer.args.train.wandb.enable = True
    run = SimpleNamespace(step=6)
    definitions, rows = [], []

    def log(data, step, commit):
        assert step >= run.step
        assert not commit
        rows.append((data, step))

    mock = SimpleNamespace(run=run, define_metric=lambda *args, **kwargs: definitions.append((args, kwargs)), log=log)
    monkeypatch.setitem(sys.modules, "wandb", mock)
    validator.evaluate(SimpleNamespace(global_step=5, epoch=0))
    assert rows[0][0]["eval/global_step"] == 5 and rows[0][1] == 6
    assert run.step == 6
    assert definitions == [(("eval/global_step",), {}), (("eval/*",), {"step_metric": "eval/global_step"})]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("phase", ["directory", "logging"])
@pytest.mark.parametrize("validation_loss_harness", ["action", "video"], indirect=True)
def test_validation_restores_state_after_persistence_failure(validation_loss_harness, monkeypatch, phase):
    from pathlib import Path

    validator, _ = validation_loss_harness
    condition = validator.runtime.condition_model
    before_cpu, before_condition = torch.get_rng_state(), condition.rng_state_dict()

    def failed_publish(*args, **kwargs):
        torch.randn(7)
        torch.randn(7, generator=condition.generator)
        raise OSError("validation log failed")

    if phase == "directory":
        monkeypatch.setattr(Path, "mkdir", failed_publish)
    else:
        monkeypatch.setattr(validator, "_publish", failed_publish)
    with pytest.raises(OSError, match="validation log failed"):
        validator.evaluate(SimpleNamespace(global_step=1, epoch=0))
    assert validator.runtime.model.training
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert all(torch.equal(value, condition.rng_state_dict()[key]) for key, value in before_condition.items())
