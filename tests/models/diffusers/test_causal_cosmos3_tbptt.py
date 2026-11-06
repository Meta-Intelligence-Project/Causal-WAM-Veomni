"""Opt-in historical gradient truncation against a full-shape autograd oracle."""

import copy
from types import MethodType

import pytest
import torch
import yaml
from torch.distributed.fsdp import FSDPModule

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.causal_cosmos3_serving.test_native import window
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample
from veomni.models import build_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import (
    Cosmos3Config,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.utils.recompute_utils import apply_recompute_policy, build_policy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def decision_inputs(modes=("policy",), histories=None):
    """Use multiple spatial patches to keep every historical latent indivisible."""
    histories = histories or [1024] * len(modes)
    samples = []
    for index, (mode, history) in enumerate(zip(modes, histories, strict=True)):
        _, sample = window(history_steps=history, horizon=32)
        sample["mode"] = mode
        sample["video_latents"] = sample["video_latents"].repeat(1, 1, 2, 4)
        sample["video_latents"] += torch.arange(8).reshape(1, 1, 2, 4) * 0.003 + index * 0.07
        sample["states"] = sample["states"].clone()
        sample["states"][0, :2] = torch.tensor([0.12 + 0.03 * index, -0.23])
        sample["text_token_ids"] = torch.tensor([1, 2 + index])
        if mode == "forward_dynamics":
            sample["condition_frame_indexes_action"] = list(range(32))
        elif mode == "inverse_dynamics":
            sample["condition_frame_indexes_vision"] = list(range(sample["video_latents"].shape[1]))
        sample["resolution"] = "256"
        sample["token_length"] = 2 + 2 * sample["video_latents"].shape[1] + 32 + 1
        samples.append(sample)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32, seed=17, shift=1)
    return condition.prepare(samples)


def history_masks(layout, keep, *, video_only=False):
    """Independent old/recent definition; do not use the production partitioner."""
    old, recent = [], []
    for sample, role, modality, frame in zip(
        layout.sample_ids, layout.roles, layout.modalities, layout.frame_ids, strict=True
    ):
        if (video_only and modality != 1) or (not video_only and role == -1):
            continue
        is_old = modality == 1 and role == 0 and frame < max(0, layout.decision_windows[sample] - keep)
        old.append(is_old)
        recent.append(modality == 1 and role == 0 and not is_old)
    return torch.tensor(old), torch.tensor(recent)


def install_full_shape_oracle(model, keep):
    """Retain all GEMM/attention shapes, detaching only historical hidden edges."""
    for layer in model.layers:
        original = layer.forward

        def forward(self, und, gen, tables, plan, cache_capture=None, history_partition=None, _original=original):
            assert history_partition is None
            old, _ = history_masks(plan.layout, keep)
            entry = torch.where(old[:, None], gen.detach(), gen)
            text, hidden = _original(und, entry, tables, plan, cache_capture)
            return text, torch.where(old[:, None], hidden.detach(), hidden)

        layer.forward = MethodType(forward, layer)


def training_result(template, inputs, targets, backend, recompute="none", *, oracle_keep=None):
    model = copy.deepcopy(template).train()
    if oracle_keep is not None:
        install_full_shape_oracle(model, oracle_keep)
    if recompute != "none":
        model.gradient_checkpointing_enable()
        if recompute == "sac":
            report = apply_recompute_policy(
                model,
                build_policy(
                    enabled=True,
                    enable_reentrant=False,
                    early_stop=True,
                    recompute_layers=-1,
                    selective_recompute_layers=2,
                ),
            )
            assert report.bound_blocks == 2
    data = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    data["video_tokens"].requires_grad_()
    memory_outputs = []
    handles = []
    if model.config.enable_tbptt and recompute == "none":

        def retain_memory(module, args, output):
            output.retain_grad()
            memory_outputs.append(output)

        for name in ("add_k_proj", "add_v_proj"):
            handles.append(getattr(model.layers[0].self_attn, name).register_forward_hook(retain_memory))
    output = model(**data, targets=targets, attention_backend=backend)
    loss = sum(output.loss.values())
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    assert all(value is None or torch.isfinite(value).all() for value in gradients.values())
    if memory_outputs:
        old, _ = history_masks(inputs["layout"], model.config.tbptt_keep_history_latents)
        if old.any():
            assert len(memory_outputs) == 2
            assert all(value.grad is not None and value.grad[old].abs().sum() > 0 for value in memory_outputs)
    for handle in handles:
        handle.remove()
    return dict(
        video=output.video.detach(),
        action=output.action.detach(),
        loss=loss.detach(),
        gradients=gradients,
        video_gradient=data["video_tokens"].grad.detach(),
    )


def assert_training_equal(actual, expected, *, exact=False):
    for key in ("video", "action", "loss", "video_gradient"):
        torch.testing.assert_close(
            actual[key], expected[key], atol=0 if exact else 2e-6, rtol=0 if exact else 2e-5, msg=key
        )
    assert actual["gradients"].keys() == expected["gradients"].keys()
    for name, value in actual["gradients"].items():
        reference = expected["gradients"][name]
        assert (value is None) == (reference is None), name
        if value is not None:
            torch.testing.assert_close(
                value, reference, atol=0 if exact else 2e-6, rtol=0 if exact else 2e-5, msg=name
            )


def training_model(*, enabled=False, keep=16):
    model = Cosmos3Model(
        tiny_config(
            action_dim=55,
            state_dim=55,
            train_generation_only=True,
            enable_tbptt=enabled,
            tbptt_keep_history_latents=keep,
        )
    )
    model.freeze_model()
    return model


@pytest.mark.parametrize("enabled,keep", [(False, 16), (True, 65)])
@pytest.mark.parametrize("recompute", ["none", "sac"])
def test_tbptt_disabled_or_full_history_preserves_training(enabled, keep, recompute):
    inputs, targets = decision_inputs(("policy", "inverse_dynamics"))
    torch.manual_seed(871)
    reference = training_model()
    candidate = copy.deepcopy(reference)
    candidate.config.enable_tbptt = enabled
    candidate.config.tbptt_keep_history_latents = keep
    expected = training_result(reference, inputs, targets, "grouped", recompute)
    actual = training_result(candidate, inputs, targets, "grouped", recompute)
    assert_training_equal(actual, expected, exact=True)


@pytest.mark.parametrize(
    "modes,backend,recompute,keep",
    [
        (("policy",), "dense", "none", 32),
        (("forward_dynamics",), "grouped", "checkpoint", 16),
        (("inverse_dynamics",), "grouped", "sac", 16),
        (("policy", "forward_dynamics", "inverse_dynamics"), "grouped", "sac", 32),
        (("policy", "forward_dynamics", "inverse_dynamics"), "dense", "checkpoint", 16),
        (("policy",), "grouped", "sac", 16),
    ],
)
def test_tbptt_matches_independent_truncated_gradient_oracle(modes, backend, recompute, keep):
    inputs, targets = decision_inputs(modes)
    torch.manual_seed(871)
    reference = training_model()
    candidate = copy.deepcopy(reference)
    candidate.config.enable_tbptt = True
    candidate.config.tbptt_keep_history_latents = keep
    expected = training_result(reference, inputs, targets, backend, recompute, oracle_keep=keep)
    actual = training_result(candidate, inputs, targets, backend, recompute)
    assert_training_equal(actual, expected)
    old, recent = history_masks(inputs["layout"], keep, video_only=True)
    assert not actual["video_gradient"][old].any()
    assert actual["video_gradient"][recent].abs().sum() > 0
    for name in (
        "state_proj_in.weight",
        "state_modality_embed",
        "layers.0.self_attn.k_norm_und_for_gen.weight",
        "layers.0.self_attn.add_k_proj.weight",
        "layers.0.self_attn.add_v_proj.weight",
    ):
        assert actual["gradients"][name] is not None and actual["gradients"][name].abs().sum() > 0, name


def test_tbptt_partitions_mixed_history_lengths_without_changing_sample_boundaries():
    inputs, targets = decision_inputs(("policy", "forward_dynamics", "inverse_dynamics"), histories=[0, 48, 1024])
    torch.manual_seed(871)
    reference = training_model()
    candidate = copy.deepcopy(reference)
    candidate.config.enable_tbptt = True
    candidate.config.tbptt_keep_history_latents = 16
    expected = training_result(reference, inputs, targets, "grouped", oracle_keep=16)
    actual = training_result(candidate, inputs, targets, "grouped")
    assert_training_equal(actual, expected)


def test_tbptt_checkpoint_does_not_recompute_old_history_mlp():
    inputs, targets = decision_inputs()
    model = training_model(enabled=True).train()
    model.gradient_checkpointing_enable()
    apply_recompute_policy(
        model,
        build_policy(
            enabled=True,
            enable_reentrant=False,
            early_stop=True,
            recompute_layers=-1,
            selective_recompute_layers=2,
        ),
    )
    calls = [{"active": 0, "old": 0} for _ in model.layers]

    def record(index):
        def hook(module, args):
            # Old MLP outputs are constants for the following layer; checkpoint
            # recomputation should finish before evaluating this branch again.
            calls[index]["active" if torch.is_grad_enabled() else "old"] += 1

        return hook

    handles = [layer.mlp_moe_gen.register_forward_pre_hook(record(index)) for index, layer in enumerate(model.layers)]
    try:
        output = model(**inputs, targets=targets, attention_backend="grouped")
        sum(output.loss.values()).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [{"active": 2, "old": 1}] * len(model.layers)


@pytest.mark.parametrize("training,grad_enabled", [(False, False), (False, True), (True, False)])
def test_tbptt_does_not_change_evaluation_or_no_grad(training, grad_enabled):
    inputs, targets = decision_inputs()
    torch.manual_seed(871)
    reference = training_model().train(training)
    candidate = copy.deepcopy(reference)
    candidate.config.enable_tbptt = True
    with torch.set_grad_enabled(grad_enabled):
        left = reference(**inputs, targets=targets, attention_backend="grouped")
        right = candidate(**inputs, targets=targets, attention_backend="grouped")
        assert torch.equal(left.video, right.video) and torch.equal(left.action, right.action)
        assert all(torch.equal(left.loss[key], right.loss[key]) for key in left.loss)


def test_tbptt_rejects_non_decision_training():
    sample = dict(payload_sample("policy", actions=32), resolution="256")
    inputs, targets = CausalCosmos3ConditionModel(device="cpu", dtype=torch.float32).prepare([sample])
    model = Cosmos3Model(tiny_config(enable_tbptt=True)).train()
    with pytest.raises(ValueError, match="TBPTT requires Action decision-window"):
        model(**inputs, targets=targets, attention_backend="grouped")
    model.eval()
    assert torch.isfinite(sum(model(**inputs, targets=targets, attention_backend="grouped").loss.values()))


@pytest.mark.parametrize("enabled,keep", [(True, 16), (False, 16), (True, 65)])
def test_tbptt_rejects_separately_sharded_attention_before_projection(enabled, keep):
    inputs, targets = decision_inputs()
    model = training_model(enabled=enabled, keep=keep).train()
    attention = model.layers[0].self_attn
    original_class = type(attention)
    # fully_shard installs this same mixin/MRO in place. No process group or
    # accelerator is needed to exercise the unsupported wrapping boundary.
    attention.__class__ = type(f"FSDP{original_class.__name__}", (FSDPModule, original_class), {})
    calls = {"attention": 0, "projection": 0}

    def enter_attention(module, args):
        calls["attention"] += 1

    def enter_projection(module, args):
        calls["projection"] += 1
        if enabled and keep < 65:
            pytest.fail("TBPTT reached an independently sharded attention projection")

    handles = [attention.register_forward_pre_hook(enter_attention)]
    handles += [
        child.register_forward_pre_hook(enter_projection)
        for child in attention.modules()
        if isinstance(child, torch.nn.Linear)
    ]
    try:
        if enabled and keep < 65:
            with pytest.raises(ValueError, match="TBPTT requires decoder-layer FSDP wrapping"):
                model(**inputs, targets=targets, attention_backend="grouped")
            assert calls == {"attention": 0, "projection": 0}
        else:
            output = model(**inputs, targets=targets, attention_backend="grouped")
            sum(output.loss.values()).backward()
            assert calls["attention"] == 1 and calls["projection"] > 0
            assert model.layers[0].self_attn.add_k_proj.weight.grad is not None
    finally:
        for handle in handles:
            handle.remove()
        attention.__class__ = original_class


def test_tbptt_config_yaml_and_serialization(tmp_path):
    options = yaml.safe_load("model:\n  model_config:\n    enable_tbptt: true\n    tbptt_keep_history_latents: 16\n")
    base = tiny_config()
    base.save_pretrained(tmp_path / "base")
    config = build_config(str(tmp_path / "base"), **options["model"]["model_config"])
    assert config.enable_tbptt is True and config.tbptt_keep_history_latents == 16
    config.save_pretrained(tmp_path / "trained")
    restored = build_config(str(tmp_path / "trained"))
    assert restored.enable_tbptt is True and restored.tbptt_keep_history_latents == 16
    legacy = config.to_dict()
    legacy.pop("enable_tbptt")
    legacy.pop("tbptt_keep_history_latents")
    restored = Cosmos3Config.from_dict(legacy)
    assert restored.enable_tbptt is False and restored.tbptt_keep_history_latents == 16
    restored, unused = Cosmos3Config.from_dict(config.to_dict(), return_unused_kwargs=True, untouched_option=73)
    assert restored.enable_tbptt is True and unused == {"untouched_option": 73}


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"enable_tbptt": "true"}, "enable_tbptt must be a bool"),
        ({"enable_tbptt": 1}, "enable_tbptt must be a bool"),
        ({"enable_tbptt": None}, "enable_tbptt must be a bool"),
        ({"tbptt_keep_history_latents": 0}, "tbptt_keep_history_latents must be a positive integer"),
        ({"tbptt_keep_history_latents": -1}, "tbptt_keep_history_latents must be a positive integer"),
        ({"tbptt_keep_history_latents": True}, "tbptt_keep_history_latents must be a positive integer"),
        ({"tbptt_keep_history_latents": 1.5}, "tbptt_keep_history_latents must be a positive integer"),
        ({"tbptt_keep_history_latents": "16"}, "tbptt_keep_history_latents must be a positive integer"),
    ],
)
def test_tbptt_invalid_config_is_rejected_before_model_allocation(overrides, error, tmp_path):
    with pytest.raises(ValueError, match=error):
        Cosmos3Config(**overrides)
    config = tiny_config()
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match=error):
        build_config(str(tmp_path), **overrides)
    config.update(overrides)
    with pytest.raises(ValueError, match=error):
        Cosmos3Model(config)
