"""Fixed tensor contracts and causal gradients, independent of a data pipeline."""

import copy
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.models import build_config, build_foundation_model
from veomni.models.checkpoint.convert import get_checkpoint_tensor_converter
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
    CausalLayout,
    generation_attention,
    prepare_attention,
    understanding_attention,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers import Cosmos3Attention, TimeEmbedder
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video, unpatchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def tiny_config(variant="edge", **overrides):
    config = build_config(
        str(
            Path(__file__).parents[3]
            / "configs"
            / "model_configs"
            / "causal_cosmos3"
            / f"causal_cosmos3_{variant}.json"
        )
    )
    config.update(
        dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mrope_section=[2, 1, 1],
            vocab_size=32,
            latent_channels=2,
            action_interface="shared",
            action_dim=4,
            state_dim=4,
        )
    )
    config.update(overrides)
    if variant == "super" and "num_attention_heads" not in overrides:
        config.num_attention_heads = 8
    return config


def action_weight_loading():
    return {
        "skip_patterns": ["audio_*", "action_*"],
        "initialize_missing_patterns": ["action_*", "state_*"],
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires CUDA")),
        pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU")),
    ],
)
def test_time_embedding_precision_and_gradients(device, dtype):
    module = TimeEmbedder(32).to(device=device, dtype=dtype)
    # CUDA/NPU follow the source FP32 MLP; CPU retains its homogeneous dtype path.
    compute_dtype = dtype if device == "cpu" else torch.float32
    reference = torch.nn.Sequential(
        copy.deepcopy(module.linear_1), torch.nn.SiLU(), copy.deepcopy(module.linear_2)
    ).to(dtype=compute_dtype)
    times = torch.tensor([0.0, 0.383567, 0.824859], device=device, requires_grad=True)
    reference_times = times.detach().clone().requires_grad_()
    freqs = torch.exp(-math.log(10000) * torch.arange(128, device=device, dtype=torch.float32) / 128)
    phases = reference_times[:, None] * freqs[None, :]
    expected = reference(torch.cat((phases.cos(), phases.sin()), dim=-1).to(compute_dtype)).to(dtype)
    output = module(times)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if device != "cpu":
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            under_autocast = module(times)
        torch.testing.assert_close(under_autocast, output, rtol=0, atol=0)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(times.grad, reference_times.grad, rtol=0, atol=0)
    for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad.to(dtype), rtol=0, atol=0)


@pytest.fixture
def weight_loading_logs(caplog, monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer as modeling

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(modeling.logger, "handlers", [caplog.handler])
    with caplog.at_level("WARNING", logger=modeling.logger.name):
        yield caplog


def fixed_inputs(task="policy", device="cpu", dtype=torch.float32):
    # One text prefix, three clean frames, three current frames, two actions,
    # and one immutable state per prediction block. Duplicate for packing tests.
    video_roles = (2, 2, 2) if task == "id" else (2, 1, 1)
    action_roles = (2, 2) if task == "fd" else (1, 1)
    roles = (-1, -1, 0, 0, 0, *video_roles, *action_roles, 3, 3)
    modalities = (0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3)
    blocks = (-1, -1, -1, 0, 1, -1, 0, 1, 0, 1, 0, 1)
    layout = CausalLayout((0,) * 12 + (1,) * 12, blocks * 2, roles * 2, modalities * 2, (1, 2))
    generator = torch.Generator().manual_seed(20261019)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=dtype)

    return dict(
        input_ids=torch.tensor([1, 2, 3, 4], device=device),
        video_tokens=rand(12, 8),
        video_timesteps=torch.tensor([0, 0, 0, 0, 400, 700] * 2, device=device),
        action_tokens=rand(4, 4),
        action_timesteps=torch.tensor([400, 700] * 2, device=device),
        action_mask=torch.tensor([[True, True, False, False]] * 4, device=device),
        state_tokens=rand(4, 4),
        state_mask=torch.tensor([[True, False, True, False]] * 4, device=device),
        position_ids=torch.tensor([blocks * 2, tuple(range(12)) * 2, (0,) * 24], device=device),
        layout=layout,
    )


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
def test_configuration_registry_and_roundtrip(variant, tmp_path):
    config = tiny_config(variant)
    model = build_foundation_model(config, init_device="cpu", torch_dtype="float32")
    config.save_pretrained(tmp_path)
    restored = build_config(str(tmp_path))
    assert restored.to_dict() == config.to_dict()
    assert isinstance(model, Cosmos3Model)
    assert set(model._no_split_modules) == {"Cosmos3DecoderLayer"}
    assert (model.layers[0].mlp.gate_proj is None) == (variant == "edge")
    # Super-style attention width is independent of hidden_size.
    config.num_attention_heads = 8
    wider = Cosmos3Model(config)
    assert wider.layers[0].self_attn.to_q.weight.shape == (64, 32)


def test_patch_order_and_padding():
    x = torch.arange(2 * 3 * 3 * 5).reshape(2, 3, 3, 5).float().requires_grad_()
    tokens = patchify_video(x, 2)
    expected = torch.stack([x[:, 0, h, w] for h, w in ((0, 0), (0, 1), (1, 0), (1, 1))]).flatten()
    torch.testing.assert_close(tokens[0], expected)
    restored = unpatchify_video(tokens, x.shape, 2)
    torch.testing.assert_close(restored, x)
    restored.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_grouped_attention_outputs_and_shared_kv_gradients(task):
    layout = fixed_inputs(task)["layout"]
    dense = prepare_attention(layout, "cpu", "dense")
    grouped = prepare_attention(layout, "cpu", "grouped", max_kv_tokens=13)
    torch.manual_seed(42)
    inputs = [
        torch.randn(n, heads, 8, dtype=torch.float64, requires_grad=True) for n, heads in ((20, 4), (24, 2), (24, 2))
    ]
    reference = generation_attention(*inputs, dense)
    actual = generation_attention(*inputs, grouped)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(reference, inputs, grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_model_forward_backward_and_checkpointing(variant, task):
    torch.manual_seed(7)
    dense = Cosmos3Model(tiny_config(variant))
    grouped = copy.deepcopy(dense)
    grouped.gradient_checkpointing_enable()
    inputs = fixed_inputs(task)
    inputs["video_tokens"].requires_grad_()
    reference = dense(**inputs)
    actual = grouped(**inputs, attention_backend="grouped")
    torch.testing.assert_close(actual.video, reference.video, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual.action, reference.action, atol=3e-6, rtol=3e-5)
    # Only current targets contribute loss; clean frames must still receive gradients.
    video_roles = torch.tensor([r for r, m in zip(inputs["layout"].roles, inputs["layout"].modalities) if m == 1])
    action_roles = torch.tensor([r for r, m in zip(inputs["layout"].roles, inputs["layout"].modalities) if m == 2])

    def loss(output):
        v = output.video[video_roles == 1].float().square().sum()
        a = output.action[action_roles == 1].float().square().sum()
        return v + a

    loss(reference).backward()
    loss(actual).backward()
    assert inputs["video_tokens"].grad[:2].abs().sum() > 0
    for (name, expected), (_, current) in zip(dense.named_parameters(), grouped.named_parameters(), strict=True):
        if expected.grad is not None:
            torch.testing.assert_close(current.grad, expected.grad, atol=3e-5, rtol=3e-4, msg=name)
    assert torch.count_nonzero(grouped.action_proj_in.weight.grad[:, 2:]) == 0
    assert torch.count_nonzero(grouped.state_proj_in.weight.grad[:, [1, 3]]) == 0


@pytest.mark.parametrize("checkpointing", [False, True])
def test_generation_only_training_updates_gen_and_keeps_und_frozen(checkpointing):
    model = Cosmos3Model(tiny_config(train_generation_only=True))
    model.freeze_model()
    if checkpointing:
        model.gradient_checkpointing_enable()
    und_requires_grad = []
    handles = [
        layer.mlp.register_forward_hook(lambda _module, _args, output: und_requires_grad.append(output.requires_grad))
        for layer in model.layers
    ]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    outputs = model(**fixed_inputs())
    (outputs.video.square().mean() + outputs.action.square().mean()).backward()
    optimizer.step()
    for handle in handles:
        handle.remove()
    assert len(und_requires_grad) >= model.config.num_hidden_layers
    assert not any(und_requires_grad)
    for name in (
        "embed_tokens.weight",
        "lm_head.weight",
        "layers.0.self_attn.to_q.weight",
        "layers.0.mlp.up_proj.weight",
    ):
        parameter = model.get_parameter(name)
        assert not parameter.requires_grad
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    for name in (
        "action_proj_out.weight",
        "proj_out.weight",
        "layers.0.self_attn.add_q_proj.weight",
        "layers.0.self_attn.k_norm_und_for_gen.weight",
    ):
        parameter = model.get_parameter(name)
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert not torch.equal(parameter, before[name])


@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
@pytest.mark.parametrize("action_interface", ["shared", "domain"])
def test_split_stream_outputs_preserve_packed_order(variant, action_interface):
    model = Cosmos3Model(tiny_config(variant, action_interface=action_interface)).eval()
    inputs = fixed_inputs()
    if action_interface == "domain":
        inputs["action_domains"] = torch.tensor([0, 1, 1, 2])
    actual = model(**inputs, output_text_logits=True, output_hidden_states=True)
    hidden = actual.last_hidden_state
    plan = prepare_attention(inputs["layout"], "cpu")
    ti, vi, ai, _ = plan.modality_indexes
    assert hidden.shape == (len(inputs["layout"].sample_ids), model.config.hidden_size)
    torch.testing.assert_close(actual.video, model.proj_out(hidden[vi]), rtol=0, atol=0)
    torch.testing.assert_close(actual.text_logits, model.lm_head(hidden[ti]), rtol=0, atol=0)
    expected_action = (
        model.action_proj_out(hidden[ai], inputs["action_domains"])
        if action_interface == "domain"
        else model.action_proj_out(hidden[ai])
    ).masked_fill(~inputs["action_mask"], 0)
    torch.testing.assert_close(actual.action, expected_action, rtol=0, atol=0)
    predictions = model(**inputs)
    assert predictions.last_hidden_state is None and predictions.text_logits is None
    torch.testing.assert_close(predictions.video, actual.video, rtol=0, atol=0)
    torch.testing.assert_close(predictions.action, actual.action, rtol=0, atol=0)


@pytest.mark.parametrize("text_only", [False, True])
def test_split_streams_allow_an_empty_branch(text_only):
    model = Cosmos3Model(tiny_config())
    inputs = fixed_inputs()
    layout = inputs["layout"]
    keep = [i for i, modality in enumerate(layout.modalities) if (modality == 0) == text_only]
    inputs["layout"] = CausalLayout(
        tuple(layout.sample_ids[i] for i in keep),
        tuple(layout.block_ids[i] for i in keep),
        tuple(layout.roles[i] for i in keep),
        tuple(layout.modalities[i] for i in keep),
        layout.history_blocks,
    )
    inputs["position_ids"] = inputs["position_ids"][:, keep]
    if text_only:
        inputs["video_tokens"] = inputs["video_tokens"][:0]
        inputs["video_timesteps"] = inputs["video_timesteps"][:0]
        for name in ("action_tokens", "action_timesteps", "action_mask", "state_tokens", "state_mask"):
            inputs.pop(name)
    else:
        inputs["input_ids"] = inputs["input_ids"][:0]
    output = model(**inputs, output_text_logits=True, output_hidden_states=True)
    assert output.video.shape == inputs["video_tokens"].shape
    assert output.text_logits.shape == (len(inputs["input_ids"]), model.config.vocab_size)
    assert output.last_hidden_state.shape == (len(keep), model.config.hidden_size)
    loss = output.text_logits.float().square().sum() + output.video.float().square().sum()
    if output.action is not None:
        loss = loss + output.action.float().square().sum()
    loss.backward()


def test_causal_isolation_and_invalid_dimensions():
    torch.manual_seed(11)
    model = Cosmos3Model(tiny_config()).eval()
    inputs = fixed_inputs()
    baseline = model(**inputs, output_hidden_states=True)
    changed = copy.deepcopy(inputs)
    # Change future targets, current clean copies, and a different packed sample.
    changed["video_tokens"][[1, 2, 5, 6, 7, 8, 9, 10, 11]] += 100
    changed["action_tokens"][[1, 2, 3]] += 100
    result = model(**changed, output_hidden_states=True)
    torch.testing.assert_close(result.video[4], baseline.video[4], atol=0, rtol=0)
    torch.testing.assert_close(result.action[0], baseline.action[0], atol=0, rtol=0)
    # State and condition cannot absorb targets across multiple transformer layers.
    changed = copy.deepcopy(inputs)
    changed["action_tokens"] += 100
    changed["video_tokens"][[4, 5]] += 100
    result = model(**changed, output_hidden_states=True)
    torch.testing.assert_close(
        result.last_hidden_state[[2, 3, 4, 5, 10, 11]],
        baseline.last_hidden_state[[2, 3, 4, 5, 10, 11]],
        atol=0,
        rtol=0,
    )
    changed = copy.deepcopy(inputs)
    changed["action_tokens"][~inputs["action_mask"]] = float("nan")
    changed["state_tokens"][~inputs["state_mask"]] = float("nan")
    result = model(**changed)
    torch.testing.assert_close(result.video, baseline.video, atol=0, rtol=0)
    torch.testing.assert_close(result.action, baseline.action, atol=0, rtol=0)
    assert torch.count_nonzero(result.action[~inputs["action_mask"]]) == 0


def test_native_weight_loading_and_configured_partial_load(tmp_path):
    original = Cosmos3Model(tiny_config(action_interface="domain", state_dim=0, num_embodiment_domains=2))
    save_file(original.state_dict(), tmp_path / "model.safetensors")
    restored = Cosmos3Model(original.config)
    load_model_weights(restored, str(tmp_path), init_device="cpu")
    assert restored.weight_loading_report["initialized"] == []
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, atol=0, rtol=0)
    shared = Cosmos3Model(tiny_config(weight_loading=action_weight_loading()))
    load_model_weights(shared, str(tmp_path), init_device="cpu")
    assert len(shared.weight_loading_report["initialized"]) == 8
    assert len(shared.weight_loading_report["skipped"]) == 5
    torch.testing.assert_close(shared.layers[0].self_attn.to_k.weight, original.layers[0].self_attn.to_k.weight)
    shared.save_pretrained(tmp_path)
    reloaded_config = build_config(str(tmp_path))
    assert reloaded_config.action_interface == "shared"
    assert reloaded_config.weight_loading == {}
    assert "weight_loading" not in reloaded_config.to_dict()
    load_model_weights(shared, str(tmp_path), init_device="cpu")
    assert shared.weight_loading_report["mode"] == "complete"
    assert shared.weight_loading_report["initialized"] == []
    broken = dict(shared.state_dict())
    del broken["layers.0.self_attn.to_k.weight"]
    save_file(broken, tmp_path / "model.safetensors")
    shared.config.weight_loading = {"initialize_missing_patterns": []}
    with pytest.raises(ValueError, match="Missing Cosmos3 weights"):
        load_model_weights(shared, str(tmp_path), init_device="cpu")


@pytest.mark.parametrize(
    "policy",
    [
        True,
        [],
        {"skip": ["action_*"]},
        {"skip_patterns": "action_*"},
        {"skip_patterns": [7]},
        {"initialize_missing_patterns": [""]},
    ],
)
def test_weight_loading_rejects_invalid_rules(policy):
    model = Cosmos3Model(tiny_config(weight_loading=policy))
    with pytest.raises(ValueError, match="weight_loading"):
        get_checkpoint_tensor_converter(model)


def test_weight_loading_rules_are_not_exported_before_or_after_loading(tmp_path):
    config = tiny_config(weight_loading=action_weight_loading())
    config.save_pretrained(tmp_path)
    assert config.weight_loading == action_weight_loading()
    assert "weight_loading" not in config.to_dict()
    assert build_config(str(tmp_path)).weight_loading == {}
    # 标准 model_config 覆盖仍能将本次规则传入模型配置。
    restored = build_config(str(tmp_path), weight_loading=action_weight_loading())
    assert restored.weight_loading == action_weight_loading()


@pytest.mark.parametrize("strict", [False, True])
def test_skipping_parameters_initializes_missing_unless_explicitly_restricted(tmp_path, weight_loading_logs, strict):
    source = Cosmos3Model(tiny_config(action_interface="domain", state_dim=0))
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    policy = {"skip_patterns": ["action_*"]}
    if strict:
        policy["initialize_missing_patterns"] = []
    model = Cosmos3Model(tiny_config(weight_loading=policy))
    if strict:
        with pytest.raises(ValueError, match="Missing Cosmos3 weights") as error:
            load_model_weights(model, str(tmp_path), init_device="cpu")
        assert "state_proj_in.weight" in str(error.value)
        assert not weight_loading_logs.records
    else:
        load_model_weights(model, str(tmp_path), init_device="cpu")
        assert len(model.weight_loading_report["initialized"]) == 8
        assert "state_proj_in.weight" in weight_loading_logs.text
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_default_missing_initialization_warns_and_preserves_loaded_parameters(tmp_path, weight_loading_logs):
    source = Cosmos3Model(tiny_config())
    state = source.state_dict()
    save_file(state, tmp_path / "model.safetensors")
    model = Cosmos3Model(tiny_config())
    load_model_weights(model, str(tmp_path), init_device="cpu")
    assert model.weight_loading_report["mode"] == "complete"
    assert not weight_loading_logs.records

    missing = ["layers.0.self_attn.to_k.weight", "norm.weight", "state_proj_in.weight"]
    for name in missing:
        del state[name]
    save_file(state, tmp_path / "model.safetensors")
    load_model_weights(model, str(tmp_path), init_device="cpu")
    assert model.weight_loading_report["mode"] == "partial"
    assert model.weight_loading_report["initialized"] == missing
    assert len(weight_loading_logs.records) == 1
    message = weight_loading_logs.records[0].getMessage()
    assert "Initialized 3 missing Cosmos3 parameter(s)" in message
    assert all(name in message for name in missing)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    torch.testing.assert_close(model.norm.weight, torch.ones_like(model.norm.weight), atol=0, rtol=0)
    for name, tensor in state.items():
        torch.testing.assert_close(model.get_parameter(name), tensor, atol=0, rtol=0)


def test_partial_loading_initializes_only_the_requested_missing_parameter(tmp_path):
    source = Cosmos3Model(tiny_config())
    with torch.no_grad():
        source.norm_moe_gen.weight.fill_(2)
    state = source.state_dict()
    del state["norm.weight"]
    save_file(state, tmp_path / "model.safetensors")
    model = Cosmos3Model(tiny_config(weight_loading={"initialize_missing_patterns": ["norm.weight"]}))
    load_model_weights(model, str(tmp_path), init_device="cpu")
    assert model.weight_loading_report["initialized"] == ["norm.weight"]
    torch.testing.assert_close(model.norm.weight, torch.ones_like(model.norm.weight), atol=0, rtol=0)
    for name, tensor in state.items():
        torch.testing.assert_close(model.get_parameter(name), tensor, atol=0, rtol=0)


@pytest.mark.parametrize("interface", ["domain", "shared"])
def test_missing_projection_weights_use_source_initialization(tmp_path, interface):
    torch.manual_seed(71)
    model = Cosmos3Model(
        tiny_config(
            hidden_size=256,
            num_hidden_layers=1,
            action_interface=interface,
            action_dim=55,
            state_dim=55,
            num_embodiment_domains=2,
        )
    )
    projections = {
        "proj_in": model.proj_in,
        "proj_out": model.proj_out,
        "time_embedder.linear_1": model.time_embedder.linear_1,
        "time_embedder.linear_2": model.time_embedder.linear_2,
        "action_proj_in": model.action_proj_in,
        "action_proj_out": model.action_proj_out,
        "state_proj_in": model.state_proj_in,
    }
    missing_weights = {}
    missing_biases = []
    for name, projection in projections.items():
        if isinstance(projection, torch.nn.Linear):
            missing_weights[f"{name}.weight"] = projection.in_features
            missing_biases.append(f"{name}.bias")
        else:
            missing_weights[f"{name}.fc.weight"] = projection.input_size
            missing_biases.append(f"{name}.bias.weight")
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for name in (*missing_weights, *missing_biases):
        del state[name]
    # 已加载的值刻意放到初始化范围外，确保补初始化不会覆盖已有参数。
    state["action_modality_embed"].fill_(4)
    save_file(state, tmp_path / "model.safetensors")
    load_model_weights(model, str(tmp_path), init_device="cpu")
    assert model.weight_loading_report["initialized"] == sorted([*missing_weights, *missing_biases])
    for name, input_width in missing_weights.items():
        values = model.get_parameter(name).detach().float().flatten() * input_width**0.5
        assert values.abs().max() <= 3, name
        assert values.mean().abs() < 0.08, name
        assert 0.90 < values.std() < 1.08, name
    for name in missing_biases:
        assert torch.count_nonzero(model.get_parameter(name)) == 0, name
    for name, tensor in state.items():
        torch.testing.assert_close(model.get_parameter(name), tensor, atol=0, rtol=0, msg=name)


def test_skipped_checkpoint_parameters_are_audited_and_duplicates_rejected():
    model = Cosmos3Model(tiny_config(weight_loading={"skip_patterns": ["audio_*"]}))
    converter = get_checkpoint_tensor_converter(model)
    assert converter.should_skip_without_loading("audio_proj_in.weight")
    assert not converter.should_skip_without_loading("layers.0.self_attn.to_k.weight")
    with pytest.raises(ValueError, match="not configured to skip"):
        converter.record_skip_without_loading("norm.weight")
    converter.record_skip_without_loading("audio_proj_in.weight")
    assert converter.skipped == {"audio_proj_in.weight"}
    with pytest.raises(ValueError, match="Duplicate Cosmos3 weight"):
        converter.record_skip_without_loading("audio_proj_in.weight")


@pytest.mark.parametrize("mode", ["domain", "shared", "warm_start"])
def test_action_embedding_truncation_and_state_normal_initialization(mode, tmp_path):
    torch.manual_seed(42)
    config = tiny_config(hidden_size=4096, num_hidden_layers=0, num_embodiment_domains=2)
    if mode == "domain":
        config.action_interface = "domain"
    model = Cosmos3Model(config)
    if mode == "warm_start":
        source_config = copy.deepcopy(config)
        source_config.action_interface, source_config.state_dim = "domain", 0
        source = Cosmos3Model(source_config)
        save_file(source.state_dict(), tmp_path / "model.safetensors")
        model.config.weight_loading = action_weight_loading()
        load_model_weights(model, str(tmp_path), init_device="cpu")
    # Large fixed-seed samples catch an ordinary normal in place of truncation,
    # or accidentally applying action's truncation policy to state as well.
    action = model.action_modality_embed.detach() * config.hidden_size**0.5
    state = model.state_modality_embed.detach() * config.hidden_size**0.5
    assert action.abs().max() <= 3
    assert state.abs().max() > 3
    for embedding in (action, state):
        assert embedding.mean().abs() < 0.06
        assert 0.94 < embedding.std() < 1.06
    # Complete reload must preserve trained values, even outside init bounds.
    with torch.no_grad():
        model.action_modality_embed.fill_(1)
    model.save_pretrained(tmp_path)
    restored = Cosmos3Model(config)
    load_model_weights(restored, str(tmp_path), init_device="cpu")
    torch.testing.assert_close(restored.action_modality_embed, model.action_modality_embed, atol=0, rtol=0)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend UND TND validation")
@pytest.mark.parametrize("samples", [(0, 1, 0, 1, 1), (0,) * 7 + (1,) * 2053], ids=["interleaved", "long"])
def test_npu_und_packed_causal_forward_backward(samples, monkeypatch):
    import torch_npu

    torch.manual_seed(29)
    n = len(samples)
    layout = CausalLayout(samples, (-1,) * n, (-1,) * n, (0,) * n, (1, 1))
    inputs = [(torch.randn(n, heads, 128) * 0.2).to("npu", torch.bfloat16).requires_grad_() for heads in (4, 2, 2)]
    reference_inputs = [x.detach().float().cpu().requires_grad_() for x in inputs]
    # Independent dense mask: same sample and no later text position.
    ids, positions = torch.tensor(samples), torch.arange(n)
    visible = (ids[:, None] == ids[None, :]) & (positions[:, None] >= positions[None, :])
    q, k, v = reference_inputs
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(2, dim=1).transpose(0, 1),
        v.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=visible,
    ).transpose(0, 1)
    calls = []
    original = torch_npu.npu_fusion_attention

    def record(query, key, value, **kwargs):
        calls.append((query.shape[1], key.shape[1], kwargs["input_layout"], kwargs["sparse_mode"]))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", record)
    actual = understanding_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd"))
    assert calls == [(4, 2, "TND", 2)]
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.float().cpu())
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float().cpu(), expected_grad, atol=0.02, rtol=0.04)
    # A query cannot read a future text token or another packed sample.
    key_grad, value_grad = torch.autograd.grad(
        understanding_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd"))[0].float().sum(), inputs[1:]
    )
    assert torch.count_nonzero(key_grad[1:]) == 0
    assert torch.count_nonzero(value_grad[1:]) == 0


def test_und_empty_text():
    layout = CausalLayout((0,), (-1,), (1,), (1,), (1,))
    query = torch.empty(0, 4, 8, requires_grad=True)
    key, value = (torch.empty(0, 2, 8) for _ in range(2))
    output = understanding_attention(query, key, value, prepare_attention(layout, "cpu"))
    assert output.shape == query.shape
    output.sum().backward()
    assert query.grad is not None


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend packed model checkpointing")
@pytest.mark.parametrize("variant", ["edge", "nano", "super"])
def test_npu_packed_model_checkpointing(variant):
    torch.manual_seed(31)
    model = Cosmos3Model(tiny_config(variant, head_dim=128, mrope_section=[24, 20, 20])).to("npu", torch.bfloat16)
    checkpointed = copy.deepcopy(model)
    checkpointed.gradient_checkpointing_enable()
    inputs = fixed_inputs(device="npu", dtype=torch.bfloat16)
    expected = model(**inputs, attention_backend="npu_tnd")
    actual = checkpointed(**inputs, attention_backend="npu_tnd")
    torch.testing.assert_close(actual.video, expected.video, atol=0, rtol=0)
    torch.testing.assert_close(actual.action, expected.action, atol=0, rtol=0)
    for output in (expected, actual):
        (output.video.float().square().sum() + output.action.float().square().sum()).backward()
    assert model.layers[0].self_attn.to_q.weight.grad.float().norm() > 0
    for (name, reference), (_, parameter) in zip(
        model.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0, msg=name)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend TND validation")
def test_npu_grouped_tnd_forward_backward():
    torch.manual_seed(19)
    layout = fixed_inputs()["layout"]
    inputs = [
        (torch.randn(n, heads, 128) * 0.2).to("npu", torch.bfloat16).requires_grad_()
        for n, heads in ((20, 4), (24, 2), (24, 2))
    ]
    oracle_inputs = [x.detach().float().cpu().requires_grad_() for x in inputs]
    expected = generation_attention(*oracle_inputs, prepare_attention(layout, "cpu", "dense"))
    actual = generation_attention(*inputs, prepare_attention(layout, "npu", "npu_tnd", max_kv_tokens=13))
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.002, rtol=0.02)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, oracle_inputs, grad.float().cpu())
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float().cpu(), expected_grad, atol=0.02, rtol=0.04)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend fused rotary validation")
def test_npu_edge_rotary_matches_source_fused_formula():
    import torch_npu

    ops = eager_ops_config()
    ops.rotary_pos_emb_implementation = "torch_npu"
    with ops_config_scope(ops):
        attention = Cosmos3Attention(tiny_config(head_dim=128, mrope_section=[24, 20, 20]))
    torch.manual_seed(8)
    q = torch.randn(3, 4, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(3, 2, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    angles = torch.randn(3, 64, device="npu").repeat(1, 2)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    actual = attention.apply_rope(q, k, cos, sin)
    expected = tuple(
        torch_npu.npu_rotary_mul(x.unsqueeze(0), cos[None, :, None], sin[None, :, None], "half").squeeze(0)
        for x in (q, k)
    )
    gradients = tuple(torch.randn_like(x) for x in actual)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual, (q, k), gradients)
    expected_grad = torch.autograd.grad(expected, (q, k), gradients)
    for output, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(output.float(), reference.float(), atol=0.03125, rtol=0.02)
