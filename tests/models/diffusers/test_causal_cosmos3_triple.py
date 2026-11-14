"""Three-expert contracts: detached conditioning, cache reuse and causal inputs."""

import copy
import inspect
from dataclasses import fields, is_dataclass, replace
from datetime import timedelta
from itertools import zip_longest
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import fixed_inputs
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample, video_payload_sample
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
    CausalCosmos3ConditionModel,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import (
    CausalLayout,
    generation_attention,
    prepare_attention,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.triple_attention import prepare_branch_attention
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import Cosmos3Config
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def triple_config(**overrides):
    options = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
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
        action_num_hidden_layers=2,
        action_layer_map=[2, 4],
    )
    options.update(overrides)
    return Cosmos3Config(**options)


def triple_inputs(task="inverse_dynamics"):
    inputs = fixed_inputs("fd" if task == "forward_dynamics" else "id")
    inputs["task_modes"] = (task, task)
    inputs["video_timesteps"] = inputs["video_timesteps"].float()
    inputs["action_timesteps"] = inputs["action_timesteps"].float()
    if task == "forward_dynamics":
        inputs["action_timesteps"] = torch.zeros_like(inputs["action_timesteps"])
    else:
        # The first three rows of each sample are the teacher-forcing clean
        # stream; only the fixed current future carries nonzero noise time.
        inputs["video_timesteps"] = torch.tensor([0, 0, 0, 0, 900, 900] * 2, dtype=torch.float32)
    return inputs


def video_inputs():
    inputs = fixed_inputs()
    layout = inputs["layout"]
    keep = [i for i, modality in enumerate(layout.modalities) if modality < 2]
    inputs["layout"] = replace(
        layout,
        **{
            field: tuple(getattr(layout, field)[i] for i in keep)
            for field in ("sample_ids", "block_ids", "roles", "modalities", "frame_ids")
        },
    )
    inputs["position_ids"] = inputs["position_ids"][:, keep]
    for key in ("action_tokens", "action_timesteps", "action_mask", "state_tokens", "state_mask"):
        inputs.pop(key)
    inputs["task_modes"] = ("video", "video")
    return inputs


def supervised_triple_inputs(task="inverse_dynamics", *, conditioning="both"):
    """Use production noising/packing rather than manufacturing video labels."""
    sample = {
        **payload_sample("fd" if task == "forward_dynamics" else "id"),
        "mode": task,
        "resolution": "256",
    }
    if task == "forward_dynamics":
        sample["fdm_conditioning"] = conditioning
        if conditioning == "action":
            # Dataset tokenizes the empty prompt before condition preparation.
            sample["text_token_ids"] = torch.tensor([0])
            sample["token_length"] -= 1
    generator = torch.Generator().manual_seed(101)
    sample["video_latents"] = torch.randn(sample["video_latents"].shape, generator=generator)
    sample["action"] = torch.randn(sample["action"].shape, generator=generator)
    sample["states"] = torch.randn(sample["states"].shape, generator=generator)
    condition = CausalCosmos3ConditionModel(device="cpu", dtype="float32", expert_layout="triple")
    return condition.prepare(
        [sample],
        sigmas=[torch.tensor([0.35])],
        action_sigmas=[torch.tensor([0.65])],
        epsilon_video=[torch.randn(sample["video_latents"].shape, generator=generator)],
        epsilon_action=[torch.randn(sample["action"].shape, generator=generator)],
    )


def assert_tree_close(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for field in fields(actual):
            assert_tree_close(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            assert_tree_close(actual[name], expected[name])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_close(left, right)
    else:
        assert actual == expected


def assert_gradient_partition(model, *, action):
    active = []
    for name, parameter in model.named_parameters():
        has_gradient = parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        if name.startswith("action_expert.") == action:
            active.append(has_gradient)
        else:
            assert not has_gradient, name
    assert any(active)


def test_triple_dimensions_layer_mapping_and_frozen_reasoner(tmp_path):
    config = triple_config()
    model = Cosmos3Model(config)
    model.freeze_model()
    assert config.action_layer_map == [2, 4]
    assert config.action_hidden_size == 16
    assert config.action_num_hidden_layers == 2
    assert model.proj_in.out_features == 32
    assert hasattr(model, "action_expert")
    reasoner = [model.embed_tokens, model.norm, model.lm_head]
    for layer in model.layers:
        reasoner.extend([layer.mlp, layer.input_layernorm, layer.post_attention_layernorm])
        reasoner.extend(
            getattr(layer.self_attn, name) for name in ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k")
        )
    assert all(not parameter.requires_grad for module in reasoner for parameter in module.parameters())
    action_blocks = next(
        module for name, module in model.action_expert.named_modules() if name in ("blocks", "layers")
    )
    assert len(action_blocks) == 2
    assert type(action_blocks[0]).__name__ in model._no_split_modules
    config.save_pretrained(tmp_path)
    restored = Cosmos3Config.from_pretrained(tmp_path)
    assert restored.expert_layout == "triple"
    assert restored.action_layer_map == config.action_layer_map


@pytest.mark.parametrize("mapping", [[2, 2], [2, 5], [4, 2], [2]])
def test_invalid_action_layer_mapping_is_rejected(mapping):
    with pytest.raises(ValueError, match="layer|map"):
        triple_config(action_layer_map=mapping)


def test_edge_defaults_use_half_width_and_even_video_layers():
    config = Cosmos3Config(expert_layout="triple", action_interface="shared")
    assert config.action_hidden_size == config.hidden_size // 2
    assert config.action_intermediate_size == config.intermediate_size // 2
    assert config.action_num_hidden_layers == config.num_hidden_layers // 2
    assert config.action_layer_map == list(range(2, config.num_hidden_layers + 1, 2))


@pytest.mark.parametrize("dimension", ["action_hidden_size", "action_intermediate_size", "action_num_hidden_layers"])
def test_invalid_action_dimensions_raise_configuration_error(dimension):
    with pytest.raises(ValueError, match="Action|action"):
        triple_config(**{dimension: 0, "action_layer_map": None})


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_idm_gradient_isolation(backend, checkpointing):
    torch.manual_seed(17)
    model = Cosmos3Model(triple_config()).train()
    model.freeze_model()
    if checkpointing:
        model.gradient_checkpointing_enable()
    output = model(**triple_inputs(), attention_backend=backend)
    output.action.float().square().mean().backward()
    assert_gradient_partition(model, action=True)


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_fdm_video_loss_trains_video_and_clean_action_encoder(backend, checkpointing):
    torch.manual_seed(19)
    model = Cosmos3Model(triple_config()).train()
    model.freeze_model()
    if checkpointing:
        model.gradient_checkpointing_enable()
    output = model(**triple_inputs("forward_dynamics"), attention_backend=backend)
    output.video.float().square().mean().backward()
    assert model.proj_in.weight.grad is not None and model.proj_in.weight.grad.abs().sum() > 0
    for projection in (model.action_expert.proj_in, model.action_expert.state_proj_in):
        assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.action_expert.proj_out.parameters())
    assert all(parameter.grad is None for parameter in model.action_expert.norm.parameters())
    assert all(not parameter.requires_grad for parameter in model.embed_tokens.parameters())


@pytest.mark.parametrize("conditioning", ["text", "action", "both"])
def test_fdm_optimizer_updates_only_the_effective_condition_encoder(conditioning):
    from veomni.trainer.causal_cosmos3 import _TripleExpertOptimizerGuard

    torch.manual_seed(149)
    model = Cosmos3Model(triple_config()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.1)
    runtime = SimpleNamespace(model=model, optimizer=optimizer, model_config=model.config)
    guard = _TripleExpertOptimizerGuard(runtime)
    # Populate nonzero momentum before the FDM step, including the action
    # decoder. A zero FDM action loss must not move its parameters or state.
    idm_inputs, idm_targets = supervised_triple_inputs()
    guard.record(idm_targets, task_modes=idm_inputs["task_modes"], video_conditioning=idm_inputs["video_conditioning"])
    initial = model(**idm_inputs, targets=idm_targets, attention_backend="grouped")
    sum(initial.loss.values()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = _expert_optimizer_snapshot(model, optimizer)

    inputs, targets = supervised_triple_inputs("forward_dynamics", conditioning=conditioning)
    guard.reset()
    guard.record(targets, task_modes=inputs["task_modes"], video_conditioning=inputs["video_conditioning"])
    hook = model.action_expert.proj_out.register_forward_pre_hook(
        lambda *_: pytest.fail("FDM must not execute the action denoising decoder")
    )
    try:
        output = model(**inputs, targets=targets, attention_backend="grouped")
    finally:
        hook.remove()
    torch.testing.assert_close(output.action, inputs["action_tokens"], atol=0, rtol=0)
    assert output.loss["action_loss"].item() == 0
    sum(output.loss.values()).backward()
    optimizer.step()
    after = _expert_optimizer_snapshot(model, optimizer)
    denoise_only = {id(parameter) for parameter in guard.parameters[1]}
    condition_shared = {id(parameter) for parameter in guard.parameters[2]}
    updated = [False, False]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        old_parameter, old_state = before[name]
        new_parameter, new_state = after[name]
        if id(parameter) in denoise_only or (conditioning == "text" and id(parameter) in condition_shared):
            assert parameter.grad is None, name
            torch.testing.assert_close(new_parameter, old_parameter, rtol=0, atol=0, msg=name)
            assert_tree_close(new_state, old_state)
        else:
            updated[int(id(parameter) in condition_shared)] |= not torch.equal(new_parameter, old_parameter)
    assert updated[0]
    assert updated[1] == (conditioning != "text")


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("objective", ["video", "action"])
def test_idm_production_losses_train_only_their_own_expert(backend, checkpointing, objective):
    torch.manual_seed(103)
    model = Cosmos3Model(triple_config()).train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    inputs, targets = supervised_triple_inputs()
    output = model(**inputs, targets=targets, attention_backend=backend)
    assert output.loss[objective + "_loss"].item() > 0
    output.loss[objective + "_loss"].backward()
    assert_gradient_partition(model, action=objective == "action")


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("conditioning", ["text", "action", "both"])
def test_fdm_text_only_excludes_every_action_and_state_path(backend, conditioning):
    torch.manual_seed(107)
    model = Cosmos3Model(triple_config()).eval()
    inputs, _ = supervised_triple_inputs("forward_dynamics", conditioning=conditioning)
    inputs.pop("idm_video_inputs", None)
    perturbed = {
        **inputs,
        "action_tokens": inputs["action_tokens"] + 20,
        "state_tokens": inputs["state_tokens"] - 20,
    }
    with torch.no_grad():
        original = model(**inputs, expert_operation="video", attention_backend=backend).video
        changed = model(**perturbed, expert_operation="video", attention_backend=backend).video
    if conditioning == "text":
        torch.testing.assert_close(original, changed, atol=0, rtol=0)
    else:
        assert not torch.equal(original, changed)


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("conditioning", ["text", "action", "both"])
def test_fdm_action_only_excludes_every_text_information_path(backend, conditioning):
    torch.manual_seed(108)
    model = Cosmos3Model(triple_config()).eval()
    inputs, _ = supervised_triple_inputs("forward_dynamics", conditioning=conditioning)
    inputs.pop("idm_video_inputs", None)
    # This deliberately bypasses Dataset's empty prompt: the attention contract
    # must block text in the video and detached action/state features alike.
    changed_text = {**inputs, "input_ids": (inputs["input_ids"] + 13) % model.config.vocab_size}
    with torch.no_grad():
        original = model(**inputs, expert_operation="video", attention_backend=backend).video
        changed = model(**changed_text, expert_operation="video", attention_backend=backend).video
    if conditioning == "action":
        torch.testing.assert_close(original, changed, atol=0, rtol=0)
    else:
        assert not torch.equal(original, changed)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_idm_cache_source_cannot_change_supervised_video_fm_or_its_gradients(checkpointing):
    torch.manual_seed(109)
    reference = Cosmos3Model(triple_config()).train()
    if checkpointing:
        reference.gradient_checkpointing_enable()
    predicted_cache_model = copy.deepcopy(reference)
    inputs, targets = supervised_triple_inputs()
    before_inputs, before_targets = copy.deepcopy(inputs), copy.deepcopy(targets)
    metadata = {**inputs["triple_training"], "seed": 113}
    gt_output = reference(
        **{**inputs, "triple_training": {**metadata, "rollout_cache_probability": 0.0}},
        targets=targets,
        attention_backend="grouped",
    )
    predicted_output = predicted_cache_model(
        **{**inputs, "triple_training": {**metadata, "rollout_cache_probability": 1.0}},
        targets=targets,
        attention_backend="grouped",
    )
    assert_tree_close(inputs, before_inputs)
    assert_tree_close(targets, before_targets)
    torch.testing.assert_close(predicted_output.video, gt_output.video, atol=0, rtol=0)
    torch.testing.assert_close(predicted_output.loss["video_loss"], gt_output.loss["video_loss"], atol=0, rtol=0)
    assert not torch.equal(predicted_output.video_cache.prediction, gt_output.video_cache.prediction)
    gt_output.loss["video_loss"].backward()
    predicted_output.loss["video_loss"].backward()
    for (name, parameter), (_, other) in zip(
        reference.named_parameters(), predicted_cache_model.named_parameters(), strict=True
    ):
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(other.grad, parameter.grad, atol=0, rtol=0, msg=name)


def test_idm_gt_visual_cache_matches_a_separate_final_sigma_prefill():
    torch.manual_seed(127)
    model = Cosmos3Model(triple_config()).eval()
    inputs, targets = supervised_triple_inputs()
    inputs["triple_training"]["rollout_cache_probability"] = 0.0
    cache_inputs = {**inputs, **inputs["idm_video_inputs"], "video_conditioning": (False,)}
    cache_inputs.pop("idm_video_inputs")
    with torch.no_grad():
        actual = model(**inputs, targets=targets, attention_backend="grouped")
        expected = model(**cache_inputs, expert_operation="video", attention_backend="grouped")
    assert_tree_close(actual.video_cache, expected.video_cache)
    assert not torch.equal(actual.video, actual.video_cache.prediction)
    assert all(not tensor.requires_grad for pair in actual.video_cache.layers for tensor in pair)


def test_idm_predicted_visual_cache_reencodes_the_final_euler_state():
    torch.manual_seed(131)
    model = Cosmos3Model(triple_config()).eval()
    inputs, targets = supervised_triple_inputs()
    inputs["triple_training"].update(rollout_cache_probability=1.0, partial_video_steps=4)
    tokens, times, velocities = [], [], []
    hooks = [
        model.proj_in.register_forward_pre_hook(lambda _module, args: tokens.append(args[0].detach().clone())),
        model.time_embedder.register_forward_pre_hook(lambda _module, args: times.append(args[0].detach().clone())),
        model.proj_out.register_forward_hook(
            lambda _module, _args, output: velocities.append(output.detach().clone())
        ),
    ]
    try:
        with torch.no_grad():
            actual = model(**inputs, targets=targets, attention_backend="grouped")
    finally:
        for hook in hooks:
            hook.remove()
    # Four rollout evaluations, a fresh final-state cache, then supervised FM.
    assert len(tokens) == len(times) == len(velocities) == 6
    cache_times = inputs["idm_video_inputs"]["video_timesteps"]
    torch.testing.assert_close(times[-2], cache_times * model.config.timestep_scale, atol=0, rtol=0)
    torch.testing.assert_close(times[-1], inputs["video_timesteps"] * model.config.timestep_scale, atol=0, rtol=0)
    schedule = torch.linspace(1.0, 0.9, 5)
    # Loss excludes padded video positions; rollout still predicts the whole
    # future, so identify all non-history current rows from layout metadata.
    active = torch.tensor(
        [
            role == 1 and block >= 0
            for role, block, modality in zip(
                inputs["layout"].roles, inputs["layout"].block_ids, inputs["layout"].modalities, strict=True
            )
            if modality == 1
        ]
    )
    expected_final = tokens[3] + (schedule[-1] - schedule[-2]) * velocities[3]
    torch.testing.assert_close(tokens[4][active], expected_final[active], atol=0, rtol=0)
    cache_inputs = {
        **inputs,
        "video_tokens": tokens[-2],
        "video_timesteps": cache_times,
        "video_conditioning": (False,),
    }
    cache_inputs.pop("idm_video_inputs")
    with torch.no_grad():
        expected = model(**cache_inputs, expert_operation="video", attention_backend="grouped")
    assert_tree_close(actual.video_cache, expected.video_cache)


def test_video_mode_completely_skips_action_modules():
    model = Cosmos3Model(triple_config()).eval()
    calls = []
    hooks = [
        module.register_forward_pre_hook(lambda _module, _args: calls.append(True))
        for name, module in model.named_modules()
        if name.startswith("action_expert.")
    ]
    try:
        output = model(**video_inputs(), attention_backend="grouped")
    finally:
        for hook in hooks:
            hook.remove()
    assert output.video.shape == (12, 8)
    assert output.action is None
    assert not calls


def test_video_mode_rejects_state_tokens_without_actions():
    model = Cosmos3Model(triple_config()).eval()
    inputs = fixed_inputs()
    layout = inputs["layout"]
    keep = [i for i, modality in enumerate(layout.modalities) if modality != 2]
    inputs["layout"] = replace(
        layout,
        **{
            field: tuple(getattr(layout, field)[i] for i in keep)
            for field in ("sample_ids", "block_ids", "roles", "modalities", "frame_ids")
        },
    )
    inputs["position_ids"] = inputs["position_ids"][:, keep]
    for key in ("action_tokens", "action_timesteps", "action_mask"):
        inputs.pop(key)
    inputs["task_modes"] = ("video", "video")
    with pytest.raises(ValueError, match="[Pp]ure video|state"):
        model(**inputs)


def test_fdm_action_condition_cache_cannot_read_video():
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs("forward_dynamics")
    with torch.no_grad():
        original = model(**inputs, expert_operation="action_condition", attention_backend="grouped")
        perturbed = {**inputs, "video_tokens": inputs["video_tokens"] + 100}
        changed = model(**perturbed, expert_operation="action_condition", attention_backend="grouped")
    assert_tree_close(original.action_condition_cache, changed.action_condition_cache)


def test_idm_video_prediction_is_independent_of_noisy_action():
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs()
    with torch.no_grad():
        original = model(**inputs, expert_operation="video", attention_backend="grouped")
        perturbed = {**inputs, "action_tokens": inputs["action_tokens"] + 100}
        changed = model(**perturbed, expert_operation="video", attention_backend="grouped")
    torch.testing.assert_close(original.video, changed.video, rtol=0, atol=0)


def test_mixed_fdm_idm_full_execution_rejects_joint():
    torch.manual_seed(29)
    model = Cosmos3Model(triple_config()).eval()
    forward, inverse = triple_inputs("forward_dynamics"), triple_inputs()
    mixed = dict(inverse)
    mixed["layout"] = replace(inverse["layout"], roles=forward["layout"].roles[:12] + inverse["layout"].roles[12:])
    mixed["video_timesteps"] = torch.cat((forward["video_timesteps"][:6], inverse["video_timesteps"][6:]))
    mixed["action_timesteps"] = torch.cat((forward["action_timesteps"][:2], inverse["action_timesteps"][2:]))
    mixed["task_modes"] = ("forward_dynamics", "inverse_dynamics")
    with torch.no_grad(), pytest.raises(ValueError, match="joint|single|one task"):
        model(**mixed, attention_backend="grouped")


@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_idm_state_conditions_action_without_entering_video(backend):
    torch.manual_seed(139)
    model = Cosmos3Model(triple_config()).eval()
    inputs, targets = supervised_triple_inputs()
    changed_state = {**inputs, "state_tokens": inputs["state_tokens"] + 20}
    with torch.no_grad():
        original = model(**inputs, targets=targets, attention_backend=backend)
        changed = model(**changed_state, targets=targets, attention_backend=backend)
    torch.testing.assert_close(original.video, changed.video, atol=0, rtol=0)
    assert_tree_close(original.video_cache, changed.video_cache)
    assert not torch.equal(original.action, changed.action)


def test_idm_cannot_read_same_block_or_future_clean_teacher_forcing_copies():
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs()
    with torch.no_grad():
        original = model(**inputs, attention_backend="grouped").action
        clean_future = inputs["video_tokens"].clone()
        # The fixed current future remains unchanged. Only unavailable clean
        # teacher-forcing copies are perturbed, in both packed samples.
        clean_future[[1, 2, 7, 8]] += 100
        changed = model(**{**inputs, "video_tokens": clean_future}, attention_backend="grouped").action
    # Later blocks may use earlier real frames as history; compare block zero.
    torch.testing.assert_close(original[[0, 2]], changed[[0, 2]], rtol=0, atol=0)


def test_noisy_video_condition_retains_its_time():
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs()
    with torch.no_grad():
        noisy_time = model(**inputs, expert_operation="video", attention_backend="grouped").video
        clean_times = {**inputs, "video_timesteps": torch.zeros_like(inputs["video_timesteps"])}
        zero_time = model(**clean_times, expert_operation="video", attention_backend="grouped").video
    assert not torch.equal(noisy_time, zero_time)


@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_cached_action_matches_full_and_skips_video_layers(backend):
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs()
    with torch.no_grad():
        expected = model(**inputs, attention_backend=backend).action
        text = model(**inputs, expert_operation="text", attention_backend=backend).text_cache
        conditions = model(
            **inputs, expert_operation="action_condition", text_cache=text, attention_backend=backend
        ).action_condition_cache
        visual = model(
            **inputs,
            expert_operation="video",
            text_cache=text,
            action_condition_cache=conditions,
            attention_backend=backend,
        ).video_cache
        calls = []
        hooks = [layer.register_forward_pre_hook(lambda _module, _args: calls.append(True)) for layer in model.layers]
        try:
            cached = model(
                **inputs,
                expert_operation="action",
                text_cache=text,
                video_cache=visual,
                action_condition_cache=conditions,
                attention_backend=backend,
            ).action
        finally:
            for hook in hooks:
                hook.remove()
    torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-6)
    assert not calls
    assert not any("cache" in name for name in model.state_dict())


@pytest.mark.parametrize("mapping", [[2, 4], [1, 2, 3, 4]])
def test_action_consumes_the_mapped_video_attention_keys_and_values(monkeypatch, mapping):
    model = Cosmos3Model(triple_config(action_num_hidden_layers=len(mapping), action_layer_map=mapping)).eval()
    expected_keys, expected_values, consumed = {}, {}, {}
    hooks = []

    def record_rope(attention, index):
        original = attention.apply_rope

        def wrapped(query, key, cos, sin):
            output = original(query, key, cos, sin)
            if len(key) == 12:
                expected_keys[index] = output[1].detach().clone()
            return output

        return wrapped

    def record_values(index):
        def hook(_module, _args, output):
            expected_values[index] = output.detach().view(12, 2, 8).clone()

        return hook

    def record_action(index, signature):
        def hook(_module, args, kwargs):
            source = signature.bind(*args, **kwargs).arguments.get("video_kv")
            if source is not None:
                consumed[index] = tuple(tensor.detach().clone() for tensor in source)

        return hook

    for index, layer in enumerate(model.layers):
        monkeypatch.setattr(layer.self_attn, "apply_rope", record_rope(layer.self_attn, index))
        hooks.append(layer.self_attn.add_v_proj.register_forward_hook(record_values(index)))
    for index, layer in enumerate(model.action_expert.layers):
        hooks.append(
            layer.register_forward_pre_hook(record_action(index, inspect.signature(layer.forward)), with_kwargs=True)
        )
    try:
        with torch.no_grad():
            model(**triple_inputs(), attention_backend="grouped")
    finally:
        for hook in hooks:
            hook.remove()
    assert len(consumed) == len(mapping)
    for index, source_layer in enumerate(mapping):
        indexes, key, value = consumed[index]
        assert len(indexes) == 12
        torch.testing.assert_close(key, expected_keys[source_layer - 1], rtol=0, atol=0)
        torch.testing.assert_close(value, expected_values[source_layer - 1], rtol=0, atol=0)


def test_dense_and_grouped_match_outputs_and_action_gradients():
    torch.manual_seed(23)
    dense = Cosmos3Model(triple_config()).train()
    dense.freeze_model()
    grouped = copy.deepcopy(dense)
    inputs = triple_inputs()
    reference = dense(**inputs, attention_backend="dense")
    actual = grouped(**inputs, attention_backend="grouped")
    torch.testing.assert_close(actual.action, reference.action, rtol=1e-5, atol=1e-6)
    reference.action.float().square().mean().backward()
    actual.action.float().square().mean().backward()
    for (name, left), (_, right) in zip(dense.named_parameters(), grouped.named_parameters(), strict=True):
        if name.startswith("action_expert.") and left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-4, atol=1e-6)


def test_policy_is_rejected_by_triple_forward():
    model = Cosmos3Model(triple_config())
    inputs = fixed_inputs()
    inputs["task_modes"] = ("policy", "policy")
    with pytest.raises(ValueError, match="policy|task|mode"):
        model(**inputs)


def test_triple_full_weights_roundtrip_and_caches_are_not_saved(tmp_path):
    model = Cosmos3Model(triple_config())
    model.config.save_pretrained(tmp_path)
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    restored = Cosmos3Model(triple_config(weight_loading={"initialize_missing_patterns": []}))
    load_model_weights(restored, str(tmp_path), init_device="cpu")
    assert restored.weight_loading_report["mode"] == "complete"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_legacy_pretrained_loading_preserves_video_and_initializes_only_action(tmp_path):
    legacy = Cosmos3Model(triple_config(expert_layout="dual"))
    legacy.config.save_pretrained(tmp_path)
    save_file(legacy.state_dict(), str(tmp_path / "model.safetensors"))
    policy = {
        "skip_patterns": ["action_*", "state_*"],
        "initialize_missing_patterns": ["action_expert.*"],
    }
    triple = Cosmos3Model(triple_config(weight_loading=policy))
    load_model_weights(triple, str(tmp_path), init_device="cpu")
    report = triple.weight_loading_report
    assert report["initialized"]
    assert all(name.startswith("action_expert.") for name in report["initialized"])
    assert report["skipped"]
    for name, value in legacy.state_dict().items():
        if not name.startswith(("action_", "state_")):
            torch.testing.assert_close(triple.state_dict()[name], value, rtol=0, atol=0)


def test_full_resume_rejects_missing_action_weights(tmp_path):
    model = Cosmos3Model(triple_config())
    model.config.save_pretrained(tmp_path)
    state = model.state_dict()
    missing = next(name for name in state if name.startswith("action_expert."))
    del state[missing]
    save_file(state, str(tmp_path / "model.safetensors"))
    restored = Cosmos3Model(triple_config(weight_loading={"initialize_missing_patterns": []}))
    with pytest.raises(ValueError, match="Missing Cosmos3 weights"):
        load_model_weights(restored, str(tmp_path), init_device="cpu")


def branch_layout():
    modes = ("forward_dynamics", "inverse_dynamics", "video")
    lanes = []
    for sample, mode in enumerate(modes):
        rows = [(sample, -1, -1, 0, -1)] * 2
        for frame in range(5):
            rows += [(sample, (frame - 1) // 2, 0, 1, frame)] * 2
        for frame in range(5):
            role = 2 if frame in (0, 2) else 1
            rows += [(sample, (frame - 1) // 2, role, 1, frame)] * 2
        if mode != "video":
            for step in range(4):
                role = 2 if mode == "forward_dynamics" or step == 1 else 1
                rows += [(sample, step // 2, role, 2, -1)]
            rows += [(sample, block, 3, 3, -1) for block in (0, 1)]
        lanes.append(rows)
    rows = [row for group in zip_longest(*lanes) for row in group if row is not None]
    s, b, r, m, f = map(tuple, zip(*rows, strict=True))
    return CausalLayout(s, b, r, m, (3, 3, 3), f, (2, 2, 2)), modes


@pytest.mark.parametrize("history", [0, 1, 3, 9])
@pytest.mark.parametrize("backend", ["dense", "grouped"])
def test_video_visibility_matches_main_causal_mask(history, backend):
    # attention.py is unchanged from main. Compare against its original mask,
    # including the condition/target mixture within a single video block.
    layout, modes = branch_layout()
    layout = replace(layout, history_blocks=(history,) * len(modes))
    legacy = prepare_attention(layout, "cpu", "dense")
    candidate = prepare_branch_attention(layout, "cpu", backend, (1,), modes, "video")
    legacy_rows = torch.searchsorted(legacy.gen_indexes, candidate.gen_indexes)
    video_or_text = torch.tensor([modality < 2 for modality in layout.modalities])
    expected = legacy.mask[legacy_rows][:, video_or_text]
    if candidate.mask is not None:
        actual = candidate.mask
    else:
        actual = torch.zeros(len(candidate.gen_indexes), len(layout.roles), dtype=torch.bool)
        for chunk in candidate.chunks:
            qstart = kstart = 0
            for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
                queries = chunk.query_indexes[qstart:qend]
                keys = chunk.kv_indexes[kstart:kend]
                actual[queries[:, None], keys[None, :]] = True
                qstart, kstart = qend, kend
    assert torch.equal(actual[:, video_or_text], expected)


@pytest.mark.parametrize("backend", ["dense", "grouped"])
@pytest.mark.parametrize("partial_condition", [False, True])
def test_pure_video_predictions_and_gradients_match_main_path(backend, partial_condition):
    torch.manual_seed(41)
    config = triple_config(action_interface="none", state_dim=0, train_generation_only=True)
    triple = Cosmos3Model(config).train()
    legacy_config = copy.deepcopy(config)
    legacy_config.expert_layout = "dual"
    legacy = Cosmos3Model(legacy_config).train()
    legacy.load_state_dict(triple.state_dict(), strict=True)
    legacy.freeze_model()
    sample = video_payload_sample(frames=5, height=3, width=3, history=3)
    if partial_condition:
        sample["condition_frame_indexes_vision"] = [0, 2]
    generator = torch.Generator().manual_seed(43)
    inputs, _, _ = pack_model_inputs(
        [sample],
        [torch.randn(sample["video_latents"].shape, generator=generator)],
        timesteps=[torch.tensor([400.0])],
        has_action=False,
        patch_size=config.patch_size,
    )
    expected = legacy(**inputs, attention_backend=backend).video
    actual = triple(**inputs, attention_backend=backend, task_modes=("video",)).video
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.randn(actual.shape, generator=generator)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    legacy_parameters = dict(legacy.named_parameters())
    for name, parameter in triple.named_parameters():
        reference = legacy_parameters[name]
        assert parameter.requires_grad == reference.requires_grad
        if reference.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=3e-6, rtol=3e-5, msg=name)


def branch_visibility_reference(
    layout, queries, modes, branch, conditioning, video_conditioning=None, text_conditioning=None
):
    result = torch.zeros(len(queries), len(layout.roles), dtype=torch.bool)
    rows = list(
        zip(layout.sample_ids, layout.block_ids, layout.roles, layout.modalities, layout.frame_ids, strict=True)
    )
    for offset, query in enumerate(queries):
        s, b, r, m, _ = rows[query]
        for key, (ks, kb, kr, km, frame) in enumerate(rows):
            if s != ks:
                continue
            visible = km == 0 and (text_conditioning is None or text_conditioning[s])
            state = km == 3 and kb == b
            known = km == 2 and kr == 2 and kb == b
            past = km == 1 and kr == 0 and kb < b and max(0, 1 + b * 2) - 3 <= frame < max(0, 1 + b * 2)
            if branch == "video":
                visible |= past
                if r == 0:
                    visible |= km == 1 and kr == 0 and kb == b
                else:
                    visible |= km == 1 and kr in ((2,) if r == 2 else (1, 2)) and kb == b
                    if conditioning and (video_conditioning is None or video_conditioning[s]):
                        visible |= state or (modes[s] == "forward_dynamics" and known)
            elif m == 3:
                visible |= state
            elif branch == "condition" or r == 2:
                visible |= state or known
            else:
                visible |= past or state or (kb == b and km in (1, 2) and kr in (1, 2))
            result[offset, key] = visible
    return result


@pytest.mark.parametrize("branch,modalities", [("video", (1,)), ("condition", (2, 3)), ("action", (2, 3))])
@pytest.mark.parametrize("conditioning", [False, True])
def test_branch_visibility_reference_and_gradients(branch, modalities, conditioning):
    layout, modes = branch_layout()
    dense = prepare_branch_attention(layout, "cpu", "dense", modalities, modes, branch, conditioning)
    grouped = prepare_branch_attention(layout, "cpu", "grouped", modalities, modes, branch, conditioning)
    expected = branch_visibility_reference(layout, dense.gen_indexes.tolist(), modes, branch, conditioning)
    assert torch.equal(dense.mask, expected)
    assert torch.equal(dense.gen_indexes, grouped.gen_indexes)
    restored = torch.zeros_like(expected)
    for chunk in grouped.chunks:
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            qs = chunk.query_indexes[qstart:qend]
            ks = chunk.kv_indexes[kstart:kend]
            restored[qs[:, None], ks[None, :]] = True
            qstart, kstart = qend, kend
    assert torch.equal(restored, expected)
    torch.manual_seed(17)
    q = torch.randn(len(dense.gen_indexes), 4, 8)
    k = torch.randn(len(layout.roles), 2, 8)
    v = torch.randn_like(k)
    probe = torch.randn_like(q)
    outcomes = []
    for plan in (dense, grouped):
        inputs = [item.detach().clone().requires_grad_(True) for item in (q, k, v)]
        output = generation_attention(*inputs, plan)
        grads = torch.autograd.grad((output * probe).sum(), inputs)
        outcomes.append((output, grads))
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=2e-6, rtol=2e-5)
    for left, right in zip(outcomes[0][1], outcomes[1][1], strict=True):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)


def test_branch_grouped_does_not_build_dense_mask(monkeypatch):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import triple_attention

    monkeypatch.setattr(triple_attention, "_dense_branch_mask", lambda *args: pytest.fail("dense allocation"))
    layout, modes = branch_layout()
    plan = prepare_branch_attention(layout, "cpu", "grouped", (1,), modes, "video")
    assert plan.mask is None


@pytest.mark.parametrize("branch,modalities", [("video", (1,)), ("condition", (2, 3)), ("action", (2, 3))])
def test_attention_applies_condition_gates_per_sample_in_dense_and_grouped(branch, modalities):
    layout, modes = branch_layout()
    enabled, text_enabled = (False, True, False), (True, False, True)
    plans = [
        prepare_branch_attention(
            layout,
            "cpu",
            backend,
            modalities,
            modes,
            branch,
            video_conditioning=enabled,
            text_conditioning=text_enabled,
        )
        for backend in ("dense", "grouped")
    ]
    expected = branch_visibility_reference(
        layout, plans[0].gen_indexes.tolist(), modes, branch, True, enabled, text_enabled
    )
    assert torch.equal(plans[0].mask, expected)
    restored = torch.zeros_like(expected)
    for chunk in plans[1].chunks:
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            qs = chunk.query_indexes[qstart:qend]
            ks = chunk.kv_indexes[kstart:kend]
            restored[qs[:, None], ks[None, :]] = True
            qstart, kstart = qend, kend
    assert torch.equal(restored, expected)


def test_branch_empty_queries():
    layout, _ = branch_layout()
    plan = prepare_branch_attention(layout, "cpu", "grouped", (2, 3), ("video",) * 3, "action")
    assert len(plan.gen_indexes) == 0 and not plan.chunks


@pytest.mark.parametrize("task", ["forward_dynamics", "inverse_dynamics"])
@pytest.mark.parametrize("selected", [False, True])
def test_step_level_cache_mix_selects_all_idm_samples_and_leaves_fdm_fixed(task, selected):
    model = Cosmos3Model(triple_config()).eval()
    inputs = triple_inputs(task)
    shared_seed = 83
    step = next(
        step
        for step in range(20)
        if bool(torch.rand((), generator=torch.Generator().manual_seed(shared_seed + 1000003 * step)) < 0.5)
        == selected
    )
    options = {"seed": 79, "video_stop_sigma": 0.9}
    # A selected global step must match the all-samples rollout reference.
    # FDM's known conditioning is fixed even when the shared step is selected.
    reference_probability = float(selected and task == "inverse_dynamics")
    with torch.no_grad():
        expected = model(
            **inputs,
            triple_training={**options, "rollout_cache_probability": reference_probability},
            attention_backend="grouped",
        )
        actual = model(
            **inputs,
            triple_training={
                **options,
                "rollout_cache_probability": 0.5,
                "mix_seed": shared_seed,
                "optimizer_step": step,
            },
            attention_backend="grouped",
        )
    assert_tree_close(actual.video_cache, expected.video_cache)
    torch.testing.assert_close(actual.action, expected.action, rtol=0, atol=0)


def _fsdp2_wrap(model, mesh):
    # Match the production wrapping boundary; freeze before flattening/sharding.
    model.freeze_model()
    for module in list(model.modules()):
        if type(module).__name__ in model._no_split_modules:
            fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _fsdp2_assert_parameters(model, reference):
    for (name, actual), (expected_name, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == expected_name
        gathered = actual.full_tensor() if hasattr(actual, "full_tensor") else actual
        expected_gathered = expected.full_tensor() if hasattr(expected, "full_tensor") else expected
        torch.testing.assert_close(gathered, expected_gathered, rtol=2e-5, atol=2e-6)


def _fsdp2_training_step(model, optimizer, inputs, task):
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = inputs if isinstance(inputs, tuple) else (inputs, None)
    output = model(**inputs, targets=targets, attention_backend="grouped")
    prediction = output.action if task == "inverse_dynamics" else output.video
    loss = sum(output.loss.values()) if targets is not None else prediction.float().square().mean()
    loss.backward()
    optimizer.step()
    return prediction.detach()


def _fsdp2_triple_worker(rank, checkpoint_path, endpoint, *, only_task=None):
    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState, _ModelStrictLoadPlanner

    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        with ops_config_scope(eager_ops_config()):
            torch.manual_seed(41)
            reference = Cosmos3Model(
                triple_config(num_hidden_layers=2, action_num_hidden_layers=1, action_layer_map=[2])
            )
            reference.freeze_model()
            model = _fsdp2_wrap(copy.deepcopy(reference), init_device_mesh("cpu", (2,)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
            tasks = (only_task,) if only_task is not None else ("forward_dynamics", "inverse_dynamics", "video")
            for task in tasks:
                torch.manual_seed(43)
                inputs = video_inputs() if task == "video" else supervised_triple_inputs(task)
                expected = _fsdp2_training_step(reference, reference_optimizer, inputs, task)
                actual = _fsdp2_training_step(model, optimizer, inputs, task)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                _fsdp2_assert_parameters(model, reference)

            if only_task == "forward_dynamics":
                assert all(parameter.grad is None for parameter in model.action_expert.proj_out.parameters())
                assert all(
                    not optimizer.state.get(parameter) for parameter in model.action_expert.proj_out.parameters()
                )
            model_state, optimizer_state = get_state_dict(model, optimizer)
            assert not any("cache" in name for name in model_state)
            parallel = SimpleNamespace(dp_mode="fsdp2")
            if only_task is not None:
                # Match the production checkpointer's sparse-optimizer
                # contract. Unused FDM decoder parameters have no Adam state.
                state = {
                    "model": ModelState(model, parallel_state=parallel),
                    "optimizer": OptimizerState(model, optimizer, parallel_state=parallel),
                }
            else:
                state = {"model": model_state, "optimizer": optimizer_state}
            dcp.save(state, checkpoint_id=checkpoint_path)
            if only_task == "forward_dynamics":
                metadata = dcp.FileSystemReader(checkpoint_path).read_metadata()
                assert not any(
                    key.startswith("optimizer.state.action_expert.proj_out.") for key in metadata.state_dict_metadata
                )

            torch.manual_seed(47)
            restored = Cosmos3Model(
                triple_config(num_hidden_layers=2, action_num_hidden_layers=1, action_layer_map=[2])
            )
            restored = _fsdp2_wrap(restored, init_device_mesh("cpu", (2,)))
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
            if only_task is not None:
                state = {
                    "model": ModelState(restored, parallel_state=parallel),
                    "optimizer": OptimizerState(restored, restored_optimizer, parallel_state=parallel, load=True),
                }
                dcp.load(state, checkpoint_id=checkpoint_path, planner=_ModelStrictLoadPlanner(strict_model=True))
            else:
                restored_state, restored_optimizer_state = get_state_dict(restored, restored_optimizer)
                state = {"model": restored_state, "optimizer": restored_optimizer_state}
                dcp.load(state, checkpoint_id=checkpoint_path)
                incompatible = set_state_dict(
                    restored,
                    restored_optimizer,
                    model_state_dict=state["model"],
                    optim_state_dict=state["optimizer"],
                )
                assert not incompatible.missing_keys and not incompatible.unexpected_keys
            _fsdp2_assert_parameters(restored, reference)
            torch.manual_seed(53)
            resume_task = only_task or "inverse_dynamics"
            inputs = video_inputs() if resume_task == "video" else supervised_triple_inputs(resume_task)
            expected = _fsdp2_training_step(model, optimizer, inputs, resume_task)
            actual = _fsdp2_training_step(restored, restored_optimizer, inputs, resume_task)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            _fsdp2_assert_parameters(restored, model)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_two_rank_fsdp2_training_and_full_dcp_resume(tmp_path):
    spawn_with_rendezvous(_fsdp2_triple_worker, args=(str(tmp_path / "checkpoint"),), nprocs=2)


def _fsdp2_fdm_checkpoint_worker(rank, checkpoint_path, endpoint):
    _fsdp2_triple_worker(rank, checkpoint_path, endpoint, only_task="forward_dynamics")


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_two_rank_fsdp2_fdm_checkpoint_restores_unused_decoder_optimizer_state(tmp_path):
    spawn_with_rendezvous(_fsdp2_fdm_checkpoint_worker, args=(str(tmp_path / "fdm-checkpoint"),), nprocs=2)


def _fdm_condition_fsdp2_worker(rank, checkpointing, endpoint):
    from veomni.trainer.causal_cosmos3 import _TripleExpertOptimizerGuard

    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        with ops_config_scope(eager_ops_config()):
            torch.manual_seed(67)
            reference = Cosmos3Model(
                triple_config(num_hidden_layers=2, action_num_hidden_layers=1, action_layer_map=[2])
            )
            reference.freeze_model()
            if checkpointing:
                reference.gradient_checkpointing_enable()
            model = _fsdp2_wrap(copy.deepcopy(reference), init_device_mesh("cpu", (2,)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
            task = "forward_dynamics"
            guards = [
                _TripleExpertOptimizerGuard(
                    SimpleNamespace(
                        model=candidate,
                        optimizer=optim,
                        model_config=candidate.config,
                        parallel_state=SimpleNamespace(dp_group=dist.group.WORLD),
                    )
                )
                for candidate, optim in ((model, optimizer), (reference, reference_optimizer))
            ]
            calls = {"video_text": 0, "action": 0}

            def record(name):
                def hook(_module, _args):
                    calls[name] += 1

                return hook

            hooks = [layer.register_forward_pre_hook(record("video_text")) for layer in model.layers]
            hooks += [layer.register_forward_pre_hook(record("action")) for layer in model.action_expert.layers]
            try:
                # Every rank runs FDM. Different conditioning choices must
                # keep the action-encoder FSDP trace aligned, including a rank
                # whose video mask excludes every action/state key.
                for pair in (("text", "action"), ("action", "text"), ("both", "text"), ("both", "both")):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    inputs, targets = supervised_triple_inputs(task, conditioning=pair[rank])
                    for guard in guards:
                        guard.reset()
                        guard.record(
                            targets,
                            task_modes=inputs["task_modes"],
                            video_conditioning=inputs["video_conditioning"],
                        )
                    expected = reference(**inputs, targets=targets, attention_backend="grouped")
                    calls.update(video_text=0, action=0)
                    actual = model(**inputs, targets=targets, attention_backend="grouped")
                    torch.testing.assert_close(actual.video, expected.video, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.action, expected.action, rtol=1e-5, atol=1e-6)
                    assert calls == {"video_text": 4, "action": 1}
                    assert set(actual.loss) == {"video_loss", "action_loss"}
                    sum(expected.loss.values()).backward()
                    if pair[rank] == "text":
                        assert_gradient_partition(reference, action=False)
                    else:
                        assert reference.action_expert.proj_in.weight.grad.abs().sum() > 0
                    sum(actual.loss.values()).backward()
                    for parameter in reference.parameters():
                        if parameter.requires_grad:
                            if parameter.grad is None:
                                parameter.grad = torch.zeros_like(parameter)
                            dist.all_reduce(parameter.grad)
                            parameter.grad.div_(2)
                    reference_optimizer.step()
                    optimizer.step()
                    _fsdp2_assert_parameters(model, reference)
            finally:
                for hook in hooks:
                    hook.remove()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("checkpointing", [False, True])
def test_two_rank_fsdp2_fdm_condition_choices_and_encoder_gradients(checkpointing):
    spawn_with_rendezvous(_fdm_condition_fsdp2_worker, args=(checkpointing,), nprocs=2)


def _local_tensor_snapshot(tensor):
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return local.detach().clone()


def _expert_optimizer_snapshot(model, optimizer):
    return {
        name: (
            _local_tensor_snapshot(parameter),
            {
                key: _local_tensor_snapshot(value) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for key, value in optimizer.state.get(parameter, {}).items()
            },
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _guard_fsdp2_worker(rank, endpoint):
    from veomni.trainer.causal_cosmos3 import _TripleExpertOptimizerGuard

    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        with ops_config_scope(eager_ops_config()):
            torch.manual_seed(97)
            model = Cosmos3Model(triple_config(num_hidden_layers=2, action_num_hidden_layers=1, action_layer_map=[2]))
            model = _fsdp2_wrap(model, init_device_mesh("cpu", (2,)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
            runtime = SimpleNamespace(
                model=model,
                optimizer=optimizer,
                model_config=model.config,
                parallel_state=SimpleNamespace(dp_group=dist.group.WORLD),
            )
            guard = _TripleExpertOptimizerGuard(runtime)
            scenarios = [
                # (task, per-microbatch conditions, suppress video loss,
                #  global [video FM, action FM, conditioned FDM video FM])
                ("inverse_dynamics", ("both",), False, (True, True, False)),
                ("forward_dynamics", (("action", "text")[rank],), False, (True, False, True)),
                ("forward_dynamics", ("text",), False, (True, False, False)),
                ("inverse_dynamics", ("both",), True, (False, True, False)),
                ("forward_dynamics", ("text", "action"), False, (True, False, True)),
            ]
            parameter_groups = {
                id(parameter): group for group, parameters in enumerate(guard.parameters) for parameter in parameters
            }
            for step, (task, microbatch_conditions, suppress_video, global_activity) in enumerate(scenarios):
                before = _expert_optimizer_snapshot(model, optimizer)
                optimizer.zero_grad(set_to_none=True)
                guard.reset()
                for conditioning in microbatch_conditions:
                    inputs, targets = supervised_triple_inputs(task, conditioning=conditioning)
                    if suppress_video:
                        targets["loss_masks"]["video_loss_mask"].zero_()
                    guard.record(
                        targets,
                        task_modes=inputs["task_modes"],
                        video_conditioning=inputs["video_conditioning"],
                    )
                    output = model(**inputs, targets=targets, attention_backend="grouped")
                    (sum(output.loss.values()) / len(microbatch_conditions)).backward()
                if len(microbatch_conditions) == 2:
                    # A text-only FDM microbatch cannot activate its encoder;
                    # the later action-conditioned microbatch activates it.
                    assert guard.active.tolist() == [1, 0, 1]
                optimizer.step()
                assert guard.prepared
                assert guard.active.tolist() == list(map(int, global_activity))
                after = _expert_optimizer_snapshot(model, optimizer)
                group_activity = (global_activity[0], global_activity[1], global_activity[1] or global_activity[2])
                updated = [False, False, False]
                momentum = [False, False, False]
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    expert = parameter_groups[id(parameter)]
                    old_parameter, old_state = before[name]
                    new_parameter, new_state = after[name]
                    if not group_activity[expert]:
                        assert parameter.grad is None, name
                        torch.testing.assert_close(new_parameter, old_parameter, rtol=0, atol=0, msg=name)
                        assert_tree_close(new_state, old_state)
                    else:
                        updated[expert] |= not torch.equal(new_parameter, old_parameter)
                        state = optimizer.state.get(parameter, {})
                        if "exp_avg" in state:
                            momentum[expert] |= bool(torch.count_nonzero(_local_tensor_snapshot(state["exp_avg"])))
                for expert, active in enumerate(group_activity):
                    if active:
                        assert updated[expert] and momentum[expert], (step, expert)
                if step == 0:
                    # Subsequent pure-task steps now have nonzero AdamW
                    # momentum in both experts, not only decoupled decay.
                    assert all(momentum)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_two_rank_optimizer_guard_freezes_inactive_weights_and_momentum():
    spawn_with_rendezvous(_guard_fsdp2_worker, nprocs=2)
