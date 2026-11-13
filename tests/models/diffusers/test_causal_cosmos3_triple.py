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
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.attention import CausalLayout, generation_attention
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
def test_fdm_gradient_isolation(backend, checkpointing):
    torch.manual_seed(19)
    model = Cosmos3Model(triple_config()).train()
    model.freeze_model()
    if checkpointing:
        model.gradient_checkpointing_enable()
    output = model(**triple_inputs("forward_dynamics"), attention_backend=backend)
    output.video.float().square().mean().backward()
    assert_gradient_partition(model, action=False)
    assert all(not parameter.requires_grad for parameter in model.embed_tokens.parameters())


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


def test_mixed_fdm_idm_batch_matches_each_task_separately():
    torch.manual_seed(29)
    model = Cosmos3Model(triple_config()).eval()
    forward, inverse = triple_inputs("forward_dynamics"), triple_inputs()
    mixed = dict(inverse)
    mixed["layout"] = replace(inverse["layout"], roles=forward["layout"].roles[:12] + inverse["layout"].roles[12:])
    mixed["video_timesteps"] = torch.cat((forward["video_timesteps"][:6], inverse["video_timesteps"][6:]))
    mixed["action_timesteps"] = torch.cat((forward["action_timesteps"][:2], inverse["action_timesteps"][2:]))
    mixed["task_modes"] = ("forward_dynamics", "inverse_dynamics")
    with torch.no_grad():
        actual = model(**mixed, attention_backend="grouped")
        expected_forward = model(**forward, attention_backend="grouped")
        expected_inverse = model(**inverse, attention_backend="grouped")
    torch.testing.assert_close(actual.video[:6], expected_forward.video[:6], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.action[2:], expected_inverse.action[2:], rtol=1e-5, atol=1e-6)


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


def branch_visibility_reference(layout, queries, modes, branch, conditioning):
    result = torch.zeros(len(queries), len(layout.roles), dtype=torch.bool)
    rows = list(
        zip(layout.sample_ids, layout.block_ids, layout.roles, layout.modalities, layout.frame_ids, strict=True)
    )
    for offset, query in enumerate(queries):
        s, b, r, m, _ = rows[query]
        for key, (ks, kb, kr, km, frame) in enumerate(rows):
            if s != ks:
                continue
            visible = km == 0
            state = km == 3 and kb == b
            known = km == 2 and kr == 2 and kb == b
            past = km == 1 and kr == 0 and kb < b and max(0, 1 + b * 2) - 3 <= frame < max(0, 1 + b * 2)
            if branch == "video":
                visible |= past
                if r == 0:
                    visible |= km == 1 and kr == 0 and kb == b
                else:
                    visible |= km == 1 and kr in (1, 2) and kb == b
                    if conditioning:
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
    output = model(**inputs, attention_backend="grouped")
    prediction = output.action if task == "inverse_dynamics" else output.video
    prediction.float().square().mean().backward()
    optimizer.step()
    return prediction.detach()


def _fsdp2_triple_worker(rank, checkpoint_path, endpoint):
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
            for task in ("forward_dynamics", "inverse_dynamics", "video"):
                torch.manual_seed(43)
                inputs = video_inputs() if task == "video" else triple_inputs(task)
                expected = _fsdp2_training_step(reference, reference_optimizer, inputs, task)
                actual = _fsdp2_training_step(model, optimizer, inputs, task)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                _fsdp2_assert_parameters(model, reference)

            model_state, optimizer_state = get_state_dict(model, optimizer)
            assert not any("cache" in name for name in model_state)
            dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=checkpoint_path)

            torch.manual_seed(47)
            restored = Cosmos3Model(
                triple_config(num_hidden_layers=2, action_num_hidden_layers=1, action_layer_map=[2])
            )
            restored = _fsdp2_wrap(restored, init_device_mesh("cpu", (2,)))
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
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
            inputs = triple_inputs()
            expected = _fsdp2_training_step(model, optimizer, inputs, "inverse_dynamics")
            actual = _fsdp2_training_step(restored, restored_optimizer, inputs, "inverse_dynamics")
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            _fsdp2_assert_parameters(restored, model)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_two_rank_fsdp2_training_and_full_dcp_resume(tmp_path):
    spawn_with_rendezvous(_fsdp2_triple_worker, args=(str(tmp_path / "checkpoint"),), nprocs=2)


def _supervised_targets(inputs, task):
    generator = torch.Generator().manual_seed(61)
    video_roles = [
        role
        for role, modality in zip(inputs["layout"].roles, inputs["layout"].modalities, strict=True)
        if modality == 1
    ]
    video_mask = torch.tensor([role == 1 and task == "forward_dynamics" for role in video_roles])
    action_mask = (
        inputs["action_mask"].clone() if task == "inverse_dynamics" else torch.zeros_like(inputs["action_mask"])
    )
    return dict(
        spans=[
            SimpleNamespace(current_video_rows=slice(3, 6), action_rows=slice(0, 2)),
            SimpleNamespace(current_video_rows=slice(9, 12), action_rows=slice(2, 4)),
        ],
        video=[torch.randn(2, 3, 2, 2, generator=generator) for _ in range(2)],
        action=[torch.randn(2, 4, generator=generator) for _ in range(2)],
        video_times=[inputs["video_timesteps"][span] / 1000 for span in (slice(3, 6), slice(9, 12))],
        action_times=[inputs["action_timesteps"][span] / 1000 for span in (slice(0, 2), slice(2, 4))],
        loss_masks={"video_loss_mask": video_mask, "action_loss_mask": action_mask},
    )


def _mixed_rank_fsdp2_worker(rank, checkpointing, endpoint):
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
            task = "forward_dynamics" if rank == 0 else "inverse_dynamics"
            inputs = triple_inputs(task)
            targets = _supervised_targets(inputs, task)
            calls = {"video_text": 0, "action": 0}

            def record(name):
                def hook(_module, _args):
                    calls[name] += 1

                return hook

            hooks = [layer.register_forward_pre_hook(record("video_text")) for layer in model.layers]
            hooks += [layer.register_forward_pre_hook(record("action")) for layer in model.action_expert.layers]
            try:
                scenarios = [
                    ({"rollout_cache_probability": 0.0}, 4),
                    ({"rollout_cache_probability": 1.0}, 12),
                ]
                # Pick a selected and a skipped optimizer step from a shared
                # stream. Rank-local noise seeds deliberately differ; they
                # must not decide which collectives to enter.
                shared_seed = 83
                steps = {}
                for step in range(20):
                    draw = torch.rand((), generator=torch.Generator().manual_seed(shared_seed + 1000003 * step))
                    steps.setdefault(bool(draw < 0.5), step)
                assert set(steps) == {False, True}
                for selected in (False, True):
                    scenarios.append(
                        (
                            {
                                "rollout_cache_probability": 0.5,
                                "mix_seed": shared_seed,
                                "optimizer_step": steps[selected],
                            },
                            12 if selected else 4,
                        )
                    )
                for options, expected_video_text_calls in scenarios:
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    metadata = {"video_stop_sigma": 0.9, "seed": 71 + rank, **options}
                    expected = reference(
                        **inputs, targets=targets, triple_training=metadata, attention_backend="grouped"
                    )
                    calls.update(video_text=0, action=0)
                    actual = model(**inputs, targets=targets, triple_training=metadata, attention_backend="grouped")
                    torch.testing.assert_close(actual.video, expected.video, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual.action, expected.action, rtol=1e-5, atol=1e-6)
                    assert calls == {"video_text": expected_video_text_calls, "action": 2}
                    assert set(actual.loss) == {"video_loss", "action_loss"}
                    sum(expected.loss.values()).backward()
                    assert_gradient_partition(reference, action=rank == 1)
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
def test_mixed_rank_fsdp2_tasks_and_rollout_trace(checkpointing):
    spawn_with_rendezvous(_mixed_rank_fsdp2_worker, args=(checkpointing,), nprocs=2)


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
            mixed_task = "forward_dynamics" if rank == 0 else "inverse_dynamics"
            scenarios = [
                ((mixed_task,), (True, True)),
                (("forward_dynamics",), (True, False)),
                (("inverse_dynamics",), (False, True)),
                ((mixed_task,), (True, True)),
                (("forward_dynamics", "inverse_dynamics"), (True, True)),
            ]
            for step, (microbatch_tasks, global_activity) in enumerate(scenarios):
                before = _expert_optimizer_snapshot(model, optimizer)
                optimizer.zero_grad(set_to_none=True)
                guard.reset()
                for task in microbatch_tasks:
                    inputs = triple_inputs(task)
                    targets = _supervised_targets(inputs, task)
                    guard.record(targets)
                    output = model(**inputs, targets=targets, attention_backend="grouped")
                    (sum(output.loss.values()) / len(microbatch_tasks)).backward()
                if len(microbatch_tasks) == 2:
                    # Both ranks see FD then ID: the last microbatch alone
                    # cannot activate video, so this proves local OR semantics.
                    assert guard.active.tolist() == [1, 1]
                optimizer.step()
                assert guard.prepared
                assert guard.active.tolist() == list(map(int, global_activity))
                after = _expert_optimizer_snapshot(model, optimizer)
                updated = [False, False]
                momentum = [False, False]
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    expert = int(name.startswith("action_expert."))
                    old_parameter, old_state = before[name]
                    new_parameter, new_state = after[name]
                    if not global_activity[expert]:
                        assert parameter.grad is None, name
                        torch.testing.assert_close(new_parameter, old_parameter, rtol=0, atol=0, msg=name)
                        assert_tree_close(new_state, old_state)
                    else:
                        updated[expert] |= not torch.equal(new_parameter, old_parameter)
                        state = optimizer.state.get(parameter, {})
                        if "exp_avg" in state:
                            momentum[expert] |= bool(torch.count_nonzero(_local_tensor_snapshot(state["exp_avg"])))
                for expert, active in enumerate(global_activity):
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
