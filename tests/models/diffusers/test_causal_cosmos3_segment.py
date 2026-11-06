"""Episode-segment recurrence against a full-shape stop-gradient reference."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.layers import rotary_tables
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.segment import SegmentMemory, _prepare
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.utils.recompute_utils import apply_recompute_policy, build_policy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def positions(frames, patches=2):
    return torch.tensor(
        [
            [100.0 + frame for frame in frames for _ in range(patches)],
            [0.0] * (len(frames) * patches),
            [float(patch) for _ in frames for patch in range(patches)],
        ],
        dtype=torch.float32,
    )


def segment_payload(*, prefix=(-4, -3, -2, -1, 0), clean=(1, 2, 3, 4), anchors=(0, 2, 4), backend="grouped"):
    """Small spatially nontrivial segment; cold prefix includes the first anchor."""
    generator = torch.Generator().manual_seed(123 + (clean[0] if clean else 0))
    patches, features = 2, 8
    decisions, video_conditions, action_conditions = [], [], []
    future_frames, action_frames, state_frames = [], [], []
    video_times, action_times = [], []
    for index, anchor in enumerate(anchors):
        vc, ac = index % 3 == 2, index % 3 == 1
        decisions.append(
            dict(
                anchor_frame=anchor,
                video_start=4 * index,
                video_stop=4 * (index + 1),
                action_start=4 * index,
                action_stop=4 * (index + 1),
                state_row=index,
                video_condition=vc,
                action_condition=ac,
            )
        )
        video_conditions.extend([vc] * 4)
        action_conditions.extend([ac] * 4)
        future_frames.extend((anchor + 1, anchor + 2))
        action_frames.extend(anchor + (i + 1) / 2 for i in range(4))
        state_frames.append(anchor)
        video_times.extend([200.0 + index * 100] * 4)
        action_times.extend([200.0 + index * 100] * 4)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    d = len(decisions)
    sample = dict(
        input_ids=torch.tensor([1, 2, 3]),
        context_id="episode0/prompt0",
        history_latents=5,
        patches_per_frame=patches,
        prefix_frame_ids=tuple(prefix),
        clean_frame_ids=tuple(clean),
        prefix_video_tokens=rand(len(prefix) * patches, features),
        prefix_position_ids=positions(prefix),
        clean_video_tokens=rand(len(clean) * patches, features),
        clean_position_ids=positions(clean),
        future_video_tokens=rand(d * 4, features),
        future_video_timesteps=torch.tensor(video_times),
        future_position_ids=positions(future_frames),
        action_tokens=rand(d * 4, 4),
        action_timesteps=torch.tensor(action_times),
        action_mask=torch.ones(d * 4, 4, dtype=torch.bool),
        action_position_ids=positions(action_frames, 1),
        state_tokens=rand(d, 4),
        state_mask=torch.ones(d, 4, dtype=torch.bool),
        state_position_ids=positions(state_frames, 1),
        decisions=tuple(decisions),
        attention_backend=backend,
    )
    spans = [
        SimpleNamespace(current_video_rows=slice(4 * i, 4 * (i + 1)), action_rows=slice(4 * i, 4 * (i + 1)))
        for i in range(d)
    ]
    sample["targets"] = dict(
        video=[rand(2, 2, 2, 4) for _ in range(d)],
        action=[rand(4, 4) for _ in range(d)],
        video_times=[torch.full((2,), 0.2 + i * 0.1) for i in range(d)],
        action_times=[torch.full((4,), 0.2 + i * 0.1) for i in range(d)],
        spans=spans,
        loss_masks=dict(
            video_loss_mask=~torch.tensor(video_conditions),
            action_loss_mask=(~torch.tensor(action_conditions))[:, None].expand(-1, 4).clone(),
        ),
    )
    return sample


def model_for_segment():
    model = Cosmos3Model(tiny_config(enable_tbptt=True, tbptt_keep_history_latents=4, train_generation_only=True))
    model.freeze_model()
    return model.train()


def attention(query, key, value, mask):
    repeats = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query.transpose(0, 1),
        key.repeat_interleave(repeats, 1).transpose(0, 1),
        value.repeat_interleave(repeats, 1).transpose(0, 1),
        attn_mask=mask,
        dropout_p=0,
    ).transpose(0, 1)


def reference_layer(layer, und, joined, cos, sin, text, mask):
    a = layer.self_attn
    un, gn = layer.input_layernorm(und), layer.input_layernorm_moe_gen(joined)
    qu = a.norm_q(a.to_q(un).view(-1, a.num_heads, a.head_dim))
    raw_ku = a.to_k(un).view(-1, a.num_kv_heads, a.head_dim)
    ku, vu = a.norm_k(raw_ku), a.to_v(un).view(-1, a.num_kv_heads, a.head_dim)
    qu, ku = a.apply_rope(qu, ku, cos[:text], sin[:text])
    cross = ku
    if a.k_norm_und_for_gen is not None:
        cross = a.k_norm_und_for_gen(raw_ku)
        _, cross = a.apply_rope(cross, cross, cos[:text], sin[:text])
    qg = a.norm_added_q(a.add_q_proj(gn).view(-1, a.num_heads, a.head_dim))
    kg = a.norm_added_k(a.add_k_proj(gn).view(-1, a.num_kv_heads, a.head_dim))
    vg = a.add_v_proj(gn).view(-1, a.num_kv_heads, a.head_dim)
    qg, kg = a.apply_rope(qg, kg, cos[text:], sin[text:])
    unew = und + a.to_out(attention(qu, ku, vu, torch.ones(text, text, dtype=torch.bool).tril()).flatten(1))
    joined = joined + a.to_add_out(attention(qg, torch.cat((cross, kg)), torch.cat((vu, vg)), mask).flatten(1))
    joined = joined + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(joined))
    return unew + layer.mlp(layer.post_attention_layernorm(unew)), joined


def reference(model, sample, memory=None, *, owners=None):
    """Full old+active shapes, independently built masks, per-layer boundary detach."""
    p, m, text = sample["patches_per_frame"], sample["history_latents"], len(sample["input_ids"])
    old_frames = sample["prefix_frame_ids"] if memory is None else memory.frame_ids
    old_positions = sample["prefix_position_ids"] if memory is None else memory.position_ids
    all_frames = old_frames + sample["clean_frame_ids"]
    old_rows, clean_rows = len(old_frames) * p, len(sample["clean_frame_ids"]) * p
    future_rows, action_rows = len(sample["future_video_tokens"]), len(sample["action_tokens"])
    vconds = [d["video_condition"] for d in sample["decisions"] for _ in range(d["video_stop"] - d["video_start"])]
    aconds = [d["action_condition"] for d in sample["decisions"] for _ in range(d["action_stop"] - d["action_start"])]
    with torch.no_grad():
        old = model.proj_in(sample["prefix_video_tokens"])
        old = old + model.time_embedder(torch.zeros(len(old)))
    clean = model.proj_in(sample["clean_video_tokens"]) + model.time_embedder(torch.zeros(clean_rows))
    future = model.proj_in(sample["future_video_tokens"]) + model.time_embedder(
        sample["future_video_timesteps"].masked_fill(torch.tensor(vconds), 0) * model.config.timestep_scale
    )
    actions = model.action_proj_in(sample["action_tokens"].masked_fill(~sample["action_mask"], 0))
    actions = (
        actions
        + model.action_modality_embed
        + model.time_embedder(
            sample["action_timesteps"].masked_fill(torch.tensor(aconds), 0) * model.config.timestep_scale
        )
    )
    states = model.state_proj_in(sample["state_tokens"].masked_fill(~sample["state_mask"], 0))
    states = states + model.state_modality_embed + model.time_embedder(torch.zeros(len(states)))
    active = torch.cat((clean, future, actions, states))
    und = model.embed_tokens(sample["input_ids"])
    text_pos = sample.get("text_position_ids", torch.arange(text, dtype=torch.float32).expand(3, -1))
    all_positions = torch.cat(
        (
            text_pos,
            old_positions,
            sample["clean_position_ids"],
            sample["future_position_ids"],
            sample["action_position_ids"],
            sample["state_position_ids"],
        ),
        dim=1,
    )
    cos, sin = rotary_tables(all_positions, model.config, active.dtype)
    # Metadata is assembled row-by-row independently of the production plan.
    labels = [("clean", frame, False) for frame in all_frames for _ in range(p)]
    labels += [
        ("video", i, vc)
        for i, d in enumerate(sample["decisions"])
        for vc in [d["video_condition"]] * (d["video_stop"] - d["video_start"])
    ]
    labels += [
        ("action", i, ac)
        for i, d in enumerate(sample["decisions"])
        for ac in [d["action_condition"]] * (d["action_stop"] - d["action_start"])
    ]
    labels += [("state", i, True) for i in range(len(sample["decisions"]))]
    mask = torch.zeros(len(labels), text + len(labels), dtype=torch.bool)
    for q, (kind, index, condition) in enumerate(labels):
        mask[q, :text] = True
        for k, (key_kind, key_index, key_condition) in enumerate(labels):
            if kind == "clean":
                visible = key_kind == "clean" and index - m < key_index <= index
            elif kind == "state":
                visible = key_kind == "state" and key_index == index
            elif key_kind == "clean":
                anchor = sample["decisions"][index]["anchor_frame"]
                visible = anchor - m < key_index <= anchor
            else:
                visible = key_index == index and (key_kind == "state" or not condition or key_condition)
            mask[q, text + k] = visible
    saved = []
    for index, layer in enumerate(model.layers):
        if memory is not None:
            old = memory.layer_inputs[index]
        joined = torch.cat((old.detach(), active))
        saved.append(torch.cat((old.detach(), active[:clean_rows].detach()))[-m * p :].clone())
        if owners is None:
            und, joined = reference_layer(layer, und, joined, cos, sin, text, mask)
        else:
            # The complete episode is present at every layer. Independently
            # recompute each query chunk, detaching its earlier clean entries;
            # no production cache or attention-plan routing is used.
            result, next_und = torch.zeros_like(joined), None
            for owner in sorted(set(owners)):
                stopped = torch.tensor(
                    [
                        row < old_rows or (labels[row][0] == "clean" and earlier < owner)
                        for row, earlier in enumerate(owners)
                    ]
                )
                entry = torch.where(stopped[:, None], joined.detach(), joined)
                text_result, hidden = reference_layer(layer, und, entry, cos, sin, text, mask)
                selected = torch.tensor([row for row, value in enumerate(owners) if value == owner])
                chosen = hidden.index_select(0, selected)
                result = result.index_copy(0, selected, chosen.detach() if owner < 0 else chosen)
                if next_und is None:
                    next_und = text_result
            und, joined = next_und, result
        old, active = joined[:old_rows].detach(), joined[old_rows:]
    normalized = model.norm_moe_gen(active)
    output = SimpleNamespace(
        video=model.proj_out(normalized[clean_rows : clean_rows + future_rows]),
        action=model.action_proj_out(normalized[clean_rows + future_rows : clean_rows + future_rows + action_rows]),
    )
    output.action = output.action.masked_fill(~sample["action_mask"], 0)
    output.loss = model.loss(output, sample["targets"])
    output.segment_memory = SegmentMemory(
        frame_ids=all_frames[-m:],
        position_ids=torch.cat((old_positions, sample["clean_position_ids"]), 1)[:, -m * p :].clone(),
        layer_inputs=tuple(saved),
        input_ids=sample["input_ids"].cpu().clone(),
        text_position_ids=text_pos.cpu().clone(),
        context_id=sample["context_id"],
        history_latents=m,
        patches_per_frame=p,
        model_id=id(model),
    )
    return output


def combine_segments(samples):
    """Full-episode payload, with independent query ownership for the oracle."""
    combined = copy.deepcopy(samples[0])
    fields = (
        "clean_video_tokens",
        "clean_position_ids",
        "future_video_tokens",
        "future_video_timesteps",
        "future_position_ids",
        "action_tokens",
        "action_timesteps",
        "action_mask",
        "action_position_ids",
        "state_tokens",
        "state_mask",
        "state_position_ids",
    )
    for field in fields:
        combined[field] = torch.cat([sample[field] for sample in samples], dim=1 if "position_ids" in field else 0)
    combined["clean_frame_ids"] = sum((sample["clean_frame_ids"] for sample in samples), ())
    decisions, spans, clean_owners, video_owners, action_owners, state_owners = [], [], [], [], [], []
    voffset = aoffset = doffset = 0
    p = combined["patches_per_frame"]
    for owner, sample in enumerate(samples):
        clean_owners.extend([owner] * len(sample["clean_frame_ids"]) * p)
        video_owners.extend([owner] * len(sample["future_video_tokens"]))
        action_owners.extend([owner] * len(sample["action_tokens"]))
        state_owners.extend([owner] * len(sample["state_tokens"]))
        for decision in sample["decisions"]:
            d = {
                **decision,
                "video_start": decision["video_start"] + voffset,
                "video_stop": decision["video_stop"] + voffset,
                "action_start": decision["action_start"] + aoffset,
                "action_stop": decision["action_stop"] + aoffset,
                "state_row": decision["state_row"] + doffset,
            }
            decisions.append(d)
            spans.append(
                SimpleNamespace(
                    current_video_rows=slice(d["video_start"], d["video_stop"]),
                    action_rows=slice(d["action_start"], d["action_stop"]),
                )
            )
        voffset += len(sample["future_video_tokens"])
        aoffset += len(sample["action_tokens"])
        doffset += len(sample["decisions"])
    combined["decisions"] = tuple(decisions)
    combined["targets"] = {
        key: sum((sample["targets"][key] for sample in samples), [])
        for key in ("video", "action", "video_times", "action_times")
    }
    combined["targets"]["spans"] = spans
    combined["targets"]["loss_masks"] = {
        key: torch.cat([sample["targets"]["loss_masks"][key] for sample in samples])
        for key in ("video_loss_mask", "action_loss_mask")
    }
    owners = [-1] * len(combined["prefix_frame_ids"]) * p + clean_owners + video_owners + action_owners + state_owners
    return combined, owners


def gradients(model):
    return {
        name: None if value.grad is None else value.grad.detach().clone() for name, value in model.named_parameters()
    }


def assert_gradients(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        expected = right[name]
        if value is None or expected is None:
            assert value is expected, name
        else:
            torch.testing.assert_close(
                value, expected, atol=3e-6, rtol=3e-5, msg=lambda message, name=name: f"{name}: {message}"
            )


@pytest.mark.parametrize(
    "backend,recompute", [("dense", "none"), ("grouped", "none"), ("grouped", "checkpoint"), ("grouped", "sac")]
)
def test_segment_matches_independent_full_shape_reference(backend, recompute):
    torch.manual_seed(41)
    actual_model = model_for_segment()
    expected_model = copy.deepcopy(actual_model)
    if recompute != "none":
        actual_model.gradient_checkpointing_enable()
        if recompute == "sac":
            apply_recompute_policy(
                actual_model,
                build_policy(
                    enabled=True,
                    enable_reentrant=False,
                    early_stop=True,
                    recompute_layers=-1,
                    selective_recompute_layers=2,
                ),
            )
    sample = segment_payload(backend=backend)
    expected = reference(expected_model, sample)
    actual = actual_model(stream_segment=sample)
    torch.testing.assert_close(actual.video, expected.video, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.action, expected.action, atol=2e-6, rtol=2e-5)
    for key in expected.loss:
        torch.testing.assert_close(actual.loss[key], expected.loss[key], atol=2e-6, rtol=2e-5)
    sum(actual.loss.values()).backward()
    sum(expected.loss.values()).backward()
    assert_gradients(gradients(actual_model), gradients(expected_model))
    for actual_hidden, expected_hidden in zip(
        actual.segment_memory.layer_inputs, expected.segment_memory.layer_inputs
    ):
        assert not actual_hidden.requires_grad and actual_hidden.grad_fn is None
        torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-6, rtol=2e-5)
    assert actual.segment_memory.frame_ids == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("recompute", ["none", "sac"])
def test_segment_continuation_uses_old_hidden_after_optimizer_updates(recompute):
    torch.manual_seed(9)
    actual_model = model_for_segment()
    expected_model = copy.deepcopy(actual_model)
    if recompute == "sac":
        actual_model.gradient_checkpointing_enable()
        apply_recompute_policy(
            actual_model,
            build_policy(
                enabled=True,
                enable_reentrant=False,
                early_stop=True,
                recompute_layers=-1,
                selective_recompute_layers=2,
            ),
        )
    # SGD keeps this a stale-hidden-state reference check: Adam's normalization
    # can amplify harmless near-zero FP32 reduction differences across updates.
    actual_optim = torch.optim.SGD([p for p in actual_model.parameters() if p.requires_grad], lr=0.003)
    expected_optim = torch.optim.SGD([p for p in expected_model.parameters() if p.requires_grad], lr=0.003)
    actual_memory = expected_memory = None
    for step, (prefix, clean, anchors) in enumerate(
        [
            ((-4, -3, -2, -1, 0), (1, 2, 3, 4), (0, 2, 4)),
            ((), (5, 6, 7, 8), (6, 8)),
            ((), (9, 10), (10,)),
        ]
    ):
        sample = segment_payload(prefix=prefix, clean=clean, anchors=anchors)
        actual = actual_model(stream_segment=sample, stream_memory=actual_memory)
        expected = reference(expected_model, sample, expected_memory)
        torch.testing.assert_close(actual.video, expected.video, atol=2e-6, rtol=2e-5)
        sum(actual.loss.values()).backward()
        sum(expected.loss.values()).backward()
        assert_gradients(gradients(actual_model), gradients(expected_model))
        actual_optim.step()
        expected_optim.step()
        actual_optim.zero_grad(set_to_none=True)
        expected_optim.zero_grad(set_to_none=True)
        actual_memory, expected_memory = actual.segment_memory, expected.segment_memory
        assert len(actual_memory.frame_ids) == 5
        assert all(value.grad_fn is None and not value.requires_grad for value in actual_memory.layer_inputs)
        assert actual.segment_stats["reused_history_latents"] == (0 if step == 0 else 5)
    assert actual_memory.frame_ids == (6, 7, 8, 9, 10)


def test_fixed_weight_segments_match_full_episode_unroll_at_detach_boundaries():
    torch.manual_seed(27)
    actual_model = model_for_segment()
    full_model = copy.deepcopy(actual_model)
    samples = [
        segment_payload(clean=(1, 2), anchors=(0, 2)),
        segment_payload(prefix=(), clean=(3, 4, 5, 6), anchors=(4, 6)),
    ]
    actual_video, actual_action, memory, total = [], [], None, 0.0
    for sample in samples:
        output = actual_model(stream_segment=sample, stream_memory=memory)
        loss = sum(output.loss.values())
        total += loss.detach()
        loss.backward()
        actual_video.append(output.video.detach())
        actual_action.append(output.action.detach())
        memory = output.segment_memory
    complete, owners = combine_segments(samples)
    expected = reference(full_model, complete, owners=owners)
    # Equal decision counts per segment: sum of segment means is twice the
    # complete episode's decision mean. No target is omitted at a boundary.
    full_loss = len(samples) * sum(expected.loss.values())
    torch.testing.assert_close(total, full_loss.detach(), atol=2e-6, rtol=2e-5)
    full_loss.backward()
    torch.testing.assert_close(torch.cat(actual_video), expected.video, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.cat(actual_action), expected.action, atol=2e-6, rtol=2e-5)
    assert_gradients(gradients(actual_model), gradients(full_model))


def test_segment_checkpoint_calls_each_layer_once_and_skips_cold_mlp_recompute():
    model = model_for_segment()
    model.gradient_checkpointing_enable()
    apply_recompute_policy(
        model,
        build_policy(
            enabled=True, enable_reentrant=False, early_stop=True, recompute_layers=-1, selective_recompute_layers=2
        ),
    )
    calls = [dict(active=0, cold=0) for _ in model.layers]
    sample = segment_payload()
    old_rows = len(sample["prefix_video_tokens"])

    def record(index):
        def hook(module, args):
            calls[index]["cold" if len(args[0]) == old_rows else "active"] += 1

        return hook

    handles = [layer.mlp_moe_gen.register_forward_pre_hook(record(i)) for i, layer in enumerate(model.layers)]
    try:
        output = model(stream_segment=sample)
        assert calls == [dict(active=1, cold=1)] * len(model.layers)
        sum(output.loss.values()).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [dict(active=2, cold=1)] * len(model.layers)


def test_segment_later_decision_trains_earlier_clean_but_not_previous_segment():
    torch.manual_seed(43)
    model = model_for_segment()
    first = segment_payload()
    first["clean_video_tokens"].requires_grad_()
    one = model(stream_segment=first)
    # Last decision sees early clean within the same unroll and sends gradients there.
    one.action[-4:].square().sum().backward()
    assert first["clean_video_tokens"].grad[:2].abs().sum() > 0
    before = first["clean_video_tokens"].grad.clone()
    second = segment_payload(prefix=(), clean=(5, 6, 7, 8), anchors=(6, 8))
    two = model(stream_segment=second, stream_memory=one.segment_memory)
    sum(two.loss.values()).backward()
    assert torch.equal(first["clean_video_tokens"].grad, before)


def test_segment_current_queries_do_not_leak_between_decisions_or_into_conditions():
    torch.manual_seed(4)
    model = model_for_segment().eval()
    sample = segment_payload()
    with torch.no_grad():
        initial = model(stream_segment=sample)
        changed = copy.deepcopy(sample)
        changed["future_video_tokens"][:4] += 100
        changed["action_tokens"][:4] += 100
        altered = model(stream_segment=changed)
        torch.testing.assert_close(initial.video[4:], altered.video[4:], atol=0, rtol=0)
        torch.testing.assert_close(initial.action[4:], altered.action[4:], atol=0, rtol=0)
        for left, right in zip(initial.segment_memory.layer_inputs, altered.segment_memory.layer_inputs):
            assert torch.equal(left, right)
        changed = copy.deepcopy(sample)
        # FD decision1 has conditioned actions: perturb its noisy video only.
        changed["future_video_tokens"][4:8] += 100
        altered = model(stream_segment=changed)
        assert torch.equal(initial.action[4:8], altered.action[4:8])
        changed = copy.deepcopy(sample)
        changed["clean_video_tokens"][-4:] += 100
        altered = model(stream_segment=changed)
        assert torch.equal(initial.video[:8], altered.video[:8])
        assert torch.equal(initial.action[:8], altered.action[:8])
        changed = copy.deepcopy(sample)
        changed["action_tokens"][8:] += 100
        altered = model(stream_segment=changed)
        assert torch.equal(initial.video[8:], altered.video[8:])


def test_segment_warm_single_decision_and_repeated_anchor_have_empty_new_clean():
    model = model_for_segment().eval()
    sample = segment_payload(clean=(), anchors=(0,))
    with torch.no_grad():
        first = model(stream_segment=sample)
        continued = segment_payload(prefix=(), clean=(), anchors=(0,))
        continued["future_video_tokens"] = sample["future_video_tokens"]
        continued["action_tokens"] = sample["action_tokens"]
        continued["state_tokens"] = sample["state_tokens"]
        second = model(stream_segment=continued, stream_memory=first.segment_memory)
    torch.testing.assert_close(first.video, second.video, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(first.action, second.action, atol=2e-6, rtol=2e-5)
    assert second.segment_stats["new_clean_latents"] == 0
    assert second.segment_memory is first.segment_memory


def test_segment_single_observation_without_cold_prefix_and_backend_fallback():
    sample = segment_payload(prefix=(), clean=(0,), anchors=(0,))
    sample["history_latents"] = 1
    sample.pop("attention_backend")
    output = model_for_segment()(stream_segment=sample, attention_backend="grouped")
    assert output.segment_memory.frame_ids == (0,)
    assert output.segment_stats["cold_prefix_latents"] == 0
    assert "attention_backend" not in sample
    sum(output.loss.values()).backward()


def test_segment_reuses_padding_and_condition_supervision_masks_without_mutation():
    model = model_for_segment()
    sample = segment_payload()
    masks = sample["targets"]["loss_masks"]
    masks["video_loss_mask"][-2:] = False
    masks["action_loss_mask"][-2:] = False
    before = {name: value.clone() for name, value in masks.items()}
    output = model(stream_segment=sample)
    changed = copy.deepcopy(sample)
    changed["targets"]["video"][-1][:, -1] += 10000
    changed["targets"]["action"][-1][-2:] += 10000
    adjusted = model(stream_segment=changed)
    for name in output.loss:
        assert torch.equal(output.loss[name], adjusted.loss[name])
    assert all(torch.equal(masks[name], value) for name, value in before.items())


@pytest.mark.parametrize("fault", ["context", "text", "gap", "graph", "geometry", "model", "prefix"])
def test_segment_invalid_memory_is_rejected_instead_of_rebuilding(fault):
    model = model_for_segment()
    first = model(stream_segment=segment_payload())
    memory = first.segment_memory
    second = segment_payload(prefix=(), clean=(5, 6, 7, 8), anchors=(6, 8))
    if fault == "context":
        second["context_id"] = "another episode"
    elif fault == "text":
        second["input_ids"] = torch.tensor([1, 2, 4])
    elif fault == "gap":
        second["clean_frame_ids"] = (6, 7, 8, 9)
    elif fault == "graph":
        memory = replace(
            memory, layer_inputs=(memory.layer_inputs[0].clone().requires_grad_(), *memory.layer_inputs[1:])
        )
    elif fault == "geometry":
        memory = replace(memory, history_latents=4)
    elif fault == "model":
        memory = replace(memory, model_id=-1)
    else:
        second["prefix_frame_ids"] = (-4, -3, -2, -1, 0)
        second["prefix_video_tokens"] = torch.zeros(10, 8)
        second["prefix_position_ids"] = positions(second["prefix_frame_ids"])
    with pytest.raises(ValueError):
        model(stream_segment=second, stream_memory=memory)


@pytest.mark.parametrize("fault", ["shape", "not_tensor", "integer", "grad_leaf", "grad_nonleaf"])
def test_segment_malformed_layer_memory_is_rejected_before_projection(fault):
    model = model_for_segment()
    memory = model(stream_segment=segment_payload()).segment_memory
    value = memory.layer_inputs[0]
    if fault == "shape":
        value = value[:-1]
    elif fault == "not_tensor":
        value = value.tolist()
    elif fault == "integer":
        value = value.to(torch.long)
    elif fault == "grad_leaf":
        value = value.clone().requires_grad_()
        assert value.is_leaf and value.requires_grad and value.grad_fn is None
    else:
        value = value.clone().requires_grad_() * 2
        assert not value.is_leaf and value.grad_fn is not None
    memory = replace(memory, layer_inputs=(value, *memory.layer_inputs[1:]))
    continued = segment_payload(prefix=(), clean=(5, 6, 7, 8), anchors=(6, 8))
    with pytest.raises(ValueError, match="Episode layer memory must be detached"):
        _prepare(model, continued, memory)

    def poison_projection(module, args):
        raise AssertionError("Malformed memory reached model projection before preflight validation")

    handle = model.proj_in.register_forward_pre_hook(poison_projection)
    try:
        with pytest.raises(ValueError, match="Episode layer memory must be detached"):
            model(stream_segment=continued, stream_memory=memory)
    finally:
        handle.remove()
