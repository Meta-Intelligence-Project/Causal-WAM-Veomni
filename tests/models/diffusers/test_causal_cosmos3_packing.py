"""Final causal registration, native token budgets and attention boundaries."""

from dataclasses import asdict

import pytest
import torch

from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.data.causal_cosmos3.processing import Geometry
from veomni.models.diffusers.causal_cosmos3 import modeling_causal_cosmos3, mrope, packing
from veomni.models.diffusers.causal_cosmos3.attention import dense_generation_mask, prepare_attention
from veomni.models.diffusers.causal_cosmos3.layers import rotary_tables
from veomni.models.diffusers.causal_cosmos3.modeling_causal_cosmos3 import Cosmos3Model, patchify_video
from veomni.models.diffusers.causal_cosmos3.packing import (
    SampleSpans,
    build_packed_layout,
    build_position_ids,
    pack_model_inputs,
)
from veomni.utils.device import IS_NPU_AVAILABLE


def sample_input(task="policy", *, actions=32, text=2, height=2, width=2, history=1):
    geometry = Geometry()
    frames = 1 + actions // geometry.actions_per_latent
    return dict(
        text_token_ids=torch.arange(text),
        video_latents=torch.zeros(2, frames, height, width),
        action=torch.zeros(actions, 4),
        states=torch.zeros(actions // geometry.actions_per_block, 4),
        block_size=geometry.actions_per_block // geometry.actions_per_latent,
        history_blocks=history,
        action_frame_ids=1 + torch.arange(actions) // geometry.actions_per_latent,
        state_latent_indexes=torch.arange(actions // geometry.actions_per_block)
        * (geometry.actions_per_block // geometry.actions_per_latent),
        conditioning_fps=7.5,
        conditioning_fps_action=30.0,
        condition_frame_indexes_vision=list(range(frames)) if task == "id" else [0],
        condition_frame_indexes_action=list(range(actions)) if task == "fd" else [],
        token_length=sum(geometry.token_layout(actions, height * 16, width * 16, text).values()),
    )


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_final_spans_and_roles(task):
    sample = sample_input(task)
    layout, (spans,), packed_length = build_packed_layout([sample])
    assert packed_length == len(layout.sample_ids) == 41
    assert asdict(spans) == dict(
        text=slice(0, 2),
        clean_video=slice(2, 5),
        current_video=slice(5, 8),
        action=slice(8, 40),
        state=slice(40, 41),
    )
    assert layout.sample_ids == (0,) * 41
    assert layout.block_ids == (-1, -1, -1, 0, 0, -1, 0, 0) + (0,) * 33
    assert layout.modalities == (0,) * 2 + (1,) * 6 + (2,) * 32 + (3,)
    assert layout.roles[spans.clean_video] == (0, 0, 0)
    assert layout.roles[spans.current_video] == ((2, 2, 2) if task == "id" else (2, 1, 1))
    assert layout.roles[spans.action] == ((2,) * 32 if task == "fd" else (1,) * 32)
    assert layout.roles[spans.state] == (3,)


def test_variable_pack_offsets_and_patch_padding():
    first = sample_input()
    second = sample_input("id", actions=64, text=5, height=3, width=5, history=2)
    layout, spans, packed_length = build_packed_layout([first, second])
    assert packed_length == len(layout.sample_ids) == 172
    assert asdict(spans[1]) == dict(
        text=slice(41, 46),
        clean_video=slice(46, 76),
        current_video=slice(76, 106),
        action=slice(106, 170),
        state=slice(170, 172),
    )
    assert layout.sample_ids == (0,) * 41 + (1,) * 131
    assert layout.history_blocks == (1, 2)
    assert layout.block_ids[spans[1].clean_video] == (-1,) * 6 + (0,) * 12 + (1,) * 12
    assert layout.block_ids[spans[1].state] == (0, 1)
    assert len(layout.sample_ids) == first["token_length"] + second["token_length"]
    single, _, _ = build_packed_layout([second])
    section = slice(spans[1].text.start, spans[1].state.stop)
    for key in ("block_ids", "roles", "modalities"):
        assert getattr(layout, key)[section] == getattr(single, key)


def test_action_blocks_follow_explicit_frame_ids():
    sample = sample_input(actions=64)
    sample["action_frame_ids"] = torch.repeat_interleave(torch.arange(1, 5), torch.tensor([8, 8, 24, 24]))
    layout, (spans,), _ = build_packed_layout([sample])
    assert layout.block_ids[spans.action] == (0,) * 16 + (1,) * 48


def test_partial_final_block_keeps_its_state():
    sample = sample_input(actions=64)
    sample["video_latents"] = sample["video_latents"][:, :4]
    sample["action"] = sample["action"][:48]
    sample["action_frame_ids"] = sample["action_frame_ids"][:48]
    sample["token_length"] -= 2 + 16
    layout, (spans,), _ = build_packed_layout([sample])
    assert layout.block_ids[spans.current_video] == (-1, 0, 0, 1)
    assert layout.block_ids[spans.state] == (0, 1)


def test_registration_only_reads_payload_shapes():
    sample = sample_input()
    expected, _, expected_length = build_packed_layout([sample])
    for key in ("text_token_ids", "video_latents", "action", "states"):
        sample[key] = sample[key].to("meta")
    before = sample.copy()
    actual, _, actual_length = build_packed_layout([sample])
    assert actual == expected
    assert actual_length == expected_length
    assert sample.keys() == before.keys()
    assert all(value is before[key] for key, value in sample.items())


def test_registered_layout_preserves_causal_and_sample_boundaries():
    layout, spans, _ = build_packed_layout([sample_input(actions=64), sample_input("id")])
    plan = prepare_attention(layout, "cpu", "grouped", max_kv_tokens=13)
    visible = dense_generation_mask(layout, "cpu")
    gen = plan.gen_indexes.tolist()
    action = visible[gen.index(spans[0].action.start)]
    assert action[spans[0].text].all()
    assert not action[spans[0].clean_video.start + 1 : spans[0].clean_video.stop].any()
    assert not action[spans[1].text.start :].any()
    state = visible[gen.index(spans[0].state.start)]
    assert state.sum() == 3  # Two text tokens and this block's state.
    assert state[spans[0].state.start]
    # Reconstruct all grouped visibility independently of the dense oracle.
    grouped = torch.zeros_like(visible)
    for chunk in plan.chunks:
        qstart = kstart = 0
        for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            qi, ki = chunk.query_indexes[qstart:qend], chunk.kv_indexes[kstart:kend]
            grouped[qi[:, None], ki[None, :]] = True
            qstart, kstart = qend, kend
    torch.testing.assert_close(grouped, visible)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("text_token_ids", torch.empty(0, dtype=torch.long), "text_token_ids"),
        ("video_latents", torch.zeros(2, 0, 2, 2), "video_latents"),
        ("block_size", 0, "block_size"),
        ("history_blocks", 0, "history_blocks"),
        ("states", torch.zeros(0, 4), "one state"),
        ("action_frame_ids", torch.ones(31, dtype=torch.long), "match action"),
        ("action_frame_ids", torch.zeros(32, dtype=torch.long), "after the independent"),
        ("action_frame_ids", torch.full((32,), 3), "after the independent"),
        ("action_frame_ids", torch.ones(32), "integer"),
        ("action_frame_ids", torch.ones(32, device="meta", dtype=torch.long), "CPU"),
        ("condition_frame_indexes_vision", [], "independent first"),
        ("condition_frame_indexes_vision", [0, 3], "valid frame"),
        ("condition_frame_indexes_action", [32], "valid action"),
        ("token_length", 42, "token_length"),
    ],
)
def test_invalid_registration_metadata(key, value, match):
    sample = sample_input()
    sample[key] = value
    with pytest.raises(ValueError, match=match):
        build_packed_layout([sample])


def test_empty_pack_and_invalid_patch_size():
    with pytest.raises(ValueError, match="nonempty"):
        build_packed_layout([])
    with pytest.raises(ValueError, match="positive patch_size"):
        build_packed_layout([sample_input()], patch_size=0)


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_positions_follow_time_geometry_and_spatial_patch_order(task):
    sample = sample_input(task, actions=64, text=5, height=3, width=5)
    layout, (span,), packed_length = build_packed_layout([sample])
    positions = build_position_ids([sample], (span,), packed_length)
    assert positions.shape == (3, len(layout.sample_ids))
    assert positions.dtype == torch.float32 and positions.device.type == "cpu"
    torch.testing.assert_close(positions[:, span.text], torch.arange(5).float().expand(3, -1), atol=0, rtol=0)

    # 5 个 latent，每个 latent 有 2x3 个 patch；独立首 latent 的相对时间为 0。
    offset = 15005
    times = torch.tensor([offset + time for time in (0, 3.2, 6.4, 9.6, 12.8)])
    spatial = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], dtype=torch.float32)
    video = positions[:, span.clean_video].reshape(3, 5, 6)
    torch.testing.assert_close(video[0], times[:, None].expand(-1, 6), atol=0, rtol=0)
    torch.testing.assert_close(video[1:], spatial[:, None, :].expand(-1, 5, -1), atol=0, rtol=0)
    torch.testing.assert_close(positions[:, span.clean_video], positions[:, span.current_video], atol=0, rtol=0)
    actions = torch.tensor([offset + 0.2 * (step + 1) for step in range(64)])
    torch.testing.assert_close(positions[0, span.action], actions, atol=0, rtol=0)
    torch.testing.assert_close(positions[0, span.state], times[[0, 2]], atol=0, rtol=0)
    assert torch.count_nonzero(positions[1:, span.action]) == 0
    assert torch.count_nonzero(positions[1:, span.state]) == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(patch_size=1),
        dict(patch_size=2, base_fps=30.0, temporal_compression=8, temporal_modality_margin=700),
        dict(patch_size=3, temporal_modality_margin=0),
    ],
)
def test_positions_are_independent_of_pack_neighbors_and_payload_values(options):
    samples = [sample_input(), sample_input("id", actions=64, text=5, height=3, width=5)]
    samples[0].update(conditioning_fps=15.0, conditioning_fps_action=60.0)
    samples[1].update(conditioning_fps=29.97 / 4, conditioning_fps_action=29.97)
    patch_size = options["patch_size"]
    for sample in samples:
        _, frames, height, width = sample["video_latents"].shape
        patches = ((height + patch_size - 1) // patch_size) * ((width + patch_size - 1) // patch_size)
        sample["token_length"] = (
            len(sample["text_token_ids"]) + 2 * frames * patches + len(sample["action"]) + len(sample["states"])
        )
    _, spans, packed_length = build_packed_layout(samples, patch_size=patch_size)
    packed = build_position_ids(samples, spans, packed_length, **options)
    for sample, span in zip(samples, spans, strict=True):
        # payload 放在 meta 设备仍可构造坐标，证明只读取形状和 CPU 元数据。
        meta = dict(sample)
        for key in ("text_token_ids", "video_latents", "action", "states"):
            meta[key] = meta[key].to(device="meta")
        _, single_spans, single_length = build_packed_layout([meta], patch_size=patch_size)
        single = build_position_ids([meta], single_spans, single_length, **options)
        torch.testing.assert_close(packed[:, span.text.start : span.state.stop], single, atol=0, rtol=0)


def test_state_positions_follow_explicit_latent_anchors():
    sample = sample_input(actions=64)
    sample["state_latent_indexes"] = torch.tensor([0, 3])
    _, (span,), packed_length = build_packed_layout([sample])
    positions = build_position_ids([sample], (span,), packed_length)
    video = positions[:, span.clean_video]
    torch.testing.assert_close(positions[:, span.state], video[:, [0, 3]], atol=0, rtol=0)


@pytest.mark.parametrize(
    "sample_order,segment_order",
    [
        ((0, 1), ("state", "text", "action", "clean_video", "current_video")),
        ((1, 0), ("text", "clean_video", "current_video", "action", "state")),
    ],
)
def test_positions_follow_reordered_sample_and_modality_spans(sample_order, segment_order):
    samples = [sample_input(), sample_input("id", actions=64, text=5, height=3, width=5)]
    _, original_spans, packed_length = build_packed_layout(samples)
    expected = build_position_ids(samples, original_spans, packed_length)
    sections = [{}, {}]
    cursor = 0
    for sample_id in sample_order:
        for name in segment_order:
            original = getattr(original_spans[sample_id], name)
            length = original.stop - original.start
            sections[sample_id][name] = slice(cursor, cursor + length)
            cursor += length
    # spans 仍与 samples 一一对应；实际存放顺序由各字段的全局 slice 决定。
    reordered_spans = tuple(SampleSpans(**section) for section in sections)
    assert cursor == packed_length
    actual = build_position_ids(samples, reordered_spans, packed_length)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    for original, reordered in zip(original_spans, reordered_spans, strict=True):
        for name in segment_order:
            torch.testing.assert_close(
                actual[:, getattr(reordered, name)], expected[:, getattr(original, name)], atol=0, rtol=0
            )


def test_position_creation_ignores_default_float_dtype():
    sample = sample_input()
    _, spans, packed_length = build_packed_layout([sample])
    expected = build_position_ids([sample], spans, packed_length)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual = build_position_ids([sample], spans, packed_length)
    finally:
        torch.set_default_dtype(previous)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_position_helpers_use_cpu_under_a_different_default_device():
    sample = sample_input()
    _, spans, packed_length = build_packed_layout([sample])
    expected = build_position_ids([sample], spans, packed_length)
    with torch.device("meta"):
        actual = build_position_ids([sample], spans, packed_length)
    assert actual.device.type == "cpu" and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mrope_helpers_can_be_replaced_independently_of_packing(monkeypatch):
    sample = sample_input()
    _, (span,), packed_length = build_packed_layout([sample])
    expected = build_position_ids([sample], (span,), packed_length)
    original_text = mrope.get_3d_mrope_ids_text_tokens
    original_vae = mrope.get_3d_mrope_ids_vae_tokens

    def shifted_text(*args, **kwargs):
        positions, next_offset = original_text(*args, **kwargs)
        return positions + 100, next_offset + 100

    def shifted_vae(*args, **kwargs):
        positions, next_offset = original_vae(*args, **kwargs)
        positions[0] += 7
        return positions, next_offset + 7

    monkeypatch.setattr(mrope, "get_3d_mrope_ids_text_tokens", shifted_text)
    monkeypatch.setattr(mrope, "get_3d_mrope_ids_vae_tokens", shifted_vae)
    actual = build_position_ids([sample], (span,), packed_length)
    # 替换文本算法的返回游标后，生成模态起点也随之变化；state 继续引用视频锚点。
    expected[:, span.text] += 100
    for section in (span.clean_video, span.current_video, span.action, span.state):
        expected[0, section] += 107
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_replaced_mrope_helpers_must_preserve_fp32(monkeypatch, dtype):
    sample = sample_input()
    _, spans, packed_length = build_packed_layout([sample])
    original = mrope.get_3d_mrope_ids_vae_tokens

    def quantized_positions(*args, **kwargs):
        positions, next_offset = original(*args, **kwargs)
        return positions.to(dtype=dtype), next_offset

    monkeypatch.setattr(mrope, "get_3d_mrope_ids_vae_tokens", quantized_positions)
    with pytest.raises(ValueError, match="helpers must return float32"):
        build_position_ids([sample], spans, packed_length)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend FP32 coordinates"))],
)
def test_fractional_positions_remain_distinct_through_autocast_and_rope(device):
    sample = sample_input()
    _, (span,), packed_length = build_packed_layout([sample])
    expected = build_position_ids([sample], (span,), packed_length)
    config = tiny_config()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        positions = build_position_ids([sample], (span,), packed_length).to(device=device)
        cos, sin = rotary_tables(positions, config, torch.bfloat16)
    assert positions.dtype == torch.float32
    torch.testing.assert_close(positions.cpu(), expected, atol=0, rtol=0)
    assert (positions[0, span.action].diff() > 0).all()
    # 大约 15000 的偏移下，0.2 的间隔应在坐标和旋转表中都保留下来。
    first = span.action.start
    assert not torch.equal(cos[first], cos[first + 1]) or not torch.equal(sin[first], sin[first + 1])
    expected_cos, expected_sin = rotary_tables(positions, config, torch.float32)
    torch.testing.assert_close(cos, expected_cos.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_rejects_positions_quantized_before_entry(dtype):
    sample = sample_input()
    _, spans, packed_length = build_packed_layout([sample])
    positions = build_position_ids([sample], spans, packed_length)
    with pytest.raises(ValueError, match="float32 precision before RoPE"):
        rotary_tables(positions.to(dtype=dtype), tiny_config(), torch.bfloat16)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("conditioning_fps", 0.0, "FPS"),
        ("conditioning_fps", float("nan"), "FPS"),
        ("conditioning_fps_action", float("inf"), "FPS"),
        ("conditioning_fps_action", -30.0, "FPS"),
        ("state_latent_indexes", torch.tensor([-1]), "valid video latents"),
        ("state_latent_indexes", torch.tensor([3]), "valid video latents"),
        ("state_latent_indexes", torch.tensor([0, 1]), "match states"),
    ],
)
def test_invalid_position_metadata(key, value, match):
    sample = sample_input()
    _, spans, packed_length = build_packed_layout([sample])
    sample[key] = value
    with pytest.raises(ValueError, match=match):
        build_position_ids([sample], spans, packed_length)


def test_constructed_positions_reach_network_rope_and_backward(monkeypatch):
    sample = sample_input()
    layout, spans, packed_length = build_packed_layout([sample])
    seen = []
    original = modeling_causal_cosmos3.rotary_tables

    def checked_rotary(positions, config, dtype):
        seen.append(positions.dtype)
        return original(positions, config, dtype)

    monkeypatch.setattr(modeling_causal_cosmos3, "rotary_tables", checked_rotary)
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config())
    video = torch.randn_like(sample["video_latents"], requires_grad=True)
    tokens = patchify_video(video, 2)
    output = model(
        input_ids=sample["text_token_ids"],
        video_tokens=torch.cat((tokens, tokens)),
        video_timesteps=torch.zeros(2 * len(tokens)),
        action_tokens=torch.randn_like(sample["action"]),
        action_timesteps=torch.full((len(sample["action"]),), 400.0),
        action_mask=torch.ones_like(sample["action"], dtype=torch.bool),
        state_tokens=sample["states"],
        state_mask=torch.ones_like(sample["states"], dtype=torch.bool),
        position_ids=build_position_ids([sample], spans, packed_length),
        layout=layout,
        attention_backend="grouped",
    )
    (output.video.square().mean() + output.action.square().mean()).backward()
    assert seen == [torch.float32]
    assert torch.isfinite(video.grad).all() and video.grad.abs().sum() > 0


def payload_sample(task="policy", **geometry):
    sample = sample_input(task, **geometry)
    for key in ("video_latents", "action", "states"):
        sample[key] = torch.arange(sample[key].numel(), dtype=torch.float32).reshape(sample[key].shape) / 100
    sample.update(
        action_mask=torch.tensor([True, False, True, True]),
        state_mask=torch.tensor([True, True, False, True]),
        action_valid_steps=torch.arange(len(sample["action"])) < len(sample["action"]) - 7,
        num_valid_video_latents=sample["video_latents"].shape[1] - 1,
    )
    return sample


def latent_timesteps(sample):
    return torch.tensor(
        [900.0]
        + [
            125.5 + 250 * ((frame - 1) // sample["block_size"]) for frame in range(1, sample["video_latents"].shape[1])
        ],
        dtype=torch.float32,
    )


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_payloads_restore_conditions_and_retain_input_values(task, dtype):
    sample = payload_sample(task, actions=64, height=3, width=5)
    for key in ("video_latents", "action", "states"):
        sample[key] = sample[key].to(dtype=dtype)
    current_video = sample["video_latents"] + 10
    current_action = sample["action"] + 20
    original = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    inputs, (span,), _ = pack_model_inputs(
        [sample], [current_video], [current_action], timesteps=[latent_timesteps(sample)]
    )
    # 独立检查每个视频 patch 的源数据及补零顺序，避免参考也调用 patchify。
    rows = []
    for current in (False, True):
        for frame in range(5):
            source = (
                sample["video_latents"]
                if not current or frame in sample["condition_frame_indexes_vision"]
                else current_video
            )
            for top in range(0, 3, 2):
                for left in range(0, 5, 2):
                    rows.append(
                        torch.cat(
                            [
                                source[:, frame, h, w] if h < 3 and w < 5 else torch.zeros(2, dtype=dtype)
                                for h in range(top, top + 2)
                                for w in range(left, left + 2)
                            ]
                        )
                    )
    torch.testing.assert_close(inputs["video_tokens"], torch.stack(rows), atol=0, rtol=0)
    torch.testing.assert_close(
        inputs["action_tokens"], sample["action"] if task == "fd" else current_action, atol=0, rtol=0
    )
    torch.testing.assert_close(inputs["state_tokens"], sample["states"], atol=0, rtol=0)
    assert inputs["video_tokens"].dtype == inputs["action_tokens"].dtype == dtype
    assert inputs["input_ids"].dtype == torch.int64 and inputs["position_ids"].dtype == torch.float32
    assert inputs["action_mask"].shape == sample["action"].shape
    assert torch.equal(inputs["action_mask"], sample["action_mask"].expand_as(sample["action"]))
    assert torch.equal(inputs["state_mask"], sample["state_mask"].expand_as(sample["states"]))
    # 时间 padding 不删 token、不关闭特征通道，也不在组装时清零无效特征值。
    assert inputs["action_mask"][-1, 0] and not sample["action_valid_steps"][-1]
    assert inputs["action_tokens"][-1, 1] != 0
    assert span.current_video.stop - span.current_video.start == 30
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(sample[key], value, atol=0, rtol=0)
        else:
            assert sample[key] == value
    torch.testing.assert_close(current_video, sample["video_latents"] + 10, atol=0, rtol=0)
    torch.testing.assert_close(current_action, sample["action"] + 20, atol=0, rtol=0)


def test_payloads_follow_reordered_final_spans(monkeypatch):
    samples = [payload_sample(), payload_sample("id", actions=64, text=5, height=3, width=5)]
    current_videos = [sample["video_latents"] + 10 for sample in samples]
    current_actions = [sample["action"] + 20 for sample in samples]
    times = [latent_timesteps(sample) for sample in samples]
    expected, original_spans, expected_masks = pack_model_inputs(
        samples, current_videos, current_actions, timesteps=times
    )
    old_layout, _, length = build_packed_layout(samples)
    sections, permutation, cursor = [{}, {}], [], 0
    for sample_id in (1, 0):
        for name in ("state", "action", "current_video", "text", "clean_video"):
            section = getattr(original_spans[sample_id], name)
            count = section.stop - section.start
            sections[sample_id][name] = slice(cursor, cursor + count)
            permutation.extend(range(section.start, section.stop))
            cursor += count
    reordered_spans = tuple(SampleSpans(**section) for section in sections)
    reordered_layout = packing.CausalLayout(
        **{
            key: tuple(getattr(old_layout, key)[i] for i in permutation)
            for key in ("sample_ids", "block_ids", "roles", "modalities")
        },
        history_blocks=old_layout.history_blocks,
    )
    monkeypatch.setattr(
        packing, "build_packed_layout", lambda *args, **kwargs: (reordered_layout, reordered_spans, length)
    )
    actual, spans, actual_masks = pack_model_inputs(samples, current_videos, current_actions, timesteps=times)
    assert spans == reordered_spans and actual["layout"] == reordered_layout and cursor == length
    torch.testing.assert_close(actual["position_ids"], expected["position_ids"][:, permutation], atol=0, rtol=0)
    for modality, keys in (
        (0, ("input_ids",)),
        (1, ("video_tokens", "video_timesteps")),
        (2, ("action_tokens", "action_mask", "action_timesteps")),
        (3, ("state_tokens", "state_mask")),
    ):
        source_rows = {i: row for row, i in enumerate(i for i, m in enumerate(old_layout.modalities) if m == modality)}
        rows = [source_rows[i] for i in permutation if old_layout.modalities[i] == modality]
        for key in keys:
            torch.testing.assert_close(actual[key], expected[key][rows], atol=0, rtol=0)
        if modality in (1, 2):
            key = "video_loss_mask" if modality == 1 else "action_loss_mask"
            torch.testing.assert_close(actual_masks[key], expected_masks[key][rows], atol=0, rtol=0)


def test_sparse_conditions_preserve_clean_and_current_gradients():
    sample = payload_sample()
    sample["condition_frame_indexes_vision"] = [0, 2]
    sample["condition_frame_indexes_action"] = [1, 7, 31]
    video = sample["video_latents"].requires_grad_()
    action = sample["action"].requires_grad_()
    states = sample["states"].requires_grad_()
    current_video = torch.full_like(video, 10, requires_grad=True)
    current_action = torch.full_like(action, 20, requires_grad=True)
    inputs, _, _ = pack_model_inputs([sample], [current_video], [current_action], timesteps=[latent_timesteps(sample)])
    sum(inputs[key].sum() for key in ("video_tokens", "action_tokens", "state_tokens")).backward()
    # clean video 全部保留一份梯度，current 的条件帧额外回传到 clean。
    torch.testing.assert_close(video.grad[:, 0], torch.full_like(video[:, 0], 2))
    torch.testing.assert_close(video.grad[:, 1], torch.ones_like(video[:, 1]))
    torch.testing.assert_close(video.grad[:, 2], torch.full_like(video[:, 2], 2))
    assert current_video.grad[:, [0, 2]].count_nonzero() == 0
    assert current_video.grad[:, 1].eq(1).all()
    assert action.grad[[1, 7, 31]].eq(1).all() and current_action.grad[[1, 7, 31]].count_nonzero() == 0
    assert action.grad[0].count_nonzero() == 0 and current_action.grad[0].eq(1).all()
    assert states.grad.eq(1).all()


@pytest.mark.parametrize(
    "failure", ["count", "video_shape", "action_shape", "video_device", "mask_dtype", "mask_shape"]
)
def test_packing_rejects_misaligned_payloads(failure):
    sample = payload_sample()
    videos, actions = [sample["video_latents"]], [sample["action"]]
    match = "shapes must match"
    if failure == "count":
        videos, match = [], "per sample"
    elif failure == "video_shape":
        videos = [videos[0][:, :-1]]
    elif failure == "action_shape":
        actions = [actions[0][:-1]]
    elif failure == "video_device":
        videos, match = [videos[0].to(device="meta")], "share one device"
    elif failure == "mask_dtype":
        sample["action_mask"], match = sample["action_mask"].float(), "boolean valid-feature"
    else:
        sample["state_mask"], match = torch.ones_like(sample["states"], dtype=torch.bool), "boolean valid-feature"
    with pytest.raises(ValueError, match=match):
        pack_model_inputs([sample], videos, actions, timesteps=[latent_timesteps(sample)])


def test_mixed_pack_matches_independent_sample_outputs_and_parameter_gradients():
    samples = [
        payload_sample("fd"),
        payload_sample("id", text=5),
        payload_sample("policy", actions=64, height=3, width=5),
    ]
    videos = [sample["video_latents"] + 0.25 for sample in samples]
    actions = [sample["action"] + 0.5 for sample in samples]
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config())

    def forward(group, current_videos, current_actions):
        inputs, _, masks = pack_model_inputs(
            group, current_videos, current_actions, timesteps=[latent_timesteps(sample) for sample in group]
        )
        return model(**inputs, attention_backend="grouped"), masks

    def supervised_probe(output, masks):
        return (output.video.square() * masks["video_loss_mask"][:, None]).sum() + (
            output.action.square() * masks["action_loss_mask"]
        ).sum()

    packed, packed_masks = forward(samples, videos, actions)
    supervised_probe(packed, packed_masks).backward()
    expected_gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    separate = [
        forward([sample], [video], [action]) for sample, video, action in zip(samples, videos, actions, strict=True)
    ]
    torch.testing.assert_close(packed.video, torch.cat([out.video for out, _ in separate]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(packed.action, torch.cat([out.action for out, _ in separate]), atol=1e-6, rtol=1e-5)
    sum(supervised_probe(out, masks) for out, masks in separate).backward()
    for name, p in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(p.grad, expected_gradients[name], atol=2e-5, rtol=1e-4, msg=name)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend payload packing and FP32 positions")
@pytest.mark.parametrize("uniform", [False, True])
def test_npu_packing_preserves_fp32_positions_and_tnd_backward(monkeypatch, uniform):
    sample = payload_sample(actions=64, height=3, width=5)
    times = [torch.tensor([600.25]) if uniform else latent_timesteps(sample)]
    expected, _, expected_masks = pack_model_inputs(
        [sample], [sample["video_latents"] + 0.25], [sample["action"] + 0.5], timesteps=times
    )
    # 仅视频提前在设备上；其他 payload 和索引元数据保持 loader 的 CPU 格式。
    sample["video_latents"] = sample["video_latents"].to(device="npu").requires_grad_()
    current_video = (sample["video_latents"].detach() + 0.25).requires_grad_()
    current_action = (sample["action"] + 0.5).requires_grad_()

    def reject_cpu_copy(*args, **kwargs):
        raise AssertionError("Payload packing must not copy NPU data back to CPU")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", reject_cpu_copy)
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            inputs, _, loss_masks = pack_model_inputs(
                [sample], [current_video], [current_action], timesteps=[times[0].to(device="npu")]
            )
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "npu"
            torch.testing.assert_close(value.cpu(), expected[key], atol=0, rtol=0)
    for key, value in loss_masks.items():
        assert value.device.type == "npu" and value.dtype == torch.bool
        torch.testing.assert_close(value.cpu(), expected_masks[key], atol=0, rtol=0)
    assert inputs["video_timesteps"].dtype == inputs["action_timesteps"].dtype == torch.float32
    assert sample["action_frame_ids"].device.type == "cpu" and inputs["position_ids"].dtype == torch.float32
    assert inputs["position_ids"][0, inputs["layout"].modalities.index(2) :][:32].diff().gt(0).all()
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config()).to(device="npu", dtype=torch.bfloat16)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        output = model(**inputs, attention_backend="npu_tnd")
        loss = (output.video.float().square() * loss_masks["video_loss_mask"][:, None]).mean()
        loss = loss + (output.action.float().square() * loss_masks["action_loss_mask"]).mean()
    loss.backward()
    for value in (sample["video_latents"], current_video, current_action):
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


@pytest.mark.parametrize("task", ["fd", "id", "policy"])
def test_packed_timesteps_and_supervision_follow_task_and_padding(task):
    sample = payload_sample(task, actions=64, height=3, width=5)
    times = latent_timesteps(sample)
    inputs, _, masks = pack_model_inputs(
        [sample], [sample["video_latents"] + 10], [sample["action"] + 20], timesteps=[times]
    )
    expected_video_times = [0.0] * 30 + ([0.0] * 30 if task == "id" else [0.0] * 6 + [125.5] * 12 + [375.5] * 12)
    expected_action_times = [0.0] * 64 if task == "fd" else [125.5] * 32 + [375.5] * 32
    torch.testing.assert_close(inputs["video_timesteps"], torch.tensor(expected_video_times), atol=0, rtol=0)
    torch.testing.assert_close(inputs["action_timesteps"], torch.tensor(expected_action_times), atol=0, rtol=0)
    expected_video_mask = [False] * 30 + ([False] * 30 if task == "id" else [False] * 6 + [True] * 18 + [False] * 6)
    assert masks["video_loss_mask"].tolist() == expected_video_mask
    expected_action_mask = torch.zeros(64, 4, dtype=torch.bool)
    if task != "fd":
        expected_action_mask[:57, [0, 2, 3]] = True
    torch.testing.assert_close(masks["action_loss_mask"], expected_action_mask, atol=0, rtol=0)
    assert inputs["video_timesteps"].dtype == inputs["action_timesteps"].dtype == torch.float32
    assert set(masks) == {"video_loss_mask", "action_loss_mask"} and not set(masks).intersection(inputs)
    # 补齐步仍然有输入、正常的扩散时间和有效特征定义，只有监督 mask 为 False。
    assert inputs["action_mask"][-1, 0] and not masks["action_loss_mask"][-1].any()
    assert inputs["action_timesteps"][-1] == (0 if task == "fd" else 375.5)
    torch.testing.assert_close(times, latent_timesteps(sample), atol=0, rtol=0)


@pytest.mark.parametrize("uniform", [False, True])
def test_timestep_mapping_uses_explicit_frames_for_partial_blocks(uniform):
    sample = payload_sample(actions=64)
    sample["video_latents"] = sample["video_latents"][:, :4]
    sample["action"] = sample["action"][:48]
    sample["action_frame_ids"] = torch.tensor([1] * 8 + [2] * 8 + [3] * 32)
    sample["action_valid_steps"] = torch.arange(48) < 41
    sample["num_valid_video_latents"] = 4
    sample["token_length"] -= 2 + 16
    times = torch.tensor([600.25]) if uniform else latent_timesteps(sample)
    inputs, _, masks = pack_model_inputs([sample], [sample["video_latents"]], [sample["action"]], timesteps=[times])
    expected = [600.25] * 48 if uniform else [125.5] * 16 + [375.5] * 32
    torch.testing.assert_close(inputs["action_timesteps"], torch.tensor(expected), atol=0, rtol=0)
    assert masks["video_loss_mask"].tolist() == [False] * 4 + [False, True, True, True]
    assert masks["action_loss_mask"].sum() == 41 * 3


def test_supervision_combines_sparse_conditions_and_explicit_valid_steps():
    sample = payload_sample()
    sample["condition_frame_indexes_vision"] = [0, 2]
    sample["condition_frame_indexes_action"] = [1, 7]
    sample["action_valid_steps"][:] = True
    sample["action_valid_steps"][[0, 3, 31]] = False
    inputs, _, masks = pack_model_inputs(
        [sample], [sample["video_latents"]], [sample["action"]], timesteps=[latent_timesteps(sample)]
    )
    assert masks["video_loss_mask"].tolist() == [False, False, False, False, True, False]
    assert inputs["action_timesteps"][[1, 7]].eq(0).all()
    expected = torch.tensor([step not in (0, 1, 3, 7, 31) for step in range(32)])
    torch.testing.assert_close(masks["action_loss_mask"][:, 0], expected, atol=0, rtol=0)
    assert not masks["action_loss_mask"][:, 1].any()


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("timesteps", [], "One timestep"),
        ("timesteps", [torch.ones(3, dtype=torch.bfloat16)], "float32"),
        ("timesteps", [torch.ones(2)], "float32"),
        ("timesteps", [torch.ones(1, 3)], "float32"),
        ("action_valid_steps", torch.ones(32), "boolean mask"),
        ("action_valid_steps", torch.ones(31, dtype=torch.bool), "matching action"),
        ("num_valid_video_latents", torch.tensor(2), "integer"),
        ("num_valid_video_latents", 4, "latent range"),
        ("num_valid_video_latents", 0, "latent range"),
    ],
)
def test_invalid_timestep_and_supervision_metadata(key, value, match):
    sample = payload_sample()
    times = [latent_timesteps(sample)]
    if key == "timesteps":
        times = value
    else:
        sample[key] = value
    with pytest.raises(ValueError, match=match):
        pack_model_inputs([sample], [sample["video_latents"]], [sample["action"]], timesteps=times)
