"""Final causal registration, native token budgets and attention boundaries."""

from dataclasses import asdict

import pytest
import torch

from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.data.causal_cosmos3.processing import Geometry
from veomni.models.diffusers.causal_cosmos3 import modeling_causal_cosmos3, mrope
from veomni.models.diffusers.causal_cosmos3.attention import dense_generation_mask, prepare_attention
from veomni.models.diffusers.causal_cosmos3.layers import rotary_tables
from veomni.models.diffusers.causal_cosmos3.modeling_causal_cosmos3 import Cosmos3Model, patchify_video
from veomni.models.diffusers.causal_cosmos3.packing import SampleSpans, build_packed_layout, build_position_ids
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
