# SPDX-License-Identifier: OpenMDW-1.1
"""按最终布局登记 token 区间，并构造 FP32 mRoPE 坐标。"""

import math
from dataclasses import dataclass

import torch

from . import mrope
from .attention import CausalLayout


@dataclass(frozen=True)
class SampleSpans:
    """Final global token slices, with exclusive stops, for one packed sample."""

    text: slice
    clean_video: slice
    current_video: slice
    action: slice
    state: slice


def _cpu_indexes(indexes: torch.Tensor, name: str) -> list[int]:
    if not isinstance(indexes, torch.Tensor) or indexes.device.type != "cpu" or indexes.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional CPU index tensor")
    if indexes.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indexes")
    return indexes.tolist()


def build_packed_layout(samples, *, patch_size=2):
    """Return ``(CausalLayout, tuple[SampleSpans, ...], packed_length)`` in final sample order.

    Each sample follows the action loader protocol after VAE encoding:
    ``text_token_ids`` [L], ``video_latents`` [C,T,H,W], ``action`` [A,D],
    ``states`` [S,D], CPU integer ``action_frame_ids`` [A], condition index
    lists, block/history sizes and ``token_length``. Block sizes count latents
    after the independent first one.

    Positions are registered directly as text, clean video, current video,
    action, state. Only shapes and CPU metadata are read; payloads are neither
    moved nor modified. This function does not generate RoPE coordinates,
    noise, timesteps or loss masks, and does not pad or select pack members.
    """
    if not samples or patch_size < 1:
        raise ValueError("A nonempty pack and a positive patch_size are required")

    # 前四个列表逐 token 保存元数据，相同下标对应最终混合序列中的同一个 token：
    # sample_ids：所属样本；block_ids：所属预测块。
    # roles：文本=-1、clean=0、target=1、condition=2、state=3。
    # modalities：文本=0、视频=1、动作=2、状态=3。
    # histories 和 spans 每个样本各保存一项，分别记录允许访问的历史 clean block 数量，
    # 以及五段内容在整个 pack 中的全局切片区间。
    sample_ids, block_ids, roles, modalities, histories, spans = [], [], [], [], [], []
    # 整个 pack 中下一个可用的 token 位置，跨样本连续累加。
    cursor = 0

    def register_span(modality, blocks, token_roles, sample_id):
        """登记一段 token 的标签，返回其在整个 pack 中的左闭右开区间 [start, stop)。"""
        nonlocal cursor
        start = cursor
        count = len(blocks)
        sample_ids.extend([sample_id] * count)
        block_ids.extend(blocks)
        roles.extend(token_roles)
        modalities.extend([modality] * count)
        cursor += count
        return slice(start, cursor)

    for sample_id, sample in enumerate(samples):
        text, video, action, states = (sample[key] for key in ("text_token_ids", "video_latents", "action", "states"))
        if text.ndim != 1 or not text.numel():
            raise ValueError("text_token_ids must be a nonempty one-dimensional tensor")
        if video.ndim != 4 or min(video.shape) < 1:
            raise ValueError("video_latents must have positive [C,T,H,W] dimensions")
        if action.ndim != 2 or not len(action) or states.ndim != 2:
            raise ValueError("action and states must be [steps, features] tensors with nonempty action")

        block, history = sample["block_size"], sample["history_blocks"]
        if block < 1 or history < 1:
            raise ValueError("block_size and history_blocks must be positive")
        _, frames, height, width = video.shape
        patches = ((height + patch_size - 1) // patch_size) * ((width + patch_size - 1) // patch_size)
        video_length, action_length = frames * patches, len(action)
        state_length = (frames - 1 + block - 1) // block
        if len(states) != state_length:
            raise ValueError("states must contain one state per prediction block")

        action_frames = _cpu_indexes(sample["action_frame_ids"], "action_frame_ids")
        if len(action_frames) != action_length or any(frame < 1 or frame >= frames for frame in action_frames):
            raise ValueError(
                "action_frame_ids must match action and reference latents after the independent first one"
            )
        video_conditions = set(sample["condition_frame_indexes_vision"])
        action_conditions = set(sample["condition_frame_indexes_action"])
        if 0 not in video_conditions or any(frame < 0 or frame >= frames for frame in video_conditions):
            raise ValueError("Video conditions must include the independent first latent and use valid frame indexes")
        if any(step < 0 or step >= action_length for step in action_conditions):
            raise ValueError("Action conditions must use valid action indexes")

        # 同一 latent 的所有空间 patch 共享 block 编号和 current video 角色。
        # 独立首 latent 的 block 为 -1，后续预测块从 0 开始编号。
        video_blocks = [(frame - 1) // block for frame in range(frames) for _ in range(patches)]
        current_roles = [2 if frame in video_conditions else 1 for frame in range(frames) for _ in range(patches)]
        action_blocks = [(frame - 1) // block for frame in action_frames]
        action_roles = [2 if step in action_conditions else 1 for step in range(action_length)]
        sample_start = cursor
        # 按最终序列顺序登记各段，同时保存当前样本的五个区间。
        # clean/current video 共享 block 排布；clean video 的 role 固定为 0。
        spans.append(
            SampleSpans(
                text=register_span(0, [-1] * text.numel(), [-1] * text.numel(), sample_id),
                clean_video=register_span(1, video_blocks, [0] * video_length, sample_id),
                current_video=register_span(1, video_blocks, current_roles, sample_id),
                action=register_span(2, action_blocks, action_roles, sample_id),
                state=register_span(3, range(state_length), [3] * state_length, sample_id),
            )
        )
        if cursor - sample_start != sample["token_length"]:
            raise ValueError("Final token count differs from the pre-packing token_length")
        histories.append(history)

    # CausalLayout 沿用现有注意力接口，提供逐 token 标签及每个样本的历史范围。
    # SampleSpans 单独保存各样本的分段区间，供后续组装模态数据、位置坐标时直接定位，
    # 避免重新扫描 token 标签。构建完成后，将列表转为不可变的 tuple。
    # 最终 cursor 就是实际 token 总数，一并返回，供后续组装直接分配空间。
    return (
        CausalLayout(
            sample_ids=tuple(sample_ids),
            block_ids=tuple(block_ids),
            roles=tuple(roles),
            modalities=tuple(modalities),
            history_blocks=tuple(histories),
        ),
        tuple(spans),
        cursor,
    )


def build_position_ids(
    samples,
    spans,
    packed_length,
    *,
    patch_size=2,
    base_fps=24.0,
    temporal_compression=4,
    temporal_modality_margin=15000,
):
    """按最终区间构造 CPU FP32 的 [3, N_packed] mRoPE 坐标。

    spans 与 samples 一一对应，记录同一 pack 的最终区间，且 patch_size 与登记时相同。
    packed_length 使用 build_packed_layout 返回的实际 token 总数。
    每个样本提供视频/动作 FPS，以及 CPU 整数 state_latent_indexes。
    使用与旧 causal 一致的 FPS 缩放；每个样本独立从文本坐标 0 开始。
    迁移坐标到模型设备时只指定 device，保留 FP32，不跟随模态张量转为 BF16。
    """
    if not samples or len(samples) != len(spans) or patch_size < 1:
        raise ValueError("A nonempty pack, one SampleSpans per sample and a positive patch_size are required")
    if any(not math.isfinite(value) or value <= 0 for value in (base_fps, temporal_compression)):
        raise ValueError("base_fps and temporal_compression must be finite and positive")
    if not math.isfinite(temporal_modality_margin):
        raise ValueError("temporal_modality_margin must be finite")

    # 显式指定 CPU 和 FP32，避免默认 dtype、默认设备或混合精度影响坐标构造。
    positions = torch.zeros((3, packed_length), dtype=torch.float32, device="cpu")
    for sample, span in zip(samples, spans, strict=True):
        video_fps, action_fps = sample["conditioning_fps"], sample["conditioning_fps_action"]
        if any(not math.isfinite(fps) or fps <= 0 for fps in (video_fps, action_fps)):
            raise ValueError("Video and action conditioning FPS must be finite and positive")
        _, frames, height, width = sample["video_latents"].shape
        grid_h = (height + patch_size - 1) // patch_size
        grid_w = (width + patch_size - 1) // patch_size
        text_length = sample["text_token_ids"].numel()
        state_latents = _cpu_indexes(sample["state_latent_indexes"], "state_latent_indexes")
        if len(state_latents) != len(sample["states"]) or any(index < 0 or index >= frames for index in state_latents):
            raise ValueError("state_latent_indexes must match states and reference valid video latents")

        # 坐标算法保留在独立 mrope 模块，原函数的默认设备由调用方明确限定为 CPU。
        with torch.device("cpu"):
            text_positions, text_end = mrope.get_3d_mrope_ids_text_tokens(
                text_length, temporal_offset=0, use_float_positions=True
            )
            # video/action 共用文本结束后的时间起点，不随视频返回的下一游标串行推进。
            offset = text_end + temporal_modality_margin
            video_positions, _ = mrope.get_3d_mrope_ids_vae_tokens(
                grid_t=frames,
                grid_h=grid_h,
                grid_w=grid_w,
                temporal_offset=offset,
                reset_spatial_indices=True,
                fps=video_fps,
                base_fps=base_fps,
                temporal_compression_factor=temporal_compression,
                actual_temporal_compression_factor=temporal_compression,
            )
            action_positions, _ = mrope.get_3d_mrope_ids_vae_tokens(
                grid_t=len(sample["action"]),
                grid_h=1,
                grid_w=1,
                temporal_offset=offset,
                reset_spatial_indices=True,
                fps=action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=temporal_compression,
                start_frame_offset=1,
            )
            # 模块替换也必须保持 FP32；写入 FP32 容器无法补救上游已发生的精度损失。
            if any(value.dtype != torch.float32 for value in (text_positions, video_positions, action_positions)):
                raise ValueError("mRoPE helpers must return float32 coordinates for packed action inputs")
            state_times = video_positions[0].view(frames, grid_h * grid_w)[state_latents, 0]

        # 本层只负责按已登记的最终区间填入坐标；clean/current 共用一份视频坐标结果。
        positions[:, span.text] = text_positions
        positions[:, span.clean_video] = video_positions
        positions[:, span.current_video] = video_positions
        positions[:, span.action] = action_positions
        # 直接取视频锚点，避免独立推算 state 时间产生舍入差异或忽略显式映射。
        positions[0, span.state] = state_times

    return positions
