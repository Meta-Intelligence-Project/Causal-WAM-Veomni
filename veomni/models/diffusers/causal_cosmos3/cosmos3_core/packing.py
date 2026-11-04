# SPDX-License-Identifier: OpenMDW-1.1
"""按最终布局登记 token 区间，构造 FP32 坐标并组装模态输入。"""

import math
from dataclasses import dataclass, replace

import torch

from . import mrope
from .attention import CausalLayout
from .patching import patchify_video


@dataclass(frozen=True)
class SampleSpans:
    """每样本的全局区间，以及 packing 拼接后对应的模态输出行区间。

    前五项指向完整混合序列；后两项分别指向 outputs.video / outputs.action。
    build_packed_layout 只登记前五项，pack_model_inputs 在实际模态拼接时补齐后两项。
    所有 slice 均为左闭右开区间，spans 的样本顺序始终与输入 samples 一致。
    纯视频的 action/state 是空区间，action_rows 保持 None。
    """

    text: slice
    clean_video: slice
    current_video: slice
    action: slice
    state: slice
    current_video_rows: slice | None = None
    action_rows: slice | None = None


class _ModalityBuffer:
    """按段同步收集一个模态的字段，追加时直接返回该段的模态行区间。"""

    def __init__(self, *field_names):
        self.parts = {name: [] for name in field_names}
        self.num_rows = 0

    def append(self, tensors):
        length = len(tensors[next(iter(self.parts))])
        rows = slice(self.num_rows, self.num_rows + length)
        for name, parts in self.parts.items():
            parts.append(tensors[name])
        self.num_rows += length
        return rows

    def finish(self):
        # append 只保存引用；每个字段在这里统一拼接一次。
        return {name: torch.cat(parts, dim=0) for name, parts in self.parts.items()}


def _segment_order(spans):
    """从最终布局取得唯一的段顺序，输入、时间步和 mask 共用这一顺序。"""
    segments = []
    for sample_id, span in enumerate(spans):
        for name in ("text", "clean_video", "current_video", "action", "state"):
            section = getattr(span, name)
            if section.start < section.stop:
                segments.append((section.start, sample_id, name))
    # 只对完整的段排序一次，保留样本和 clean/current 的物理重排能力。
    segments.sort(key=lambda segment: segment[0])
    return [(sample_id, name) for _, sample_id, name in segments]


def _cpu_indexes(indexes: list[int] | torch.Tensor, name: str) -> list[int]:
    """Validate Python integer lists; retain compatibility with CPU tensor callers."""
    if isinstance(indexes, list):
        if any(type(index) is not int for index in indexes):
            raise ValueError(f"{name} must be a one-dimensional list of integer indexes")
        return indexes
    if not isinstance(indexes, torch.Tensor) or indexes.device.type != "cpu" or indexes.ndim != 1:
        raise ValueError(f"{name} must be a list of integers or a one-dimensional CPU index tensor")
    if indexes.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indexes")
    return indexes.tolist()


def build_packed_layout(samples, *, has_action=True, patch_size=2):
    """Return ``(CausalLayout, tuple[SampleSpans, ...], packed_length)`` in final sample order.

    Each sample supplies ``text_token_ids`` [L], ``video_latents`` [C,T,H,W],
    video condition indexes, block/history sizes and ``token_length``.
    With has_action=True (the default), it also supplies ``action`` [A,D],
    ``states`` [S,D], ``action_frame_ids`` [A] (a Python integer list or CPU
    tensor), and action condition indexes. With has_action=False, action/state
    fields are absent and their spans have zero length. Block sizes count
    latents after the independent first one.

    Positions are registered directly as text, clean video, current video,
    action, state. Only shapes and CPU metadata are read; payloads are neither
    moved nor modified. This function does not generate RoPE coordinates,
    noise, timesteps or loss masks, and does not pad or select pack members.
    """
    if not samples or patch_size < 1:
        raise ValueError("A nonempty pack and a positive patch_size are required")

    # 前五个列表逐 token 保存元数据，相同下标对应最终混合序列中的同一个 token：
    # sample_ids：所属样本；block_ids：所属预测块。
    # roles：文本=-1、clean=0、target=1、condition=2、state=3。
    # modalities：文本=0、视频=1、动作=2、状态=3。
    # frame_ids：视频 latent 帧编号，其他模态为 -1。
    # 每样本记录历史 latent 帧预算、block 大小和五段内容的全局切片。
    sample_ids, block_ids, roles, modalities, frame_ids = [], [], [], [], []
    histories, block_sizes, spans, windows = [], [], [], []
    # 整个 pack 中下一个可用的 token 位置，跨样本连续累加。
    cursor = 0

    def register_span(modality, blocks, token_roles, sample_id, frames=None):
        """登记一段 token 的标签，返回其在整个 pack 中的左闭右开区间 [start, stop)。"""
        nonlocal cursor
        start = cursor
        count = len(blocks)
        sample_ids.extend([sample_id] * count)
        block_ids.extend(blocks)
        roles.extend(token_roles)
        modalities.extend([modality] * count)
        frame_ids.extend([-1] * count if frames is None else frames)
        cursor += count
        return slice(start, cursor)

    for sample_id, sample in enumerate(samples):
        if has_action:
            if sample.get("mode") == "video" or "action" not in sample or "states" not in sample:
                raise ValueError("Action packing requires action and states")
        elif sample.get("mode") not in (None, "video") or "action" in sample or "states" in sample:
            raise ValueError("Video packing requires video-only samples")
        text, video = sample["text_token_ids"], sample["video_latents"]
        if text.ndim != 1 or not text.numel():
            raise ValueError("text_token_ids must be a nonempty one-dimensional tensor")
        if video.ndim != 4 or min(video.shape) < 1:
            raise ValueError("video_latents must have positive [C,T,H,W] dimensions")
        block, history = sample["block_size"], sample["history_blocks"]
        if block < 1 or history < 1:
            raise ValueError("block_size and history_blocks must be positive")
        _, frames, height, width = video.shape
        observed = sample.get("observed_latents", 0) if sample.get("layout_mode") == "decision_window" else 0
        if sample.get("layout_mode") == "decision_window" and (
            not has_action
            or type(observed) is not int
            or not 1 <= observed < frames
            or frames - observed != block
            or history != observed
        ):
            raise ValueError("Decision windows require observed history and one complete future group")
        patches = ((height + patch_size - 1) // patch_size) * ((width + patch_size - 1) // patch_size)
        video_length = frames * patches
        video_conditions = set(sample["condition_frame_indexes_vision"])
        if 0 not in video_conditions or any(frame < 0 or frame >= frames for frame in video_conditions):
            raise ValueError("Video conditions must include the independent first latent and use valid frame indexes")

        # 独立首 latent 的 block 为 -1；同帧所有 patch 共享 block 和角色。
        video_blocks = [
            min(frame, observed) if observed else (frame - 1) // block
            for frame in range(frames)
            for _ in range(patches)
        ]
        video_frames = [frame for frame in range(frames) for _ in range(patches)]
        current_roles = [2 if frame in video_conditions else 1 for frame in range(frames) for _ in range(patches)]
        if observed:
            if not set(range(observed)) <= video_conditions:
                raise ValueError("Decision-window history must be entirely observed")
            current_roles[: observed * patches] = [0] * (observed * patches)
        sample_start = cursor
        text_span = register_span(0, [-1] * text.numel(), [-1] * text.numel(), sample_id)
        clean_span = (
            slice(cursor, cursor)
            if observed
            else register_span(1, video_blocks, [0] * video_length, sample_id, video_frames)
        )
        current_span = register_span(1, video_blocks, current_roles, sample_id, video_frames)
        action_span = state_span = slice(cursor, cursor)
        if has_action:
            action, states = sample["action"], sample["states"]
            if action.ndim != 2 or not len(action) or states.ndim != 2:
                raise ValueError("action and states must be [steps, features] tensors with nonempty action")
            action_length = len(action)
            state_length = 1 if observed else (frames - 1 + block - 1) // block
            if len(states) != state_length:
                raise ValueError("states must contain one state per prediction block")
            action_frames = _cpu_indexes(sample["action_frame_ids"], "action_frame_ids")
            if len(action_frames) != action_length or any(frame < 1 or frame >= frames for frame in action_frames):
                raise ValueError(
                    "action_frame_ids must match action and reference latents after the independent first one"
                )
            action_conditions = set(sample["condition_frame_indexes_action"])
            if any(step < 0 or step >= action_length for step in action_conditions):
                raise ValueError("Action conditions must use valid action indexes")
            action_blocks = (
                [observed] * action_length if observed else [(frame - 1) // block for frame in action_frames]
            )
            action_roles = [2 if step in action_conditions else 1 for step in range(action_length)]
            action_span = register_span(2, action_blocks, action_roles, sample_id)
            state_span = register_span(
                3, [observed] if observed else range(state_length), [3] * state_length, sample_id
            )
        spans.append(
            SampleSpans(
                text=text_span,
                clean_video=clean_span,
                current_video=current_span,
                action=action_span,
                state=state_span,
            )
        )
        if cursor - sample_start != sample["token_length"]:
            raise ValueError("Final token count differs from the pre-packing token_length")
        histories.append(history)
        block_sizes.append(block)
        windows.append(observed)

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
            frame_ids=tuple(frame_ids),
            block_sizes=tuple(block_sizes),
            decision_windows=tuple(windows) if any(windows) else (),
        ),
        tuple(spans),
        cursor,
    )


def build_position_ids(
    samples,
    spans,
    packed_length,
    *,
    has_action=True,
    patch_size=2,
    base_fps=24.0,
    temporal_compression=4,
    temporal_modality_margin=15000,
):
    """按最终区间构造 CPU FP32 的 [3, N_packed] mRoPE 坐标。

    spans 与 samples 一一对应，记录同一 pack 的最终区间，且 patch_size 与登记时相同。
    packed_length 使用 build_packed_layout 返回的实际 token 总数。
    每个样本提供视频 FPS；仅 has_action=True 时读取动作 FPS 和 state_latent_indexes。
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
        video_fps = sample["conditioning_fps"]
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError("Video conditioning FPS must be finite and positive")
        _, frames, height, width = sample["video_latents"].shape
        grid_h = (height + patch_size - 1) // patch_size
        grid_w = (width + patch_size - 1) // patch_size
        text_length = sample["text_token_ids"].numel()

        with torch.device("cpu"):
            text_positions, text_end = mrope.get_3d_mrope_ids_text_tokens(
                text_length, temporal_offset=0, use_float_positions=True
            )
            # 视频与动作共用文本结束后的时间起点。
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
            if any(value.dtype != torch.float32 for value in (text_positions, video_positions)):
                raise ValueError("mRoPE helpers must return float32 coordinates for packed inputs")
            positions[:, span.text] = text_positions
            if span.clean_video.stop > span.clean_video.start:
                positions[:, span.clean_video] = video_positions
            positions[:, span.current_video] = video_positions

            if has_action:
                action_fps = sample["conditioning_fps_action"]
                if not math.isfinite(action_fps) or action_fps <= 0:
                    raise ValueError("Action conditioning FPS must be finite and positive")
                state_latents = _cpu_indexes(sample["state_latent_indexes"], "state_latent_indexes")
                if len(state_latents) != len(sample["states"]) or any(
                    index < 0 or index >= frames for index in state_latents
                ):
                    raise ValueError("state_latent_indexes must match states and reference valid video latents")
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
                    start_frame_offset=sample.get("action_start_frame_offset", 1),
                )
                if action_positions.dtype != torch.float32:
                    raise ValueError("mRoPE helpers must return float32 coordinates for packed inputs")
                positions[:, span.action] = action_positions
                # 状态直接取视频锚点坐标，避免独立计算的舍入差异。
                positions[0, span.state] = video_positions[0].view(frames, grid_h * grid_w)[state_latents, 0]

    return positions


def pack_model_inputs(
    samples,
    current_videos,
    current_actions=None,
    *,
    timesteps,
    has_action=True,
    patch_size=2,
    base_fps=24.0,
    temporal_compression=4,
    temporal_modality_margin=15000,
):
    """组装一个 pack，返回 ``(model_inputs, spans, loss_masks)``。

    has_action=False 时 current_actions 必须为 None，只生成文本和视频字段。
    Action samples 保留原始 clean video/action、states、文本及 CPU 索引元数据，
    另需一维 bool action_mask/state_mask，分别表示有效特征通道。
    current_videos/current_actions 与 samples 一一对应，形状与各自 clean 数据相同；
    目标位置使用 current，条件位置恢复 clean，不修改任何输入或切断梯度。
    timesteps 同样逐样本提供一维 FP32 Tensor，长度为视频 latent 数或 1；
    调用方负责给出符合 block 采样规则的原始 0..1000 时间步，本层只展开和映射。

    所有视频张量须已在同一设备，其他 payload 和 mask 搬到该设备且不指定 dtype。
    布局保留 CPU tuple，坐标在 CPU 生成后仅搬设备，始终保持 FP32。
    返回的模态张量按各段在最终布局中的位置排序；spans 仍与 samples 一一对应。
    视频时间步展开到 patch，动作时间步按 action_frame_ids 绑定视频 latent，
    clean/condition 时间步为零，state 的零时间步继续由模型处理。
    loss_masks 的 True 表示参与监督：视频排除 clean、condition 和纯时间 padding，
    动作另需同时满足 action_valid_steps 与有效特征 mask。
    从最终 spans 取得一次段顺序，同段的输入、时间步和 mask 同步追加；追加时直接返回
    current_video_rows/action_rows，定位对应模态预测和监督 mask，不通过全局起点查表。
    时间 padding 元数据保持在原样本中；本函数不随机采样、不加噪、不构造目标或计算 loss。
    """
    if has_action:
        if current_actions is None or len(current_actions) != len(samples):
            raise ValueError("One current video and action tensor per sample are required")
    elif current_actions is not None:
        raise ValueError("Video packing requires current_actions=None")
    layout, spans, packed_length = build_packed_layout(samples, has_action=has_action, patch_size=patch_size)
    if len(current_videos) != len(samples):
        raise ValueError("One current video and action tensor per sample are required")
    if len(timesteps) != len(samples):
        raise ValueError("One timestep tensor per sample is required")
    positions = build_position_ids(
        samples,
        spans,
        packed_length,
        has_action=has_action,
        patch_size=patch_size,
        base_fps=base_fps,
        temporal_compression=temporal_compression,
        temporal_modality_margin=temporal_modality_margin,
    )
    device = samples[0]["video_latents"].device
    sample_parts = []
    for i, (sample, span, current_video, times) in enumerate(
        zip(samples, spans, current_videos, timesteps, strict=True)
    ):
        video = sample["video_latents"]
        if current_video.shape != video.shape:
            raise ValueError("Current video/action shapes must match the corresponding clean payloads")
        if video.device != device or current_video.device != device:
            raise ValueError("All clean/current video tensors in a pack must already share one device")
        frames = video.shape[1]
        if times.dtype != torch.float32 or times.ndim != 1 or times.numel() not in (1, frames):
            raise ValueError("timesteps must be float32 [video_latents] or [1] tensors")
        valid_latents = sample["num_valid_video_latents"]
        if type(valid_latents) is not int or not 1 <= valid_latents <= frames:
            raise ValueError("num_valid_video_latents must be an integer within the video latent range")

        # 条件恢复发生在 patchify 前；时间 padding 只影响监督范围。
        video_condition = torch.zeros(frames, dtype=torch.bool, device=device)
        video_condition[sample["condition_frame_indexes_vision"]] = True
        current_video = torch.where(video_condition[None, :, None, None], video, current_video)
        patches = (span.current_video.stop - span.current_video.start) // frames
        frame_times = times.to(device=device, non_blocking=True).expand(frames)
        video_times = frame_times.masked_fill(video_condition, 0).repeat_interleave(patches)
        video_loss = (~video_condition & (torch.arange(frames, device=device) < valid_latents)).repeat_interleave(
            patches
        )
        parts = dict(
            text=dict(input_ids=sample["text_token_ids"].to(device=device, non_blocking=True)),
            clean_video=dict(
                video_tokens=patchify_video(video, patch_size),
                video_timesteps=torch.zeros_like(video_times),
                video_loss_mask=torch.zeros_like(video_loss),
            ),
            current_video=dict(
                video_tokens=patchify_video(current_video, patch_size),
                video_timesteps=video_times,
                video_loss_mask=video_loss,
            ),
        )
        if has_action:
            action, states, current_action = sample["action"], sample["states"], current_actions[i]
            if current_action.shape != action.shape:
                raise ValueError("Current video/action shapes must match the corresponding clean payloads")
            for key, payload in (("action_mask", action), ("state_mask", states)):
                mask = sample[key]
                if mask.dtype != torch.bool or mask.shape != (payload.shape[1],):
                    raise ValueError(f"{key} must be a one-dimensional boolean valid-feature mask")
            valid_steps = sample["action_valid_steps"]
            if valid_steps.dtype != torch.bool or valid_steps.shape != (len(action),):
                raise ValueError("action_valid_steps must be a one-dimensional boolean mask matching action steps")
            action = action.to(device=device, non_blocking=True)
            current_action = current_action.to(device=device, non_blocking=True)
            action_mask = sample["action_mask"].to(device=device, non_blocking=True).expand_as(action)
            state_mask = sample["state_mask"].to(device=device, non_blocking=True).expand_as(states)
            action_condition = torch.zeros(len(action), dtype=torch.bool, device=device)
            action_condition[sample["condition_frame_indexes_action"]] = True
            current_action = torch.where(action_condition[:, None], action, current_action)
            action_frames = torch.as_tensor(sample["action_frame_ids"], device=device, dtype=torch.long)
            action_times = frame_times.index_select(0, action_frames).masked_fill(action_condition, 0)
            action_loss = (
                ~action_condition[:, None] & valid_steps.to(device=device, non_blocking=True)[:, None] & action_mask
            )
            parts.update(
                action=dict(
                    action_tokens=current_action,
                    action_timesteps=action_times,
                    action_mask=action_mask,
                    action_loss_mask=action_loss,
                ),
                state=dict(state_tokens=states.to(device=device, non_blocking=True), state_mask=state_mask),
            )
        sample_parts.append(parts)

    """
    方便模型对不同模态使用不同的投影层，按照类别进行组装

    组装前：样本 A → 文本、clean视频、current视频、动作、状态
    样本 B → 文本、clean视频、current视频、动作、状态

    组装后：文本张量 → A文本、B文本
            视频张量 → A_clean、A_current、B_clean、B_current
            动作张量 → A动作、B动作
            状态张量 → A状态、B状态
    """
    text = _ModalityBuffer("input_ids")
    video = _ModalityBuffer("video_tokens", "video_timesteps", "video_loss_mask")
    action = _ModalityBuffer("action_tokens", "action_timesteps", "action_mask", "action_loss_mask")
    state = _ModalityBuffer("state_tokens", "state_mask")
    buffers = dict(text=text, clean_video=video, current_video=video, action=action, state=state)
    video_rows, action_rows = [None] * len(samples), [None] * len(samples)

    # 同一份最终段顺序驱动所有模态；append 返回实际追加的行区间，无需再查表推导。
    for sample_id, name in _segment_order(spans):
        rows = buffers[name].append(sample_parts[sample_id][name])  # 把sample按照key，塞到类别里
        if name == "current_video":
            video_rows[sample_id] = rows
        elif name == "action":
            action_rows[sample_id] = rows

    inputs = {}
    for buffer in (text, video, action, state):
        if buffer.num_rows:
            inputs.update(buffer.finish())
    spans = tuple(
        replace(span, current_video_rows=vr, action_rows=ar)
        for span, vr, ar in zip(spans, video_rows, action_rows, strict=True)
    )
    # 网络只接收模型字段；监督 mask 单独返回，并与对应模态的预测行顺序保持一致。
    loss_masks = {key: inputs.pop(key) for key in ("video_loss_mask", "action_loss_mask") if key in inputs}
    inputs.update(layout=layout, position_ids=positions.to(device=device, non_blocking=True))
    return inputs, spans, loss_masks
