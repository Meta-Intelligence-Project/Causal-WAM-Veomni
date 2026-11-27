# SPDX-License-Identifier: OpenMDW-1.1
"""Finite frame anchors with O(episodes) indexing and on-demand LeRobot v3 reads."""

import copy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, get_worker_info

from ..readers.lerobot import (
    ActionColumns,
    decode_window_video,
    plan_frame_window,
    read_frame_window,
    read_info,
    read_metadata,
    read_source_fps,
)
from ..transforms.layout import VIEW_DESCRIPTION, VIEW_ROLES, camera_layout, format_prompt, resolve_camera_roles
from ..transforms.processing import Geometry, num_valid_video_latents, video_token_layout
from .dataset import MODES


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameCandidateConfig:
    """Candidate anchor enumeration and the maximum complete blocks per training sample."""

    mode: str = "frame_anchor"
    max_blocks: int = 4
    anchor_stride: int = 1

    def __post_init__(self):
        if self.mode != "frame_anchor":
            raise ValueError("Frame candidates require mode=frame_anchor")
        if type(self.max_blocks) is not int or self.max_blocks < 1:
            raise ValueError("max_blocks must be a positive integer")
        if type(self.anchor_stride) is not int or self.anchor_stride < 1:
            raise ValueError("anchor_stride must be a positive integer")


@dataclass(frozen=True)
class FrameLabelConfig:
    """Task and subtask IDs are root-local.

    'auto' chooses subtask_index/subtask_id from info.json, or task text when neither
    exists. Explicitly named columns/tables must exist. Unknown IDs raise.
    Text tables are projected and loaded once per root.
    """

    task_column: str = "task_index"
    task_table: str = "meta/tasks.parquet"
    task_id_column: str = "task_index"
    task_text_column: str = "task"
    task_records_key: str | None = None
    subtask_column: str | None = "auto"
    subtask_table: str = "meta/subtasks.parquet"
    subtask_id_column: str = "subtask_index"
    subtask_text_column: str = "subtask"
    subtask_records_key: str | None = None


class LeRobotFrameDataset(Dataset, ABC):
    """Shared frame indexing, labels, preparation and checkpointed sampling RNG.

    Concrete Action/Video datasets use the same complete-block window for raw
    inspection and training, with text chosen from the actual video coverage.
    """

    def __init__(
        self,
        root,
        *,
        source,
        tokenize=None,
        cameras=None,
        resolution="256",
        viewpoint="concat_view",
        candidates=None,
        decode_threads=0,
        tolerance_s=None,
        episodes=None,
        prompt_formatter=None,
        video=True,
        labels=None,
        video_stride=4,
        seed=0,
        rank=0,
        full_history=True,
        history_range=(1, 32),
        subtask_mode: Literal["task", "subtask_concat", "subtask_single"] = "subtask_concat",
    ):
        self.candidates = candidates or FrameCandidateConfig()
        if subtask_mode not in ("task", "subtask_concat", "subtask_single"):
            raise ValueError("subtask_mode must be 'task', 'subtask_concat' or 'subtask_single'")
        self.subtask_mode = subtask_mode
        if type(video_stride) is not int or video_stride < 1:
            raise ValueError("video_stride must be a positive integer")
        if (
            len(history_range) != 2
            or any(type(v) is not int for v in history_range)
            or not 1 <= history_range[0] <= history_range[1]
        ):
            raise ValueError("Invalid history_range")
        self.seed, self.rank, self.rng = seed, rank, None
        self.full_history, self.history_range = full_history, history_range
        self.labels = FrameLabelConfig(**labels) if isinstance(labels, dict) else labels or FrameLabelConfig()
        self.video_stride = video_stride
        self.root = str(Path(root).resolve())
        self.source, self.tokenize = source, tokenize
        self.resolution, self.viewpoint = str(resolution), viewpoint
        self.decode_threads, self.tolerance_s, self.video = decode_threads, tolerance_s, video
        self.prompt_formatter = prompt_formatter
        if viewpoint not in VIEW_ROLES:
            raise ValueError(f"Unsupported viewpoint: {viewpoint}")
        self.info = read_info(self.root)
        self.tolerance_s = 1 / float(self.info["fps"]) if tolerance_s is None else tolerance_s
        self.roles = resolve_camera_roles(self.info["features"], cameras, viewpoint, root=self.root)
        self.info, self.episodes = read_metadata(
            self.root, [self.roles[role] for role in VIEW_ROLES[viewpoint]], info=self.info
        )
        self.cameras = camera_layout(self.info, self.roles, self.resolution, viewpoint)
        self.episodes["source_fps"] = read_source_fps(self.root, self.info, self.episodes)
        ids = self.episodes["episode_index"]
        available_episode_ids = set(map(int, ids))
        # 如果外面(例如采样的数据集)指定了当前lerobotv3中的episode，则替换，否则用全量
        self.selected_episodes = available_episode_ids if episodes is None else set(episodes)
        # 判断外部指定episode是否有不存在的
        if unknown := self.selected_episodes - available_episode_ids:
            raise ValueError(f"Unknown episodes in {self.root}: {sorted(unknown)}")
        # 判断有效长度是否>0
        episode_lengths = self.episodes["dataset_to_index"] - self.episodes["dataset_from_index"]
        if (episode_lengths <= 0).any():
            raise ValueError("Episode length must be positive")
        # 找到对应episodes在文件中的行号
        self.episode_rows = np.flatnonzero(np.isin(ids, list(self.selected_episodes)))
        starts, counts = self._candidate_ranges(episode_lengths[self.episode_rows])
        keep = counts > 0
        self.episode_rows = self.episode_rows[keep]
        self.anchor_starts = starts[keep]
        counts = counts[keep]
        # 只保留 episode 的候选数量前缀和，不展开每个窗口。
        self.cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        if not len(self):
            raise ValueError(f"No legal frame candidates in {self.root}")
        subtask = self.labels.subtask_column if self.subtask_mode != "task" else None
        if subtask == "auto":
            subtask = next((name for name in ("subtask_index", "subtask_id") if name in self.info["features"]), None)
        self.subtask_column = subtask
        self.task_texts = self._load_texts("task")
        self.subtask_texts = self._load_texts("subtask") if subtask is not None else {}

    def _candidate_ranges(self, episode_lengths):
        """Enumerate strided anchors; concrete datasets filter episodes by block geometry."""
        counts = (episode_lengths - 1) // self.candidates.anchor_stride + 1
        return np.zeros(len(episode_lengths), dtype=np.int64), counts

    def _load_texts(self, kind):
        config = self.labels
        path = Path(self.root) / getattr(config, f"{kind}_table")
        id_column, text_column = getattr(config, f"{kind}_id_column"), getattr(config, f"{kind}_text_column")
        records_key = getattr(config, f"{kind}_records_key")
        columns = [id_column, text_column]

        def records():
            if path.suffix == ".jsonl":
                with path.open() as file:
                    for line in file:
                        if not line.strip():
                            continue
                        parent = json.loads(line)
                        children = parent[records_key] if records_key else [parent]
                        for child in children:
                            yield {name: child[name] for name in columns}
            else:
                if records_key is not None:
                    raise ValueError("records_key is only supported for JSONL text definitions")
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(columns=columns, use_threads=False):
                        yield from batch.to_pylist()

        texts = {}
        for row in records():
            label, text = row[id_column], row[text_column]
            if type(label) is not int or not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: invalid {kind} ID/text")
            if label in texts:
                raise ValueError(f"{path}: duplicate {kind} ID {label}")
            texts[label] = text
        return texts

    def __len__(self):
        return int(self.cumulative[-1])

    def locate(self, index):
        """Map a valid sampler index to (metadata row, episode-local t)."""
        # 二分法找到index这个样本(帧)对应的episode
        slot = int(np.searchsorted(self.cumulative, index, side="right")) - 1
        # 由于anchor_stride(采样间隔)的存在，需要反算到真实帧号
        anchor_frame_index = (
            int(self.anchor_starts[slot]) + int(index - self.cumulative[slot]) * self.candidates.anchor_stride
        )
        return int(self.episode_rows[slot]), anchor_frame_index

    @abstractmethod
    def plan_read_window(self, index, *, columns=None):
        """Subclasses add block geometry and max_blocks to this storage plan."""
        row, anchor = self.locate(index)
        ep = {name: values[row].item() for name, values in self.episodes.items()}
        return plan_frame_window(
            self.root,
            ep["episode_index"],
            anchor,
            info=self.info,
            ep=ep,
            columns=columns,
            cameras=self.cameras,
            video_stride=self.video_stride,
        )

    def _caption(self, task_id, subtask_ids, video_offsets):
        task_id = int(task_id)
        # 如果只使用全局task，则直接取了返回
        if task_id not in self.task_texts:
            raise ValueError(f"Unknown task ID {task_id}")
        if subtask_ids is None:
            return self.task_texts[task_id], task_id, None, "task"
        # 如果使用subtask，则使用下面的逻辑
        subtask_ids = np.asarray(subtask_ids, dtype=np.int64)
        for subtask_id in np.unique(subtask_ids):
            if subtask_id not in self.subtask_texts:
                raise ValueError(f"Unknown subtask ID {subtask_id}")
        # 按完整逐帧标签划连续段，再统计实际视频采样点；重复 ID 不合并，并列选最早段。
        change_indices = np.flatnonzero(subtask_ids[1:] != subtask_ids[:-1]) + 1
        boundaries = np.concatenate(([0], change_indices, [len(subtask_ids)]))
        counts = np.diff(np.searchsorted(video_offsets, boundaries, side="left"))
        subtask_id = int(subtask_ids[boundaries[counts.argmax()]])
        return self.subtask_texts[subtask_id], task_id, subtask_id, "subtask"

    def _concat_subtask_caption(self, task_id, subtask_ids, block_span):
        """首尾段覆盖至少一个 block 才保留，中间段全部保留，按时序拼接；无保留段时使用 task。"""
        task_id = int(task_id)
        if task_id not in self.task_texts:
            raise ValueError(f"Unknown task ID {task_id}")
        task_caption = self.task_texts[task_id]
        if subtask_ids is None:
            return task_caption, task_id, [], "task"
        subtask_ids = np.asarray(subtask_ids, dtype=np.int64)
        for subtask_id in np.unique(subtask_ids):
            if subtask_id not in self.subtask_texts:
                raise ValueError(f"Unknown subtask ID {subtask_id}")
        # 最后一帧是末端观测，不增加 subtask 的覆盖时长，也不单独形成一段。
        subtask_ids = subtask_ids[:-1]
        change_indices = np.flatnonzero(subtask_ids[1:] != subtask_ids[:-1]) + 1
        boundaries = np.concatenate(([0], change_indices, [len(subtask_ids)]))
        selected_ids = []
        last_segment = len(boundaries) - 2
        for segment, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if segment in (0, last_segment) and end - start < block_span:
                continue
            selected_ids.append(int(subtask_ids[start]))
        if not selected_ids:
            return task_caption, task_id, [], "task"
        captions = []
        for subtask_id in selected_ids:
            text = self.subtask_texts[subtask_id].strip()
            if text[-1].isalnum():
                text += "."
            captions.append(text)
        caption = " ".join(captions)
        return caption, task_id, selected_ids, "subtask"

    def read_sample(self, index, *, video=None):
        """Read labels and payloads once, then select caption from the actual video window."""
        window = self.plan_read_window(index)
        label_columns = [self.labels.task_column] + ([self.subtask_column] if self.subtask_column else [])
        sample = read_frame_window(
            window,
            label_columns=label_columns,
            subtask_column=self.subtask_column if self.subtask_mode == "subtask_single" else None,
            video=self.video if video is None else video,
            decode_threads=self.decode_threads,
            tolerance_s=self.tolerance_s,
        )
        # 接下来我们通过上面读取的sample，来处理caption
        window = sample["window"]
        frame_labels = sample.pop("frame_labels")
        task_id = sample["anchor_labels"][self.labels.task_column]
        # 未使用或没有 subtask 时，明确传入 None。
        subtask_ids = frame_labels[self.subtask_column] if self.subtask_column is not None else None
        if self.subtask_mode == "subtask_concat":
            caption, task_id, subtask_indices, kind = self._concat_subtask_caption(
                task_id, subtask_ids, window.block_span
            )
            subtask_id = subtask_indices[0] if len(subtask_indices) == 1 else None
        else:
            task_id = int(task_id)
            if task_id not in self.task_texts:
                raise ValueError(f"Unknown task ID {task_id}")
            subtask_id = None
            caption, kind = self.task_texts[task_id], "task"
            if self.subtask_column is not None:
                subtask_id = int(sample["anchor_labels"][self.subtask_column])
                if subtask_id not in self.subtask_texts:
                    raise ValueError(f"Unknown subtask ID {subtask_id}")
                caption, kind = self.subtask_texts[subtask_id], "subtask"
            subtask_indices = [] if subtask_id is None else [subtask_id]
        row, _ = self.locate(index)
        source_fps = float(self.episodes["source_fps"][row])
        sample.update(
            sample_kind="frame_anchor",
            episode=window.episode,
            caption_kind=kind,
            task_index=task_id,
            subtask_index=subtask_id,
            subtask_indices=subtask_indices,
            source=self.source,
            subset=self.root,
            candidate_index=int(index),
            caption=caption,
            storage_fps=window.fps,
            source_fps=source_fps,
            conditioning_fps=source_fps / window.video_stride,
            viewpoint=self.viewpoint,
            additional_view_description=VIEW_DESCRIPTION[self.viewpoint],
            resolution=self.resolution,
            image_size=(window.height, window.width),
        )
        return sample

    def _decode_sample_video(self, sample, real_intervals):
        """Decode the prepared timeline, repeating the actual window endpoint."""
        window = sample["window"]
        if self.video and window.cameras:
            queries = sample["timestamps"][:: window.video_stride].numpy()
            queries = np.minimum(queries, float(sample["timestamps"][real_intervals]))
            sample["video"] = decode_window_video(
                window, queries, tolerance_s=self.tolerance_s, decode_threads=self.decode_threads
            )

    def prepare_sample(self, sample):
        """Format the completed sample; tokenize only when a callback is supplied."""
        prompt = (self.prompt_formatter or format_prompt)(sample)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty task prompt is required")
        sample["prompt"] = prompt
        if self.tokenize is not None:
            ids = torch.as_tensor(self.tokenize(prompt), dtype=torch.long)
            if ids.ndim != 1:
                raise ValueError("tokenize must return a 1D sequence, including final special tokens")
            sample["text_token_ids"] = ids
        return sample

    def make_sample(self, index):
        """Return video data and text without model-dependent length alignment."""
        sample = self.read_sample(index)
        return self.prepare_sample(sample)

    def __getitem__(self, index):
        index = index % self.__len__()
        row, anchor = self.locate(index)
        episode = int(self.episodes["episode_index"][row])
        try:
            return self.make_sample(index)
        except Exception as error:
            error.add_note(f"source={self.source} root={self.root} episode={episode} t={anchor} index={index}")
            logger.warning("读取样本失败，尝试下一条", exc_info=True)
            return self.__getitem__(index + 1)

    def _get_rng(self):
        if self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, worker.id if worker else 0, 0])
            )
        return self.rng

    def select_history(self, num_latents, *, explicit=None, rng=None):
        """History is measured in latent frames, matching existing causal training."""
        if explicit is not None:
            if type(explicit) is not int or explicit < 1:
                raise ValueError("Explicit history must be a positive integer")
            return explicit
        if self.full_history:
            return num_latents
        rng = self._get_rng() if rng is None else rng
        return int(rng.integers(self.history_range[0], self.history_range[1] + 1))

    def state_dict(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state) if self.rng is not None else None)

    def load_state_dict(self, state):
        self.rng = None
        if state["rng"] is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class ActionFrameDataset(LeRobotFrameDataset):
    """Per-frame anchors with bidirectional complete blocks and existing action preparation."""

    def __init__(
        self,
        root,
        *,
        source,
        processor=None,
        tokenize=None,
        cameras=None,
        resolution="256",
        viewpoint="concat_view",
        columns=None,
        candidates=None,
        geometry=None,
        decode_threads=0,
        tolerance_s=None,
        episodes=None,
        prompt_formatter=None,
        video=True,
        labels=None,
        video_stride=None,
        seed=0,
        rank=0,
        mode="policy",
        mode_weights=(1, 1, 1),
        full_history=True,
        history_range=(1, 32),
        subtask_mode: Literal["task", "subtask_concat", "subtask_single"] = "subtask_concat",
    ):
        self.columns = columns or ActionColumns()
        self.processor = processor
        self.geometry = geometry or Geometry()
        if video_stride is not None and video_stride != self.geometry.video_stride:
            raise ValueError("video_stride must match the action geometry")
        self.mode = mode
        weights = np.asarray(mode_weights, dtype=float)
        if (
            mode not in (*MODES, "joint")
            or weights.shape != (3,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Invalid task mode or mode_weights (FD, ID, policy)")
        self.mode_weights = weights / weights.sum()
        if processor is not None:
            for key, value in {
                "actions_per_block": self.geometry.actions_per_block,
                "video_stride": self.geometry.video_stride,
                "temporal_compression_factor": self.geometry.temporal_compression,
            }.items():
                if key in processor.statistics_geometry and processor.statistics_geometry[key] != value:
                    raise ValueError(f"Statistics {key} differs from the configured block geometry")
        super().__init__(
            root,
            source=source,
            tokenize=tokenize,
            cameras=cameras,
            resolution=resolution,
            viewpoint=viewpoint,
            candidates=candidates,
            decode_threads=decode_threads,
            tolerance_s=tolerance_s,
            episodes=episodes,
            prompt_formatter=prompt_formatter,
            video=video,
            labels=labels,
            video_stride=self.geometry.video_stride if video_stride is None else video_stride,
            seed=seed,
            rank=rank,
            full_history=full_history,
            history_range=history_range,
            subtask_mode=subtask_mode,
        )

    def _candidate_ranges(self, episode_lengths):
        """Keep every frame of episodes that can contain at least one aligned block."""
        offset = self.columns.target_offset
        minimum = max(0, -offset) + self.geometry.actions_per_block + max(1, offset)
        starts, counts = super()._candidate_ranges(episode_lengths)
        return starts, np.where(episode_lengths >= minimum, counts, 0)

    def plan_read_window(self, index):
        window = super().plan_read_window(index, columns=self.columns)
        window.block_span = self.geometry.actions_per_block
        window.max_blocks = self.candidates.max_blocks
        return window

    def make_sample(self, index, *, mode=None, history=None):
        if self.processor is None or not callable(self.tokenize):
            raise ValueError("Prepared action samples require processor and tokenize; use read_sample for raw data")
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown explicit task: {mode}")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        # 双向选窗与 payload 复用一次 Parquet 读取，随后仅解码最终窗口的视频。
        raw = self.read_sample(index, video=False)
        total = len(raw["action"])
        latents = 1 + total // self.geometry.actions_per_latent
        if mode is None:
            mode = str(self._get_rng().choice(MODES, p=self.mode_weights)) if self.mode == "joint" else self.mode
        history = self.select_history(latents, explicit=history)
        self._decode_sample_video(raw, raw["real_action_steps"])
        sample = dict(raw, **self.processor(raw, self.geometry))
        sample.pop("state")
        sample.update(
            mode=mode,
            history_blocks=history,
            block_size=self.geometry.actions_per_block // self.geometry.actions_per_latent,
            conditioning_fps_action=sample["source_fps"],
            template_id=self.processor.template.template_id,
        )
        sample = self.prepare_sample(sample)
        layout = self.geometry.token_layout(total, *sample["image_size"], len(sample["text_token_ids"]))
        sample.update(
            token_layout=layout,
            token_length=sum(layout.values()),
            condition_frame_indexes_vision=list(range(latents)) if mode == "inverse_dynamics" else [0],
            condition_frame_indexes_action=list(range(total)) if mode == "forward_dynamics" else [],
        )
        return sample


class VideoFrameDataset(LeRobotFrameDataset):
    """Bidirectional complete video blocks using video-specific geometry and source FPS."""

    def __init__(
        self,
        root,
        *,
        source,
        candidates=None,
        latent_block_size=2,
        temporal_compression=4,
        spatial_compression=16,
        patch_size=2,
        video_stride=None,
        seed=0,
        rank=0,
        full_history=False,
        history_range=(1, 32),
        **kwargs,
    ):
        sizes = (latent_block_size, temporal_compression, spatial_compression, patch_size)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Video block, compression and patch sizes must be positive integers")
        self.latent_block_size = latent_block_size
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.patch_size = patch_size
        self.auto_video_stride = video_stride is None
        super().__init__(
            root,
            source=source,
            candidates=candidates,
            video_stride=1 if video_stride is None else video_stride,
            seed=seed,
            rank=rank,
            full_history=full_history,
            history_range=history_range,
            **kwargs,
        )

    def _candidate_ranges(self, episode_lengths):
        strides = (
            np.where(self.episodes["source_fps"][self.episode_rows] > 16, 2, 1)
            if self.auto_video_stride
            else self.video_stride
        )
        spans = strides * self.temporal_compression * self.latent_block_size
        starts, counts = super()._candidate_ranges(episode_lengths)
        return starts, np.where(episode_lengths > spans, counts, 0)

    def plan_read_window(self, index):
        window = super().plan_read_window(index)
        if self.auto_video_stride and window.video_stride == 1:
            row, _ = self.locate(index)
            window.video_stride = 2 if self.episodes["source_fps"][row] > 16 else 1
        window.block_span = window.video_stride * self.temporal_compression * self.latent_block_size
        window.max_blocks = self.candidates.max_blocks
        return window

    def make_sample(self, index, *, history=None):
        if not callable(self.tokenize):
            raise ValueError("Prepared video samples require tokenize; use read_sample for raw data")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        raw = self.read_sample(index, video=False)
        window = raw["window"]
        total = window.past + window.future
        latents = 1 + total // (window.video_stride * self.temporal_compression)
        sample = dict(
            raw,
            mode="video",
            history_blocks=self.select_history(latents, explicit=history),
            block_size=self.latent_block_size,
            condition_frame_indexes_vision=[0],
            num_valid_video_latents=num_valid_video_latents(total, window.video_stride, self.temporal_compression),
        )
        self._decode_sample_video(sample, total)
        sample = self.prepare_sample(sample)
        layout = video_token_layout(
            total // window.video_stride + 1,
            *sample["image_size"],
            len(sample["text_token_ids"]),
            temporal_compression=self.temporal_compression,
            spatial_compression=self.spatial_compression,
            patch_size=self.patch_size,
        )
        sample.update(token_layout=layout, token_length=sum(layout.values()))
        return sample
