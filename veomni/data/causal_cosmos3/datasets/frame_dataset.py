# SPDX-License-Identifier: OpenMDW-1.1
"""Finite frame anchors with O(episodes) indexing and on-demand LeRobot v3 reads."""

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

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


@dataclass(frozen=True)
class FrameCandidateConfig:
    mode: str = "frame_anchor"
    past_frames: int = 0
    future_frames: int = 128
    anchor_stride: int = 1

    def __post_init__(self):
        if self.mode != "frame_anchor":
            raise ValueError("Frame candidates require mode=frame_anchor")
        if any(type(v) is not int or v < 0 for v in (self.past_frames, self.future_frames)):
            raise ValueError("past_frames/future_frames must be nonnegative integers")
        if type(self.anchor_stride) is not int or self.anchor_stride < 1:
            raise ValueError("anchor_stride must be a positive integer")


@dataclass(frozen=True)
class FrameLabelConfig:
    """Columns/table keys are root-local; optional episode columns scope IDs further.

    'auto' chooses subtask_index/subtask_id from info.json, or task text when neither
    exists. Explicitly named columns/tables must exist. Unknown IDs raise.
    Text tables are projected and loaded once per root.
    """

    task_column: str = "task_index"
    task_table: str = "meta/tasks.parquet"
    task_id_column: str = "task_index"
    task_text_column: str = "task"
    task_episode_column: str | None = None
    task_records_key: str | None = None
    subtask_column: str | None = "auto"
    subtask_table: str = "meta/subtasks.parquet"
    subtask_id_column: str = "subtask_index"
    subtask_text_column: str = "subtask"
    subtask_episode_column: str | None = None
    subtask_records_key: str | None = None


class LeRobotFrameDataset(Dataset):
    """Video windows and anchor text with O(episodes) indexing.

    Raw reads preserve [t-k,t+a], padding only at episode boundaries. Action/video
    subclasses add their own forward candidates, block alignment and training fields.
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
        tolerance_s=1e-4,
        episodes=None,
        prompt_formatter=None,
        video=True,
        labels=None,
        video_stride=4,
        seed=0,
        rank=0,
        full_history=True,
        history_range=(1, 32),
    ):
        self.candidates = candidates or FrameCandidateConfig()
        if type(video_stride) is not int or video_stride < 1:
            raise ValueError("video_stride must be a positive integer")
        if self.candidates.past_frames % video_stride:
            raise ValueError("past_frames must be divisible by video_stride to include the anchor")
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
        lengths = self.episodes["dataset_to_index"] - self.episodes["dataset_from_index"]
        if (lengths <= 0).any():
            raise ValueError("Episode length must be positive")
        # 找到对应episodes在文件中的行号
        self.episode_rows = np.flatnonzero(np.isin(ids, list(self.selected_episodes)))
        starts, counts = self._candidate_ranges(lengths[self.episode_rows])
        keep = counts > 0
        self.episode_rows = self.episode_rows[keep]
        self.anchor_starts = starts[keep]
        counts = counts[keep]
        # 只保留 episode 的候选数量前缀和，不展开每个窗口。
        self.cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        if not len(self):
            raise ValueError(f"No legal frame candidates in {self.root}")
        subtask = self.labels.subtask_column
        if subtask == "auto":
            subtask = next((name for name in ("subtask_index", "subtask_id") if name in self.info["features"]), None)
        self.subtask_column = subtask
        self.task_texts = self._load_texts("task")
        self.subtask_texts = self._load_texts("subtask") if subtask is not None else {}

    def _candidate_ranges(self, lengths):
        """Raw inspection permits every frame, including terminal observations."""
        return np.zeros(len(lengths), dtype=np.int64), (lengths - 1) // self.candidates.anchor_stride + 1

    def _forward_candidate_counts(self, available):
        """Count complete forward starts, retaining one window for short episodes."""
        return np.where(
            available > 0,
            np.maximum(1, (available - self.candidates.future_frames) // self.candidates.anchor_stride + 1),
            0,
        )

    def _load_texts(self, kind):
        config = self.labels
        path = Path(self.root) / getattr(config, f"{kind}_table")
        id_column, text_column = getattr(config, f"{kind}_id_column"), getattr(config, f"{kind}_text_column")
        episode_column = getattr(config, f"{kind}_episode_column")
        records_key = getattr(config, f"{kind}_records_key")
        columns = [id_column, text_column] + ([episode_column] if episode_column else [])

        def records():
            if path.suffix == ".jsonl":
                with path.open() as file:
                    for line in file:
                        if not line.strip():
                            continue
                        parent = json.loads(line)
                        children = parent[records_key] if records_key else [parent]
                        for child in children:
                            row = {name: child[name] for name in (id_column, text_column)}
                            if episode_column:
                                row[episode_column] = child.get(episode_column, parent.get(episode_column))
                            yield row
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
            if episode_column and type(row[episode_column]) is not int:
                raise ValueError(f"{path}: invalid episode ID")
            key = (row[episode_column], label) if episode_column else label
            if key in texts:
                raise ValueError(f"{path}: duplicate {kind} ID {key}")
            texts[key] = text
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

    def plan_read_window(self, index, *, columns=None):
        row, anchor = self.locate(index)
        ep = {name: values[row].item() for name, values in self.episodes.items()}
        return plan_frame_window(
            self.root,
            ep["episode_index"],
            anchor,
            self.candidates.past_frames,
            self.candidates.future_frames,
            info=self.info,
            ep=ep,
            columns=columns,
            cameras=self.cameras,
            video_stride=self.video_stride,
        )

    def _label_id(self, value, kind):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{kind}: expected an integer ID, got {value!r}")
        return int(value)

    def _caption(self, labels, episode):
        task = self._label_id(labels[self.labels.task_column], "task")
        task_key = (episode, task) if self.labels.task_episode_column else task
        if task_key not in self.task_texts:
            raise ValueError(f"Unknown task ID {task_key}")
        if self.subtask_column is None:
            return self.task_texts[task_key], task, None, "task"
        subtask = self._label_id(labels[self.subtask_column], "subtask")
        subtask_key = (episode, subtask) if self.labels.subtask_episode_column else subtask
        if subtask_key not in self.subtask_texts:
            raise ValueError(f"Unknown subtask ID {subtask_key}")
        return self.subtask_texts[subtask_key], task, subtask, "subtask"

    def read_sample(self, index, *, video=None, block_size=None):
        """Read anchor text and payloads; optional forward action blocks pad once."""
        window = self.plan_read_window(index)
        label_columns = [self.labels.task_column] + ([self.subtask_column] if self.subtask_column else [])
        sample = read_frame_window(
            window,
            label_columns=label_columns,
            video=self.video if video is None else video,
            decode_threads=self.decode_threads,
            tolerance_s=self.tolerance_s,
            block_size=block_size,
        )
        caption, task, subtask, kind = self._caption(sample["anchor_labels"], window.episode)
        row, _ = self.locate(index)
        source_fps = float(self.episodes["source_fps"][row])
        sample.update(
            sample_kind="frame_anchor",
            episode=window.episode,
            caption_kind=kind,
            task_index=task,
            subtask_index=subtask,
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
        row, anchor = self.locate(index)
        episode = int(self.episodes["episode_index"][row])
        try:
            return self.make_sample(index)
        except Exception as error:
            error.add_note(f"source={self.source} root={self.root} episode={episode} t={anchor} index={index}")
            raise

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
    """Forward frame candidates and existing per-block action training semantics.

    Raw reads retain the requested horizon. Prepared samples repeat only the short
    tail needed for complete blocks and include policy/history/packing metadata.
    Training and model validation entry points remain gated during data review.
    """

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
        tolerance_s=1e-4,
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
    ):
        self.columns = columns or ActionColumns()
        self.processor = processor
        self.geometry = geometry or Geometry()
        candidates = candidates or FrameCandidateConfig()
        if candidates.past_frames != 0:
            raise ValueError("ActionFrameDataset currently requires past_frames=0")
        if candidates.future_frames < 1 or candidates.future_frames % self.geometry.actions_per_block:
            raise ValueError("future_frames must be a positive multiple of actions_per_block")
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
        )

    def _candidate_ranges(self, lengths):
        """Match finite frame_window starts, including one padded short window."""
        offset = self.columns.target_offset
        starts = np.full(len(lengths), max(0, -offset), dtype=np.int64)
        available = np.minimum(lengths, lengths + 1 - offset) - starts - 1
        return starts, self._forward_candidate_counts(available)

    def plan_read_window(self, index):
        return super().plan_read_window(index, columns=self.columns)

    def make_sample(self, index, *, mode=None, history=None):
        if self.processor is None or not callable(self.tokenize):
            raise ValueError("Prepared action samples require processor and tokenize; use read_sample for raw data")
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown explicit task: {mode}")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        # 只读一次 Parquet。视频按实际有效终点解码，避免 offset/短窗口中混入后续真实帧。
        raw = self.read_sample(index, video=False, block_size=self.geometry.actions_per_block)
        total = len(raw["action"])
        latents = 1 + total // self.geometry.actions_per_latent
        if mode is None or history is None:
            self._get_rng()
        if mode is None:
            mode = str(self.rng.choice(MODES, p=self.mode_weights)) if self.mode == "joint" else self.mode
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
    """Forward frame candidates with existing causal video training semantics.

    Raw reads keep the requested horizon; prepared samples align the real prefix
    to video latent blocks. Training/model-validation entry points remain gated.
    """

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
        candidates = candidates or FrameCandidateConfig()
        if candidates.past_frames != 0:
            raise ValueError("VideoFrameDataset currently requires past_frames=0")
        if candidates.future_frames < 1:
            raise ValueError("VideoFrameDataset requires positive future_frames")
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

    def _candidate_ranges(self, lengths):
        """Keep complete forward windows and one window for each short episode."""
        available = lengths - 1
        return np.zeros(len(lengths), dtype=np.int64), self._forward_candidate_counts(available)

    def plan_read_window(self, index):
        window = super().plan_read_window(index)
        if self.auto_video_stride and window.video_stride == 1:
            row, _ = self.locate(index)
            stride = 2 if self.episodes["source_fps"][row] > 16 else 1
            window = replace(window, video_stride=stride)
        return window

    def make_sample(self, index, *, history=None):
        if not callable(self.tokenize):
            raise ValueError("Prepared video samples require tokenize; use read_sample for raw data")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        raw = self.read_sample(index, video=False)
        window = raw["window"]
        real = int(raw["observation_valid_steps"].sum()) - 1
        block = window.video_stride * self.temporal_compression * self.latent_block_size
        total = ((real + block - 1) // block) * block
        timestamps = np.empty(total + 1, dtype=np.float64)
        timestamps[: real + 1] = raw["timestamps"][: real + 1].numpy()
        timestamps[real + 1 :] = timestamps[real] + np.arange(1, total - real + 1) / window.fps
        latents = 1 + total // (window.video_stride * self.temporal_compression)
        sample = dict(
            raw,
            timestamps=torch.from_numpy(timestamps),
            observation_valid_steps=torch.arange(total + 1) <= real,
            video_valid_steps=torch.arange(0, total + 1, window.video_stride) <= real,
            mode="video",
            history_blocks=self.select_history(latents, explicit=history),
            block_size=self.latent_block_size,
            condition_frame_indexes_vision=[0],
            num_valid_video_latents=num_valid_video_latents(real, window.video_stride, self.temporal_compression),
        )
        self._decode_sample_video(sample, real)
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
