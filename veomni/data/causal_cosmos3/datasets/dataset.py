# SPDX-License-Identifier: OpenMDW-1.1
"""Shared LeRobot candidates and task-specific on-demand samples."""

import copy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset, get_worker_info

from ..readers.lerobot import ActionColumns, episode_frame_bounds, plan_window, read_metadata, read_window
from ..transforms.layout import VIEW_DESCRIPTION, VIEW_ROLES, camera_layout, format_prompt
from ..transforms.processing import Geometry, num_valid_video_latents, video_token_layout


logger = logging.getLogger(__name__)
MODES = ("forward_dynamics", "inverse_dynamics", "policy")


def resolve_action_modes(mode):
    """Expand joint or an explicit task list, preserving the configured order."""
    if isinstance(mode, str):
        modes = MODES if mode == "joint" else (mode,)
    elif isinstance(mode, (list, tuple)):
        modes = tuple(mode)
    else:
        raise ValueError("Task mode must be a task name, 'joint', or a nonempty task list")
    if (
        not modes
        or any(not isinstance(task, str) or task not in MODES for task in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("Task mode must contain unique tasks from forward_dynamics, inverse_dynamics, policy")
    return modes


@dataclass(frozen=True)
class CandidateConfig:
    mode: str = "frame_window"
    max_actions: int = 160
    start_stride: int = 1
    overlap: int = 0
    use_subtask: bool = True

    def validate(self, geometry=None):
        if self.mode not in ("frame_window", "segmented") or self.start_stride < 1:
            raise ValueError("Unknown candidate mode or invalid start stride")
        if self.max_actions != -1 and (
            self.max_actions < 1 or (geometry is not None and self.max_actions % geometry.actions_per_block)
        ):
            raise ValueError("max_actions must be -1 or a positive multiple of actions_per_block")
        if self.overlap < 0 or (self.max_actions != -1 and self.overlap >= self.max_actions):
            raise ValueError("overlap must be nonnegative and smaller than max_actions unless max_actions is -1")


def segment_windows(first, stop, maximum, overlap):
    """Keep short ranges whole (the reader pads); right-align long tails."""
    available = stop - first - 1
    if available < 1:
        return []
    if available < maximum:
        return [(first, available)]
    tail = first + available - maximum
    starts = list(range(first, tail + 1, maximum - overlap))
    if starts[-1] != tail:
        starts.append(tail)
    return [(start, maximum) for start in starts]


# 每个实例管理一个 root：初始化建候选索引，按需读取单个窗口；混合采样和 packing 由外层负责。
class LeRobotDataset(Dataset, ABC):
    """Shared LeRobot metadata, explicit windows, history and prompt preparation.

    Subclasses own make_sample and any restrictions on legal frame bounds.
    No action columns, processor or geometry are required by this base.
    """

    def __init__(
        self,
        root,
        *,
        source,
        tokenize,
        cameras,
        resolution="256",
        viewpoint="concat_view",
        candidates=None,
        seed=0,
        rank=0,
        full_history=True,
        history_range=(1, 32),
        decode_threads=0,
        tolerance_s=1e-4,
        episodes=None,
        prompt_formatter=None,
        video=True,
    ):
        # 公共初始化只依赖 metadata，不读取动作数据、模板或统计量。
        self.root = str(Path(root).resolve())
        self.source, self.tokenize = source, tokenize
        self.candidates = candidates or CandidateConfig()
        self.candidates.validate()
        self.resolution, self.viewpoint = str(resolution), viewpoint
        self.seed, self.rank, self.rng = seed, rank, None
        if not 1 <= history_range[0] <= history_range[1]:
            raise ValueError("Invalid history_range")
        self.full_history, self.history_range = full_history, history_range
        self.decode_threads, self.tolerance_s, self.video = decode_threads, tolerance_s, video
        self.prompt_formatter = prompt_formatter
        if viewpoint not in VIEW_ROLES:
            raise ValueError(f"Unsupported viewpoint: {viewpoint}")
        # episodes 保存长度、文件位置等数值信息；相机布局按分辨率桶和视角角色确定。
        self.info, self.episodes = read_metadata(self.root, [cameras[role] for role in VIEW_ROLES[viewpoint]])
        self.cameras = camera_layout(self.info, cameras, self.resolution, viewpoint)
        self.tasks = {}
        task_path = Path(self.root) / "meta/tasks.parquet"
        if task_path.exists():
            with pq.ParquetFile(task_path) as parquet:
                for batch in parquet.iter_batches(use_threads=False):
                    for row in batch.to_pylist():
                        self.tasks[int(row["task_index"])] = row.get("task", row.get("__index_level_0__", ""))
        # task 文本按 task_index 查找；subtask 文本在当前 root 内去重，编号 0 表示无子任务文本。
        self.texts = [""]
        self.subtasks = {}
        self._load_episode_annotations()
        self._build_windows(episodes)

    def _load_episode_annotations(self):
        """读取子任务区间、去重文本和源帧率，供后续窗口生成使用。"""
        ids = self.episodes["episode_index"]
        id_to_row = {int(ep): i for i, ep in enumerate(ids)}
        text_ids = {"": 0}

        def intern(text):
            if text not in text_ids:
                text_ids[text] = len(self.texts)
                self.texts.append(text)
            return text_ids[text]

        path = Path(self.root) / "meta/episodes.jsonl"
        if path.exists():
            with path.open() as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    i = id_to_row.get(record["episode_index"])
                    if i is None:
                        continue
                    if "source_fps" in record:
                        self.episodes["source_fps"][i] = float(record["source_fps"])
                    if not self.candidates.use_subtask:
                        continue
                    length = int(self.episodes["dataset_to_index"][i] - self.episodes["dataset_from_index"][i])
                    parsed, previous = [], 0
                    try:
                        for sub in record.get("action_config") or []:
                            start, stop, text = sub["start_frame"], sub["end_frame"], sub["action_text"]
                            if (
                                type(start) is not int
                                or type(stop) is not int
                                or not previous <= start < stop <= length
                                or not isinstance(text, str)
                                or not text.strip()
                            ):
                                raise ValueError("invalid subtask bounds/text")
                            parsed.append((start, stop, intern(text)))
                            previous = stop
                    except (ValueError, KeyError, TypeError) as error:
                        logger.warning("%s episode %s: fallback to episode/task text: %s", self.root, ids[i], error)
                        parsed = []
                    if parsed:
                        # 每行 [start_frame, end_frame, text_id]，形状为 [子任务数, 3]。
                        self.subtasks[i] = np.asarray(parsed, dtype=np.int64)
        fps = self.episodes["source_fps"]
        fps[np.isnan(fps)] = float(self.info["fps"])
        if not np.isfinite(fps).all() or (fps <= 0).any() or float(self.info["fps"]) <= 0:
            raise ValueError("Storage/source FPS must be positive and finite")

    def _subtask_text_id(self, episode_row, start):
        """按滑窗起点查找子任务文本编号；未命中时返回 0，取样时回退到 task 文本。"""
        subtasks = self.subtasks.get(episode_row)
        if subtasks is not None:
            # 找到最后一个起点不晚于 start 的子任务，再检查是否位于其结束边界内。
            t = int(np.searchsorted(subtasks[:, 0], start, side="right")) - 1
            if t >= 0 and start < subtasks[t, 1]:
                return int(subtasks[t, 2])
        return 0

    def _build_windows(self, selected):
        """所有模式统一展开窗口索引：一行一个窗口，提前确定起点、有效步数和文本。"""
        ids = self.episodes["episode_index"]
        records = []
        selected = None if selected is None else set(selected)
        self.selected_episodes = set(map(int, ids)) if selected is None else selected
        c = self.candidates
        for i, ep in enumerate(ids):
            # selected 使用真实 episode 编号；窗口中保存的 i 是元数据行号。
            if selected is not None and int(ep) not in selected:
                continue
            length = int(self.episodes["dataset_to_index"][i] - self.episodes["dataset_from_index"][i])
            # 子类限定合法范围；公共枚举保留 N 个间隔对应 N+1 帧的语义。
            first, stop = self._episode_bounds(length)
            available = stop - first - 1
            if available < 1:
                continue
            if c.mode == "frame_window" and c.max_actions != -1:
                # 按固定步长展开滑窗，可跨子任务边界；文本只由窗口起点决定。
                frame_intervals = min(available, c.max_actions)
                # 长 episode 只生成容得下的完整窗口；短 episode 保留一个有效短窗口。
                window_count = max(1, (available - c.max_actions) // c.start_stride + 1)
                for k in range(window_count):
                    start = first + k * c.start_stride
                    records.append((i, start, frame_intervals, self._subtask_text_id(i, start)))
                continue
            # segmented 或不限长度时按 subtask 取范围；没有合法 subtask 时使用整个 episode。
            ranges = self.subtasks.get(i)
            ranges = ranges if ranges is not None else np.array([[0, length, 0]])
            for lo, hi, text_id in ranges:
                subtask = text_id != 0
                a, b = max(first, int(lo)), min(stop, int(hi) + int(subtask))
                if c.max_actions == -1:
                    # 完整范围不切窗；统一由 reader 在尾部补齐到整 block。
                    if b - a > 1:
                        records.append((i, a, b - a - 1, int(text_id)))
                    continue
                # 短窗口保留全部有效步，读取时再 padding；长窗口由右对齐窗口覆盖尾部。
                for start, frame_intervals in segment_windows(a, b, c.max_actions, c.overlap):
                    records.append((i, start, frame_intervals, int(text_id)))
        # 所有模式只登记真实范围；block 补齐由 plan_window/read_window 统一处理。
        # 每行 [episode 元数据行号, 实际起点, 有效原始帧间隔数, 文本编号]；不存视频或 action 内容。
        self.windows = np.asarray(records, dtype=np.int64).reshape(-1, 4)
        if not len(self.windows):
            raise ValueError(f"No legal candidates in {self.root}")

    def __len__(self):
        # 返回候选窗口总数，不是 episode 数或原始帧数。
        return len(self.windows)

    def get_window_metadata(self, index):
        """Resolve one candidate to episode metadata, start, real intervals and caption."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # index 直接对应一个窗口，各模式在此使用相同的读取逻辑。
        episode_row, start, frame_intervals, text_id = map(int, self.windows[index])
        # 从按列存储的元数据中取出当前 episode，供 plan_window 计算文件位置。
        ep = {c: v[episode_row].item() for c, v in self.episodes.items()}
        return ep, start, frame_intervals, self.texts[text_id]

    def __getitem__(self, index, **options):
        try:
            return self.make_sample(index, **options)
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except Exception as error:
            logger.warning(
                "Sample failure source=%s subset=%s candidate_index=%s: %s: %s",
                self.source,
                self.root,
                index,
                type(error).__name__,
                error,
            )
            return []

    def _episode_bounds(self, length):
        """Exclusive frame bounds; subclasses may restrict the readable range."""
        return episode_frame_bounds(length)

    @abstractmethod
    def make_sample(self, index, **options):
        """Build a task sample; errors propagate to the caller or __getitem__."""

    def _get_rng(self):
        if self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, worker.id if worker else 0, 0])
            )
        return self.rng

    def select_history(self, num_latents, *, explicit=None, rng=None):
        """Choose latent-frame history; explicit values or a supplied RNG leave training RNG untouched."""
        if explicit is not None:
            if type(explicit) is not int or explicit < 1:
                raise ValueError("Explicit history must be a positive integer")
            return explicit
        if self.full_history:
            return num_latents
        rng = self._get_rng() if rng is None else rng
        return int(rng.integers(self.history_range[0], self.history_range[1] + 1))

    def prepare_sample(self, sample, *, index, caption, source_fps):
        """Attach common identity/video metadata and tokenize the final task prompt.

        The subclass supplies its task fields before calling this method and
        computes the final token budget afterwards.
        """
        window = sample["window"]
        sample.update(
            source=self.source,
            subset=self.root,
            candidate_index=int(index),
            caption=caption or self.tasks.get(sample["task_index"], ""),
            storage_fps=window.fps,
            source_fps=source_fps,
            conditioning_fps=source_fps / window.video_stride,
            viewpoint=self.viewpoint,
            additional_view_description=VIEW_DESCRIPTION[self.viewpoint],
            resolution=self.resolution,
            image_size=(window.height, window.width),
        )
        prompt = (self.prompt_formatter or format_prompt)(sample)
        if not prompt.strip():
            raise ValueError("A nonempty task prompt is required")
        ids = torch.as_tensor(self.tokenize(prompt), dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("tokenize must return a 1D sequence, including final special tokens")
        sample.update(prompt=prompt, text_token_ids=ids)
        return sample

    def state_dict(self):
        # 仅保存任务/history 的随机状态；采样位置由外层 sampler/loader 管理。
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state) if self.rng is not None else None,
        )

    def load_state_dict(self, state):
        # 恢复当前 Dataset 的随机状态，不重新读取或重建候选索引。
        self.rng = None
        if state["rng"] is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class ActionDataset(LeRobotDataset):
    """Action window rules, FD/ID/Policy sampling and action/state processing."""

    def __init__(
        self,
        root,
        *,
        source,
        processor,
        tokenize,
        cameras,
        resolution="256",
        viewpoint="concat_view",
        columns=None,
        candidates=None,
        geometry=None,
        seed=0,
        rank=0,
        mode="joint",
        mode_weights=None,
        full_history=True,
        history_range=(1, 32),
        decode_threads=0,
        tolerance_s=1e-4,
        episodes=None,
        prompt_formatter=None,
        video=True,
    ):
        self.processor = processor
        self.columns = columns or ActionColumns()
        self.geometry = geometry or Geometry()
        candidates = candidates or CandidateConfig()
        candidates.validate(self.geometry)
        for key, value in {
            "actions_per_block": self.geometry.actions_per_block,
            "video_stride": self.geometry.video_stride,
            "temporal_compression_factor": self.geometry.temporal_compression,
        }.items():
            if key in processor.statistics_geometry and processor.statistics_geometry[key] != value:
                raise ValueError(f"Statistics {key} differs from the configured block geometry")
        self.modes = resolve_action_modes(mode)
        weights = np.ones(len(self.modes)) if mode_weights is None else np.asarray(mode_weights, dtype=float)
        # Single-task strings historically accepted the three joint weights, but
        # never sampled from them. Keep that configuration compatibility.
        lengths = (1, len(MODES)) if isinstance(mode, str) and mode != "joint" else (len(self.modes),)
        if (
            weights.ndim != 1
            or len(weights) not in lengths
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isfinite(weights.sum())
            or weights.sum() <= 0
        ):
            raise ValueError("mode_weights must match mode tasks, be finite and nonnegative, and have a positive sum")
        self.mode_weights = np.ones(1) if len(self.modes) == 1 else weights / weights.sum()
        super().__init__(
            root,
            source=source,
            tokenize=tokenize,
            cameras=cameras,
            resolution=resolution,
            viewpoint=viewpoint,
            candidates=candidates,
            seed=seed,
            rank=rank,
            full_history=full_history,
            history_range=history_range,
            decode_threads=decode_threads,
            tolerance_s=tolerance_s,
            episodes=episodes,
            prompt_formatter=prompt_formatter,
            video=video,
        )

    def _episode_bounds(self, length):
        # N 个 action 对应 N+1 个 state；偏移后的 target 也必须在 episode 内。
        return episode_frame_bounds(length, columns=self.columns)

    def plan_read_window(self, index):
        """Plan action/video reads and padding; return the plan, caption and source FPS."""
        ep, start, frame_intervals, caption = self.get_window_metadata(index)
        window = plan_window(
            self.root,
            int(ep["episode_index"]),
            start,
            frame_intervals,
            info=self.info,
            ep=ep,
            columns=self.columns,
            cameras=self.cameras,
            block_size=self.geometry.actions_per_block,
            video_stride=self.geometry.video_stride,
        )
        return window, caption, float(ep["source_fps"])

    def num_video_latents(self, window):
        """Count padded video latents for training and validation history."""
        return 1 + window.actions // self.geometry.actions_per_latent

    def make_sample(self, index, *, mode=None, history=None):
        """Build a window with optional explicit task/history before tokenization.

        Errors propagate to the caller; __getitem__ handles logging and skipping.
        Explicit task/history do not initialize or consume the training sampling RNG.
        """
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown explicit task: {mode}")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        window, caption, source_fps = self.plan_read_window(index)
        raw = read_window(window, video=self.video, tolerance_s=self.tolerance_s, decode_threads=self.decode_threads)
        if mode is None or history is None:
            self._get_rng()
        # 这里随机选择训练任务和 history；候选编号由外层采样逻辑决定。
        if mode is None:
            mode = self.modes[0] if len(self.modes) == 1 else str(self.rng.choice(self.modes, p=self.mode_weights))
        latents = self.num_video_latents(window)
        history = self.select_history(latents, explicit=history)
        # processor 负责模板映射、相对 action 编码、归一化和有效步 mask。
        sample = dict(raw, **self.processor(raw, self.geometry))
        sample.pop("state")  # full absolute trajectory is replaced by block states and anchors
        sample.update(
            conditioning_fps_action=source_fps,
            mode=mode,
            history_blocks=history,
            block_size=self.geometry.actions_per_block // self.geometry.actions_per_latent,
            template_id=self.processor.template.template_id,
        )
        sample = self.prepare_sample(sample, index=index, caption=caption, source_fps=source_fps)
        ids = sample["text_token_ids"]
        layout = self.geometry.token_layout(window.actions, window.height, window.width, len(ids))
        sample.update(token_layout=layout, token_length=sum(layout.values()))
        sample["condition_frame_indexes_vision"] = list(range(latents)) if mode == "inverse_dynamics" else [0]
        sample["condition_frame_indexes_action"] = list(range(window.actions)) if mode == "forward_dynamics" else []
        return sample


class VideoDataset(LeRobotDataset):
    """Video/task-only LeRobot samples with deterministic source-FPS downsampling."""

    def __init__(
        self,
        root,
        *,
        latent_block_size=2,
        temporal_compression=4,
        spatial_compression=16,
        patch_size=2,
        full_history=False,
        **kwargs,
    ):
        sizes = (latent_block_size, temporal_compression, spatial_compression, patch_size)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Video block, compression and patch sizes must be positive integers")
        self.latent_block_size = latent_block_size
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.patch_size = patch_size
        super().__init__(root, full_history=full_history, **kwargs)

    def plan_read_window(self, index):
        """Plan video reads and padding with deterministic FPS-based downsampling."""
        ep, start, frame_intervals, caption = self.get_window_metadata(index)
        source_fps = float(ep["source_fps"])
        stride = 2 if source_fps > 16 else 1
        window = plan_window(
            self.root,
            int(ep["episode_index"]),
            start,
            frame_intervals,
            info=self.info,
            ep=ep,
            columns=None,
            cameras=self.cameras,
            block_size=stride * self.temporal_compression * self.latent_block_size,
            video_stride=stride,
        )
        return window, caption, source_fps

    def num_video_latents(self, window):
        """Count padded video latents after this window's FPS-based downsampling."""
        return 1 + (window.video_frames - 1) // self.temporal_compression

    def make_sample(self, index, *, history=None):
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        window, caption, source_fps = self.plan_read_window(index)
        raw = read_window(window, video=self.video, tolerance_s=self.tolerance_s, decode_threads=self.decode_threads)
        latents = self.num_video_latents(window)
        history = self.select_history(latents, explicit=history)
        sample = dict(
            raw,
            mode="video",
            history_blocks=history,
            block_size=self.latent_block_size,
            condition_frame_indexes_vision=[0],
            num_valid_video_latents=num_valid_video_latents(
                window.real_intervals, window.video_stride, self.temporal_compression
            ),
        )
        sample = self.prepare_sample(sample, index=index, caption=caption, source_fps=source_fps)
        layout = video_token_layout(
            window.video_frames,
            window.height,
            window.width,
            len(sample["text_token_ids"]),
            temporal_compression=self.temporal_compression,
            spatial_compression=self.spatial_compression,
            patch_size=self.patch_size,
        )
        sample.update(token_layout=layout, token_length=sum(layout.values()))
        return sample


class StatefulConcatDataset(ConcatDataset):
    """PyTorch concatenation with only child state forwarding added."""

    def state_dict(self):
        return {str(i): ds.state_dict() for i, ds in enumerate(self.datasets)}

    def load_state_dict(self, state):
        for i, ds in enumerate(self.datasets):
            ds.load_state_dict(state[str(i)])


class UniformSource(IterableDataset):
    """Uniform candidates inside ONE weighted source; VeOmni chooses the source."""

    def __init__(self, dataset, seed, rank):
        self.dataset, self.seed, self.rank, self.rng = dataset, seed, rank, None

    def __iter__(self):
        if self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, worker.id if worker else 0, 1])
            )
        while True:
            sample = self.dataset[int(self.rng.integers(len(self.dataset)))]
            yield sample if isinstance(sample, dict) else None

    def state_dict(self):
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state) if self.rng is not None else None,
            dataset=self.dataset.state_dict(),
        )

    def load_state_dict(self, state):
        self.rng = None
        if state["rng"] is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.dataset.load_state_dict(state["dataset"])
