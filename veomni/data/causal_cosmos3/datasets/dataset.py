# SPDX-License-Identifier: OpenMDW-1.1
"""Action candidates and on-demand samples, with complete short-range tail padding."""

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset, get_worker_info

from ..readers.lerobot import ActionColumns, plan_window, read_metadata, read_window
from ..transforms.layout import VIEW_DESCRIPTION, VIEW_ROLES, camera_layout, format_prompt
from ..transforms.processing import Geometry


logger = logging.getLogger(__name__)
MODES = ("forward_dynamics", "inverse_dynamics", "policy")


@dataclass(frozen=True)
class CandidateConfig:
    mode: str = "frame_window"
    max_actions: int = 160
    start_stride: int = 1
    overlap: int = 0
    use_subtask: bool = True
    borrow_subtask_frames: bool = False

    def validate(self, geometry):
        if self.mode not in ("frame_window", "segmented") or self.start_stride < 1:
            raise ValueError("Unknown candidate mode or invalid start stride")
        if self.max_actions != -1 and (self.max_actions < 1 or self.max_actions % geometry.actions_per_block):
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
class ActionDataset(Dataset):
    """One subset: numeric episode/window arrays, interned captions, no open resources.

    tokenize receives the final prompt and returns actual 1D token IDs including
    special tokens. Optional prompt_formatter(sample) fixes prompt augmentation
    before tokenization and packing. __getitem__ logs sample errors and returns [];
    successful samples return dict. Resource exhaustion always propagates.
    """

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
        mode_weights=(1, 1, 1),
        full_history=True,
        history_range=(1, 32),
        decode_threads=0,
        tolerance_s=1e-4,
        episodes=None,
        prompt_formatter=None,
        video=True,
    ):
        # 保存读取/处理配置；此阶段只读取 metadata，不加载视频或 action 内容。
        self.root = str(Path(root).resolve())
        self.source, self.processor, self.tokenize = source, processor, tokenize
        self.columns, self.candidates = columns or ActionColumns(), candidates or CandidateConfig()
        self.geometry = geometry or Geometry()
        self.resolution, self.viewpoint = str(resolution), viewpoint
        self.candidates.validate(self.geometry)
        for key, value in {
            "actions_per_block": self.geometry.actions_per_block,
            "video_stride": self.geometry.video_stride,
            "temporal_compression_factor": self.geometry.temporal_compression,
        }.items():
            if key in processor.statistics_geometry and processor.statistics_geometry[key] != value:
                raise ValueError(f"Statistics {key} differs from the configured block geometry")
        self.seed, self.rank, self.rng = seed, rank, None
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
        if not 1 <= history_range[0] <= history_range[1]:
            raise ValueError("Invalid history_range")
        self.full_history, self.history_range = full_history, history_range
        self.decode_threads, self.tolerance_s, self.video = decode_threads, tolerance_s, video
        self.prompt_formatter = prompt_formatter
        if viewpoint not in VIEW_ROLES:
            raise ValueError(f"Unsupported action viewpoint: {viewpoint}")
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
        c, block, offset = self.candidates, self.geometry.actions_per_block, self.columns.target_offset
        for i, ep in enumerate(ids):
            # selected 使用真实 episode 编号；窗口中保存的 i 是元数据行号。
            if selected is not None and int(ep) not in selected:
                continue
            length = int(self.episodes["dataset_to_index"][i] - self.episodes["dataset_from_index"][i])
            # n 步 action 需要 n+1 个 state；同时保证偏移后的 target 不越出 episode。
            first, stop = max(0, -offset), min(length, length + 1 - offset)
            available = stop - first - 1
            if available < 1:
                continue
            if c.mode == "frame_window" and c.max_actions != -1:
                # 按固定步长展开滑窗，可跨子任务边界；文本只由窗口起点决定。
                actions = min(available, c.max_actions)
                # 长 episode 只生成容得下的完整窗口；短 episode 保留一个有效短窗口。
                window_count = max(1, (available - c.max_actions) // c.start_stride + 1)
                for k in range(window_count):
                    start = first + k * c.start_stride
                    records.append((i, start, actions, self._subtask_text_id(i, start)))
                continue
            # segmented 或不限长度时按 subtask 取范围；没有合法 subtask 时使用整个 episode。
            ranges = self.subtasks.get(i)
            ranges = ranges if ranges is not None else np.array([[0, length, 0]])
            for lo, hi, text_id in ranges:
                subtask = text_id != 0
                a, b = max(first, int(lo)), min(stop, int(hi) + int(subtask))
                if c.max_actions == -1:
                    # 完整范围不切窗、不借帧；读取时再补齐到整 block。
                    if b - a > 1:
                        records.append((i, a, b - a - 1, int(text_id)))
                    continue
                if subtask and c.borrow_subtask_frames:
                    # 允许借帧时，先向后、再向前扩展到整 block，且不跨出 episode。
                    # 借帧只调整边界，仍使用当前子任务的 text_id。
                    missing = max(1, (b - a - 1 + block - 1) // block) * block + 1 - (b - a)
                    after = min(missing, stop - b)
                    b += after
                    a -= min(missing - after, a - first)
                # 短窗口保留全部有效步，读取时再 padding；长窗口由右对齐窗口覆盖尾部。
                for start, actions in segment_windows(a, b, c.max_actions, c.overlap):
                    records.append((i, start, actions, int(text_id)))
        # 每行 [episode 元数据行号, 实际起点, 有效 action 数, 文本编号]；不存视频或 action 内容。
        self.windows = np.asarray(records, dtype=np.int64).reshape(-1, 4)
        if not len(self.windows):
            raise ValueError(f"No legal candidates in {self.root}")

    def __len__(self):
        # 返回候选窗口总数，不是 episode 数或原始帧数。
        return len(self.windows)

    def window(self, index):
        # 将候选编号转为文件路径、行偏移、时间偏移及补齐长度；此处不读取样本内容。
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # index 直接对应一个窗口，各模式在此使用相同的读取逻辑。
        episode_row, start, actions, text_id = map(int, self.windows[index])
        # 从按列存储的元数据中取出当前 episode，供 plan_window 计算文件位置。
        ep = {c: v[episode_row].item() for c, v in self.episodes.items()}
        window = plan_window(
            self.root,
            int(ep["episode_index"]),
            start,
            actions,
            info=self.info,
            ep=ep,
            columns=self.columns,
            cameras=self.cameras,
            block_size=self.geometry.actions_per_block,
            video_stride=self.geometry.video_stride,
        )
        return window, self.texts[text_id], float(ep["source_fps"])

    def __getitem__(self, index, *, mode=None, history=None):
        try:
            return self.make_sample(index, mode=mode, history=history)
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

    def make_sample(self, index, *, mode=None, history=None):
        """Build a window with optional explicit task/history before tokenization.

        Errors propagate to the caller; __getitem__ handles logging and skipping.
        Explicit task/history do not initialize or consume the training sampling RNG.
        """
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown explicit task: {mode}")
        if history is not None and (type(history) is not int or history < 1):
            raise ValueError("Explicit history must be a positive integer")
        window, caption, source_fps = self.window(index)
        raw = read_window(window, video=self.video, tolerance_s=self.tolerance_s, decode_threads=self.decode_threads)
        if (mode is None or history is None) and self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, worker.id if worker else 0, 0])
            )
        # 这里随机选择训练任务和 history；候选编号由外层采样逻辑决定。
        if mode is None:
            mode = str(self.rng.choice(MODES, p=self.mode_weights)) if self.mode == "joint" else self.mode
        if history is None:
            history = (
                1 + window.actions // self.geometry.actions_per_latent
                if self.full_history
                else int(self.rng.integers(self.history_range[0], self.history_range[1] + 1))
            )
        # processor 负责模板映射、相对 action 编码、归一化和有效步 mask。
        sample = dict(raw, **self.processor(raw, self.geometry))
        sample.pop("state")  # full absolute trajectory is replaced by block states and anchors
        sample.update(
            source=self.source,
            subset=self.root,
            candidate_index=int(index),
            caption=caption or self.tasks.get(raw["task_index"], ""),  # 无 subtask 文本时回退到 task 文本。
            storage_fps=window.fps,
            source_fps=source_fps,
            conditioning_fps=source_fps / self.geometry.video_stride,
            conditioning_fps_action=source_fps,
            mode=mode,
            history_blocks=history,
            block_size=self.geometry.actions_per_block // self.geometry.actions_per_latent,
            template_id=self.processor.template.template_id,
            viewpoint=self.viewpoint,
            additional_view_description=VIEW_DESCRIPTION[self.viewpoint],
            resolution=self.resolution,
            image_size=(window.height, window.width),
        )
        # 先生成最终文本 token，再结合视频/action 的 token 布局计算长度，供主进程 packing。
        prompt = (self.prompt_formatter or format_prompt)(sample)
        if not prompt.strip():
            raise ValueError("A nonempty task prompt is required")
        ids = torch.as_tensor(self.tokenize(prompt), dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("tokenize must return a 1D sequence, including final special tokens")
        sample.update(prompt=prompt, text_token_ids=ids)
        layout = self.geometry.token_layout(window.actions, window.height, window.width, len(ids))
        sample.update(token_layout=layout, token_length=sum(layout.values()))
        latents = 1 + window.actions // self.geometry.actions_per_latent
        sample["condition_frame_indexes_vision"] = list(range(latents)) if mode == "inverse_dynamics" else [0]
        sample["condition_frame_indexes_action"] = list(range(window.actions)) if mode == "forward_dynamics" else []
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
