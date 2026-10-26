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
        if self.max_actions < 1 or self.max_actions % geometry.actions_per_block:
            raise ValueError("max_actions must be a positive multiple of actions_per_block")
        if not 0 <= self.overlap < self.max_actions:
            raise ValueError("overlap must be smaller than max_actions")


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
    """One subset: numeric episode/range arrays, interned captions, no open resources.

    tokenize receives the final prompt and returns actual 1D token IDs including
    special tokens. Optional prompt_formatter(sample) fixes prompt augmentation
    before tokenization and packing. Failed I/O returns []; successful reads dict.
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
        self.successes = self.failures = 0
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
        self._build_index(episodes)

    def _build_index(self, selected):
        # 只建立逻辑窗口索引，不切割磁盘文件。i 是 metadata 行号，不一定等于 episode_index。
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
        records, counts = [], []
        selected = None if selected is None else set(selected)
        c, block, offset = self.candidates, self.geometry.actions_per_block, self.columns.target_offset
        for i, ep in enumerate(ids):
            if selected is not None and int(ep) not in selected:
                continue
            length = int(self.episodes["dataset_to_index"][i] - self.episodes["dataset_from_index"][i])
            # n 步 action 需要 n+1 个 state；同时保证偏移后的 target 不越出 episode。
            first, stop = max(0, -offset), min(length, length + 1 - offset)
            available = stop - first - 1
            if available < 1:
                continue
            if c.mode == "frame_window":
                # 每个 episode 仅存一条范围记录，滑窗起点在 window() 中计算。
                # 此模式不按 subtask 边界截窗；subtask 只用于按窗口起点选取文本。
                records.append((i, first, min(available, c.max_actions), 0))
                counts.append(max(1, (available - c.max_actions) // c.start_stride + 1))
                continue
            # segmented 按 subtask 切窗；没有合法 subtask 时使用整个 episode。
            ranges = self.subtasks.get(i)
            ranges = ranges if ranges is not None else np.array([[0, length, 0]])
            for lo, hi, text_id in ranges:
                subtask = text_id != 0
                a, b = max(first, int(lo)), min(stop, int(hi) + int(subtask))
                if subtask and c.borrow_subtask_frames:
                    # 允许借帧时，先向后、再向前扩展到整 block，且不跨出 episode。
                    missing = max(1, (b - a - 1 + block - 1) // block) * block + 1 - (b - a)
                    after = min(missing, stop - b)
                    b += after
                    a -= min(missing - after, a - first)
                # 短窗口保留全部有效步，读取时再 padding；长窗口由右对齐窗口覆盖尾部。
                for start, actions in segment_windows(a, b, c.max_actions, c.overlap):
                    records.append((i, start, actions, int(text_id)))
                    counts.append(1)
        # 每行 [episode 行号, 起点, 有效 action 数, 文本编号]；累计数量用于定位候选编号。
        self.ranges = np.asarray(records, dtype=np.int64).reshape(-1, 4)
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        if not len(self.cumulative):
            raise ValueError(f"No legal candidates in {self.root}")

    def __len__(self):
        # 返回候选窗口总数，不是 episode 数或原始帧数。
        return int(self.cumulative[-1])

    def window(self, index):
        # 将候选编号转为文件路径、行偏移、时间偏移及补齐长度；此处不读取样本内容。
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        r = int(np.searchsorted(self.cumulative, index, side="right"))
        i, start, actions, text_id = map(int, self.ranges[r])
        if self.candidates.mode == "frame_window":
            start += (index - (int(self.cumulative[r - 1]) if r else 0)) * self.candidates.start_stride
            ranges = self.subtasks.get(i)
            if ranges is not None:
                t = int(np.searchsorted(ranges[:, 0], start, side="right")) - 1
                if t >= 0 and start < ranges[t, 1]:
                    text_id = int(ranges[t, 2])
        ep = {c: v[i].item() for c, v in self.episodes.items()}
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

    def __getitem__(self, index):
        window, caption, source_fps = self.window(index)
        try:
            # 按窗口读取表格、解码视频并 resize/拼接视角；末尾不足部分重复最后有效值。
            raw = read_window(
                window, video=self.video, tolerance_s=self.tolerance_s, decode_threads=self.decode_threads
            )
        except (MemoryError, torch.OutOfMemoryError):
            raise
        except Exception as error:
            self.failures += 1
            if self.failures <= 3 or self.failures % 100 == 0:
                logger.warning(
                    "Read failure source=%s subset=%s episode=%s start=%s successes=%s failures=%s: %s: %s",
                    self.source,
                    self.root,
                    window.episode,
                    window.start,
                    self.successes,
                    self.failures,
                    type(error).__name__,
                    error,
                )
            # 空列表作为跳过标记，由上层继续取下一候选；OOM 不在此处吞掉。
            return []
        self.successes += 1
        if self.rng is None:
            worker = get_worker_info()
            self.rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, worker.id if worker else 0, 0])
            )
        # 这里随机选择训练任务和 history；候选编号由外层采样逻辑决定。
        mode = str(self.rng.choice(MODES, p=self.mode_weights)) if self.mode == "joint" else self.mode
        history = (
            window.actions // self.geometry.actions_per_block
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
        # 仅保存任务/history 的随机状态和读取计数；采样位置由外层 sampler/loader 管理。
        return dict(
            rng=copy.deepcopy(self.rng.bit_generator.state) if self.rng is not None else None,
            successes=self.successes,
            failures=self.failures,
        )

    def load_state_dict(self, state):
        # 恢复当前 Dataset 的随机状态和计数，不重新读取或重建候选索引。
        self.rng = None
        if state["rng"] is not None:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.successes, self.failures = state["successes"], state["failures"]


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
