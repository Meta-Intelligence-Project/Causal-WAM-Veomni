"""Build video/action sources with shared sampling and native greedy batching."""

import glob
import json
import math
from pathlib import Path

from veomni.data.data_collator import NoopDataCollator
from veomni.data.data_loader import build_native_dataloader
from veomni.data.dataset import WeightedMultiSourceDataset
from veomni.data.dynamic_batching import TextBatchingStrategy
from veomni.distributed.parallel_state import get_parallel_state

from ..readers.lerobot import ActionColumns
from ..transforms.processing import ActionProcessor, Geometry
from .dataset import ActionDataset, CandidateConfig, StatefulConcatDataset, UniformSource, VideoDataset
from .frame_dataset import ActionFrameDataset, FrameCandidateConfig, VideoFrameDataset


def token_length(sample):
    return sample["token_length"]


def collate_action_pack(samples):
    """Wrap one greedy pack as a DiT microbatch without copying tensor data."""
    return {"samples": samples}


def _read_source_manifest(path):
    """Read LeRobot roots and episode IDs from a JSON manifest."""
    for record in json.loads(path.read_text(encoding="utf-8")):
        episodes = record["episodes"]
        if episodes:
            yield str(Path(record["lerobot_path"]).resolve()), episodes


def resolve_source_roots(source):
    """Normalize directory/glob/JSON roots without mutating the supplied source."""
    selections, has_json = {}, False
    for entry in source["roots"]:
        path = Path(entry)
        if path.is_dir():
            resolved = ((str(path.resolve()), None),)
        elif path.suffix.lower() == ".json":
            if "episodes" in source or "episodes" in source.get("read", {}):
                raise ValueError("JSON manifests cannot be combined with source.episodes or read.episodes")
            has_json = True
            resolved = _read_source_manifest(path.resolve())
        else:
            matches = sorted(root for root in glob.glob(str(entry)) if Path(root).is_dir())
            if not matches:
                raise ValueError(f"No roots matched {entry}")
            resolved = ((str(Path(root).resolve()), None) for root in matches)
        for root, episodes in resolved:
            if root in selections:
                raise ValueError(f"Duplicate roots in source {source['name']}: {root}")
            selections[root] = episodes
    if not selections:
        raise ValueError(f"No selected roots in source {source['name']}")
    # 复制原配置，保留 name、weight、read 等字段，不修改传入的 source。
    resolved_source = source.copy()
    # selections 的 key 是数据集目录，value 是 episode 列表（None 表示全部）。
    # 这里只取目录作为 roots，例如 {"/data/A": [2, 4]} -> ["/data/A"]，尚未写入 JSON 的 episodes。
    resolved_source["roots"] = list(selections.keys())
    if has_json:
        # 再将完整的“目录 -> episode 选择”映射写入 episodes，供后续按目录读取。
        resolved_source["episodes"] = selections
    elif isinstance(source.get("episodes"), dict):
        episodes = {str(Path(root).resolve()): ids for root, ids in source["episodes"].items()}
        if set(episodes) != set(selections):
            raise ValueError("Episode mapping must cover exactly the resolved source roots")
        resolved_source["episodes"] = episodes
    return resolved_source


def source_read_options(source, root):
    """Resolve read options for one root of a normalized source."""
    options = dict(source.get("read", {}))
    if "episodes" in source:
        if "episodes" in options:
            raise ValueError("Specify episodes on the source or in read, not both")
        selection = source["episodes"]
        if isinstance(selection, dict):
            selection = selection[root]
        options["episodes"] = selection
    if (selection := options.get("episodes")) is not None:
        if not isinstance(selection, list) or not selection or any(type(ep) is not int for ep in selection):
            raise ValueError("episodes must be a nonempty integer list (optionally mapped by root)")
    return options


class ActionBatchingStrategy(TextBatchingStrategy):
    """Keep VeOmni's greedy buffer and oversized singletons; skip text-only filtering."""

    def __init__(self, token_budget, buffer_size=8):
        super().__init__(
            token_micro_bsz=token_budget,
            buffer_size=buffer_size,
            get_length_fn=token_length,
            physical_token_cap=token_budget,
        )

    def put_item(self, item):
        # action 样本不按文本长度过滤；超预算单样本沿用原 buffer 的单独组包行为。
        self.buffer.append(item)


def build_dataset(config, tokenize, *, template=None, prompt_formatter=None):
    """Build one stage's sources without batching or inferring optimizer step counts.

    Config paths are ordered subset roots, directory globs or JSON manifests. No HF
    dataset conversion cache. tokenize(prompt) must return actual final token IDs.
    Geometry must match the chosen VAE/network, supplied by the caller's config.
    stage defaults to action; video construction never loads action statistics.
    """
    stage = config.get("stage", "action")
    if stage not in ("action", "video"):
        raise ValueError("stage must be action or video")
    if not callable(tokenize):
        raise ValueError("The causal Cosmos3 pipeline requires an explicit tokenize(prompt) callback")
    parallel = get_parallel_state()
    if parallel.sp_size != 1:
        raise ValueError("The causal Cosmos3 pipeline currently supports DP with SP/CP=1")
    rank, seed = parallel.dp_rank, config.get("seed", 0)
    if stage == "action":
        geometry = Geometry(**config.get("geometry", {}))
    groups, names, weights = [], [], []
    subset_number = 0
    for source in config["sources"]:
        if stage == "action":
            processors = {}
            columns = ActionColumns(**source.get("columns", {}))
        candidate_values = config.get("candidates", {}) | source.get("candidates", {})
        candidate_cls = FrameCandidateConfig if candidate_values.get("mode") == "frame_anchor" else CandidateConfig
        candidates = candidate_cls(**candidate_values)
        source = resolve_source_roots(source)
        roots = source["roots"]
        subsets = []
        for root in roots:
            read_options = source_read_options(source, root)
            common = dict(
                root=root,
                source=source["name"],
                tokenize=tokenize,
                cameras=source.get("cameras"),
                resolution=source.get("resolution", "256"),
                viewpoint=source["viewpoint"],
                candidates=candidates,
                seed=seed + subset_number,
                rank=rank,
                prompt_formatter=prompt_formatter,
                **read_options,
            )
            if stage == "action":
                statistics = source["statistics"]
                if isinstance(statistics, str):
                    statistics = statistics.replace("{root}", root)
                key = statistics if isinstance(statistics, str) else None
                if key not in processors:
                    processors[key] = ActionProcessor(
                        statistics,
                        template=template,
                        state_mapping=source.get("state_mapping"),
                        action_mapping=source.get("action_mapping"),
                    )
                processor = processors[key]
                if processor.statistics is None:
                    raise ValueError("A training pipeline requires measured state/action statistics")
                if isinstance(candidates, FrameCandidateConfig):
                    subset = ActionFrameDataset(
                        **common,
                        labels=source.get("labels"),
                        columns=columns,
                        geometry=geometry,
                        processor=processor,
                        **config.get("sample", {}),
                    )
                else:
                    subset = ActionDataset(
                        **common, columns=columns, geometry=geometry, processor=processor, **config.get("sample", {})
                    )
            else:
                if isinstance(candidates, FrameCandidateConfig):
                    subset = VideoFrameDataset(
                        **common,
                        labels=source.get("labels"),
                        **config.get("geometry", {}),
                        **config.get("sample", {}),
                    )
                else:
                    subset = VideoDataset(**common, **config.get("geometry", {}), **config.get("sample", {}))
            subsets.append(subset)
            subset_number += 1
        groups.append(StatefulConcatDataset(subsets))
        names.append(source["name"])
        weights.append(source.get("weight", 1.0))
    mode = config.get("sampling", "sample_uniform")
    if mode == "sample_uniform":
        dataset = StatefulConcatDataset([subset for group in groups for subset in group.datasets])
    elif mode == "source_weighted":
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Source weights must be finite, nonnegative and have positive sum")
        dataset = WeightedMultiSourceDataset(
            [UniformSource(group, seed + i, rank) for i, group in enumerate(groups)],
            weights=weights,
            seed=seed,
            level="sample",
            source_names=names,
            source_ids=names,
            upstream_sharded=True,
            stopping_strategy="never_exhausted",
            sample_token_len_fn=token_length,
        )
    else:
        raise ValueError("sampling must be sample_uniform or source_weighted")
    return dataset


def build_action_dataloader(config, tokenize, *, template=None, prompt_formatter=None, collate_fn=None):
    """Build fixed-step input: list[packs], each pack list[sample dict] by default.

    Pass collate_fn=collate_action_pack for DiTTrainer: each pack then becomes
    {"samples": [...]}. Collation runs after greedy selection and does
    not change the number of microbatches per optimizer step.
    """
    dataset = build_dataset(config, tokenize, template=template, prompt_formatter=prompt_formatter)
    parallel, seed = get_parallel_state(), config.get("seed", 0)
    budget, microbatches = config["token_budget"], config.get("grad_accum_iter", 1)
    if min(budget, microbatches, config["train_steps"], config.get("buffer_size", 8)) < 1:
        raise ValueError("Token budget, grad_accum_iter, train steps and buffer size must be positive")
    workers = config.get("num_workers", 0)
    return build_native_dataloader(
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=microbatches * parallel.dp_size,
        dataloader_batch_size=1,
        max_seq_len=budget,
        train_steps=config["train_steps"],
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        bsz_warmup_ratio=0,
        batching_strategy=ActionBatchingStrategy(budget, config.get("buffer_size", 8)),
        seed=seed,
        collate_fn=NoopDataCollator() if collate_fn is None else collate_fn,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=config.get("prefetch_factor", 2) if workers else None,
        persistent_workers=workers > 0,
        in_order=True,
        multiprocessing_context=config.get("multiprocessing_context") if workers else None,
    )
