"""Build video/action sources with shared sampling and native greedy batching."""

import glob
import math
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

from torch import distributed as dist
from tqdm import tqdm

from veomni.data.data_collator import NoopDataCollator
from veomni.data.data_loader import build_native_dataloader
from veomni.data.dataset import WeightedMultiSourceDataset
from veomni.data.dynamic_batching import TextBatchingStrategy
from veomni.distributed.parallel_state import get_parallel_state

from ..readers.lerobot import ActionColumns
from ..transforms.processing import ActionProcessor, Geometry
from .dataset import ActionDataset, CandidateConfig, StatefulConcatDataset, UniformSource, VideoDataset


def token_length(sample):
    return sample["token_length"]


def collate_action_pack(samples):
    """Wrap one greedy pack as a DiT microbatch without copying tensor data."""
    return {"samples": samples}


def resolve_source_roots(source):
    """Expand source roots once in configured order and reject duplicate paths."""
    roots = []
    for entry in source["roots"]:
        matches = sorted(root for root in glob.glob(str(entry)) if Path(root).is_dir())
        if not matches:
            raise ValueError(f"No roots matched {entry}")
        roots.extend(str(Path(root).resolve()) for root in matches)
    if len(set(roots)) != len(roots):
        raise ValueError(f"Duplicate roots in source {source['name']}")
    return roots


def source_read_options(source, roots):
    """Resolve and validate one source's read options, keyed by subset root."""
    options = dict(source.get("read", {}))
    selections = None
    if "episodes" in source:
        if "episodes" in options:
            raise ValueError("Specify episodes on the source or in read, not both")
        selection = source["episodes"]
        if isinstance(selection, dict):
            selections = {str(Path(key).resolve()): value for key, value in selection.items()}
            if set(selections) != set(roots):
                raise ValueError("Episode mapping must cover exactly the resolved source roots")
        else:
            options["episodes"] = selection
    options_by_root = {}
    for root in roots:
        read_options = dict(options)
        if selections is not None:
            read_options["episodes"] = selections[root]
        if (selection := read_options.get("episodes")) is not None:
            if not isinstance(selection, list) or not selection or any(type(ep) is not int for ep in selection):
                raise ValueError("episodes must be a nonempty integer list (optionally mapped by root)")
        options_by_root[root] = read_options
    return options_by_root


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


def _build_subset(task):
    dataset_class, kwargs = task
    return dataset_class(**kwargs)


def build_dataset(config, tokenize, *, template=None, prompt_formatter=None):
    """Build one stage's sources without batching or inferring optimizer step counts.

    Config paths are explicit, ordered subset roots (or glob patterns). No HF
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
    with get_context("fork").Pool(processes=8) as pool:
        for source in config["sources"]:
            if stage == "action":
                processors = {}
                subset_processors = []
                columns = ActionColumns(**source.get("columns", {}))
            candidates = CandidateConfig(**(config.get("candidates", {}) | source.get("candidates", {})))
            roots = resolve_source_roots(source)
            options_by_root = source_read_options(source, roots)
            tasks, subsets = [], []
            with tqdm(
                total=len(roots),
                desc=f"Building {source['name']}",
                unit="dataset",
                disable=dist.is_initialized() and dist.get_rank() != 0,
                mininterval=1.0,
                dynamic_ncols=True,
            ) as progress:
                for root in roots:
                    read_options = options_by_root[root]
                    common = dict(
                        root=root,
                        source=source["name"],
                        tokenize=None,
                        cameras=source["cameras"],
                        resolution=source.get("resolution", "256"),
                        viewpoint=source["viewpoint"],
                        candidates=candidates,
                        seed=seed + subset_number,
                        rank=rank,
                        prompt_formatter=None,
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
                        subset_processors.append(processor)
                        tasks.append(
                            (
                                ActionDataset,
                                dict(
                                    **common,
                                    columns=columns,
                                    geometry=geometry,
                                    # 构造阶段只校验统计量几何信息，完整 processor 留在父进程。
                                    processor=SimpleNamespace(statistics_geometry=processor.statistics_geometry),
                                    **config.get("sample", {}),
                                ),
                            )
                        )
                    else:
                        tasks.append(
                            (VideoDataset, dict(**common, **config.get("geometry", {}), **config.get("sample", {})))
                        )
                    subset_number += 1
                for i, subset in enumerate(pool.imap(_build_subset, tasks)):
                    subset.tokenize = tokenize
                    subset.prompt_formatter = prompt_formatter
                    if stage == "action":
                        subset.processor = subset_processors[i]
                    subsets.append(subset)
                    progress.update(1)
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
