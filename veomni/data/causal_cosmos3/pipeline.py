"""Compose existing VeOmni loaders with action datasets and native greedy packing."""

import glob
import math
from pathlib import Path

from ...distributed.parallel_state import get_parallel_state
from ..data_collator import NoopDataCollator
from ..data_loader import build_native_dataloader
from ..dataset import WeightedMultiSourceDataset
from ..dynamic_batching import TextBatchingStrategy
from .dataset import ActionDataset, CandidateConfig, StatefulConcatDataset, UniformSource
from .lerobot import ActionColumns
from .processing import ActionProcessor, Geometry


def token_length(sample):
    return sample["token_length"]


def collate_action_pack(samples):
    """Wrap one greedy pack as a DiT microbatch without copying tensor data."""
    return {"samples": samples}


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


def build_action_dataset(config, tokenize, *, template=None, prompt_formatter=None):
    """Build action sources without batching or inferring optimizer step counts.

    Config paths are explicit, ordered subset roots (or glob patterns). No HF
    dataset conversion cache. tokenize(prompt) must return actual final token IDs.
    Geometry must match the chosen VAE/network, supplied by the caller's config.
    """
    if not callable(tokenize):
        raise ValueError("The action pipeline requires an explicit tokenize(prompt) callback")
    parallel = get_parallel_state()
    if parallel.sp_size != 1:
        raise ValueError("The action pipeline currently supports DP with SP/CP=1")
    rank, seed = parallel.dp_rank, config.get("seed", 0)
    geometry = Geometry(**config.get("geometry", {}))
    groups, names, weights = [], [], []
    subset_number = 0
    for source in config["sources"]:
        processors = {}
        columns = ActionColumns(**source.get("columns", {}))
        candidates = CandidateConfig(**(config.get("candidates", {}) | source.get("candidates", {})))
        roots = []
        for entry in source["roots"]:
            # Globs are expanded once at construction; preserve configured source order.
            matches = sorted(glob.glob(str(entry)))
            if not matches:
                raise ValueError(f"No roots matched {entry}")
            roots.extend(str(Path(root).resolve()) for root in matches)
        if len(set(roots)) != len(roots):
            raise ValueError(f"Duplicate roots in source {source['name']}")
        subsets = []
        for root in roots:
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
            subsets.append(
                ActionDataset(
                    root,
                    source=source["name"],
                    processor=processor,
                    tokenize=tokenize,
                    cameras=source["cameras"],
                    resolution=source.get("resolution", "256"),
                    viewpoint=source["viewpoint"],
                    columns=columns,
                    candidates=candidates,
                    geometry=geometry,
                    seed=seed + subset_number,
                    rank=rank,
                    prompt_formatter=prompt_formatter,
                    **config.get("sample", {}),
                    **source.get("read", {}),
                )
            )
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
    dataset = build_action_dataset(config, tokenize, template=template, prompt_formatter=prompt_formatter)
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
