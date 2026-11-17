"""CPU-only simulation of one complete sampler traversal on every DP rank.

Run from the repository root with ``python -m
scripts.causal_cosmos3.statistics.estimate_epoch_iters --help``.
The training recipe and its source YAML are read without changing either file.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
from functools import lru_cache, partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml


def positive_int(value):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return int(value)


def load_recipe(path, ranks, model_path=None, train_path=None):
    """Read both existing YAML schemas; match the training entry's precedence."""
    ranks = positive_int(ranks)
    path = Path(path).resolve()
    recipe = yaml.safe_load(path.read_text())
    data, train = recipe["data"], recipe["train"]
    accelerator = recipe["model"].get("accelerator", {})
    for dimension in ("pp_size", "tp_size", "cp_size", "ulysses_size"):
        if accelerator.get(dimension, 1) != 1:
            raise ValueError("This estimator requires pure DP (PP/TP/SP/CP sizes must be 1)")
    if train.get("micro_batch_size", 1) != 1 or train.get("dyn_bsz_runtime", "main") != "main":
        raise ValueError("Cosmos3 packing requires micro_batch_size=1 and dyn_bsz_runtime='main'")
    if train.get("training_task", "online_training") != "online_training":
        raise ValueError("Only online_training recipes are supported")
    loader = data.get("dataloader", {})
    if loader.get("type", "native") != "native" or not loader.get("in_order", True):
        raise ValueError("Exact seeded simulation requires the native, in_order=True dataloader")
    accumulation = train.get("grad_accum_iter")
    if accumulation is None:
        batch_size = positive_int(ranks if train.get("global_batch_size") is None else train["global_batch_size"])
        if batch_size % ranks:
            raise ValueError("global_batch_size must be a multiple of ranks")
        accumulation = batch_size // ranks
    accumulation = positive_int(accumulation)
    source_path = Path(data["train_path"] if train_path is None else train_path).resolve()
    sources = yaml.safe_load(source_path.read_text())
    dataset = {key: data.get(key, {}) for key in ("geometry", "candidates", "sample")}
    dataset["stage"] = data["stage"]
    dataset.update(sources)
    if dataset.get("stage") not in ("action", "video"):
        raise ValueError("data.stage must be action or video")
    if dataset.get("sampling", "sample_uniform") != "sample_uniform":
        raise ValueError("source_weighted is an infinite random stream without a finite per-rank epoch")
    tokenizer_path = (
        Path(recipe["model"]["model_path"] if model_path is None else model_path).resolve() / "text_tokenizer"
    )
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing training tokenizer: {tokenizer_path}")
    return dict(
        config=str(path),
        source_config=str(source_path),
        dataset=dataset,
        tokenizer_path=str(tokenizer_path),
        ranks=ranks,
        accumulation=accumulation,
        token_budget=positive_int(data["max_seq_len"]),
        buffer_size=positive_int(data.get("dyn_bsz_buffer_size", 200)),
        shuffle=loader.get("shuffle", True),
        seed=train.get("seed", 42),
        max_steps=train.get("max_steps"),
    )


def prepare_sources(config):
    """Resolve #24 manifests once, then freeze the configured validation split."""
    from veomni.data.causal_cosmos3.datasets.pipeline import resolve_source_roots
    from veomni.data.causal_cosmos3.datasets.split import split_dataset_config

    dataset = dict(config["dataset"])
    dataset["sources"] = [resolve_source_roots(source) for source in dataset["sources"]]
    manifest = None
    if "split_ratio" in dataset.get("validation", {}):
        dataset, _, manifest = split_dataset_config(dataset)
    return dataset, manifest


@lru_cache(maxsize=1)
def prompt_tokenizer(path):
    from veomni.data.causal_cosmos3.transforms.tokenization import build_prompt_tokenizer

    return build_prompt_tokenizer(path)


@lru_cache(maxsize=16384)
def prompt_ids(path, prompt):
    from veomni.data.causal_cosmos3.transforms.tokenization import tokenize_prompt

    return tuple(tokenize_prompt(prompt, tokenizer=prompt_tokenizer(path)))


def build_root_lengths(job, tokenize=None):
    """Reuse Dataset metadata/planning without calling __getitem__ or reading payloads.

    Windows are grouped by episode, interval count and caption. Only the task_index
    column is projected from data Parquet files when a subtask caption is absent.
    No timestamp, video, action or state arrays are read.
    """
    import pyarrow.parquet as pq

    from veomni.data.causal_cosmos3.datasets.dataset import ActionDataset, CandidateConfig, VideoDataset
    from veomni.data.causal_cosmos3.datasets.pipeline import source_read_options
    from veomni.data.causal_cosmos3.readers.lerobot import ActionColumns
    from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry, video_token_layout

    source, root = job["source"], job["root"]
    if tokenize is None:
        tokenize = partial(prompt_ids, job["tokenizer_path"])
    common = dict(
        root=root,
        source=source["name"],
        tokenize=tokenize,
        cameras=source.get("cameras"),
        resolution=source.get("resolution", "256"),
        viewpoint=source["viewpoint"],
        candidates=CandidateConfig(**(job["candidates"] | source.get("candidates", {}))),
        **source_read_options(source, root),
    )
    if job["stage"] == "action":
        stats = source["statistics"]
        if isinstance(stats, str):
            stats = stats.replace("{root}", root)
        processor = ActionProcessor(
            stats, state_mapping=source.get("state_mapping"), action_mapping=source.get("action_mapping")
        )
        if processor.statistics is None:
            raise ValueError("The action recipe requires measured statistics")
        ds = ActionDataset(
            **common,
            processor=processor,
            columns=ActionColumns(**source.get("columns", {})),
            geometry=Geometry(**job["geometry"]),
            **job["sample"],
        )
    else:
        ds = VideoDataset(**common, **job["geometry"], **job["sample"])

    @lru_cache(maxsize=4)
    def task_column(path):
        with pq.ParquetFile(path) as parquet:
            column = parquet.read(columns=["task_index"], use_threads=False)["task_index"].combine_chunks()
        if column.null_count:
            raise ValueError(f"Null task_index in {path}")
        return column.to_numpy(zero_copy_only=False)

    def text_length(window, index, caption, fps, task_index=0):
        sample = ds.prepare_sample(
            dict(window=window, task_index=int(task_index)), index=index, caption=caption, source_fps=fps
        )
        return len(sample["text_token_ids"])

    lengths = np.lib.format.open_memmap(job["output"], mode="w+", dtype=np.int64, shape=(len(ds),))
    boundaries = np.r_[0, np.flatnonzero(np.diff(ds.windows[:, 0])) + 1, len(ds)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        lo, hi = int(lo), int(hi)
        window, _, fps = ds.plan_read_window(lo)
        rows = ds.windows[lo:hi]
        base = np.empty(hi - lo, dtype=np.int64)
        for intervals in np.unique(rows[:, 2]):
            positions = np.flatnonzero(rows[:, 2] == intervals)
            planned, _, _ = ds.plan_read_window(lo + int(positions[0]))
            if job["stage"] == "action":
                layout = ds.geometry.token_layout(planned.actions, planned.height, planned.width, 0)
            else:
                layout = video_token_layout(
                    planned.video_frames,
                    planned.height,
                    planned.width,
                    0,
                    temporal_compression=ds.temporal_compression,
                    spatial_compression=ds.spatial_compression,
                    patch_size=ds.patch_size,
                )
            base[positions] = sum(layout.values())
        text = np.empty(hi - lo, dtype=np.int64)

        for text_id in np.unique(rows[:, 3]):
            positions = np.flatnonzero(rows[:, 3] == text_id)
            if text_id:
                text[positions] = text_length(window, lo, ds.texts[int(text_id)], fps)
            else:
                # row_start includes the episode's file-local offset and this window's start.
                offsets = rows[positions, 1] + window.row_start - window.start
                column = task_column(window.data_path)
                if np.any(offsets < 0) or np.any(offsets >= len(column)):
                    raise ValueError(f"Window task_index outside Parquet bounds: {root}")
                tasks = column[offsets]
                for task_index in np.unique(tasks):
                    text[positions[tasks == task_index]] = text_length(window, lo, "", fps, task_index)
        lengths[lo:hi] = base + text
    if np.any(lengths <= 0):
        raise ValueError(f"Nonpositive token length: {root}")
    summary = dict(
        source=source["name"],
        root=root,
        windows=len(ds),
        episodes=len(np.unique(ds.windows[:, 0])),
        min_tokens=int(lengths.min()),
        max_tokens=int(lengths.max()),
    )
    lengths.flush()
    return summary


def global_order(size, seed, epoch=0, shuffle=True):
    """Generate one shared permutation per seed, matching DistributedSampler."""
    import torch

    if not shuffle:
        return np.arange(size, dtype=np.int64)
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).numpy()


class ReplayShard:
    """Replay the rank's padded sampler sequence; mark only its first traversal."""

    def __init__(self, lengths, order, rank, ranks):
        self.lengths, self.order = lengths, order
        self.rank, self.ranks = rank, ranks
        self.size = math.ceil(len(order) / ranks)
        self.pass_number = 0
        self.reads = 0

    def __iter__(self):
        first = self.pass_number == 0
        self.pass_number += 1
        for position in range(self.size):
            # DistributedSampler pads by repeating the global permutation prefix.
            index = int(self.order[(self.rank + position * self.ranks) % len(self.order)])
            self.reads += 1
            yield dict(token_length=int(self.lengths[index]), first_pass=first)


def simulate_rank(lengths, order, rank, ranks, accumulation, token_budget, buffer_size):
    """Stop after a whole optimizer iter has consumed every first-pass occurrence."""
    from veomni.data.causal_cosmos3.datasets.pipeline import ActionBatchingStrategy
    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader

    if not len(lengths) or len(lengths) != len(order):
        raise ValueError("Expected equally sized, nonempty lengths and order arrays")
    shard = ReplayShard(lengths, order, rank, ranks)
    loader = DynamicBatchSizeDataLoader(
        shard,
        batching_strategy=ActionBatchingStrategy(token_budget, buffer_size),
        num_micro_batch=accumulation,
        length=-1,
    )
    remaining, trained = shard.size, 0
    iterator = iter(loader)
    try:
        for step, packs in enumerate(iterator, 1):
            for pack in packs:
                remaining -= sum(sample["first_pass"] for sample in pack)
                trained += len(pack)
            if remaining == 0:
                return dict(
                    rank=rank,
                    optimizer_iters=step,
                    first_pass_samples=shard.size,
                    trained_samples_at_completion=trained,
                    read_samples_at_completion=shard.reads,
                )
    finally:
        # Early return otherwise leaves loader -> generator -> loader cycles,
        # retaining mmaps until cyclic GC. NFS cannot remove those open files.
        loader._batch_data_iter.close()
        loader._data_iter.close()
    raise RuntimeError("Simulation ended before the first sampler traversal completed")


def initialize_worker():
    import torch

    torch.set_num_threads(1)


@lru_cache(maxsize=1)
def shared_array(path):
    return np.load(path, mmap_mode="r")


def simulate_task(task):
    length_path, order_path, rank, config = task
    order = np.load(order_path, mmap_mode="r")
    try:
        return simulate_rank(
            shared_array(length_path),
            order,
            rank,
            config["ranks"],
            config["accumulation"],
            config["token_budget"],
            config["buffer_size"],
        )
    finally:
        # All tasks for a seed close their order before the parent unlinks it.
        order._mmap.close()


def estimate(config, seeds, workers=1, epoch=0, temp_dir=None):
    """Share arrays with a bounded spawn pool; parallelize seeds when ranks are few."""
    from veomni.data.causal_cosmos3.datasets.pipeline import source_read_options

    started = time.perf_counter()
    workers = positive_int(workers)
    initialize_worker()
    dataset, split = prepare_sources(config)
    # Set before spawn so BLAS/tokenizer threads do not multiply the CPU pool.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pool_context = (
        ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker)
        if workers > 1
        else nullcontext(None)
    )
    with (
        TemporaryDirectory(prefix="cosmos3-epoch-", dir=temp_dir) as directory,
        pool_context as pool,
        ExitStack() as cleanup,
    ):
        cleanup.callback(shared_array.cache_clear)
        jobs = []
        for source in dataset["sources"]:
            for root in source["roots"]:
                # Validate selection before dispatch; keep source/root order independent of completion order.
                source_read_options(source, root)
                root_source = {**source, "roots": [root]}
                if isinstance(source.get("episodes"), dict):
                    root_source["episodes"] = {root: source["episodes"][root]}
                jobs.append(
                    dict(
                        source=root_source,
                        root=root,
                        stage=dataset["stage"],
                        geometry=dataset.get("geometry", {}),
                        candidates=dataset.get("candidates", {}),
                        sample=dataset.get("sample", {}),
                        tokenizer_path=config["tokenizer_path"],
                        output=str(Path(directory) / f"root_{len(jobs)}.npy"),
                    )
                )
        if not jobs:
            raise ValueError("No training roots after source selection and validation split")
        print(
            f"Preparing {len(jobs)} roots using {workers} CPU workers (no model weights or physical ranks).",
            flush=True,
        )
        summaries = list(pool.map(build_root_lengths, jobs) if pool else map(build_root_lengths, jobs))
        size = sum(summary["windows"] for summary in summaries)
        length_path = str(Path(directory) / "lengths.npy")
        lengths = np.lib.format.open_memmap(length_path, mode="w+", dtype=np.int64, shape=(size,))
        offset = 0
        for job, summary in zip(jobs, summaries, strict=True):
            count = summary["windows"]
            part = np.load(job["output"], mmap_mode="r")
            lengths[offset : offset + count] = part
            offset += count
            del part
            Path(job["output"]).unlink()
        lengths.flush()
        oversized = int(np.count_nonzero(lengths > config["token_budget"]))
        del lengths
        prepared = time.perf_counter()
        print(
            f"Prepared {size:,} training windows across {len(jobs)} roots; simulating {len(seeds)} seeds.", flush=True
        )
        runs = []
        # Fill the pool across seeds when ranks < workers, with about 256 MiB
        # of live permutation arrays at most (one larger permutation is unavoidable).
        seed_batch = max(1, min(math.ceil(workers / config["ranks"]), (256 * 1024**2) // (size * 8)))
        for start in range(0, len(seeds), seed_batch):
            batch = seeds[start : start + seed_batch]
            order_paths, tasks = [], []
            for index, seed in enumerate(batch, start):
                # Distinct paths keep cached worker mmaps independent between seeds.
                order_path = str(Path(directory) / f"order_{index}.npy")
                np.save(order_path, global_order(size, seed, epoch, config["shuffle"]))
                order_paths.append(order_path)
                tasks.extend((length_path, order_path, rank, config) for rank in range(config["ranks"]))
            batch_results = list(pool.map(simulate_task, tasks) if pool else map(simulate_task, tasks))
            shared_array.cache_clear()
            for index, (seed, order_path) in enumerate(zip(batch, order_paths, strict=True)):
                Path(order_path).unlink()
                results = batch_results[index * config["ranks"] : (index + 1) * config["ranks"]]
                maximum = max(result["optimizer_iters"] for result in results)
                runs.append(
                    dict(
                        seed=seed,
                        optimizer_iters=maximum,
                        slowest_ranks=[result["rank"] for result in results if result["optimizer_iters"] == maximum],
                        per_rank=results,
                    )
                )
                print(
                    f"seed={seed}: {maximum:,} optimizer iters; slowest ranks={runs[-1]['slowest_ranks']}", flush=True
                )
    values = [run["optimizer_iters"] for run in runs]
    return dict(
        definition="All first-traversal sampler occurrences consumed on every DP rank, at optimizer boundaries",
        config=config["config"],
        source_config=config["source_config"],
        stage=dataset["stage"],
        ranks=config["ranks"],
        grad_accum_iter=config["accumulation"],
        token_budget=config["token_budget"],
        buffer_size=config["buffer_size"],
        sampler_epoch=epoch,
        shuffle=config["shuffle"],
        seeds=seeds,
        workers=workers,
        max_inflight_seeds=min(seed_batch, len(seeds)),
        dataset_windows=size,
        sampler_samples_per_rank=math.ceil(size / config["ranks"]),
        sampler_padding_samples=math.ceil(size / config["ranks"]) * config["ranks"] - size,
        oversized_windows=oversized,
        validation_split=split,
        roots=summaries,
        interval=dict(min=min(values), median=statistics.median(values), max=max(values)),
        configured_max_steps=config["max_steps"],
        max_steps_covers_all_runs=config["max_steps"] is None or config["max_steps"] >= max(values),
        timings_seconds=dict(preparation=prepared - started, simulation=time.perf_counter() - prepared),
        assumptions=[
            "Fresh sampler epoch and empty packing buffer; this is not a resumed data cursor",
            "Every planned window is readable; runtime read/decode failures are not simulated",
            "Interval is the observed seed range, not a confidence interval",
        ],
        runs=runs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing configs/dit training YAML")
    parser.add_argument(
        "--model-path", "--model.model_path", help="Optional checkpoint root override, as in the training launcher"
    )
    parser.add_argument(
        "--train-path", "--data.train_path", help="Optional source YAML override, as in the training launcher"
    )
    parser.add_argument("--ranks", type=int, required=True, help="Total DP ranks, including HSDP replica groups")
    parser.add_argument("--seed-start", type=int, help="First seed; defaults to train.seed")
    parser.add_argument("--runs", type=int, help="Number of consecutive seeds; default: 10")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seeds, instead of --seed-start/--runs")
    parser.add_argument("--sampler-epoch", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="CPU pool size; default 1 avoids spawn overhead on small jobs"
    )
    parser.add_argument("--temp-dir", help="Parent directory for temporary shared NumPy arrays")
    parser.add_argument("--output", type=Path, default=Path("outputs/causal_cosmos3_epoch_iters.json"))
    args = parser.parse_args(argv)
    config = load_recipe(args.config, args.ranks, args.model_path, args.train_path)
    if args.seeds is not None:
        if args.seed_start is not None or args.runs is not None:
            parser.error("--seeds cannot be combined with --seed-start/--runs")
        seeds = args.seeds
    else:
        start = config["seed"] if args.seed_start is None else args.seed_start
        seeds = list(range(start, start + positive_int(10 if args.runs is None else args.runs)))
    if args.sampler_epoch < 0 or any(not 0 <= seed < 2**63 for seed in seeds):
        parser.error("Seeds must be in [0, 2**63); sampler epoch must be nonnegative")
    report = estimate(config, seeds, args.workers, args.sampler_epoch, args.temp_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    interval = report["interval"]
    print(f"One full epoch: [{interval['min']:,}, {interval['max']:,}] optimizer iters; median={interval['median']:g}")
    if not report["max_steps_covers_all_runs"]:
        print(f"Configured max_steps={config['max_steps']} is insufficient for some simulated traversals.")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
