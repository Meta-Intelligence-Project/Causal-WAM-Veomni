"""Finite, deterministic validation iteration over existing action datasets."""

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

from .dataset import MODES, ActionDataset
from .pipeline import collate_action_pack


VALIDATION_VERSION = "causal-cosmos3-validation-v2"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validation_seed(seed, identity, stream):
    """Derive rank/order/step independent 63-bit seeds without Python hash()."""
    return int(fingerprint([VALIDATION_VERSION, seed, identity, stream])[:16], 16) % (2**63)


def action_subsets(dataset):
    """Visit the existing concat/weighted/uniform source wrappers without reading."""
    if isinstance(dataset, ActionDataset):
        return [dataset]
    if hasattr(dataset, "dataset"):
        return action_subsets(dataset.dataset)
    children = getattr(dataset, "datasets", getattr(dataset, "_datasets", ()))
    return [subset for child in children for subset in action_subsets(child)]


def iter_validation_batches(dataset, rank=0, world_size=1, *, seed=42):
    """Yield singleton packs with fixed task/history/seeds and equal DP rounds.

    Only the assigned candidate is materialized. Each call starts a complete
    validation pass; dataset sampling RNG and training loader state are unused.
    Failed reads yield empty zero-weight packs for coordinated evaluator skips.
    """
    subsets = sorted(action_subsets(dataset), key=lambda ds: (ds.source, ds.root))
    cumulative = np.cumsum([len(ds) for ds in subsets])
    if not len(cumulative) or not cumulative[-1]:
        raise ValueError("Validation requires at least one legal window")
    window_count = int(cumulative[-1])
    modes = MODES if subsets[0].mode == "joint" else (subsets[0].mode,)
    for mode in modes:
        for turn in range(math.ceil(window_count / world_size)):
            offset = turn * world_size + rank
            candidate = offset % window_count
            subset_id = int(np.searchsorted(cumulative, candidate, side="right"))
            ds = subsets[subset_id]
            index = candidate - (int(cumulative[subset_id - 1]) if subset_id else 0)
            window, caption, fps = ds.window(index)
            identity = dict(
                source=ds.source,
                root=ds.root,
                episode=window.episode,
                candidate_index=index,
                caption=caption,
                start=window.start,
                real_actions=window.real_actions,
                actions=window.actions,
            )
            history_seed = validation_seed(seed, identity, "history")
            history = (
                window.actions // ds.geometry.actions_per_block
                if ds.full_history
                else int(np.random.default_rng(history_seed).integers(ds.history_range[0], ds.history_range[1] + 1))
            )
            record = dict(
                identity=identity,
                candidate_index=index,
                mode=mode,
                history_blocks=history,
                history_seed=history_seed,
                seeds={
                    stream: validation_seed(seed, {**identity, "mode": mode}, stream)
                    for stream in ("time", "video", "action")
                },
                window=asdict(window),
                caption=caption,
                source_fps=fps,
                geometry=asdict(ds.geometry),
                resolution=ds.resolution,
                viewpoint=ds.viewpoint,
            )
            sample = ds.make_sample(index, mode=mode, history=history)
            samples = [sample] if sample else []
            yield dict(
                batch=collate_action_pack(samples), record=record, weight=int(offset < window_count and bool(samples))
            )
