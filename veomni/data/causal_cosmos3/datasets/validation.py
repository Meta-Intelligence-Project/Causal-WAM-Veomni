"""Finite, deterministic validation iteration over existing video/action datasets."""

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

from .dataset import MODES, ActionDataset, LeRobotDataset
from .pipeline import collate_action_pack


VALIDATION_VERSION = "causal-cosmos3-validation-v3"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validation_seed(seed, identity, stream):
    """Derive rank/order/step independent 63-bit seeds without Python hash()."""
    return int(fingerprint([VALIDATION_VERSION, seed, identity, stream])[:16], 16) % (2**63)


def dataset_subsets(dataset):
    """Visit the existing concat/weighted/uniform source wrappers without reading."""
    if isinstance(dataset, LeRobotDataset):
        return [dataset]
    if hasattr(dataset, "dataset"):
        return dataset_subsets(dataset.dataset)
    children = getattr(dataset, "datasets", getattr(dataset, "_datasets", ()))
    return [subset for child in children for subset in dataset_subsets(child)]


def iter_validation_batches(dataset, rank=0, world_size=1, *, seed=42):
    """Yield singleton packs with fixed task/history/seeds and equal DP rounds.

    Only the assigned candidate is materialized. Each call starts a complete
    validation pass; dataset sampling RNG and training loader state are unused.
    Failed reads yield empty zero-weight packs for coordinated evaluator skips.
    """
    subsets = sorted(dataset_subsets(dataset), key=lambda ds: (ds.source, ds.root))
    cumulative = np.cumsum([len(ds) for ds in subsets])
    if not len(cumulative) or not cumulative[-1]:
        raise ValueError("Validation requires at least one legal window")
    window_count = int(cumulative[-1])
    is_action = isinstance(subsets[0], ActionDataset)
    if any(isinstance(ds, ActionDataset) != is_action for ds in subsets):
        raise ValueError("Validation requires a single dataset stage")
    if is_action:
        modes = MODES if subsets[0].mode == "joint" else (subsets[0].mode,)
    else:
        modes = ("video",)
    streams = ("time", "video", "action") if is_action else ("time", "video")
    for mode in modes:
        for turn in range(math.ceil(window_count / world_size)):
            offset = turn * world_size + rank
            candidate = offset % window_count
            subset_id = int(np.searchsorted(cumulative, candidate, side="right"))
            ds = subsets[subset_id]
            index = candidate - (int(cumulative[subset_id - 1]) if subset_id else 0)
            window, caption, fps = ds.plan_read_window(index)
            identity = dict(
                source=ds.source,
                root=ds.root,
                episode=window.episode,
                candidate_index=index,
                caption=caption,
                start=window.start,
            )
            if is_action:
                # Preserve existing Action identities and their derived seeds.
                identity.update(real_actions=window.real_actions, actions=window.actions)
                geometry = asdict(ds.geometry)
            else:
                identity.update(real_intervals=window.real_intervals, intervals=window.intervals)
                geometry = dict(
                    latent_block_size=ds.latent_block_size,
                    temporal_compression=ds.temporal_compression,
                    spatial_compression=ds.spatial_compression,
                    patch_size=ds.patch_size,
                )
            history_seed = validation_seed(seed, identity, "history")
            history = ds.select_history(ds.num_video_latents(window), rng=np.random.default_rng(history_seed))
            record = dict(
                identity=identity,
                candidate_index=index,
                mode=mode,
                history_blocks=history,
                history_seed=history_seed,
                seeds={stream: validation_seed(seed, {**identity, "mode": mode}, stream) for stream in streams},
                window=asdict(window),
                caption=caption,
                source_fps=fps,
                geometry=geometry,
                resolution=ds.resolution,
                viewpoint=ds.viewpoint,
            )
            options = dict(mode=mode, history=history) if is_action else dict(history=history)
            sample = ds.__getitem__(index, **options)
            samples = [sample] if sample else []
            yield dict(
                batch=collate_action_pack(samples), record=record, weight=int(offset < window_count and bool(samples))
            )
