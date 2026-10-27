"""Seeded episode partitions from one complete source configuration."""

import copy
import hashlib
import json
import math
from decimal import ROUND_CEILING, Decimal

from ..readers.lerobot import ActionColumns, read_metadata
from .pipeline import resolve_source_roots, source_read_options


SPLIT_VERSION = "causal-cosmos3-episode-split-v3"


def _fingerprint(value):
    """Keep split identities independent of validation runtime imports."""
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def split_action_config(config):
    """Partition complete sources using the sole source-level strategy."""
    return split_action_config_by_source(config)


def _select_validation(episodes, ratio, seed, max_episodes):
    """Select a quota from physical episode identities without consuming RNG."""
    count = (
        min(
            len(episodes) - 1,
            max(1, int((Decimal(str(ratio)) * len(episodes)).to_integral_value(rounding=ROUND_CEILING))),
            max_episodes if max_episodes is not None else len(episodes) - 1,
        )
        if len(episodes) >= 2
        else 0
    )
    ranked = sorted(episodes, key=lambda item: (_fingerprint([SPLIT_VERSION, seed, *item]), item))
    return set(ranked[:count])


def split_action_config_by_source(config):
    """Pool each source's (root, episode) pairs before selecting validation.

    Select ceil(ratio * N), capped by max_episodes_per_source when configured
    and by N-1 to retain training data. Sources with one episode stay in training.
    Individual roots may fall entirely on either side. Recompute from metadata
    on every call without payload reads, split persistence or global RNG draws.
    """
    options = config.get("validation", {})
    ratio, seed = options.get("split_ratio"), options.get("seed", 42)
    if type(ratio) not in (int, float) or not 0 < ratio < 1 or not math.isfinite(ratio):
        raise ValueError("validation.split_ratio must be a finite number strictly between 0 and 1")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("validation.seed must be an integer in [0, 2**63)")
    max_episodes = options.get("max_episodes_per_source")
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("validation.max_episodes_per_source must be a positive integer or null")
    names = [source["name"] for source in config["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("Automatic episode split requires unique source names")
    training, evaluation = copy.deepcopy(config), copy.deepcopy(config)
    training["sources"], evaluation["sources"] = [], []
    evaluation["validation"].pop("split_ratio")
    evaluation["validation"].pop("max_episodes_per_source", None)
    records, source_records, seen = [], [], set()
    for source in config["sources"]:
        roots = resolve_source_roots(source)
        partitions = {"train": {}, "validation": {}}
        root_records = []
        offset = ActionColumns(**source.get("columns", {})).target_offset
        for root in roots:
            if root in seen:
                raise ValueError(f"Automatic episode split requires unique roots across sources: {root}")
            seen.add(root)
            selection = source_read_options(source, root, roots).get("episodes")
            _, metadata = read_metadata(root)
            known = set(map(int, metadata["episode_index"]))
            if selection is not None and (unknown := set(selection) - known):
                raise ValueError(f"Unknown episodes in {root}: {sorted(unknown)}")
            selected = known if selection is None else set(selection)
            eligible, excluded = [], []
            for episode, start, stop in zip(
                metadata["episode_index"], metadata["dataset_from_index"], metadata["dataset_to_index"], strict=True
            ):
                episode, length = int(episode), int(stop - start)
                if episode not in selected:
                    continue
                available = min(length, length + 1 - offset) - max(0, -offset) - 1
                (eligible if available >= 1 else excluded).append(episode)
            eligible.sort()
            root_records.append(
                dict(
                    source=source["name"],
                    root=root,
                    eligible_episodes=eligible,
                    excluded_episodes=sorted(excluded),
                )
            )
        identities = [(record["root"], ep) for record in root_records for ep in record["eligible_episodes"]]
        selected = _select_validation(identities, ratio, seed, max_episodes)
        for record in root_records:
            root = record["root"]
            record["validation_episodes"] = [ep for ep in record["eligible_episodes"] if (root, ep) in selected]
            record["train_episodes"] = [ep for ep in record["eligible_episodes"] if (root, ep) not in selected]
            if record["train_episodes"]:
                partitions["train"][root] = record["train_episodes"]
            if record["validation_episodes"]:
                partitions["validation"][root] = record["validation_episodes"]
        records.extend(root_records)
        source_records.append(
            dict(
                source=source["name"],
                eligible_count=sum(len(record["eligible_episodes"]) for record in root_records),
                train_count=sum(len(record["train_episodes"]) for record in root_records),
                validation_count=sum(len(record["validation_episodes"]) for record in root_records),
            )
        )
        for name, destination in (("train", training), ("validation", evaluation)):
            if partitions[name]:
                child = copy.deepcopy(source)
                child["roots"] = list(partitions[name])
                child["episodes"] = partitions[name]
                if "read" in child:
                    child["read"].pop("episodes", None)
                destination["sources"].append(child)
    if not evaluation["sources"]:
        raise ValueError("Automatic validation requires at least one source with two eligible episodes")
    manifest = dict(
        version=SPLIT_VERSION,
        seed=seed,
        split_ratio=ratio,
        max_episodes_per_source=max_episodes,
        stratify="source",
        sources=sorted(source_records, key=lambda record: record["source"]),
        roots=sorted(records, key=lambda record: record["root"]),
        train_count=sum(len(record["train_episodes"]) for record in records),
        validation_count=sum(len(record["validation_episodes"]) for record in records),
    )
    manifest["split_id"] = _fingerprint(manifest)
    return training, evaluation, manifest
