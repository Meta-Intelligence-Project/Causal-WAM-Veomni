"""Numerical and file-level checks for independent LeRobot dataset statistics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.readers.lerobot import ActionColumns, plan_window, read_metadata, read_window
from veomni.data.causal_cosmos3.statistics import (
    QuantileAccumulator,
    compute_statistics,
    discover_dataset_roots,
    save_statistics,
)
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def make_root(root, shift=0, lengths=(35, 33, 20)):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = dict(
        codebase_version="v3.0",
        fps=30,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        features={
            key: dict(dtype="float32", shape=[55])
            for key in ("state_unified", "action_unified", "mask_state", "mask_action")
        },
    )
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, records, offset = [], [], 0
    for episode, length in enumerate(lengths):
        t = np.arange(length, dtype=np.float32)
        state = np.zeros((length, 55), dtype=np.float32)
        state[:, 0] = t**2 + episode * 10000 + shift
        state[:, 19] = np.sin(t / 60)
        state[:, 20] = np.cos(t / 60)
        state[::2, 17:21] *= -3  # Equivalent rotations must be canonicalized.
        state[:, 54] = np.nan  # Invalid values never enter statistics.
        action = state.copy()
        action[:, 0] += 2
        action[:, 28] = 0.1 + t / 100
        sm = np.zeros_like(state)
        sm[:, 0] = sm[:, 17:21] = 1
        am = sm.copy()
        am[:, 28] = 1  # Absolute gripper requires no same-channel state.

        def vector(values):
            return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 55)

        records.append(
            pa.table(
                dict(
                    state_unified=vector(state),
                    action_unified=vector(action),
                    mask_state=vector(sm),
                    mask_action=vector(am),
                    timestamp=t.astype(np.float64) / 30,
                    task_index=np.zeros(length, dtype=np.int64),
                )
            )
        )
        episodes.append(
            dict(
                episode_index=episode,
                dataset_from_index=100 + offset,
                dataset_to_index=100 + offset + length,
                **{"data/chunk_index": 0, "data/file_index": 0},
            )
        )
        offset += length
    pq.write_table(pa.concat_tables(records), root / "data/chunk-000/file-000.parquet", row_group_size=11)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


@pytest.mark.parametrize("offset,from_state", [(0, False), (1, True), (-1, False)])
@pytest.mark.parametrize("method", ["exact", "reservoir"])
def test_batched_statistics_match_native_windows(tmp_path, offset, from_state, method):
    root = make_root(tmp_path / "dataset")
    columns = ActionColumns(
        target="state_unified" if from_state else "action_unified",
        target_mask="mask_state" if from_state else "mask_action",
        target_offset=offset,
    )
    actual = compute_statistics(root, columns=columns, method=method, reservoir_size=1000, batch_size=2, num_workers=3)
    reference = compute_statistics(root, columns=columns, method="exact", batch_size=1, num_workers=1)
    for kind in ("state", "action"):
        for key, values in reference[kind].items():
            np.testing.assert_allclose(actual[kind][key], values, rtol=1e-6, atol=1e-6)
    assert actual["provenance"]["blocks"] == (2 if offset < 0 else 4)
    assert actual["provenance"]["skipped_episodes"] == ([1, 2] if offset < 0 else [2])
    assert actual["provenance"]["episodes"] == [0, 1, 2]
    assert not actual["provenance"]["partial"]
    assert actual["action"]["low"][17:21] == [-1.0] * 4
    assert actual["action"]["high"][17:21] == [1.0] * 4

    # Separately enumerate native training windows, without the batch expansion.
    info, metadata = read_metadata(root)
    processor, geometry = ActionProcessor(None), Geometry()
    states, actions = [], []
    for i, episode in enumerate(metadata["episode_index"]):
        ep = {key: values[i].item() for key, values in metadata.items()}
        first = max(0, -offset)
        stop = min(
            ep["dataset_to_index"] - ep["dataset_from_index"],
            ep["dataset_to_index"] - ep["dataset_from_index"] + 1 - offset,
        )
        for start in range(first, stop - 32):
            window = plan_window(root, int(episode), start, 32, columns=columns, info=info, ep=ep)
            data = processor(read_window(window, video=False), geometry)
            states.append(data["states"])
            actions.append(data["action"])
    for kind, tensors in (("state", states), ("action", actions)):
        values = torch.cat(tensors).numpy().astype(np.float64)
        active = np.asarray(actual[kind]["valid"])
        np.testing.assert_allclose(np.asarray(actual[kind]["mean"])[active], values[:, active].mean(0))
        np.testing.assert_allclose(np.asarray(actual[kind]["std"])[active], values[:, active].std(0), atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(actual[kind]["q01"])[active], np.quantile(values[:, active], 0.01, axis=0)
        )
        assert actual[kind]["count"] == [len(values)]
        assert actual[kind]["valid_counts"][54] == 0
    if offset == 0:
        assert actual["state"]["mean"][0] == 2501.25
        assert actual["action"]["min"][0] == 2  # Relative to each window's anchor.
        assert actual["action"]["min"][28] == pytest.approx(0.1)  # Absolute gripper.
    path = root / "meta/causal_action_stats.json"
    save_statistics(actual, path)
    ActionProcessor(path)  # Output is directly consumable by native training/inference.
    assert json.loads(path.read_text()) == actual


@pytest.mark.parametrize("capacity", [1, 7, 128])
@pytest.mark.parametrize("batches", [1, 9])
def test_reservoir_matches_algorithm_r_and_full_moments(capacity, batches):
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    masks = np.ones_like(values, dtype=bool)
    masks[::2, 1] = False
    masks[:, 2] = False
    acc = QuantileAccumulator(method="reservoir", capacity=capacity, seed=123)
    for x, mask in zip(np.array_split(values, batches), np.array_split(masks, batches)):
        acc.update(torch.from_numpy(x), torch.from_numpy(mask))
    result = acc.finalize()
    for d in (0, 1):
        valid = values[masks[:, d], d]
        reference = valid[:capacity].copy()
        rng = np.random.default_rng(np.random.SeedSequence([123, d]))
        for i in range(capacity, len(valid)):
            slot = rng.integers(0, i + 1)
            if slot < capacity:
                reference[slot] = valid[i]
        np.testing.assert_array_equal(acc.rows[: len(reference), d], reference)
        assert result["q01"][d] == pytest.approx(np.quantile(reference, 0.01))
        assert result["mean"][d] == pytest.approx(valid.astype(np.float64).mean())
        assert result["std"][d] == pytest.approx(valid.astype(np.float64).std())
        assert result["valid_counts"][d] == len(valid)
    assert result["valid"] == [True, True, False]
    assert result["low"][2] == result["high"][2] == 0


def test_partial_bounds_empty_and_atomic_output(tmp_path):
    root = make_root(tmp_path / "dataset")
    result = compute_statistics(root, method="exact", bounds="min_max", limit=2, num_workers=1)
    assert result["provenance"]["partial"]
    assert result["provenance"]["blocks"] == 2
    for kind in ("state", "action"):
        assert result[kind]["low"][0] == result[kind]["min"][0]
        assert result[kind]["high"][0] == result[kind]["max"][0]
        assert result[kind]["low"][17:21] == [-1.0] * 4
    output = root / "meta/causal_action_stats.json"
    save_statistics(result, output)
    previous = output.read_text()
    with pytest.raises(ValueError):
        save_statistics({"bad": float("nan")}, output)
    assert output.read_text() == previous
    assert not list(output.parent.glob(".causal_stats_*"))
    short = make_root(tmp_path / "short", lengths=(32,))
    with pytest.raises(ValueError, match="No complete"):
        compute_statistics(short)


def test_discovery_and_cli_independent_datasets(tmp_path):
    roots = [make_root(tmp_path / "a"), make_root(tmp_path / "b/nested", shift=500)]
    assert discover_dataset_roots([tmp_path, roots[0]]) == roots
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    repo = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repo)
    env["PYTHON_BIN"] = sys.executable
    subprocess.run(
        [
            "bash",
            str(repo / "scripts/causal_cosmos3/statistics/run_statistics.sh"),
            "--dataset-root",
            str(tmp_path),
            "--dataset-processes",
            "2",
            "--method",
            "exact",
            "--num-workers",
            "1",
            "--batch-size",
            "2",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    outputs = [json.loads((root / "meta/causal_action_stats.json").read_text()) for root in roots]
    assert outputs[1]["state"]["mean"][0] - outputs[0]["state"]["mean"][0] == 500
    for root, data in zip(roots, outputs):
        assert data["provenance"]["dataset_root"] == str(root)
        assert data["state"]["count"] == [4]
        assert data["action"]["count"] == [128]
        assert "split" not in data["provenance"]
        assert not (root / "meta/stats.json").exists()
