"""Contiguous episode reads, lane ownership and independent decision semantics."""

import copy
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data import test_causal_cosmos3_pipeline as fixtures
from veomni.data.causal_cosmos3 import ActionDataset, ActionProcessor, Camera, CandidateConfig
from veomni.data.causal_cosmos3.datasets import episode as episode_module
from veomni.data.causal_cosmos3.datasets.dataset import MODES, StatefulConcatDataset
from veomni.data.causal_cosmos3.datasets.episode import EpisodeDataset, collate_episode_segment
from veomni.data.causal_cosmos3.datasets.pipeline import build_episode_dataset
from veomni.data.causal_cosmos3.readers import episode as reader


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def roots(tmp_path, request):
    return fixtures.roots.__wrapped__(tmp_path, request)


def action_dataset(root, *, video=False, use_subtask=False, **kwargs):
    dataset = ActionDataset(
        root,
        source="fixture",
        processor=ActionProcessor(fixtures.statistics()),
        tokenize=fixtures.tokenize,
        cameras={"head": "cam"},
        viewpoint="ego_view",
        video=video,
        candidates=CandidateConfig(max_actions=32, use_subtask=use_subtask),
        **(dict(history_steps=48, action_horizon=32, mode="policy") | kwargs),
    )
    # Keep decoded fixture canvases tiny; this changes neither temporal reads
    # nor any production camera selection code.
    dataset.cameras = (Camera("cam", 0, 0, 16, 16),)
    return dataset


def stream(dataset, *, length=4, repeat=False, rank=0, world_size=1):
    return EpisodeDataset(dataset, segment_latents=length, seed=29, repeat=repeat, rank=rank, world_size=world_size)


@pytest.mark.parametrize("roots", [(263, 117)], indirect=True)
def test_episode_segments_share_raw_reads_and_keep_every_decision(roots, monkeypatch):
    ds = action_dataset(roots[0], video=True)
    calls = {"rows": 0, "video": []}
    original = reader.read_rows

    def rows(*args, **kwargs):
        calls["rows"] += 1
        return original(*args, **kwargs)

    def decode(path, queries, output, **kwargs):
        calls["video"].append(np.asarray(queries).copy())
        for index, timestamp in enumerate(queries):
            output[:, index] = int(round(timestamp * 30)) % 251

    monkeypatch.setattr(reader, "read_rows", rows)
    monkeypatch.setattr(reader, "decode_into", decode)
    segments = list(stream(ds))
    decisions = [decision for segment in segments for decision in segment["decisions"]]
    assert [decision["window"].start for decision in decisions] == list(range(0, 262, 32))
    assert len(segments) == 5 and calls["rows"] == calls["video"].__len__() == 5
    assert [segment["reset"] for segment in segments] == [True, False, False, False, False]
    assert [segment["final"] for segment in segments] == [False, False, False, False, True]
    assert segments[0]["video"].shape == (3, 29, 16, 16)
    assert segments[1]["video"].shape == (3, 17, 16, 16)
    assert segments[-1]["video"].shape == (3, 9, 16, 16)
    assert segments[0]["video_frame_ticks"][:13] == [0] * 13
    assert segments[-1]["video_frame_ticks"][-1] == 262
    assert all(np.all(np.diff(queries) >= 0) for queries in calls["video"])
    for before, after in zip(segments, segments[1:]):
        assert before["end_tick"] == after["start_tick"]
        assert before["stream_id"] == after["stream_id"]
        assert after["segment_id"] == before["segment_id"] + 1

    ds.video = False
    for decision in decisions:
        reference = ds.make_sample(decision["candidate_index"], mode="policy")
        assert "video" not in decision
        for key in ("action", "states", "anchors", "state_mask", "action_mask", "action_valid_steps", "timestamps"):
            torch.testing.assert_close(decision[key], reference[key], rtol=0, atol=0)
        for key in ("prompt", "condition_frame_indexes_vision", "condition_frame_indexes_action", "video_frame_ticks"):
            assert decision[key] == reference[key]
    assert decisions[-1]["action_valid_steps"].sum() == 6
    assert torch.equal(decisions[-1]["anchors"][:, 0], torch.tensor([256.0]))


@pytest.mark.parametrize("roots", [(263, 117)], indirect=True)
def test_text_changes_end_context_and_reset_at_exact_next_anchor(roots):
    segments = list(stream(action_dataset(roots[0], use_subtask=True)))
    assert len(segments[0]["decisions"]) == 1
    assert segments[0]["decisions"][0]["caption"] == "first"
    assert segments[0]["final"] is True
    assert segments[1]["reset"] is True and segments[1]["segment_id"] == 0
    assert segments[1]["start_tick"] == segments[0]["end_tick"] == 32
    assert segments[0]["stream_id"] != segments[1]["stream_id"]
    assert all(d["caption"] == "second" for item in segments[1:] for d in item["decisions"])
    assert all(
        torch.equal(item["decisions"][0]["text_token_ids"], d["text_token_ids"])
        for item in segments
        for d in item["decisions"]
    )


@pytest.mark.parametrize("roots", [(117, 75)], indirect=True)
def test_task_fallback_and_feature_masks_are_selected_at_each_anchor(roots, monkeypatch, caplog):
    from veomni.data.causal_cosmos3.datasets import dataset as dataset_module

    monkeypatch.setattr(dataset_module.logger, "handlers", [*dataset_module.logger.handlers, caplog.handler])
    path = roots[0] / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    arrays = {
        name: np.asarray(table[name].combine_chunks().values).reshape(-1, 55).copy()
        for name in ("state_unified", "action_unified", "mask_state", "mask_action")
    }
    for name in ("mask_state", "mask_action"):
        arrays[name][32:, 1] = 1
        arrays[name][64:, 0] = 0
    columns = {name: pa.FixedSizeListArray.from_arrays(pa.array(value.ravel()), 55) for name, value in arrays.items()}
    columns["timestamp"] = table["timestamp"]
    columns["task_index"] = pa.array([int(i >= 32) for i in range(len(table))])
    pq.write_table(pa.table(columns), path, row_group_size=11)
    pq.write_table(
        pa.table(dict(task_index=[0, 1], task=["fallback", "new exact fallback"])), roots[0] / "meta/tasks.parquet"
    )
    ds = action_dataset(roots[0])
    segments = list(stream(ds))
    decisions = [d for item in segments for d in item["decisions"]]
    assert [d["task_index"] for d in decisions] == [0, 1, 1, 1]
    assert [d["caption"] for d in decisions] == ["fallback"] + ["new exact fallback"] * 3
    assert segments[0]["final"] and segments[1]["reset"]
    assert decisions[0]["action_mask"][:2].tolist() == [True, False]
    assert decisions[1]["action_mask"][:2].tolist() == [True, True]
    assert decisions[2]["action_mask"][:2].tolist() == [False, True]
    assert "fallback_caption='new exact fallback'" in caplog.text
    for decision in decisions:
        reference = ds.make_sample(decision["candidate_index"], mode="policy")
        torch.testing.assert_close(decision["action"], reference["action"], rtol=0, atol=0)


@pytest.mark.parametrize("roots", [(75, 41)], indirect=True)
def test_repeated_visits_rotate_every_anchor_phase_and_fixed_validation_repeats(roots):
    ds = action_dataset(roots[0])
    sequence = stream(ds, repeat=True)
    phases = {}
    for item in sequence:
        visit = item["stream_id"][4]
        phases.setdefault(visit, item["stream_id"][5])
        if len(phases) == 32:
            break
    assert [phases[i] for i in range(32)] == list(range(32))
    finite = stream(ds)
    first = [(x["stream_id"], x["start_tick"], x["end_tick"]) for x in finite]
    second = [(x["stream_id"], x["start_tick"], x["end_tick"]) for x in finite]
    assert first == second
    finite.set_epoch(3)
    assert next(iter(finite))["stream_id"][5] == 3
    assert finite.state_dict() == {"epoch": 3}
    finite.load_state_dict({"epoch": 0})
    assert next(iter(finite))["reset"]


@pytest.mark.parametrize("roots", [(12, 9)], indirect=True)
def test_short_episodes_rotate_only_legal_phases_without_overweighting_first_anchor(roots):
    data = stream(action_dataset(roots[0]), repeat=True)
    iterator = iter(data)
    phases = [next(iterator)["stream_id"][5] for _ in range(22)]
    assert phases == list(range(11)) * 2


def test_rank_and_worker_own_whole_episode_streams_without_overlap(roots, monkeypatch):
    datasets = StatefulConcatDataset([action_dataset(root) for root in roots])
    ranks = [stream(datasets, rank=rank, world_size=2) for rank in range(2)]
    identities = [{(x["subset"], x["episode"]) for x in data} for data in ranks]
    assert identities[0] and identities[1] and identities[0].isdisjoint(identities[1])
    assert list(stream(datasets, rank=2, world_size=3)) == []
    with pytest.raises(ValueError, match="owns no eligible episode"):
        stream(datasets, rank=2, world_size=3, repeat=True)
    whole = stream(datasets)
    lanes = []
    for lane in range(4):
        monkeypatch.setattr(
            episode_module, "get_worker_info", lambda lane=lane: SimpleNamespace(id=lane, num_workers=4)
        )
        items = list(whole)
        assert all(item["lane_id"] == lane for item in items)
        lanes.append({(item["subset"], item["episode"]) for item in items})
    assert lanes[0] and lanes[1] and not lanes[2] and not lanes[3]
    assert lanes[0].isdisjoint(lanes[1])


def test_real_cpu_workers_keep_one_lane_per_owned_episode(roots):
    data = stream(StatefulConcatDataset([action_dataset(root) for root in roots]))
    items = list(DataLoader(data, batch_size=None, num_workers=2))
    by_lane = {}
    for item in items:
        by_lane.setdefault(item["lane_id"], set()).add(item["subset"])
    assert set(by_lane) == {0, 1}
    assert all(len(identities) == 1 for identities in by_lane.values())
    assert by_lane[0].isdisjoint(by_lane[1])


@pytest.mark.parametrize("mode,weights", [(mode, [int(i == j) for i in range(3)]) for j, mode in enumerate(MODES)])
def test_each_decision_keeps_original_task_conditions(roots, mode, weights):
    ds = action_dataset(roots[0], mode="joint", mode_weights=weights)
    for item in stream(ds):
        for decision in item["decisions"]:
            assert decision["mode"] == mode
            reference = ds.make_sample(decision["candidate_index"], mode=mode)
            assert decision["condition_frame_indexes_vision"] == reference["condition_frame_indexes_vision"]
            assert decision["condition_frame_indexes_action"] == reference["condition_frame_indexes_action"]


def test_failed_continuation_propagates_instead_of_silent_skip(roots, monkeypatch):
    original = episode_module.read_episode_rows
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("broken continuation")
        return original(*args, **kwargs)

    monkeypatch.setattr(episode_module, "read_episode_rows", fail_second)
    iterator = iter(stream(action_dataset(roots[0]), length=2))
    assert next(iterator)["reset"]
    with pytest.raises(OSError, match="broken continuation"):
        next(iterator)


@pytest.mark.parametrize("segment_latents", [0, -1, True, 1.5, 3])
def test_invalid_segment_geometry_is_rejected(roots, segment_latents):
    with pytest.raises(ValueError, match="segment_latents"):
        stream(action_dataset(roots[0]), length=segment_latents)


def test_single_segment_collation_rejects_splitting_or_merging():
    item = {"stream_id": ("a",), "decisions": [{"mode": "policy"}]}
    assert collate_episode_segment([item]) == {"segment": item}
    for items in ([], [item, item]):
        with pytest.raises(ValueError, match="one complete segment"):
            collate_episode_segment(items)


def test_builder_reuses_existing_source_schema_and_rejects_weighted_mixing(roots, monkeypatch):
    from veomni.data.causal_cosmos3.datasets import pipeline

    monkeypatch.setattr(pipeline, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1, sp_size=1))
    config = dict(
        stage="action",
        sampling="sample_uniform",
        sample=dict(history_steps=48, action_horizon=32, mode="policy"),
        candidates=dict(use_subtask=False),
        sources=[
            dict(
                name="fixture",
                roots=[str(root) for root in roots],
                statistics=fixtures.statistics(),
                cameras=dict(head="cam"),
                viewpoint="ego_view",
                read=dict(video=False),
            )
        ],
    )
    original = copy.deepcopy(config)
    data = build_episode_dataset(config, fixtures.tokenize, segment_latents=4, repeat=False)
    assert config == original
    assert len(list(data)) > 0
    weighted = dict(config, sampling="source_weighted")
    with pytest.raises(ValueError, match="source_weighted"):
        build_episode_dataset(weighted, fixtures.tokenize, segment_latents=4)
    with pytest.raises(ValueError, match="stage: action"):
        build_episode_dataset(dict(config, stage="video"), fixtures.tokenize, segment_latents=4)


@pytest.mark.parametrize("roots", [(263, 10)], indirect=True)
def test_fifo_rank_slots_and_short_phase_cycles_are_not_candidate_uniform(roots):
    datasets = StatefulConcatDataset([action_dataset(root) for root in roots])
    ranks = [stream(datasets, length=2, repeat=True, rank=rank, world_size=2) for rank in range(2)]
    batches = [list(islice(iter(dataset), 9)) for dataset in ranks]
    # Equal rank slots are deliberate even though the candidate masses differ.
    assert len(ranks[0].datasets[0]) > 20 * len(ranks[1].datasets[1])
    assert [len(batch) for batch in batches] == [9, 9]
    assert [batch["decisions"][0]["window"].start for batch in batches[0]] == list(range(0, 257, 32))
    # A short episode revisits its whole legal phase set before a long episode
    # has completed the H phases. Actual-D normalization cannot alter this rate.
    assert [batch["decisions"][0]["window"].start for batch in batches[1]] == list(range(9))
