"""Frame identities, boundaries, annotations and shared processing on CPU."""

import copy
import json
import pickle
import random
from contextlib import nullcontext
from dataclasses import asdict, replace
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3 import (
    ActionColumns,
    ActionDataset,
    ActionFrameDataset,
    ActionProcessor,
    CandidateConfig,
    FrameCandidateConfig,
    FrameLabelConfig,
    LeRobotFrameDataset,
    VideoFrameDataset,
    build_action_dataloader,
    build_dataset,
    iter_frame_samples,
)
from veomni.data.causal_cosmos3.datasets.pipeline import FrameEpochSampler, collate_action_pack, get_samples_per_pack
from veomni.data.causal_cosmos3.datasets.split import split_dataset_config
from veomni.data.causal_cosmos3.datasets.validation import dataset_subsets, iter_validation_batches
from veomni.data.causal_cosmos3.readers.lerobot import EmptyTemporalSampleError, sample_multi_anchor_indices


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def multi_anchor(anchor, length, **kwargs):
    options = dict(block_span=32, video_stride=4, max_blocks=4, task_labels=None) | kwargs
    return sample_multi_anchor_indices(anchor, length, **options)


@pytest.mark.parametrize("count,starts", [(1, [64]), (2, [32, 64]), (3, [32, 64, 96]), (4, [0, 32, 64, 96])])
def test_multi_anchor_expands_earlier_before_later_and_sorts(count, starts):
    result = multi_anchor(64, 200, max_blocks=count)
    assert result.anchor == 64 and result.num_blocks == count
    np.testing.assert_array_equal(result.block_starts, starts)
    np.testing.assert_array_equal(result.state_indices, starts)
    np.testing.assert_array_equal(result.video_indices, np.arange(starts[0], starts[-1] + 33, 4))
    np.testing.assert_array_equal(result.action_indices, np.arange(starts[0], starts[-1] + 32))


@pytest.mark.parametrize(
    "length,anchor,starts",
    [(33, 0, [0]), (33, 32, [0]), (64, 32, [0]), (128, 96, [0, 32, 64]), (141, 140, [12, 44, 76, 108])],
)
def test_multi_anchor_terminal_anchors_and_endpoint_rollback(length, anchor, starts):
    result = multi_anchor(anchor, length)
    np.testing.assert_array_equal(result.block_starts, starts)
    np.testing.assert_array_equal(result.video_indices, np.arange(starts[0], starts[-1] + 33, 4))
    assert result.video_indices[-1] < length
    assert len(result.action_indices) == len(starts) * 32


@pytest.mark.parametrize("length,anchor", [(20, 0), (32, 0), (33, 1)])
def test_multi_anchor_rejects_unusable_starts_without_padding(length, anchor):
    with pytest.raises(EmptyTemporalSampleError):
        multi_anchor(anchor, length)


@pytest.mark.parametrize("side,starts", [("earlier", [64, 96, 128, 160]), ("later", [0, 32, 64]), ("both", [64])])
def test_multi_anchor_raw_task_boundary_stops_only_that_side(side, starts):
    labels = np.zeros(200, dtype=np.int64)
    if side in ("earlier", "both"):
        labels[:64] = 1
    if side in ("later", "both"):
        labels[96:] = 1
    result = multi_anchor(64, len(labels), task_labels=labels)
    np.testing.assert_array_equal(result.block_starts, starts)


def test_multi_anchor_checks_only_raw_task_at_block_starts():
    labels = np.zeros(200, dtype=np.int64)
    labels[65:80] = 1  # A task change inside a block must not add a new rejection rule.
    visited = []

    class Labels:
        def __getitem__(self, frame):
            visited.append(frame)
            return labels[frame]

    result = multi_anchor(64, len(labels), task_labels=Labels())
    np.testing.assert_array_equal(result.block_starts, [0, 32, 64, 96])
    assert visited == [64, 32, 96, 0]


def test_multi_anchor_language_strings_and_task_boundary_can_leave_no_block():
    labels = ["task A"] * 64 + ["task B"] * 32
    with pytest.raises(EmptyTemporalSampleError):
        multi_anchor(70, len(labels), task_labels=labels)
    result = multi_anchor(63, len(labels), task_labels=labels)
    assert result.num_blocks > 0  # The accepted block may cross the task boundary internally.


@pytest.mark.parametrize("span,stride", [(8, 1), (16, 2), (32, 4), (32, 2), (32, 8), (4, 4)])
@pytest.mark.parametrize("action", [False, True])
def test_multi_anchor_geometry_controls_video_count(span, stride, action):
    # Video derives span from stride * temporal_compression * latent_block_size;
    # action derives it from actions_per_block. Neither hard-codes 8 video samples.
    result = multi_anchor(
        2 * span, 5 * span + 1, block_span=span, video_stride=stride, target_offset=0 if action else None
    )
    np.testing.assert_array_equal(result.block_starts, np.arange(4) * span)
    np.testing.assert_array_equal(result.video_indices, np.arange(0, 4 * span + 1, stride))
    assert len(result.video_indices) == 4 * (span // stride) + 1
    if action:
        np.testing.assert_array_equal(result.action_indices, np.arange(4 * span))
        np.testing.assert_array_equal(result.state_indices, np.arange(4) * span)
    else:
        assert result.action_indices is result.state_indices is None


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_multi_anchor_generalized_endpoint_rollback(stride):
    result = multi_anchor(4, 8, block_span=4, video_stride=stride, target_offset=None)
    np.testing.assert_array_equal(result.block_starts, [0])
    np.testing.assert_array_equal(result.video_indices, np.arange(0, 5, stride))


@pytest.mark.parametrize("offset", [0, 1, 3, -2])
def test_multi_anchor_target_offsets_keep_same_blocks(offset):
    result = multi_anchor(64, 200, max_blocks=2, target_offset=offset)
    np.testing.assert_array_equal(result.block_starts, [32, 64])
    np.testing.assert_array_equal(result.action_indices, np.arange(32, 96) + offset)


@pytest.mark.parametrize(
    "anchor,length,offset", [(32, 65, 3), (64, 200, -2), (64, 200, 2**63 - 64), (64, 200, -(2**63) - 64)]
)
def test_multi_anchor_rejects_misaligned_action_targets(anchor, length, offset):
    with pytest.raises(EmptyTemporalSampleError, match="Action targets outside"):
        multi_anchor(anchor, length, target_offset=offset)


@pytest.mark.parametrize("label", [None, True, 1.0, [0]])
def test_multi_anchor_rejects_invalid_raw_task_labels(label):
    with pytest.raises(ValueError, match="Raw task/language"):
        multi_anchor(64, 200, task_labels={64: label})


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    import veomni.data.causal_cosmos3.datasets.pipeline as pipeline
    import veomni.data.data_loader as loader
    import veomni.data.dataset as datasets

    state = SimpleNamespace(dp_rank=0, dp_size=1, sp_size=1)
    for module in (pipeline, loader, datasets):
        monkeypatch.setattr(module, "get_parallel_state", lambda: state)


def tokenize(text):
    return [1, *text.encode(), 2]


@pytest.fixture
def root(tmp_path):
    for directory in ("meta/episodes/chunk-000", "data/chunk-000"):
        (tmp_path / directory).mkdir(parents=True)
    info = dict(
        codebase_version="v3.0",
        fps=30,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        features={
            key: dict(dtype="float32", shape=[55])
            for key in ("state_unified", "action_unified", "mask_state", "mask_action")
        },
    )
    info["features"].update(
        cam=dict(dtype="video", shape=[32, 32, 3], names=["height", "width", "channels"]),
        task_index=dict(dtype="int64", shape=[1]),
        subtask_index=dict(dtype="int64", shape=[1]),
    )
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table(
            dict(
                episode_index=[7, 30, 99],
                dataset_from_index=[100, 106, 107],
                dataset_to_index=[106, 107, 111],
                **{
                    "data/chunk_index": [0, 0, 0],
                    "data/file_index": [0, 0, 0],
                    "data/file_from_offset": [0, 6, 7],
                    "videos/cam/chunk_index": [0, 0, 0],
                    "videos/cam/file_index": [0, 0, 0],
                    "videos/cam/from_timestamp": [0.0, 0.2, 7 / 30],
                },
            )
        ),
        tmp_path / "meta/episodes/chunk-000/file-000.parquet",
    )
    (tmp_path / "meta/episodes.jsonl").write_text(
        json.dumps(dict(episode_index=7, source_fps=15, action_config="unused")) + "\n"
    )
    pq.write_table(pa.table(dict(task_index=[0], task=["full task"])), tmp_path / "meta/tasks.parquet")
    pq.write_table(
        pa.table(dict(subtask_index=[0, 1], subtask=["first", "second"])), tmp_path / "meta/subtasks.parquet"
    )
    state = np.arange(11 * 55, dtype=np.float32).reshape(11, 55)
    mask = np.zeros_like(state)
    mask[:, 0] = 1

    def vector(values):
        return pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 55)

    pq.write_table(
        pa.table(
            dict(
                timestamp=np.concatenate((np.arange(6), [0], np.arange(4))) / 30,
                task_index=[0] * 11,
                subtask_index=[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                state_unified=vector(state),
                action_unified=vector(state + 1000),
                mask_state=vector(mask),
                mask_action=vector(mask),
            )
        ),
        tmp_path / "data/chunk-000/file-000.parquet",
        row_group_size=3,
    )
    return tmp_path


class RawActionProbe(LeRobotFrameDataset):
    """Test-only fixed plans exercise low-level reader boundaries without training geometry."""

    def __init__(self, *args, columns, past, future, **kwargs):
        self.columns = columns
        self.reader_span = past, future
        super().__init__(*args, **kwargs)

    def plan_read_window(self, index):
        window = super().plan_read_window(index, columns=self.columns)
        return replace(window, past=self.reader_span[0], future=self.reader_span[1])

    def _caption(self, task_id, subtask_ids, video_offsets):
        # 固定窗口探针只检查 anchor 标签；完整窗口 caption 由 Action/Video 的测试覆盖。
        if subtask_ids is not None:
            anchor_offset = self.reader_span[0]
            subtask_ids = subtask_ids[anchor_offset : anchor_offset + 1]
            video_offsets = np.array([0])
        return super()._caption(task_id, subtask_ids, video_offsets)


def dataset(root, **kwargs):
    options = (
        dict(
            columns=ActionColumns(),
            video=False,
            video_stride=2,
            past=8,
            future=4,
        )
        | kwargs
    )
    return RawActionProbe(root, source="fixture", viewpoint="ego_view", tokenize=tokenize, **options)


def rewrite_labels(root, values):
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("subtask_index"), "subtask_index", pa.array(values))
    pq.write_table(table, path, row_group_size=3)


@pytest.mark.parametrize("t", [0, 2, 5])
@pytest.mark.parametrize("from_state,offset", [(False, 0), (True, -3), (True, 0), (True, 1), (True, 3)])
def test_exact_window_offsets_and_boundaries(root, t, from_state, offset):
    columns = ActionColumns(
        target="state_unified" if from_state else "action_unified",
        target_mask="mask_state" if from_state else "mask_action",
        target_offset=offset,
    )
    ds = dataset(root, columns=columns)
    sample = ds.read_sample(t)
    requested = np.arange(t - 8, t + 5)
    observations = requested.clip(0, 5)
    targets = requested[:-1] + offset
    np.testing.assert_array_equal(sample["state"][:, 0], observations * 55)
    np.testing.assert_array_equal(sample["action"][:, 0], targets.clip(0, 5) * 55 + (0 if from_state else 1000))
    np.testing.assert_array_equal(sample["observation_valid_steps"], (requested >= 0) & (requested < 6))
    np.testing.assert_array_equal(
        sample["action_valid_steps"], (requested[:-1] >= 0) & (requested[1:] < 6) & (targets >= 0) & (targets < 6)
    )
    assert sample["anchor_index"] == 8
    video_frames = np.arange(t - 8, t + 5, 2)
    np.testing.assert_array_equal(sample["video_valid_steps"], (video_frames >= 0) & (video_frames < 6))
    assert sample["source_fps"] == 15 and sample["storage_fps"] == 30
    assert sample["caption"] == ("first" if t < 2 else "second")


def test_all_anchors_sparse_ids_and_single_frame(root):
    ds = dataset(root)
    assert len(ds) == 11
    assert [ds.locate(i) for i in range(11)] == [(0, t) for t in range(6)] + [(1, 0)] + [(2, t) for t in range(4)]
    sample = ds.read_sample(6)
    assert sample["episode"] == 30
    assert sample["observation_valid_steps"].tolist() == [False] * 8 + [True] + [False] * 4
    assert not sample["action_valid_steps"].any()
    assert torch.unique(sample["state"][:, 0]).tolist() == [330]
    selected = dataset(root, episodes=[99, 7], past=0, future=0, candidates=FrameCandidateConfig(anchor_stride=2))
    assert len(selected) == 5
    sample = selected[len(selected) - 1]
    assert sample["window"].anchor == 2 and sample["episode"] == 99
    assert sample["state"].shape == (1, 55) and sample["action"].shape == (0, 55)
    assert sample["caption"] == "second"
    assert ds.state_dict() == selected.state_dict() == {"rng": None}
    with pytest.raises(ValueError, match="Unknown episodes"):
        dataset(root, episodes=[1])


def test_compact_index_and_initialization_do_not_read_payload(root, monkeypatch):
    import veomni.data.causal_cosmos3.datasets.frame_dataset as module

    original = module.read_metadata

    def metadata(*args, **kwargs):
        info, ep = original(*args, **kwargs)
        ep["dataset_to_index"] = ep["dataset_from_index"] + 100_000_000
        return info, ep

    monkeypatch.setattr(module, "read_metadata", metadata)

    def forbidden(*args, **kwargs):
        pytest.fail("Initialization read a frame payload")

    monkeypatch.setattr(module, "read_frame_window", forbidden)
    ds = dataset(root)
    assert len(ds) == 300_000_000
    assert ds.cumulative.nbytes + ds.episode_rows.nbytes + ds.anchor_starts.nbytes == 80
    assert ds.locate(100_000_003) == (1, 3)
    assert not hasattr(ds, "windows") and not hasattr(ds, "subtasks")


def test_one_projected_read_anchor_labels_and_output_ownership(root, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as module

    original, calls = module.read_rows, []

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "read_rows", read)
    ds = dataset(root)
    first = ds.read_sample(2)
    assert len(calls) == 1
    assert {"subtask_index", "task_index", "state_unified"}.issubset(calls[0][0][3])
    assert first["caption"] == "second"  # Window begins in the previous subtask.
    first["state"].zero_()
    assert ds.read_sample(2)["state"][8, 0] == 110


@pytest.mark.parametrize("subtask_id", [-1, 99])
def test_unknown_subtask_id(root, subtask_id):
    rewrite_labels(root, [subtask_id] * 11)
    ds = dataset(root)
    with pytest.raises(ValueError, match="Unknown subtask"):
        ds.make_sample(0)


def test_root_scoped_labels_and_source_column_names(root):
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.rename_columns(["subtask_id" if name == "subtask_index" else name for name in table.column_names])
    table = table.set_column(
        table.schema.get_field_index("subtask_id"), "subtask_id", pa.array([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3])
    )
    pq.write_table(table, path)
    pq.write_table(
        pa.table(dict(id=[0, 1, 2, 3], text=["zero", "one", "solo", "last"])),
        root / "meta/subtasks.parquet",
    )
    ds = dataset(
        root,
        labels=FrameLabelConfig(
            subtask_column="subtask_id",
            subtask_id_column="id",
            subtask_text_column="text",
        ),
    )
    assert ds[0]["caption"] == "zero" and ds[6]["caption"] == "solo" and ds[7]["caption"] == "last"


def test_video_only_read_and_anchor_centered_decoding(root, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as module

    seen = []

    def decode(path, queries, output, **kwargs):
        seen.append(queries.copy())
        output[:] = np.rint(queries * 30).astype(np.uint8)[None, :, None, None]

    monkeypatch.setattr(module, "decode_into", decode)
    ds = dataset(root, columns=None, video=True)
    sample = ds[0]
    assert "action" not in sample and "state" not in sample
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], [0, 0, 0, 0, 0, 2 / 30, 4 / 30])
    assert sample["video"][0, :, 0, 0].tolist() == [0, 0, 0, 0, 0, 2, 4]
    assert sample["video_valid_steps"].tolist() == [False] * 4 + [True] * 3


@pytest.mark.parametrize(
    "options",
    [
        dict(anchor_stride=0),
        dict(anchor_stride=True),
        dict(mode="frame_window"),
        dict(max_blocks=0),
        dict(max_blocks=-1),
        dict(max_blocks=True),
        dict(max_blocks=1.5),
    ],
)
def test_frame_config_validation(options):
    with pytest.raises(ValueError):
        FrameCandidateConfig(**options)


@pytest.mark.parametrize("name", ["past_frames", "future_frames"])
def test_frame_candidates_reject_window_limits(name):
    with pytest.raises(TypeError, match=name):
        FrameCandidateConfig(**{name: 32})


def config(root):
    return dict(
        stage="video",
        candidates=dict(mode="frame_anchor", max_blocks=1, anchor_stride=32),
        sample=dict(full_history=False, history_range=(2, 5)),
        sources=[dict(name="fixture", roots=[str(root)], viewpoint="ego_view", read=dict(video=False))],
    )


def test_factory_split_and_validation_boundary(long_root):
    cfg = config(long_root)
    ds = build_dataset(cfg, tokenize)
    subset = dataset_subsets(ds)[0]
    assert type(subset) is VideoFrameDataset
    assert not any(hasattr(subset, name) for name in ("columns", "processor", "geometry"))
    assert len(ds) == 7
    assert subset.seed == 0 and subset.history_range == (2, 5)
    assert ds[0]["mode"] == "video" and "token_length" in ds[0]
    identities = {
        (sample["episode"], sample["window"].anchor)
        for rank in range(3)
        for sample in iter_frame_samples(ds, rank=rank, world_size=3, raw=True)
    }
    assert identities == {(7, t) for t in (0, 32, 64, 96, 128)} | {(99, 0), (99, 32)}
    cfg["validation"] = dict(split_ratio=0.34, seed=42)
    train, validation, manifest = split_dataset_config(cfg)
    assert manifest["sources"][0]["eligible_count"] == 2
    assert manifest["roots"][0]["excluded_episodes"] == [30]
    train_ids = dataset_subsets(build_dataset(train, tokenize))[0].selected_episodes
    validation_ids = dataset_subsets(build_dataset(validation, tokenize))[0].selected_episodes
    assert not train_ids & validation_ids and train_ids | validation_ids == {7, 99}
    with pytest.raises(ValueError, match="requires full_history=True"):
        list(iter_validation_batches(ds))
    cfg["sample"]["full_history"] = True
    assert len(list(iter_validation_batches(build_dataset(cfg, tokenize)))) == len(ds)
    invalid = copy.deepcopy(cfg)
    invalid["sources"][0]["candidates"] = dict(max_actions=32)
    with pytest.raises(TypeError, match="max_actions"):
        build_dataset(invalid, tokenize)


@pytest.mark.parametrize("stage", ["action", "video"])
def test_factory_source_override_controls_complete_block_count(long_root, stage):
    cfg = config(long_root)
    cfg["stage"] = stage
    if stage == "action":
        cfg["sources"][0]["statistics"] = block_statistics()
    override = copy.deepcopy(cfg["sources"][0])
    override.update(name="override", candidates=dict(max_blocks=2))
    cfg["sources"].append(override)
    subsets = dataset_subsets(build_dataset(cfg, tokenize))
    assert [len(ds) for ds in subsets] == [7, 7]
    for ds, count in zip(subsets, (1, 2), strict=True):
        sample = ds.read_sample(0)
        window = sample["window"]
        assert ds.candidates.max_blocks == window.max_blocks == count
        assert len(sample["timestamps"]) - 1 == count * window.block_span


@pytest.mark.parametrize("stage", ["action", "video"])
def test_weighted_factory_keeps_existing_wrapper_and_resume(long_root, stage):
    cfg = config(long_root)
    cfg["stage"] = stage
    if stage == "action":
        cfg["candidates"]["max_blocks"] = 4
        cfg["sources"][0]["statistics"] = block_statistics()
    cfg["sampling"] = "source_weighted"
    ds = build_dataset(cfg, tokenize)
    iterator = iter(ds)
    list(islice(iterator, 3))
    state = copy.deepcopy(ds.state_dict())
    expected = [(s["episode"], s["window"].anchor, s["history_blocks"]) for s in islice(iterator, 4)]
    restored = build_dataset(cfg, tokenize)
    restored.load_state_dict(state)
    actual = [(s["episode"], s["window"].anchor, s["history_blocks"]) for s in islice(iter(restored), 4)]
    assert expected == actual


def test_episode_json_text_definitions_without_interval_parsing(root):
    (root / "meta/episodes.jsonl").write_text(
        json.dumps(
            dict(
                episode_index=7,
                source_fps=15,
                action_config=[dict(id=0, text="json zero", start_frame="not parsed"), dict(id=1, text="json one")],
            )
        )
        + "\n"
    )
    ds = dataset(
        root,
        episodes=[7],
        labels=FrameLabelConfig(
            subtask_table="meta/episodes.jsonl",
            subtask_records_key="action_config",
            subtask_id_column="id",
            subtask_text_column="text",
        ),
    )
    assert ds[0]["caption"] == "json zero" and ds[2]["caption"] == "json one"
    assert ds[0]["source_fps"] == 15


def test_two_roots_have_independent_label_namespaces(root, tmp_path):
    import shutil

    second = tmp_path / "second_root"
    # Copy only the three dataset directories, avoiding a recursive copy of tmp_path.
    second.mkdir()
    for name in ("meta", "data"):
        shutil.copytree(root / name, second / name)
    pq.write_table(
        pa.table(dict(subtask_index=[0, 1], subtask=["other zero", "other one"])), second / "meta/subtasks.parquet"
    )
    assert dataset(root)[0]["caption"] == "first"
    assert dataset(second)[0]["caption"] == "other zero"


def test_feature_masks_that_change_with_time_are_rejected(root):
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    masks = table["mask_action"].to_pylist()
    masks[1][1] = 1
    table = table.set_column(table.schema.get_field_index("mask_action"), "mask_action", pa.array(masks))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="constant binary feature mask"):
        dataset(root).make_sample(0)


@pytest.mark.parametrize(
    "geometry,error_type,error",
    [
        (dict(latent_block_size=-2, temporal_compression=-4), ValueError, "positive integers"),
        (dict(video_stride=True), ValueError, "video_stride"),
        (dict(actions_per_block=32), TypeError, "actions_per_block"),
    ],
)
def test_frame_video_geometry_is_validated_before_reading(root, geometry, error_type, error):
    cfg = config(root)
    cfg["geometry"] = geometry
    with pytest.raises(error_type, match=error):
        build_dataset(cfg, tokenize)


@pytest.mark.parametrize("task_id", [-1, 99])
def test_unknown_task_is_reported_even_when_subtask_exists(root, task_id):
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("task_index"), "task_index", pa.array([task_id] * len(table))
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=f"Unknown task ID {task_id}"):
        dataset(root).make_sample(0)


def block_statistics():
    return {
        name: dict(low=[-100.0] * 55, high=[20000.0] * 55, valid=[True] * 51 + [False] * 4)
        for name in ("state", "action")
    }


@pytest.fixture
def long_root(root):
    lengths = [141, 1, 51]
    starts = np.array([0, 141, 142])
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    for key, values in {
        "dataset_from_index": starts + 100,
        "dataset_to_index": starts + 100 + lengths,
        "data/file_from_offset": starts,
        "videos/cam/from_timestamp": starts / 30,
    }.items():
        table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
    pq.write_table(table, path)
    n = sum(lengths)
    state = np.arange(n * 55, dtype=np.float32).reshape(n, 55)
    mask = np.zeros_like(state)
    mask[:, 0] = 1
    pq.write_table(
        pa.table(
            dict(
                timestamp=np.concatenate([np.arange(length) / 30 for length in lengths]),
                task_index=[0] * n,
                subtask_index=[0] * 20 + [1] * 121 + [0] + [1] * 51,
                state_unified=pa.array(state.tolist()),
                action_unified=pa.array((state + 1000).tolist()),
                mask_state=pa.array(mask.tolist()),
                mask_action=pa.array(mask.tolist()),
            )
        ),
        root / "data/chunk-000/file-000.parquet",
        row_group_size=13,
    )
    return root


def action_dataset(root, **kwargs):
    options = (
        dict(
            source="fixture",
            viewpoint="ego_view",
            tokenize=tokenize,
            processor=ActionProcessor(None),
            video=False,
        )
        | kwargs
    )
    return ActionFrameDataset(root, **options)


@pytest.mark.parametrize("max_blocks,stride", [(1, 1), (2, 5), (4, 1)])
@pytest.mark.parametrize("offset", [0, 1, -2, 3])
def test_per_frame_candidates_include_terminal_anchors(long_root, max_blocks, stride, offset):
    columns = ActionColumns(target="state_unified", target_mask="mask_state", target_offset=offset)
    ds = action_dataset(
        long_root, columns=columns, candidates=FrameCandidateConfig(max_blocks=max_blocks, anchor_stride=stride)
    )
    expected = [(0, t) for t in range(0, 141, stride)] + [(2, t) for t in range(0, 51, stride)]
    assert [ds.locate(i) for i in range(len(ds))] == expected
    assert ds.episode_rows.tolist() == [0, 2]
    last_index = len(ds) - 1
    if offset > 1:
        with pytest.raises(EmptyTemporalSampleError, match="Action targets outside"):
            ds.read_sample(last_index)
        last_index -= (offset - 1 + stride - 1) // stride
    last = ds.read_sample(last_index)
    assert last["window"].past > 0
    assert last["action_valid_steps"].all()
    assert len(last["action"]) % 32 == 0


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
@pytest.mark.parametrize("offset", [0, 1, -2, 3])
def test_prepared_fields_match_existing_block_training(long_root, mode, offset):
    columns = ActionColumns(target="state_unified", target_mask="mask_state", target_offset=offset)
    new = action_dataset(long_root, columns=columns, labels=FrameLabelConfig(subtask_column=None))
    for index in (64 + max(0, -offset), len(new) - 1 - max(0, offset - 1)):
        a = new.make_sample(index, mode=mode, history=3)
        w = a["window"]
        first = w.anchor - w.past
        old = ActionDataset(
            long_root,
            source="fixture",
            viewpoint="ego_view",
            tokenize=tokenize,
            processor=ActionProcessor(None),
            columns=columns,
            video=False,
            candidates=CandidateConfig(max_actions=len(a["action"]), use_subtask=False),
        )
        old_index = next(
            i
            for i in range(len(old))
            if (old.get_window_metadata(i)[0]["episode_index"], old.get_window_metadata(i)[1]) == (w.episode, first)
        )
        b = old.make_sample(old_index, mode=mode, history=3)
        for key in (
            "action",
            "states",
            "anchors",
            "action_mask",
            "state_mask",
            "action_valid_steps",
            "state_action_indexes",
            "text_token_ids",
            "timestamps",
            "state_clip_fraction",
            "action_clip_fraction",
        ):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        for key in (
            "mode",
            "history_blocks",
            "block_size",
            "conditioning_fps",
            "conditioning_fps_action",
            "num_valid_video_latents",
            "action_frame_ids",
            "state_latent_indexes",
            "condition_frame_indexes_vision",
            "condition_frame_indexes_action",
            "token_layout",
            "token_length",
            "prompt",
        ):
            assert a[key] == b[key], key
        assert "state" not in a and "action_reference" not in a and "states_mask" not in a
    assert new.state_dict() == {"rng": None}


def test_four_blocks_use_four_states_and_keep_frame_labels(long_root):
    ds = action_dataset(long_root)
    plan = ds.plan_read_window(64)
    assert plan.past == plan.future == 0 and plan.max_blocks == 4
    sample = ds.make_sample(0, mode="policy", history=9)
    assert sample["action"].shape == (128, 55)
    assert sample["anchors"][:, 0].tolist() == [0, 32 * 55, 64 * 55, 96 * 55]
    expected = (torch.arange(128) % 32).float() * 55 + 1000
    torch.testing.assert_close(sample["action"][:, 0], expected)
    assert sample["states"].shape == (4, 55)
    assert sample["caption"] == "second"  # Dense caption follows actual video coverage.
    assert sample["condition_frame_indexes_vision"] == [0]
    assert sample["condition_frame_indexes_action"] == []
    assert sample["token_length"] == sum(sample["token_layout"].values())
    shorter = action_dataset(long_root, candidates=FrameCandidateConfig(max_blocks=1))
    assert shorter.make_sample(19)["caption"] == "second"
    assert shorter.make_sample(20)["caption"] == "second"


def test_prepared_action_uses_one_read_and_exact_bidirectional_video(long_root, monkeypatch):
    import veomni.data.causal_cosmos3.datasets.frame_dataset as frame_dataset
    import veomni.data.causal_cosmos3.readers.lerobot as reader

    calls, reads, windows = [], [], []
    original, original_window = reader.read_rows, reader.read_frame_window

    def read(*args, **kwargs):
        reads.append(args)
        return original(*args, **kwargs)

    def decode(path, queries, output, **kwargs):
        calls.append(queries.copy())
        output[:] = np.rint(queries * 30).astype(np.uint8)[None, :, None, None]

    def read_window(*args, **kwargs):
        result = original_window(*args, **kwargs)
        windows.append(result)
        return result

    monkeypatch.setattr(reader, "read_rows", read)
    monkeypatch.setattr(reader, "decode_into", decode)
    monkeypatch.setattr(frame_dataset, "read_frame_window", read_window)
    columns = ActionColumns(target="state_unified", target_mask="mask_state", target_offset=3)
    sample = action_dataset(long_root, columns=columns, video=True).make_sample(64, mode="policy", history=3)
    assert len(reads) == len(calls) == len(windows) == 1
    raw = windows[0]
    assert (raw["window"].anchor, raw["window"].past, raw["window"].future) == (64, 64, 64)
    assert raw["state"].shape == (129, 55) and raw["action"].shape == (128, 55)
    np.testing.assert_array_equal(raw["state"][:, 0], np.arange(129) * 55)
    np.testing.assert_array_equal(raw["action"][:, 0], np.arange(3, 131) * 55)
    torch.testing.assert_close(raw["timestamps"], torch.arange(129, dtype=torch.float64) / 30)
    np.testing.assert_allclose(calls[0], np.arange(0, 129, 4) / 30)
    assert sample["video"].shape == (3, 33, 256, 256)
    assert sample["video_valid_steps"].all() and sample["action_valid_steps"].all()
    assert sample["real_action_steps"] == 128 and sample["num_valid_video_latents"] == 9
    assert sample["states"].shape == (4, 55) and sample["caption"] == "second"


def test_full_forward_window_has_33_video_frames(long_root, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as reader

    monkeypatch.setattr(reader, "decode_into", lambda path, queries, output, **kwargs: output.fill(7))
    sample = action_dataset(long_root, video=True).make_sample(0)
    assert sample["video"].shape == (3, 33, 256, 256)
    assert sample["action"].shape == (128, 55) and sample["states"].shape == (4, 55)
    assert sample["num_valid_video_latents"] == 9


@pytest.mark.parametrize("bad_value", ["nonfinite", "zero_quaternion"])
def test_terminal_targets_are_discarded_before_block_encoding(long_root, bad_value):
    path = long_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    state = np.asarray(table["state_unified"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action_unified"].to_pylist(), dtype=np.float32)
    mask = np.asarray(table["mask_state"].to_pylist(), dtype=np.float32)
    mask[:, 17:21] = 1
    state[:, 17:21] = action[:, 17:21] = [0, 0, 0, 1]
    if bad_value == "nonfinite":
        action[128, 0] = np.nan
    else:
        action[128, 17:21] = 0
    for name, values in [
        ("state_unified", state),
        ("action_unified", action),
        ("mask_state", mask),
        ("mask_action", mask),
    ]:
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values.tolist()))
    pq.write_table(table, path)
    ds = action_dataset(long_root)
    sample = ds[0]
    assert sample["action_valid_steps"].sum() == 128
    assert torch.isfinite(sample["action"]).all()
    action[127] = action[128]
    table = table.set_column(
        table.schema.get_field_index("action_unified"), "action_unified", pa.array(action.tolist())
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="Nonfinite|quaternion"):
        ds.make_sample(0)


def test_raw_and_mapped_masks_are_separate(long_root):
    path = long_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    for name, values in [
        ("state_unified", [[float(i), float(i + 10)] for i in range(len(table))]),
        ("mask_state", [[0.0, 1.0]] * len(table)),
    ]:
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    pq.write_table(table, path)
    info_path = long_root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for name in ("state_unified", "mask_state"):
        info["features"][name]["shape"] = [2]
    info_path.write_text(json.dumps(info))
    ds = action_dataset(long_root, processor=ActionProcessor(None, state_mapping=[1] + [-1] * 54))
    raw, sample = ds.read_sample(0), ds[0]
    assert raw["state"].shape == (129, 2) and raw["state_mask"].tolist() == [False, True]
    assert sample["states"].shape == (4, 55) and sample["state_mask"].nonzero().flatten().tolist() == [0]
    assert sample["anchors"][0, 0] == 10 and sample["action"][0, 0] == 990


@pytest.mark.parametrize(
    "options,error",
    [
        (dict(video_stride=2), "must match"),
        (dict(mode="unknown"), "Invalid task"),
        (dict(history_range=(0, 2)), "history_range"),
    ],
)
def test_prepared_action_contract_validation(root, options, error):
    with pytest.raises(ValueError, match=error):
        action_dataset(root, **options)


def test_block_statistics_and_geometry_validation(long_root):
    stats = block_statistics()
    action_dataset(long_root, processor=ActionProcessor(stats))[0]
    stats["provenance"] = dict(action_reference="current_frame")
    with pytest.raises(ValueError, match="Current-frame statistics"):
        ActionProcessor(stats)
    stats["provenance"] = dict(geometry=dict(actions_per_block=64))
    with pytest.raises(ValueError, match="block geometry"):
        action_dataset(long_root, processor=ActionProcessor(stats))
    raw = action_dataset(long_root, processor=None)
    assert raw.read_sample(0)["state"].shape == (129, 55)
    with pytest.raises(ValueError, match="require processor and tokenize"):
        raw.make_sample(0)


def test_action_factory_prepares_policy_and_validation_requires_full_history(long_root):
    cfg = dict(
        stage="action",
        candidates=dict(mode="frame_anchor", max_blocks=4, anchor_stride=32),
        sample=dict(mode="policy", full_history=False, history_range=(2, 5)),
        sources=[
            dict(
                name="fixture",
                roots=[str(long_root)],
                viewpoint="ego_view",
                statistics=block_statistics(),
                read=dict(video=False),
            )
        ],
    )
    ds = build_dataset(cfg, tokenize)
    subset = dataset_subsets(ds)[0]
    assert type(subset) is ActionFrameDataset and len(ds) == 7
    assert subset.seed == 0 and subset.history_range == (2, 5)
    sample = ds[0]
    assert sample["mode"] == "policy" and sample["states"].shape == (4, 55)
    assert sample["caption"] == "second" and "token_length" in sample
    cfg["validation"] = dict(split_ratio=0.5, seed=42)
    _, _, manifest = split_dataset_config(cfg)
    assert manifest["sources"][0]["eligible_count"] == 2
    assert manifest["roots"][0]["excluded_episodes"] == [30]
    with pytest.raises(ValueError, match="requires full_history=True"):
        list(iter_validation_batches(ds))
    cfg["sample"]["full_history"] = True
    assert len(list(iter_validation_batches(build_dataset(cfg, tokenize)))) == len(ds)


@pytest.mark.parametrize(
    "stage,mode",
    [("action", mode) for mode in ("policy", "joint", "forward_dynamics", "inverse_dynamics")] + [("video", "video")],
)
@pytest.mark.parametrize("sampling", ["sample_uniform", "source_weighted"])
def test_frame_validation_reuses_samples_preserves_rng_and_dp_rounds(long_root, monkeypatch, stage, mode, sampling):
    import veomni.data.causal_cosmos3.readers.lerobot as reader

    cfg = config(long_root)
    cfg.update(stage=stage, sampling=sampling)
    cfg["candidates"].update(anchor_stride=17, max_blocks=2 if stage == "action" else 8)
    cfg["sample"] = dict(full_history=True)
    if stage == "action":
        cfg["sample"]["mode"] = mode
        cfg["sources"][0]["statistics"] = block_statistics()
    built = build_dataset(cfg, tokenize)
    subset = dataset_subsets(built)[0]
    window_count = len(subset)
    modes = ("forward_dynamics", "inverse_dynamics", "policy") if mode == "joint" else (mode,)
    reads = []
    original = reader.read_rows

    def read(*args, **kwargs):
        reads.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "read_rows", read)
    global_before = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state())
    expected = list(iter_validation_batches(built, seed=73))
    assert len(expected) == window_count * len(modes)
    assert len(reads) == len(expected)  # Anchor labels and payloads share one read.
    assert subset.state_dict() == {"rng": None}
    records = {}
    for item in expected:
        sample, record = item["batch"]["samples"][0], item["record"]
        index = record["candidate_index"]
        reference = subset.make_sample(index, **(dict(mode=record["mode"]) if stage == "action" else {}))
        assert item["weight"] == 1 and "history_seed" not in record
        assert sample["candidate_index"] == index
        assert "resolved_candidate_index" not in record["identity"]
        assert record["window"] == asdict(reference["window"])
        assert record["identity"]["anchor"] == reference["window"].anchor
        assert record["identity"]["future"] == reference["window"].future
        assert record["identity"]["max_blocks"] == subset.candidates.max_blocks
        assert record["caption"] == sample["caption"] == reference["caption"]
        for key in ("mode", "history_blocks", "prompt", "token_length", "condition_frame_indexes_vision"):
            assert sample[key] == reference[key]
        for key in ("timestamps", "text_token_ids", "video_valid_steps", "action", "states", "action_valid_steps"):
            if key in reference:
                torch.testing.assert_close(sample[key], reference[key], atol=0, rtol=0)
        temporal = subset.geometry.temporal_compression if stage == "action" else subset.temporal_compression
        assert sample["history_blocks"] == 1 + (len(sample["video_valid_steps"]) - 1) // temporal
        records[(record["mode"], index)] = record
    assert {record["caption"] for record in records.values()} == {"second"}
    assert subset.state_dict() == {"rng": None}
    subset._get_rng().integers(100, size=7)
    state = copy.deepcopy(subset.state_dict())
    for world_size in (4, 8):
        schedules = [list(iter_validation_batches(built, rank, world_size, seed=73)) for rank in range(world_size)]
        assert {len(items) for items in schedules} == {len(modes) * ((window_count + world_size - 1) // world_size)}
        counted = []
        for items in schedules:
            for item in items:
                record = item["record"]
                key = (record["mode"], record["candidate_index"])
                assert record == records[key]
                if item["weight"]:
                    counted.append(key)
        assert sorted(counted) == sorted(records)
    changed = next(iter_validation_batches(built, seed=74))["record"]
    first = expected[0]["record"]
    assert changed["identity"] == first["identity"] and changed["history_blocks"] == first["history_blocks"]
    assert all(changed["seeds"][stream] != seed for stream, seed in first["seeds"].items())
    assert subset.state_dict() == state
    assert random.getstate() == global_before[0] and pickle.dumps(np.random.get_state()) == global_before[1]
    assert torch.equal(torch.get_rng_state(), global_before[2])

    def failed_read(*args, **kwargs):
        raise OSError("frame validation read failed")

    monkeypatch.setattr(reader, "read_rows", failed_read)
    with pytest.raises(OSError, match="frame validation read failed"):
        next(iter_validation_batches(built))
    assert subset.state_dict() == state


@pytest.mark.parametrize("stage,mixed", [("action", False), ("video", False), ("action", True)])
def test_frame_validation_runs_existing_model_evaluator(long_root, tmp_path, monkeypatch, stage, mixed):
    import veomni.trainer.causal_cosmos3 as validation
    from tests.models.diffusers.test_causal_cosmos3_training import training_model
    from veomni.models.diffusers.causal_cosmos3.cosmos3_condition.modeling_cosmos3_condition import (
        CausalCosmos3ConditionModel,
    )

    cfg = config(long_root)
    cfg.update(stage=stage, validation=dict(seed=73))
    cfg["sample"] = dict(full_history=True)  # Use the Dataset's default task mode.
    if stage == "action":
        cfg["candidates"]["max_blocks"] = 4
        cfg["sources"][0]["statistics"] = block_statistics()
    if mixed:
        frame_source = copy.deepcopy(cfg["sources"][0])
        frame_source.update(name="a_frame", candidates=cfg["candidates"])
        old_source = copy.deepcopy(cfg["sources"][0])
        old_source.update(name="z_old", candidates=dict(mode="frame_window", max_actions=32, use_subtask=False))
        cfg["candidates"] = {}
        cfg["sources"] = [old_source, frame_source]  # Configuration and validation sort orders differ.
    built = build_dataset(cfg, tokenize)
    condition = CausalCosmos3ConditionModel(
        device="cpu", payload_dtype="float32", attention_backend="grouped", stage=stage
    )

    def encode(samples):
        # Replace only VAE weights; exercise real packing, noise, model and loss.
        return {
            "samples": [
                {
                    **sample,
                    "video_latents": torch.zeros(
                        2, 1 + (len(sample["timestamps"]) - 1) // (sample["window"].video_stride * 4), 16, 16
                    ),
                }
                for sample in samples
            ]
        }

    monkeypatch.setattr(condition, "get_condition", encode)
    model = training_model(
        vocab_size=256,
        action_dim=55,
        state_dim=55 if stage == "action" else 0,
        action_interface="shared" if stage == "action" else "none",
    )

    class Runtime:
        parallel_state = SimpleNamespace(dp_rank=0, dp_size=1, dp_group=None)

        def __call__(self, **inputs):
            return self.model(**inputs)

    runtime = Runtime()
    runtime.model, runtime.condition_model = model, condition
    trainer = SimpleNamespace(
        model=runtime,
        device="cpu",
        model_fwd_context=nullcontext(),
        validation_config=cfg,
        args=SimpleNamespace(
            model=SimpleNamespace(tokenizer_path=None),
            train=SimpleNamespace(
                global_rank=0,
                wandb=SimpleNamespace(enable=False),
                checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
            ),
        ),
    )
    monkeypatch.setattr(validation, "use_parallel_state", lambda state: nullcontext())
    monkeypatch.setattr(validation, "get_device_rng_state", torch.get_rng_state)
    monkeypatch.setattr(validation, "set_device_rng_state", torch.set_rng_state)
    validator = validation.CausalCosmos3Validator(trainer, built)
    tag = "policy" if stage == "action" else "video"
    assert validator.modes == (tag,)
    rng_before = condition.rng_state_dict()
    state = SimpleNamespace(global_step=1, epoch=0)
    first = validator.evaluate(state)
    second = validator.evaluate(state)
    assert first == second
    assert first[f"eval/{tag}/count"] == len(built)
    assert np.isfinite(first[f"eval/{tag}/loss"])
    assert first[f"eval/{tag}/padding_count"] == first[f"eval/{tag}/skipped_count"] == 0
    assert all(torch.equal(value, condition.rng_state_dict()[key]) for key, value in rng_before.items())
    assert all(subset.state_dict() == {"rng": None} for subset in dataset_subsets(built))
    assert model.training and condition.training
    assert len((tmp_path / "validation.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("stage", ["action", "video"])
@pytest.mark.parametrize("sampling", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("token_budget", [1, 2000])
def test_frame_dataloader_packs_samples_and_preserves_step_protocol(long_root, stage, sampling, token_budget):
    cfg = config(long_root)
    cfg.update(
        stage=stage,
        sampling=sampling,
        token_budget=token_budget,
        buffer_size=2,
        grad_accum_iter=2,
        train_steps=2,
        num_workers=0,
        seed=19,
    )
    if stage == "action":
        cfg["candidates"]["max_blocks"] = 4
        cfg["sources"][0]["statistics"] = block_statistics()
        cfg["sample"]["mode"] = "policy"
    original = list(build_action_dataloader(cfg, tokenize))
    collated = list(build_action_dataloader(cfg, tokenize, collate_fn=collate_action_pack))
    assert len(original) == len(collated) == cfg["train_steps"]
    for expected_step, step in zip(original, collated, strict=True):
        assert len(expected_step) == len(step) == cfg["grad_accum_iter"]
        for expected_pack, microbatch in zip(expected_step, step, strict=True):
            assert set(microbatch) == {"samples"}
            pack = microbatch["samples"]
            assert len(pack) == len(expected_pack) > 0
            if token_budget == 1:
                assert len(pack) == 1 and pack[0]["token_length"] > token_budget
            else:
                assert sum(sample["token_length"] for sample in pack) <= token_budget
            for expected, sample in zip(expected_pack, pack, strict=True):
                assert sample["sample_kind"] == "frame_anchor"
                for key in ("episode", "candidate_index", "history_blocks", "token_length"):
                    assert sample[key] == expected[key]
                assert sample["mode"] == ("policy" if stage == "action" else "video")
                assert ("action" in sample) == (stage == "action")
    if token_budget > 1:
        assert any(len(microbatch["samples"]) > 1 for step in collated for microbatch in step)


def test_action_rng_restores_modes_and_history(long_root):
    ds = action_dataset(long_root, mode="joint", full_history=False, history_range=(2, 5), seed=73)
    ds.make_sample(0)
    state = copy.deepcopy(ds.state_dict())
    expected = [(sample["mode"], sample["history_blocks"]) for sample in (ds.make_sample(0) for _ in range(8))]
    restored = action_dataset(long_root, mode="joint", full_history=False, history_range=(2, 5), seed=73)
    restored.load_state_dict(state)
    actual = [(sample["mode"], sample["history_blocks"]) for sample in (restored.make_sample(0) for _ in range(8))]
    assert actual == expected
    assert restored.select_history(9, explicit=50) == 50
    assert next(iter_frame_samples(restored, raw=True))["anchor_index"] == 0


def test_source_validation_script_checks_block_oracle(long_root, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as reader
    from scripts.causal_cosmos3 import validate_frame_dataset as script

    monkeypatch.setattr(reader, "decode_into", lambda path, queries, output, **kwargs: output.fill(0))
    monkeypatch.setattr(
        script,
        "reference_video",
        lambda path, queries, height, width: np.zeros((3, len(queries), height, width), dtype=np.uint8),
    )
    args = SimpleNamespace(
        root=str(long_root),
        source="fixture",
        viewpoint="ego_view",
        columns={},
        labels={},
        max_blocks=4,
        anchor_stride=1,
        statistics=None,
        benchmark_samples=2,
        workers=0,
    )
    result = script.validate(args)
    assert result["status"] == "passed" and len(result["checks"]) == 6
    assert result["processing"]["maximum_absolute_error"] == 0
    assert result["checks"][0]["blocks"] == 4 and result["checks"][-1]["blocks"] == 1


def video_dataset(root, **kwargs):
    return VideoFrameDataset(
        root,
        **(dict(source="fixture", viewpoint="ego_view", tokenize=tokenize, video=False) | kwargs),
    )


@pytest.mark.parametrize("video_stride,expected", [(None, 2), (1, 1), (2, 2), (4, 4)])
def test_video_auto_stride_and_explicit_stride(long_root, video_stride, expected):
    ds = video_dataset(
        long_root,
        episodes=[99],  # source_fps=30 exercises automatic downsampling.
        video_stride=video_stride,
        candidates=FrameCandidateConfig(max_blocks=1),
    )
    sample = ds.make_sample(0, history=3)
    assert sample["window"].video_stride == expected
    assert sample["conditioning_fps"] == 30 / expected
    assert len(sample["video_valid_steps"]) == 9
    assert sample["window"].past + sample["window"].future == 8 * expected


@pytest.mark.parametrize("source_fps,stride", [(5, 1), (16, 1), (16.01, 2), (30, 2)])
@pytest.mark.parametrize("max_blocks", [3, 16])
def test_video_candidates_and_prepared_fields_follow_geometry(long_root, monkeypatch, source_fps, stride, max_blocks):
    import veomni.data.causal_cosmos3.readers.lerobot as reader

    (long_root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(dict(episode_index=ep, source_fps=source_fps)) + "\n" for ep in (7, 99))
    )

    def decode(path, queries, output, **kwargs):
        output[:] = np.rint(queries * 30).astype(np.uint8)[None, :, None, None]

    monkeypatch.setattr(reader, "decode_into", decode)
    ds = video_dataset(
        long_root,
        candidates=FrameCandidateConfig(max_blocks=max_blocks, anchor_stride=5),
        labels=FrameLabelConfig(subtask_column=None),
        video=True,
    )
    assert [ds.locate(i) for i in range(len(ds))] == [(0, t) for t in range(0, 141, 5)] + [
        (2, t) for t in range(0, 51, 5)
    ]
    for index in (0, len(ds) - 1):
        sample = ds.make_sample(index, history=3)
        w = sample["window"]
        assert w.video_stride == stride and sample["history_blocks"] == 3
        total = w.past + w.future
        assert total % (stride * 8) == 0 and total <= max_blocks * stride * 8
        if index == 0:
            assert total == min(max_blocks, 140 // (stride * 8)) * stride * 8
        torch.testing.assert_close(
            sample["timestamps"], torch.arange(w.anchor - w.past, w.anchor + w.future + 1, dtype=torch.float64) / 30
        )
        expected = np.rint((sample["timestamps"][::stride].numpy() + w.video_offsets[0]) * 30).astype(np.uint8)
        np.testing.assert_array_equal(sample["video"][0, :, 0, 0], expected)
        assert sample["video_valid_steps"].all() and sample["observation_valid_steps"].all()
        assert sample["num_valid_video_latents"] == 1 + total // (stride * 4)
        assert sample["caption"] == "full task" and sample["token_length"] == sum(sample["token_layout"].values())
    assert ds.state_dict() == {"rng": None}


def test_video_uses_only_complete_blocks_and_projects_no_action_columns(long_root, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as reader

    calls, projections = [], []
    original = reader.read_rows

    def read(path, start, stop, columns, **kwargs):
        projections.append(columns)
        return original(path, start, stop, columns, **kwargs)

    def decode(path, queries, output, **kwargs):
        calls.append(queries.copy())
        output[:] = np.rint(queries * 30).astype(np.uint8)[None, :, None, None]

    monkeypatch.setattr(reader, "read_rows", read)
    monkeypatch.setattr(reader, "decode_into", decode)
    sample = video_dataset(long_root, episodes=[99], video=True, video_stride=2).make_sample(0, history=3)
    assert projections == [["timestamp", "task_index", "subtask_index"]] and len(calls) == 1
    np.testing.assert_allclose(calls[0], (142 + np.arange(0, 49, 2)) / 30)
    assert sample["video"].shape == (3, 25, 256, 256)
    torch.testing.assert_close(sample["timestamps"], torch.arange(49, dtype=torch.float64) / 30)
    assert sample["video_valid_steps"].all() and sample["num_valid_video_latents"] == 7
    assert sample["conditioning_fps"] == 15 and sample["caption"] == "second"
    assert not {
        "action",
        "state",
        "states",
        "action_mask",
        "state_mask",
        "real_action_steps",
        "action_valid_steps",
        "conditioning_fps_action",
        "condition_frame_indexes_action",
    }.intersection(sample)


def test_video_history_replay_full_history_and_frame_labels(long_root):
    ds = video_dataset(long_root, seed=73, history_range=(2, 5))
    ds.make_sample(0)
    state = copy.deepcopy(ds.state_dict())
    expected = [ds.make_sample(1)["history_blocks"] for _ in range(8)]
    restored = video_dataset(long_root, seed=73, history_range=(2, 5))
    restored.load_state_dict(state)
    assert [restored.make_sample(1)["history_blocks"] for _ in range(8)] == expected
    restored.load_state_dict({"rng": None})
    assert restored.make_sample(0, history=50)["history_blocks"] == 50
    assert restored.state_dict() == {"rng": None}
    restored.full_history = True
    sample = restored.make_sample(0)
    assert sample["history_blocks"] == 9  # 4 blocks, each containing 2 video latents.
    assert restored.state_dict() == {"rng": None}
    shorter = video_dataset(long_root, candidates=FrameCandidateConfig(max_blocks=3))
    assert shorter.make_sample(19)["caption"] == "second"
    assert shorter.make_sample(20)["caption"] == "second"


def test_prepared_video_contract_validation(root):
    with pytest.raises(ValueError, match="history_range"):
        video_dataset(root, history_range=(0, 2))


def test_video_raw_reads_do_not_require_tokenize(long_root):
    ds = video_dataset(long_root, tokenize=None)
    assert ds.read_sample(0)["caption"] == "first"
    with pytest.raises(ValueError, match="require tokenize"):
        ds.make_sample(0)


@pytest.mark.parametrize(
    "count,replicas,expected_total", [(13, 3, 24), (24, 3, 24), (25, 3, 48), (1, 3, 24), (13, 1, 16)]
)
def test_frame_epoch_sampler_covers_candidates_and_repeats_shuffled_prefix(count, replicas, expected_total):
    dataset = range(count)
    samplers = [FrameEpochSampler(dataset, 8, num_replicas=replicas, rank=rank, seed=19) for rank in range(replicas)]
    orders = []
    for epoch in (0, 1, 0):
        shards = []
        for sampler in samplers:
            sampler.set_epoch(epoch)
            indices = list(sampler)
            assert len(indices) == len(sampler) == expected_total // replicas
            assert len(indices) % 8 == 0
            shards.append(indices)
        order = [index for row in zip(*shards, strict=True) for index in row]
        shuffled = torch.randperm(count, generator=torch.Generator().manual_seed(19 + epoch)).tolist()
        assert order[:count] == shuffled
        assert set(order) == set(dataset)
        assert len(order) == expected_total
        assert order[count:] == [shuffled[i % count] for i in range(expected_total - count)]
        orders.append(order)
    assert orders[0] == orders[2]
    if count > 1:
        assert orders[0] != orders[1]


@pytest.mark.parametrize("stage", ["action", "video"])
@pytest.mark.parametrize("workers", [0, 1])
def test_fixed_frame_loader_preserves_pack_size_and_epoch_coverage(long_root, monkeypatch, stage, workers):
    import veomni.data.causal_cosmos3.datasets.pipeline as pipeline

    cfg = config(long_root)
    cfg.update(stage=stage, samples_per_pack=2, grad_accum_iter=2, num_workers=workers, seed=19)
    cfg["sample"]["full_history"] = True
    if stage == "action":
        cfg["candidates"]["max_blocks"] = 4
        cfg["sources"][0]["statistics"] = block_statistics()
        cfg["sample"]["mode"] = "policy"
    # These legacy budgets must have no influence on fixed-sample epochs.
    cfg.update(token_budget=1, max_seq_len=1, buffer_size=0, train_steps=99)
    identities = []
    for rank in range(2):
        monkeypatch.setattr(
            pipeline, "get_parallel_state", lambda rank=rank: SimpleNamespace(dp_rank=rank, dp_size=2, sp_size=1)
        )
        loader = build_action_dataloader(cfg, tokenize, collate_fn=collate_action_pack)
        assert len(loader) == 1
        for epoch in (0, 1):
            loader.set_epoch(epoch)
            steps = list(loader)
            assert len(steps) == 1
            assert len(steps[0]) == 2
            indices = []
            for pack in steps[0]:
                assert len(pack["samples"]) == 2
                for sample in pack["samples"]:
                    assert sample["token_length"] > 1
                    assert sample["mode"] == ("policy" if stage == "action" else "video")
                    assert ("action" in sample) == (stage == "action")
                    indices.append(sample["candidate_index"])
            assert indices == list(loader.sampler)
            if epoch == 0:
                identities.extend(indices)
        # Default collation still returns list[packs], with each pack a list.
        plain = build_action_dataloader(cfg, tokenize)
        assert [len(pack) for pack in next(iter(plain))] == [2, 2]
        del loader, plain
    assert set(identities) == set(range(len(build_dataset(cfg, tokenize))))
    assert len(identities) == 8


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_fixed_frame_packing_requires_positive_sample_count(root, count):
    cfg = {**config(root), "samples_per_pack": count}
    with pytest.raises(ValueError, match="positive integer"):
        get_samples_per_pack(cfg)


@pytest.mark.parametrize("update", [dict(sampling="source_weighted"), dict(candidates=dict(mode="frame_window"))])
def test_fixed_frame_packing_rejects_non_epoch_sampling(root, update):
    cfg = {**config(root), "samples_per_pack": 2, **update}
    with pytest.raises(ValueError, match="frame_anchor.*sample_uniform"):
        build_action_dataloader(cfg, tokenize)


def test_fixed_frame_packing_resolves_source_candidate_override(root):
    cfg = {**config(root), "samples_per_pack": 2}
    cfg["sources"][0]["candidates"] = dict(mode="frame_window")
    with pytest.raises(ValueError, match="frame_anchor.*sample_uniform"):
        get_samples_per_pack(cfg)
    cfg["sources"][0]["candidates"] = cfg.pop("candidates")
    assert get_samples_per_pack(cfg) == 2
    cfg.pop("samples_per_pack")
    cfg["sampling"] = "source_weighted"
    assert get_samples_per_pack(cfg) is None


@pytest.mark.parametrize("stage", ["action", "video"])
@pytest.mark.parametrize(
    "span,stride,ids,expected",
    [
        (8, 1, [0] * 4 + [1] * 13, 1),
        (8, 1, [0] * 6 + [1] * 7 + [0] * 4, 1),
        (8, 1, [0] * 8 + [1] * 8 + [0], 0),
        (8, 2, [0] * 5 + [1] + [0] * 5 + [2] * 10 + [1] * 4, 2),
        (16, 8, [0] * 8 + [1] + [2] * 15 + [1] * 9, 1),
    ],
)
def test_dense_caption_uses_actual_video_votes_and_contiguous_segments(long_root, stage, span, stride, ids, expected):
    from veomni.data.causal_cosmos3.transforms.processing import Geometry

    path = long_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    labels = [*ids, *([0] * (len(table) - len(ids)))]
    table = table.set_column(table.schema.get_field_index("subtask_index"), "subtask_index", pa.array(labels))
    pq.write_table(table, path)
    texts = ["first", "second", "third"]
    pq.write_table(pa.table(dict(subtask_index=[0, 1, 2], subtask=texts)), long_root / "meta/subtasks.parquet")
    common = dict(episodes=[7], candidates=FrameCandidateConfig(max_blocks=(len(ids) - 1) // span))
    if stage == "action":
        ds = action_dataset(
            long_root, geometry=Geometry(actions_per_block=span, video_stride=stride, temporal_compression=1), **common
        )
    else:
        ds = video_dataset(
            long_root, video_stride=stride, temporal_compression=1, latent_block_size=span // stride, **common
        )
    sample = ds.make_sample(0)
    assert sample["subtask_index"] == expected and sample["caption"] == texts[expected]
    assert sample["anchor_labels"]["subtask_index"] == 0
    assert "frame_labels" not in sample
    assert texts[expected] in sample["prompt"]
    assert sample["text_token_ids"].tolist() == tokenize(sample["prompt"])
    assert sample["token_layout"]["text"] == len(sample["text_token_ids"])


@pytest.mark.parametrize("stage", ["action", "video"])
@pytest.mark.parametrize("interior_change", [False, True])
def test_raw_task_bounds_window_independently_of_dense_subtasks(long_root, stage, interior_change):
    path = long_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    tasks = np.zeros(len(table), dtype=np.int64)
    tasks[65:80] = 1 if interior_change else 0
    tasks[32] = 0 if interior_change else 1
    table = table.set_column(table.schema.get_field_index("task_index"), "task_index", pa.array(tasks))
    pq.write_table(table, path)
    ds = action_dataset(long_root) if stage == "action" else video_dataset(long_root, video_stride=4)
    sample = ds.make_sample(64)
    expected_start = 0 if interior_change else 64
    assert sample["video_frame_indices"] == list(range(expected_start, 129, 4))
    assert sample["window"].anchor == 64 and sample["window"].past == 64 - expected_start
    assert sample["caption"] == "second"
    if stage == "action":
        assert sample["state_action_indexes"].tolist() == list(range(0, 128 - expected_start, 32))


@pytest.mark.parametrize("stage", ["action", "video"])
def test_invalid_anchor_getitem_tries_following_candidate(long_root, monkeypatch, stage):
    options = dict(episodes=[99], candidates=FrameCandidateConfig(max_blocks=1), seed=73, full_history=True)
    constructor = action_dataset if stage == "action" else video_dataset
    if stage == "video":
        options["video_stride"] = 4
    ds = constructor(long_root, **options)
    with pytest.raises(EmptyTemporalSampleError):
        ds.make_sample(19)
    calls = []
    make_sample = ds.make_sample

    def read(index, **kwargs):
        calls.append(index)
        return make_sample(index, **kwargs)

    monkeypatch.setattr(ds, "make_sample", read)
    sample = ds[19]
    assert calls == list(range(19, 33))
    assert sample["candidate_index"] == 32 and sample["video_valid_steps"].all()
    assert ds.state_dict() == {"rng": None}


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_validation_does_not_retry_read_or_label_errors(long_root, monkeypatch, error_type):
    ds = action_dataset(long_root)
    calls = []

    def broken(index, **kwargs):
        calls.append(index)
        raise error_type("broken payload")

    monkeypatch.setattr(ds, "read_sample", broken)
    with pytest.raises(error_type, match="broken payload"):
        next(iter_validation_batches(ds))
    assert calls == [0] and ds.state_dict() == {"rng": None}


@pytest.mark.parametrize("stage", ["action", "video"])
def test_validation_invalid_anchor_propagates_without_resampling(long_root, monkeypatch, stage):
    options = dict(episodes=[99], candidates=FrameCandidateConfig(max_blocks=1))
    ds = (
        action_dataset(long_root, **options)
        if stage == "action"
        else video_dataset(long_root, video_stride=4, full_history=True, **options)
    )
    ds._get_rng().integers(100, size=17)
    before = copy.deepcopy(ds.state_dict())
    calls = []
    make_sample = ds.make_sample

    def read(index, **kwargs):
        calls.append(index)
        return make_sample(index, **kwargs)

    monkeypatch.setattr(ds, "make_sample", read)
    batches = iter_validation_batches(ds, seed=73)
    for index in range(19):
        item = next(batches)
        assert item["record"]["candidate_index"] == item["batch"]["samples"][0]["candidate_index"] == index
        assert item["weight"] == 1
    with pytest.raises(EmptyTemporalSampleError):
        next(batches)
    assert calls == list(range(20)) and ds.state_dict() == before


@pytest.mark.parametrize("stage", ["action", "video"])
def test_episodes_without_a_complete_block_are_excluded(root, stage):
    with pytest.raises(ValueError, match="No legal frame candidates"):
        (action_dataset if stage == "action" else video_dataset)(root)


@pytest.mark.parametrize("workers", [0, 2])
def test_fixed_epoch_skips_bad_anchors_and_preserves_dp_steps(long_root, monkeypatch, workers):
    import veomni.data.causal_cosmos3.datasets.pipeline as pipeline

    cfg = config(long_root)
    cfg.update(stage="action", samples_per_pack=2, grad_accum_iter=2, num_workers=workers, seed=19)
    cfg["candidates"].update(anchor_stride=1)
    cfg["sample"] = dict(mode="policy", full_history=True)
    cfg["sources"][0]["statistics"] = block_statistics()
    cfg["sources"][0]["read"]["episodes"] = [99]
    requested, resolved = [], []
    for rank in range(2):
        monkeypatch.setattr(
            pipeline, "get_parallel_state", lambda rank=rank: SimpleNamespace(dp_rank=rank, dp_size=2, sp_size=1)
        )
        loader = build_action_dataloader(cfg, tokenize, collate_fn=collate_action_pack)
        assert len(loader) == 7  # ceil(51 / (2 samples * 2 DP * 2 accumulation)).
        rank_requests = list(loader.sampler)
        rank_resolved = []
        for step in loader:
            assert len(step) == 2
            for pack in step:
                assert len(pack["samples"]) == 2
                for sample in pack["samples"]:
                    rank_resolved.append(sample["candidate_index"])
                    assert sample["action_valid_steps"].all()
        assert rank_resolved == [32 if 19 <= index < 32 else index for index in rank_requests]
        requested.extend(rank_requests)
        resolved.extend(rank_resolved)
    assert set(requested) == set(range(51)) and len(requested) == len(resolved) == 56
    assert any(a != b for a, b in zip(requested, resolved, strict=True))
