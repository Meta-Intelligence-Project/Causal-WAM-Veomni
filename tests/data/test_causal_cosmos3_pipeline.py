"""Native action semantics, greedy packing and fixed-topology state restore."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from veomni.data.causal_cosmos3 import (
    ActionDataset,
    ActionProcessor,
    ActionTemplate55,
    CandidateConfig,
    build_action_dataloader,
)
from veomni.data.causal_cosmos3.dataset import segment_windows
from veomni.data.causal_cosmos3.pipeline import ActionBatchingStrategy


def tokenize(text):
    return [1] + list(text.encode()) + [2]


def statistics():
    mask = [True] * 51 + [False] * 4
    return {name: dict(low=[-100.0] * 55, high=[100.0] * 55, valid=mask) for name in ("state", "action")}


@pytest.fixture
def roots(tmp_path, request):
    result = []
    for subset, length in enumerate(getattr(request, "param", (75, 41))):
        root = tmp_path / str(subset)
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        info = dict(
            codebase_version="v3.0",
            fps=30,
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            video_path="videos/{video_key}/{chunk_index}/{file_index}.mp4",
            features={
                key: dict(dtype="float32", shape=[55])
                for key in ("state_unified", "action_unified", "mask_state", "mask_action")
            },
        )
        info["features"]["cam"] = dict(dtype="video", shape=[32, 32, 3], names=["height", "width", "channels"])
        (root / "meta/info.json").write_text(json.dumps(info))
        ep = dict(
            episode_index=[17],
            dataset_from_index=[100],
            dataset_to_index=[100 + length],
            **{
                "data/chunk_index": [0],
                "data/file_index": [0],
                "videos/cam/chunk_index": [0],
                "videos/cam/file_index": [0],
                "videos/cam/from_timestamp": [0.0],
            },
        )
        pq.write_table(pa.table(ep), root / "meta/episodes/chunk-000/file-000.parquet")
        annotations = [
            dict(start_frame=0, end_frame=20, action_text="first"),
            dict(start_frame=20, end_frame=length, action_text="second"),
        ]
        (root / "meta/episodes.jsonl").write_text(
            json.dumps(dict(episode_index=17, source_fps=5 if subset else 25, action_config=annotations)) + "\n"
        )
        pq.write_table(pa.table(dict(task_index=[0], task=["fallback"])), root / "meta/tasks.parquet")
        values = np.zeros((length, 55), dtype=np.float32)
        values[:, 0] = np.arange(length)
        mask = np.zeros_like(values)
        mask[:, 0] = 1

        def vector(x):
            return pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), 55)

        pq.write_table(
            pa.table(
                dict(
                    state_unified=vector(values),
                    action_unified=vector(values + 1),
                    mask_state=vector(mask),
                    mask_action=vector(mask),
                    timestamp=np.arange(length) / 30,
                    task_index=np.zeros(length, dtype=np.int64),
                )
            ),
            root / "data/chunk-000/file-000.parquet",
            row_group_size=11,
        )
        result.append(root)
    return result


def dataset(root, **kwargs):
    return ActionDataset(
        root,
        source="fixture",
        processor=ActionProcessor(statistics()),
        tokenize=tokenize,
        cameras={"head": "cam"},
        viewpoint="ego_view",
        video=False,
        **kwargs,
    )


def test_candidate_text_boundaries_and_short_tail(roots):
    ds = dataset(roots[0], candidates=CandidateConfig(max_actions=32))
    assert len(ds) == 43
    assert ds.window(19)[1] == "first" and ds.window(20)[1] == "second"
    assert ds.window(19)[0].start + 32 > 20  # frame_window crosses subtask
    short = dataset(roots[1], candidates=CandidateConfig(max_actions=160))
    sample = short[0]
    assert len(short) == 1 and sample["window"].actions == 64 and sample["real_action_steps"] == 40
    assert sample["source_fps"] == 5 and sample["storage_fps"] == 30
    assert sample["conditioning_fps"] == 1.25 and sample["conditioning_fps_action"] == 5
    assert sample["action_valid_steps"].sum() == 40
    assert sample["num_valid_video_latents"] == 4
    assert sample["state_latent_indexes"].tolist() == [0, 2]
    assert sample["action_frame_ids"].tolist() == sum(([i] * 16 for i in range(1, 5)), [])
    assert sample["token_length"] == len(sample["text_token_ids"]) + 2 * 5 * 64 + 64 + 2


def test_segments_and_explicit_borrow(roots):
    assert segment_windows(0, 76, 32, 5) == [(0, 32), (27, 32), (43, 32)]
    assert segment_windows(2, 50, 160, 0) == [(2, 47)]
    ds = dataset(roots[0], candidates=CandidateConfig(mode="segmented", max_actions=32, borrow_subtask_frames=True))
    assert ds.window(0)[0].start == 0 and ds.window(0)[1] == "first"
    assert ds.window(0)[0].actions == 32
    no_borrow = dataset(roots[0], candidates=CandidateConfig(mode="segmented", max_actions=32))
    assert no_borrow.window(0)[1] == "first"
    assert no_borrow.window(0)[0].real_actions == 20
    assert no_borrow.window(0)[0].actions == 32


def test_template_quaternion_absolute_fields_and_roundtrip():
    template = ActionTemplate55()
    mask = torch.ones(55, dtype=torch.bool)
    mask[51:] = False
    anchor = torch.randn(4, 55)
    target = torch.randn(4, 55)
    encoded = template.encode_action_delta(target, anchor, mask)
    assert torch.allclose(encoded[:, 28:42], target[:, 28:42])
    decoded = template.decode_action_delta(encoded, anchor, mask)
    torch.testing.assert_close(decoded, template.sanitize(target, mask), atol=1e-6, rtol=1e-5)
    assert torch.all(decoded[:, 51:] == 0)
    mask[17] = False
    with pytest.raises(ValueError, match="quaternion"):
        template.validate_valid_mask(mask)


def test_read_error_skips_but_processing_errors_propagate(roots, monkeypatch):
    import veomni.data.causal_cosmos3.dataset as module

    ds = dataset(roots[0])
    original = module.read_window

    def fail(*args, **kwargs):
        raise OSError("broken video")

    monkeypatch.setattr(module, "read_window", fail)
    assert ds[0] == [] and ds.failures == 1
    monkeypatch.setattr(module, "read_window", original)
    assert ds[0]["source"] == "fixture"
    ds.processor.statistics["action"][2][0] = False
    with pytest.raises(ValueError, match="cover"):
        ds[0]


def test_greedy_keeps_nonfitting_and_packs_oversize_alone():
    strategy = ActionBatchingStrategy(10, 1)
    for i, n in enumerate((6, 7, 4)):
        strategy.put_item(dict(token_length=n, index=i))
    assert [x["index"] for x in strategy.get_micro_batch(0)] == [0, 2]
    assert [x["index"] for x in strategy.get_micro_batch(1)] == [1]
    strategy.put_item(dict(token_length=11, index=3))
    strategy.put_item(dict(token_length=2, index=4))
    assert [x["index"] for x in strategy.get_micro_batch(2)] == [3]
    assert [x["index"] for x in strategy.get_micro_batch(3)] == [4]
    assert strategy.empty()


def config(roots, mode, workers):
    return dict(
        sampling=mode,
        token_budget=1600,
        grad_accum_iter=2,
        train_steps=5,
        buffer_size=2,
        num_workers=workers,
        seed=19,
        candidates=dict(max_actions=32),
        sample=dict(video=False, full_history=False),
        sources=[
            dict(
                name=f"source{i}",
                roots=[str(root)],
                statistics=statistics(),
                weight=i + 1,
                cameras={"head": "cam"},
                viewpoint="ego_view",
            )
            for i, root in enumerate(roots)
        ],
    )


def patch_rank(monkeypatch, rank):
    import veomni.data.causal_cosmos3.pipeline as pipeline
    import veomni.data.data_loader as loader
    import veomni.data.dataset as datasets

    state = SimpleNamespace(dp_rank=rank, dp_size=2, sp_size=1)
    for module in (pipeline, loader, datasets):
        monkeypatch.setattr(module, "get_parallel_state", lambda: state)


def signature(step):
    return [
        [
            (
                s["source"],
                s["candidate_index"],
                s["mode"],
                s["history_blocks"],
                s["token_length"],
                s["action"].tolist(),
            )
            for s in pack
        ]
        for pack in step
    ]


@pytest.mark.parametrize("mode", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Upstream dynamic-loader yield/step ordering retained by review; exact resume is not guaranteed",
)
def test_multirank_steps_and_exact_resume(roots, monkeypatch, mode, workers):
    all_ranks = []
    for rank in (0, 1):
        patch_rank(monkeypatch, rank)
        cfg = config(roots, mode, workers)
        loader = build_action_dataloader(cfg, tokenize)
        iterator = iter(loader)
        next(iterator)
        state = copy.deepcopy(loader.state_dict())
        expected = [next(iterator) for _ in range(4)]
        with pytest.raises(StopIteration):
            next(iterator)
        assert len(expected) == 4
        restored = build_action_dataloader(cfg, tokenize)
        restored.load_state_dict(state)
        actual = list(restored)
        assert [signature(s) for s in actual] == [signature(s) for s in expected]
        assert restored.step == 5
        for step in actual:
            assert len(step) == 2
            assert all(0 < sum(x["token_length"] for x in pack) <= 1600 for pack in step)
        all_ranks.append([signature(s) for s in expected])
    assert all_ranks[0] != all_ranks[1]


@pytest.mark.parametrize("mode", ["sample_uniform", "source_weighted"])
def test_failed_candidates_do_not_change_step_protocol(roots, monkeypatch, mode):
    import veomni.data.causal_cosmos3.dataset as module

    original = module.read_window

    def sometimes_fails(window, **kwargs):
        if window.start % 3 == 0:
            raise OSError("fixture failure")
        return original(window, **kwargs)

    monkeypatch.setattr(module, "read_window", sometimes_fails)
    patch_rank(monkeypatch, 0)
    cfg = config(roots, mode, 0)
    loader = build_action_dataloader(cfg, tokenize)
    result = list(loader)
    assert len(result) == 5 and all(len(step) == 2 for step in result)
    assert all(s["window"].start % 3 for step in result for pack in step for s in pack)


def test_sampling_probabilities_without_source_weight_conversion(monkeypatch):
    from torch.utils.data import ConcatDataset, Dataset
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from veomni.data.causal_cosmos3.dataset import UniformSource
    from veomni.data.dataset import WeightedMultiSourceDataset

    class Candidates(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return dict(candidate=index, token_length=1)

    groups = [ConcatDataset([Candidates(10), Candidates(20)]), ConcatDataset([Candidates(10)])]
    global_candidates = ConcatDataset(groups)
    shards = [list(StatefulDistributedSampler(global_candidates, num_replicas=2, rank=r, seed=7)) for r in (0, 1)]
    assert sorted(shards[0] + shards[1]) == list(range(40))
    assert not set(shards[0]) & set(shards[1])
    patch_rank(monkeypatch, 0)
    weighted = WeightedMultiSourceDataset(
        [UniformSource(group, i, 0) for i, group in enumerate(groups)],
        weights=[1, 1],
        level="sample",
        upstream_sharded=True,
        source_names=["a", "b"],
        seed=7,
    )
    iterator = iter(weighted)
    counts = np.zeros(2, dtype=int)
    for _ in range(12000):
        sample = next(iterator)
        counts[sample["ds_idx"]] += 1
    assert abs(counts[0] / counts.sum() - 0.5) < 0.025


def test_per_subset_statistics_and_geometry_validation(roots, monkeypatch):
    patch_rank(monkeypatch, 0)
    cfg = config(roots, "sample_uniform", 0)
    cfg["sources"] = [cfg["sources"][0]]
    cfg["sources"][0]["roots"] = list(map(str, roots))
    cfg["sources"][0]["statistics"] = "{root}/meta/causal_action_stats.json"
    for root in roots:
        (root / "meta/causal_action_stats.json").write_text(json.dumps(statistics()))
    assert len(next(iter(build_action_dataloader(cfg, tokenize)))) == 2
    bad = statistics()
    bad["provenance"] = dict(geometry=dict(actions_per_block=64))
    (roots[1] / "meta/causal_action_stats.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="block geometry"):
        build_action_dataloader(cfg, tokenize)


@pytest.mark.parametrize(
    "resolution,viewpoint,head,expected",
    [
        ("256", "concat_view", (1080, 1920), (384, 448)),
        ("480", "concat_view", (1080, 1920), (528, 640)),
        ("256", "ego_view", (480, 640), (256, 320)),
        ("480", "ego_view", (480, 640), (480, 640)),
        ("256", "concat_view", (1920, 1080), (672, 256)),
        ("480", "concat_view", (1920, 1080), (960, 352)),
    ],
)
def test_resolution_layout_and_prompt(resolution, viewpoint, head, expected):
    from veomni.data.causal_cosmos3.layout import VIEW_DESCRIPTION, camera_layout, format_prompt

    info = {"features": {"h": {"names": ["channels", "height", "width"], "shape": [3, *head]}}}
    cameras = camera_layout(info, {"head": "h", "left": "l", "right": "r"}, resolution, viewpoint)
    height = max(c.top + c.height for c in cameras)
    width = max(c.left + c.width for c in cameras)
    assert (height, width) == expected
    assert cameras[0].key == "h" and cameras[0].top == cameras[0].left == 0
    if viewpoint == "concat_view":
        assert [c.key for c in cameras] == ["h", "l", "r"]
        assert cameras[1].top == cameras[2].top == cameras[0].height
        assert cameras[1].left == 0 and cameras[2].left == cameras[1].width
        assert cameras[1].width + cameras[2].width == width
    prompt = format_prompt(
        dict(
            caption="Move.",
            viewpoint=viewpoint,
            image_size=expected,
            additional_view_description=VIEW_DESCRIPTION[viewpoint],
        )
    )
    assert prompt.count(f"This video is {height}x{width}.") == 1
    assert prompt.count(VIEW_DESCRIPTION[viewpoint]) == 1


def test_default_resolution_and_custom_prompt(roots):
    ds = dataset(roots[0], prompt_formatter=lambda sample: sample["caption"])
    sample = ds[0]
    assert sample["resolution"] == "256" and sample["image_size"] == (256, 256)
    assert sample["prompt"] == "first"  # custom formatter owns augmentation
    with pytest.raises(ValueError, match="256 or 480"):
        dataset(roots[0], resolution="720")


@pytest.mark.parametrize("length", [0, 1, 20, 31, 32, 96, 99, 100, 159, 160])
def test_segment_short_candidate_counts(length):
    assert segment_windows(7, 8 + length, 160, 17) == ([(7, length)] if length else [])


@pytest.mark.parametrize("roots", [(141, 101)], indirect=True)
def test_segment_short_subtask_reads_full_tail_without_borrow(roots):
    from veomni.data.causal_cosmos3 import read_window

    (roots[0] / "meta/episodes.jsonl").write_text(
        json.dumps(
            dict(
                episode_index=17,
                action_config=[
                    dict(start_frame=0, end_frame=100, action_text="first"),
                    dict(start_frame=100, end_frame=141, action_text="second"),
                ],
            )
        )
    )
    ds = dataset(roots[0], candidates=CandidateConfig(mode="segmented", max_actions=160))
    assert len(ds) == 2
    window, caption, _ = ds.window(0)
    assert caption == "first" and window.real_actions == 100 and window.actions == 128
    raw = read_window(window, video=False)
    assert raw["action"][96:100, 0].tolist() == [97, 98, 99, 100]
    assert raw["action"][100:, 0].tolist() == [100] * 28
    assert raw["state"][101:, 0].tolist() == [100] * 28
    sample = ds[0]
    assert sample["action_valid_steps"].tolist() == [True] * 100 + [False] * 28
    assert sample["num_valid_video_latents"] == 8  # L0 + 6 full + 1 mixed
    assert sample["token_layout"]["action"] == 128
    episode = dataset(roots[1], candidates=CandidateConfig(mode="segmented", max_actions=160, use_subtask=False))
    assert len(episode) == 1 and episode.window(0)[0].real_actions == 100


@pytest.mark.parametrize("roots", [(35, 37), (35, 36)], indirect=True)
def test_native_sampler_shards_replays_and_new_epoch(roots, monkeypatch):
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    shards = []
    for rank in (0, 1):
        patch_rank(monkeypatch, rank)
        cfg = config(roots, "sample_uniform", 0)
        loader = build_action_dataloader(cfg, tokenize)
        sampler = loader._dataloader.sampler
        assert isinstance(sampler, StatefulDistributedSampler)
        expected = StatefulDistributedSampler(sampler.dataset, num_replicas=2, rank=rank, seed=19)
        assert list(sampler) == list(expected)
        first = list(sampler)
        assert list(sampler) == first
        loader.set_epoch(1)
        expected.set_epoch(1)
        assert list(sampler) == list(expected)
        assert list(sampler) != first
        shards.extend(first)
    n = len(sampler.dataset)
    assert set(shards) == set(range(n))
    assert len(shards) == 2 * ((n + 1) // 2)


def test_unequal_rank_exhaustion_replays_same_shard(monkeypatch):
    from torch.utils.data import Dataset
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from veomni.data.data_collator import NoopDataCollator
    from veomni.data.data_loader import build_native_dataloader

    class Candidates(Dataset):
        def __init__(self, length):
            self.length = length
            self.reads = []

        def __len__(self):
            return 4

        def __getitem__(self, index):
            self.reads.append(index)
            return dict(index=index, token_length=self.length)

    counts = []
    for rank, length in ((0, 3), (1, 7)):
        patch_rank(monkeypatch, rank)
        ds = Candidates(length)
        loader = build_native_dataloader(
            ds,
            micro_batch_size=1,
            global_batch_size=2,
            dataloader_batch_size=1,
            max_seq_len=9,
            train_steps=4,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            bsz_warmup_ratio=0,
            batching_strategy=ActionBatchingStrategy(9, 1),
            collate_fn=NoopDataCollator(),
            num_workers=0,
            prefetch_factor=None,
            pin_memory=False,
            seed=19,
        )
        steps = list(loader)
        assert len(steps) == 4 and all(len(s) == 1 for s in steps)
        assert all(sum(x["token_length"] for x in s[0]) <= 9 for s in steps)
        shard = list(StatefulDistributedSampler(ds, num_replicas=2, rank=rank, seed=19))
        assert ds.reads == (shard * (len(ds.reads) // 2 + 1))[: len(ds.reads)]
        counts.append(len(ds.reads))
    assert counts[0] > counts[1]  # short rank consumes/replays more, without changing step count


@pytest.mark.parametrize("roots", [(100, 101)], indirect=True)
def test_video_tail_queries_repeat_real_endpoint_and_keep_padding_invalid(roots, monkeypatch):
    import veomni.data.causal_cosmos3.lerobot as reader

    for root, real in zip(roots, (99, 100)):
        seen = []

        def decode(path, queries, output, seen=seen, **kwargs):
            seen.extend(queries)
            output.fill(0)

        monkeypatch.setattr(reader, "decode_into", decode)
        ds = dataset(root, candidates=CandidateConfig(mode="segmented", max_actions=160, use_subtask=False))
        ds.video = True
        sample = ds[0]
        expected = np.minimum(np.arange(0, 129, 4), real) / 30
        np.testing.assert_allclose(seen, expected, rtol=0, atol=1e-7)
        assert sample["action_valid_steps"].sum().item() == real
        # 99: sampled real frames stop at 96; the repeated 99 frame is still padding.
        # 100: frame 100 is real and keeps one additional mixed latent valid.
        assert sample["num_valid_video_latents"] == (7 if real == 99 else 8)
