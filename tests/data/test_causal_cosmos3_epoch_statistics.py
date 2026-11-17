"""Offline epoch simulation against native sampler/loader and real sample lengths."""

import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from torch.utils.data import Dataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.causal_cosmos3.statistics.estimate_epoch_iters import (
    ReplayShard,
    build_root_lengths,
    estimate,
    global_order,
    load_recipe,
    prepare_sources,
    simulate_rank,
)
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.datasets.dataset import ActionDataset, CandidateConfig, VideoDataset
from veomni.data.causal_cosmos3.datasets.pipeline import ActionBatchingStrategy
from veomni.data.causal_cosmos3.readers.lerobot import ActionColumns
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry
from veomni.data.data_collator import NoopDataCollator
from veomni.data.data_loader import build_native_dataloader


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def byte_tokens(prompt):
    return [1, *prompt.encode(), 2]


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: dict(dtype="float32", shape=[55]) for key in ("state", "action", "state_mask", "action_mask")}
    features["cam"] = dict(dtype="video", shape=[32, 32, 3], names=["height", "width", "channels"])
    info = dict(
        codebase_version="v3.0",
        fps=30,
        features=features,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/{chunk_index}/{file_index}.mp4",
    )
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, annotations, offset = [], [], 0
    for episode, length in ((3, 39), (8, 69), (12, 53)):
        episodes.append(
            dict(
                episode_index=episode,
                dataset_from_index=offset,
                dataset_to_index=offset + length,
                **{
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "videos/cam/chunk_index": 0,
                    "videos/cam/file_index": 0,
                    "videos/cam/from_timestamp": 0.0,
                },
            )
        )
        annotations.append(
            dict(
                episode_index=episode,
                source_fps=12 if episode == 3 else 25,
                action_config=[dict(start_frame=2, end_frame=20, action_text="subtask")],
            )
        )
        offset += length
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in annotations))
    pq.write_table(
        pa.table(dict(task_index=[0, 1], task=["short", "a longer fallback task"])), root / "meta/tasks.parquet"
    )
    values = np.zeros((offset, 55), dtype=np.float32)
    masks = np.zeros_like(values)
    masks[:, 0] = 1

    def vectors(array):
        return pa.FixedSizeListArray.from_arrays(pa.array(array.ravel()), 55)

    pq.write_table(
        pa.table(
            dict(
                state=vectors(values),
                action=vectors(values),
                state_mask=vectors(masks),
                action_mask=vectors(masks),
                timestamp=np.concatenate([np.arange(length) / 30 for length in (39, 69, 53)]),
                task_index=np.arange(offset) % 2,
            )
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    stats = {
        key: dict(low=[-100.0] * 55, high=[100.0] * 55, valid=[True] * 51 + [False] * 4) for key in ("state", "action")
    }
    (root / "meta/stats.json").write_text(json.dumps(stats))
    return root


def source_config(root):
    return dict(
        name="fixture",
        roots=[str(root)],
        viewpoint="ego_view",
        cameras=dict(head="cam"),
        statistics="{root}/meta/stats.json",
        columns=dict(
            state="state", target="action", state_mask="state_mask", target_mask="action_mask", target_offset=1
        ),
    )


@pytest.mark.parametrize("stage", ["action", "video"])
@pytest.mark.parametrize("mode,maximum,use_subtask", [("frame_window", 32, True), ("segmented", -1, False)])
def test_metadata_lengths_match_real_samples_without_payload_reads(
    source_root, tmp_path, monkeypatch, stage, mode, maximum, use_subtask
):
    source = source_config(source_root)
    candidates = dict(mode=mode, max_actions=maximum, use_subtask=use_subtask)
    common = dict(
        root=str(source_root),
        source="fixture",
        tokenize=byte_tokens,
        cameras=source["cameras"],
        viewpoint="ego_view",
        candidates=CandidateConfig(**candidates),
        video=False,
    )
    if stage == "action":
        actual = ActionDataset(
            **common,
            columns=ActionColumns(**source["columns"]),
            geometry=Geometry(),
            processor=ActionProcessor(source_root / "meta/stats.json"),
        )
    else:
        actual = VideoDataset(**common)
    expected = [actual.make_sample(i)["token_length"] for i in range(len(actual))]
    import veomni.data.causal_cosmos3.datasets.dataset as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline preparation must not read a training sample")

    monkeypatch.setattr(module, "read_window", forbidden)
    output = tmp_path / "lengths.npy"
    summary = build_root_lengths(
        dict(
            source=source,
            root=str(source_root),
            stage=stage,
            candidates=candidates,
            geometry={},
            sample={},
            output=str(output),
        ),
        tokenize=byte_tokens,
    )
    np.testing.assert_array_equal(np.load(output), expected)
    assert summary["windows"] == len(expected)


@pytest.mark.parametrize("size,ranks", [(17, 3), (2, 7), (16, 4)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_shared_order_matches_sampler_padding(size, ranks, shuffle):
    for seed in (7, 91):
        order = global_order(size, seed, epoch=2, shuffle=shuffle)
        lengths = np.arange(1, size + 1)
        for rank in range(ranks):
            sampler = StatefulDistributedSampler(
                range(size), num_replicas=ranks, rank=rank, seed=seed, shuffle=shuffle
            )
            sampler.set_epoch(2)
            shard = ReplayShard(lengths, order, rank, ranks)
            assert [item["token_length"] - 1 for item in shard] == list(sampler)
            assert not any(item["first_pass"] for item in shard)


@pytest.mark.parametrize("accumulation,buffer_size", [(1, 1), (2, 8), (3, 2)])
def test_completion_matches_native_loader_including_replay(monkeypatch, accumulation, buffer_size):
    import veomni.data.data_loader as module

    lengths = np.array([3, 7, 11, 2, 4, 8, 2])
    ranks, seed, budget = 3, 19, 9
    order = global_order(len(lengths), seed)
    math_ceil_samples = (len(lengths) + ranks - 1) // ranks
    for rank in range(ranks):
        monkeypatch.setattr(
            module, "get_parallel_state", lambda rank=rank: SimpleNamespace(dp_size=ranks, dp_rank=rank, sp_size=1)
        )

        class Candidates(Dataset):
            reads = 0

            def __len__(self):
                return len(lengths)

            def __getitem__(self, index):
                first_pass = self.reads < math_ceil_samples
                self.reads += 1
                return dict(token_length=int(lengths[index]), first_pass=first_pass)

        dataset = Candidates()
        loader = build_native_dataloader(
            dataset,
            micro_batch_size=1,
            global_batch_size=ranks * accumulation,
            dataloader_batch_size=1,
            max_seq_len=budget,
            train_steps=100,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            bsz_warmup_ratio=0,
            batching_strategy=ActionBatchingStrategy(budget, buffer_size),
            collate_fn=NoopDataCollator(),
            num_workers=0,
            prefetch_factor=None,
            pin_memory=False,
            seed=seed,
        )
        consumed, trained = 0, 0
        completion_step = None
        for step, packs in enumerate(loader, 1):
            consumed += sum(item["first_pass"] for pack in packs for item in pack)
            trained += sum(map(len, packs))
            if consumed == math_ceil_samples:
                completion_step = step
                break
        simulated = simulate_rank(lengths, order, rank, ranks, accumulation, budget, buffer_size)
        assert simulated["optimizer_iters"] == completion_step
        assert simulated["trained_samples_at_completion"] == trained
        assert simulated["read_samples_at_completion"] == dataset.reads


def test_slowest_rank_requires_full_traversal_not_dataset_sized_sample_count():
    lengths = np.array([3, 7] * 4)
    results = [simulate_rank(lengths, np.arange(8), rank, 2, 1, 7, 1) for rank in (0, 1)]
    assert [result["optimizer_iters"] for result in results] == [2, 4]
    assert results[0]["read_samples_at_completion"] > 4
    assert max(result["optimizer_iters"] for result in results) == 4


def test_simulation_releases_mmaps_without_cyclic_gc(tmp_path):
    np.save(tmp_path / "lengths.npy", np.array([3, 7] * 4))
    np.save(tmp_path / "order.npy", np.arange(8))
    lengths = np.load(tmp_path / "lengths.npy", mmap_mode="r")
    order = np.load(tmp_path / "order.npy", mmap_mode="r")
    references = [weakref.ref(lengths), weakref.ref(order)]
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        simulate_rank(lengths, order, 0, 2, 1, 7, 1)
        del lengths, order
        assert all(reference() is None for reference in references)
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_seeded_parallel_estimate_and_json_split(source_root, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        chat_template="{% for m in messages %}{{m['content']}}{% endfor %}",
    )
    checkpoint = tmp_path / "checkpoint"
    tokenizer.save_pretrained(checkpoint / "text_tokenizer")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([dict(lerobot_path=str(source_root), episodes=[3, 8, 12])]))
    source = source_config(source_root)
    source["roots"] = [str(selection)]
    source_yaml = tmp_path / "sources.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            dict(
                sampling="sample_uniform",
                sources=[source],
                validation=dict(split_ratio=0.2, seed=73),
            )
        )
    )
    recipe = dict(
        model=dict(model_path=str(checkpoint)),
        data=dict(
            train_path=str(source_yaml),
            stage="action",
            candidates=dict(max_actions=32),
            max_seq_len=9000,
            dyn_bsz_buffer_size=8,
        ),
        train=dict(grad_accum_iter=2, global_batch_size=999, seed=42, max_steps=1),
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    before = path.read_bytes(), source_yaml.read_bytes(), selection.read_bytes()
    config = load_recipe(path, ranks=3)
    assert config["accumulation"] == 2

    # The checkpoint contains only tokenizer files. No weights or physical ranks exist.
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline estimation must not build models or initialize distributed training")

    monkeypatch.setattr("veomni.models.build_foundation_model", forbidden)
    monkeypatch.setattr("torch.distributed.init_process_group", forbidden)
    selected, _ = prepare_sources(config)
    assert len(selected["sources"][0]["episodes"][str(source_root)]) == 2
    serial = estimate(config, seeds=[7, 91, 7], workers=1)
    parallel = estimate(config, seeds=[7, 91, 7], workers=2)
    assert serial["runs"] == parallel["runs"]
    assert serial["runs"][0]["per_rank"] == serial["runs"][2]["per_rank"]
    assert serial["interval"] == parallel["interval"]
    assert serial["max_inflight_seeds"] == parallel["max_inflight_seeds"] == 1
    for run in serial["runs"]:
        assert run["optimizer_iters"] == max(rank["optimizer_iters"] for rank in run["per_rank"])
    assert before == (path.read_bytes(), source_yaml.read_bytes(), selection.read_bytes())
    direct_config = dict(
        config,
        dataset={**config["dataset"], "sources": [{**source, "roots": [str(source_root)], "episodes": [3, 8, 12]}]},
    )
    assert prepare_sources(direct_config) == prepare_sources(config)
    recipe["data"]["dataloader"] = dict(in_order=False)
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="in_order"):
        load_recipe(path, ranks=3)
    recipe["model"]["model_path"] = "/placeholder/model"
    recipe["data"]["train_path"] = "/placeholder/sources.yaml"
    recipe["data"].pop("dataloader")
    path.write_text(yaml.safe_dump(recipe))
    overridden = load_recipe(path, ranks=3, model_path=checkpoint, train_path=source_yaml)
    assert overridden["dataset"] == config["dataset"]
    one_rank = {**config, "ranks": 1}
    one_serial = estimate(one_rank, seeds=[7, 91, 7], workers=1)
    one_parallel = estimate(one_rank, seeds=[7, 91, 7], workers=2)
    assert one_serial["runs"] == one_parallel["runs"]
    assert one_parallel["max_inflight_seeds"] == 2
