"""Native action semantics, greedy packing and fixed-topology state restore."""

import copy
import json
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from veomni.data.causal_cosmos3 import (
    ActionDataset,
    ActionProcessor,
    ActionTemplate55,
    CandidateConfig,
    build_action_dataloader,
    collate_action_pack,
)
from veomni.data.causal_cosmos3.dataset import segment_windows
from veomni.data.causal_cosmos3.pipeline import ActionBatchingStrategy
from veomni.data.causal_cosmos3.tokenization import add_special_tokens, build_prompt_tokenizer, tokenize_prompt
from veomni.trainer.base import BaseTrainer
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer
from veomni.trainer.dit_trainer import (
    DiTDataArguments,
    DiTModelArguments,
    DiTTrainer,
    DiTTrainingArguments,
    VeOmniDiTArguments,
)


def tokenize(text):
    return [1] + list(text.encode()) + [2]


def statistics():
    mask = [True] * 51 + [False] * 4
    return {name: dict(low=[-100.0] * 55, high=[100.0] * 55, valid=mask) for name in ("state", "action")}


@pytest.fixture
def tokenizer_root(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    root = tmp_path / "checkpoint"
    for component in ("transformer", "vae", "text_tokenizer"):
        (root / component).mkdir(parents=True)
    vocab = {token: i for i, token in enumerate(("<unk>", "<|im_end|>", "user", "assistant", "hello", "机器人"))}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        chat_template=(
            "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' + eos_token + ' ' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )
    tokenizer.save_pretrained(root / "text_tokenizer")
    return root


@pytest.mark.parametrize("prompt,chat_ids", [("", [2, 1, 3]), ("hello", [2, 4, 1, 3]), ("机器人", [2, 5, 1, 3])])
def test_prompt_tokenizer_chat_and_boundaries(tokenizer_root, prompt, chat_ids):
    tokenizer = build_prompt_tokenizer(tokenizer_root / "text_tokenizer")
    assert tokenizer.padding_side == "right"
    assert tokenize_prompt(prompt, tokenizer=tokenizer) == chat_ids + [1, 6]


def test_prompt_tokenizer_truncates_chat_before_boundaries(tokenizer_root):
    tokenizer = build_prompt_tokenizer(tokenizer_root / "text_tokenizer")
    ids = tokenize_prompt("hello " * 5000, tokenizer=tokenizer)
    assert ids == [2] + [4] * 4095 + [1, 6]
    assert len(ids) == 4098


def test_prompt_tokenizer_preserves_special_tokens_and_existing_ids(tokenizer_root):
    tokenizer = build_prompt_tokenizer(tokenizer_root / "text_tokenizer")
    vocab, special = tokenizer.get_vocab(), dict(tokenizer.special_tokens_map)
    _, ids = add_special_tokens(tokenizer)
    assert ids == {"start_of_generation": 6, "end_of_generation": 7}
    assert tokenizer.get_vocab() == vocab
    assert tokenizer.special_tokens_map == special
    assert tokenizer.convert_tokens_to_ids("hello") == 4


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
    assert sample["state_latent_indexes"] == [0, 2]
    assert sample["action_frame_ids"] == sum(([i] * 16 for i in range(1, 5)), [])
    for key in ("state_latent_indexes", "action_frame_ids"):
        assert isinstance(sample[key], list) and all(type(index) is int for index in sample[key])
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
def test_dit_collation_preserves_packing_and_step_count(roots, monkeypatch, mode):
    patch_rank(monkeypatch, 0)
    cfg = config(roots, mode, 0)
    cfg.update(train_steps=2, token_budget=3200)
    original = list(build_action_dataloader(cfg, tokenize))
    adapted = list(build_action_dataloader(cfg, tokenize, collate_fn=collate_action_pack))

    assert len(original) == len(adapted) == cfg["train_steps"]
    assert any(len(pack) > 1 for step in original for pack in step)
    for expected, actual in zip(original, adapted, strict=True):
        assert len(actual) == cfg["grad_accum_iter"]
        assert all(set(microbatch) == {"samples"} for microbatch in actual)
        assert all(isinstance(microbatch["samples"], list) for microbatch in actual)
        assert signature([microbatch["samples"] for microbatch in actual]) == signature(expected)
        for pack, microbatch in zip(expected, actual, strict=True):
            for original_sample, sample in zip(pack, microbatch["samples"], strict=True):
                for key in ("action_frame_ids", "state_latent_indexes"):
                    assert isinstance(sample[key], list) and all(type(index) is int for index in sample[key])
                    assert sample[key] == original_sample[key]


def action_args(tmp_path, roots, monkeypatch, mode, rank=0):
    import veomni.trainer.causal_cosmos3 as causal
    import veomni.trainer.dit_trainer as dit

    patch_rank(monkeypatch, rank)
    parallel = SimpleNamespace(dp_rank=rank, dp_size=2, sp_size=1, sp_enabled=False)
    monkeypatch.setattr(dit, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(causal, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(dit, "use_parallel_state", lambda _: nullcontext())
    # Exercise the real DiT constructor and data methods; only model and
    # distributed setup are replaced for this data-only CPU integration test.
    monkeypatch.setattr(BaseTrainer, "_setup", lambda self, args: torch.device("cpu"))
    monkeypatch.setattr(DiTTrainer, "_build_model_runtime", lambda self: SimpleNamespace(parallel_state=parallel))
    monkeypatch.setattr(BaseTrainer, "_build_lr_scheduler", lambda self: None)
    monkeypatch.setattr(BaseTrainer, "_build_training_context", lambda self, runtime: None)
    monkeypatch.setattr(BaseTrainer, "_init_callbacks", lambda self: None)
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Only source construction comes from this YAML. Trainer arguments own
    # packing, RNG seeds, accumulation and optimizer step counts.
    cfg = config(roots, mode, 0)
    cfg = {key: cfg[key] for key in ("sampling", "candidates", "sample", "sources")}
    path = tmp_path / "action.yaml"
    path.write_text(yaml.safe_dump(cfg))
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused"),
        data=DiTDataArguments(
            train_path=str(path),
            max_seq_len=3200,
            dyn_bsz_buffer_size=2,
        ),
        train=DiTTrainingArguments(micro_batch_size=1, global_batch_size=4, max_steps=3, seed=19),
    )
    args.data.dataloader.num_workers = 0
    args.data.dataloader.prefetch_factor = None
    args.data.dataloader.pin_memory = False
    return args


def test_checkpoint_assets_and_real_tokenizer_reach_dataset(tmp_path, roots, monkeypatch, tokenizer_root):
    from tasks.train_causal_cosmos3 import configure_args

    args = action_args(tmp_path, roots, monkeypatch, "sample_uniform")
    args.model.model_path = str(tokenizer_root)
    args.model.condition_model_path = "condition-config"
    args.model.condition_model_cfg = {"shift": 2}
    args.train.checkpoint.load_path = "full-dcp-resume"
    original_config = args.model.config_path
    configure_args(args)
    assert args.model.model_path == str(tokenizer_root / "transformer")
    assert args.model.tokenizer_path == str(tokenizer_root / "text_tokenizer")
    assert args.model.condition_model_cfg == {"shift": 2, "vae_path": str(tokenizer_root / "vae")}
    assert args.model.config_path == original_config
    assert args.model.condition_model_path == "condition-config"
    assert args.train.checkpoint.load_path == "full-dcp-resume"

    tokenizer = build_prompt_tokenizer(args.model.tokenizer_path)
    encode = partial(tokenize_prompt, tokenizer=tokenizer)
    trainer = CausalCosmos3Trainer(args, tokenize=encode)
    microbatches = next(iter(trainer.base.train_dataloader))
    assert len(microbatches) == 2
    for batch in microbatches:
        for sample in batch["samples"]:
            assert sample["text_token_ids"].tolist() == encode(sample["prompt"])
            assert sample["token_layout"]["text"] == len(sample["text_token_ids"])
            assert sample["token_length"] == sum(sample["token_layout"].values())


def test_checkpoint_assets_require_a_root():
    from tasks.train_causal_cosmos3 import configure_args

    args = SimpleNamespace(model=SimpleNamespace(model_path=None))
    with pytest.raises(ValueError, match="checkpoint root"):
        configure_args(args)


@pytest.mark.parametrize("component", ["text_tokenizer", "vae"])
def test_checkpoint_assets_report_missing_components(tmp_path, tokenizer_root, component):
    from tasks.train_causal_cosmos3 import configure_args

    # 保留组件内容，模拟该组件不在约定的根目录下。
    (tokenizer_root / component).rename(tmp_path / "moved-component")
    args = VeOmniDiTArguments(
        model=DiTModelArguments(model_path=str(tokenizer_root), config_path="native-config"),
        data=DiTDataArguments(train_path="unused"),
    )
    with pytest.raises(FileNotFoundError, match=component):
        configure_args(args)


def test_checkpoint_assets_require_separate_native_config(tokenizer_root):
    from tasks.train_causal_cosmos3 import configure_args

    args = VeOmniDiTArguments(
        model=DiTModelArguments(model_path=str(tokenizer_root)),
        data=DiTDataArguments(train_path="unused"),
    )
    with pytest.raises(ValueError, match="native Cosmos3 architecture config"):
        configure_args(args)


@pytest.mark.parametrize("mode", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("rank", [0, 1])
def test_dit_builds_action_pipeline_from_config(tmp_path, roots, monkeypatch, mode, rank):
    args = action_args(tmp_path, roots, monkeypatch, mode, rank)
    trainer = CausalCosmos3Trainer(args, tokenize=tokenize)
    loader = trainer.base.train_dataloader
    assert type(trainer) is CausalCosmos3Trainer and isinstance(trainer, DiTTrainer)
    assert args.train.dyn_bsz is True and args.train.dataloader_batch_size == 1
    assert loader._dataloader.batch_size == 1
    assert args.train.gradient_accumulation_steps == 2
    assert args.train.global_batch_size == 4
    assert trainer.base.data_transform is tokenize
    assert trainer.base.train_steps == args.train_steps == len(loader) == 3
    if mode == "sample_uniform":
        assert args.train_steps != len(trainer.base.train_dataset) // 2 // 2

    expected_config = config(roots, mode, 0)
    expected_config.update(train_steps=3, token_budget=3200)
    expected = list(build_action_dataloader(expected_config, tokenize))
    actual = list(loader)
    assert len(actual) == 3
    assert all(len(step) == 2 for step in actual)
    assert any(len(batch["samples"]) > 1 for step in actual for batch in step)
    assert [signature([batch["samples"] for batch in step]) for step in actual] == [
        signature(step) for step in expected
    ]


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("train", "micro_batch_size", 2, "one microbatch is one pack"),
        ("train", "max_steps", None, "positive train.max_steps"),
        ("train", "max_steps", 0, "positive train.max_steps"),
        ("train", "training_task", "offline_training", "online_training"),
        ("train", "training_task", "offline_embedding", "online_training"),
        ("train", "dyn_bsz_runtime", "worker", "dyn_bsz_runtime='main'"),
        ("data", "max_seq_len", 0, "token budget"),
        ("data", "dyn_bsz_buffer_size", 0, "buffer size"),
    ],
)
def test_dit_rejects_unsupported_action_batch_config(tmp_path, roots, monkeypatch, section, key, value, message):
    args = action_args(tmp_path, roots, monkeypatch, "sample_uniform")
    setattr(getattr(args, section), key, value)
    with pytest.raises(ValueError, match=message):
        CausalCosmos3Trainer(args, tokenize=tokenize)


def test_dit_action_pipeline_requires_explicit_tokenizer(tmp_path, roots, monkeypatch):
    args = action_args(tmp_path, roots, monkeypatch, "sample_uniform")
    with pytest.raises(ValueError, match="tokenize"):
        CausalCosmos3Trainer(args, tokenize=None)


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
            conditioning_fps=7.5,
        )
    )
    structured = json.loads(prompt)
    assert structured["resolution"] == {"H": height, "W": width}
    assert "duration" not in structured
    assert structured["actions"] == [{"description": "Move."}]
    assert prompt.count(VIEW_DESCRIPTION[viewpoint]) == 1


def test_default_json_prompt_reaches_tokenizer_and_packing(roots, monkeypatch):
    patch_rank(monkeypatch, 0)
    seen = []

    def record_tokenize(prompt):
        seen.append(prompt)
        return tokenize(prompt)

    loader = build_action_dataloader(config(roots, "sample_uniform", 0), record_tokenize)
    for pack in next(iter(loader)):
        for sample in pack:
            prompt = sample["prompt"]
            structured = json.loads(prompt)
            assert prompt in seen
            assert "duration" not in structured and "time" not in structured["actions"][0]
            assert structured["actions"][0]["description"] == sample["caption"] + "."
            assert structured["fps"] == sample["source_fps"] / 4
            assert sample["text_token_ids"].tolist() == tokenize(prompt)
            assert sample["token_layout"]["text"] == len(tokenize(prompt))
            assert sample["token_length"] == sum(sample["token_layout"].values())


def test_json_prompt_padding_idle_metadata_and_no_mutation():
    from veomni.data.causal_cosmos3.layout import format_prompt

    sample = dict(
        caption="  Pick up the cup!  ",
        viewpoint="ego_view",
        image_size=(256, 320),
        conditioning_fps=7.5,
        video=torch.empty(3, 33, 0, 0),
        real_action_steps=100,
        mode="policy",
        idle_frames=0,
        idle_frames_total=100,
    )
    before = sample.copy()
    prompt = json.loads(format_prompt(sample))
    assert sample == before
    assert prompt["actions"] == [{"description": "Pick up the cup!", "idle_frame": "0 out of 100."}]
    assert "duration" not in prompt
    assert prompt["aspect_ratio"] == "5,4"
    sample.update(video=torch.empty(3, 137, 0, 0), real_action_steps=540)
    assert json.loads(format_prompt(sample)) == prompt  # 窗口长度和 padding 不改变文本条件。
    sample.pop("video")
    assert json.loads(format_prompt(sample)) == prompt  # 推理构造 prompt 无需未来视频。
    sample.update(mode="inverse_dynamics")
    assert "idle_frame" not in json.loads(format_prompt(sample))["actions"][0]
    sample.update(caption="")
    assert format_prompt(sample) == ""


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
