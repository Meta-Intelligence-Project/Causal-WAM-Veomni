"""Native action semantics, greedy packing and fixed-topology state restore."""

import copy
import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.train_causal_cosmos3 import configure_gradient_accumulation
from veomni.data.causal_cosmos3 import (
    ActionDataset,
    ActionProcessor,
    ActionTemplate55,
    CandidateConfig,
    build_action_dataloader,
    collate_action_pack,
)
from veomni.data.causal_cosmos3.datasets.dataset import segment_windows
from veomni.data.causal_cosmos3.datasets.pipeline import ActionBatchingStrategy, build_action_dataset
from veomni.data.causal_cosmos3.datasets.split import split_action_config, split_action_config_by_source
from veomni.data.causal_cosmos3.datasets.validation import (
    action_subsets,
    fingerprint,
    iter_validation_batches,
)
from veomni.data.causal_cosmos3.transforms.tokenization import (
    add_special_tokens,
    build_prompt_tokenizer,
    tokenize_prompt,
)
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


@pytest.mark.parametrize("mode", ["frame_window", "segmented"])
@pytest.mark.parametrize("use_subtask", [False, True])
@pytest.mark.parametrize("roots", [(201, 41)], indirect=True)
def test_full_episode_without_splitting(roots, mode, use_subtask):
    (roots[1] / "meta/episodes.jsonl").unlink()  # Missing subtasks fall back to the episode.
    for i, root in enumerate(roots):
        ds = dataset(
            root,
            candidates=CandidateConfig(
                mode=mode, max_actions=-1, overlap=32, use_subtask=use_subtask, borrow_subtask_frames=True
            ),
        )
        expected = [(0, 200, 224, "fallback")] if i == 0 else [(0, 40, 64, "fallback")]
        if use_subtask and i == 0:
            expected = [(0, 20, 32, "first"), (20, 180, 192, "second")]
        assert len(ds) == len(expected)
        for index, (start, real_actions, padded_actions, caption) in enumerate(expected):
            sample = ds.make_sample(index)
            assert sample["window"].start == start
            assert sample["window"].actions == padded_actions
            assert sample["real_action_steps"] == real_actions
            assert sample["action_valid_steps"].sum() == real_actions
            assert sample["caption"] == caption


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


def test_getitem_skips_errors_but_make_sample_propagates(roots, monkeypatch, caplog):
    import veomni.data.causal_cosmos3.datasets.dataset as module

    ds = dataset(roots[0])
    original = module.read_window

    def fail(*args, **kwargs):
        raise OSError("broken video")

    monkeypatch.setattr(module.logger, "handlers", [*module.logger.handlers, caplog.handler])
    monkeypatch.setattr(module, "read_window", fail)
    for _ in range(4):
        assert ds[0] == []
    assert caplog.text.count("OSError: broken video") == 4
    with pytest.raises(OSError, match="broken video"):
        ds.make_sample(0)
    assert set(ds.state_dict()) == {"rng"}
    assert not hasattr(ds, "successes") and not hasattr(ds, "failures")
    # Old checkpoint counters are ignored on restore.
    ds.load_state_dict(dict(ds.state_dict(), successes=7, failures=42))
    monkeypatch.setattr(module, "read_window", original)
    assert ds[0]["source"] == "fixture"
    ds.processor.statistics["action"][2][0] = False
    assert ds[0] == []
    with pytest.raises(ValueError, match="cover"):
        ds.make_sample(0)


def test_getitem_skips_tokenizer_errors_but_make_sample_propagates(roots):
    ds = dataset(roots[0])
    ds.tokenize = lambda prompt: [[1, 2]]
    assert ds[0] == []
    with pytest.raises(ValueError, match="tokenize must return a 1D sequence"):
        ds.make_sample(0)


@pytest.mark.parametrize("error_type", [MemoryError, torch.OutOfMemoryError])
@pytest.mark.parametrize("stage", ["read", "prompt"])
def test_sample_resource_errors_propagate(roots, monkeypatch, error_type, stage):
    import veomni.data.causal_cosmos3.datasets.dataset as module

    ds = dataset(roots[0])

    def fail(*args, **kwargs):
        raise error_type("resource exhaustion")

    if stage == "read":
        monkeypatch.setattr(module, "read_window", fail)
    else:
        ds.prompt_formatter = fail
    with pytest.raises(error_type, match="resource exhaustion"):
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
    import veomni.data.causal_cosmos3.datasets.pipeline as pipeline
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


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("cli_steps", [None, 4])
def test_action_accumulation_reads_training_yaml(tmp_path, monkeypatch, dp_size, cli_steps):
    from tasks.train_causal_cosmos3 import parse_causal_cosmos3_args

    monkeypatch.setenv("WORLD_SIZE", str(dp_size))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(dp_size))
    path = tmp_path / "training.yaml"
    path.write_text("model:\n  config_path: unused\ndata:\n  train_path: unused\ntrain:\n  grad_accum_iter: 2\n")
    argv = [str(path)]
    if cli_steps is not None:
        argv.extend(["--train.grad_accum_iter", str(cli_steps)])
    args = parse_causal_cosmos3_args(argv)
    steps = cli_steps or 2
    assert type(args) is VeOmniDiTArguments
    assert type(args.train) is DiTTrainingArguments
    assert args.train.gradient_accumulation_steps == steps
    assert args.train.global_batch_size == dp_size * steps
    assert args.train.dataloader_batch_size == 1


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "2"])
def test_action_accumulation_rejects_invalid_counts(tmp_path, monkeypatch, steps):
    from tasks.train_causal_cosmos3 import parse_causal_cosmos3_args

    monkeypatch.setenv("WORLD_SIZE", "2")
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {"model": {"config_path": "unused"}, "data": {"train_path": "unused"}, "train": {"grad_accum_iter": steps}}
        )
    )
    with pytest.raises(ValueError, match="grad_accum_iter must be a positive integer"):
        parse_causal_cosmos3_args([str(path)])


@pytest.mark.parametrize(
    "global_batch_size,steps,expected_steps",
    [(None, 2, 2), (4, 2, 2), (6, None, 3), (None, None, 1)],
)
def test_action_accumulation_preserves_compatible_batch_configs(monkeypatch, global_batch_size, steps, expected_steps):
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = VeOmniDiTArguments(
        model=DiTModelArguments(config_path="unused"),
        data=DiTDataArguments(train_path="unused"),
        train=DiTTrainingArguments(global_batch_size=global_batch_size),
    )
    configure_gradient_accumulation(args, steps)
    assert args.train.gradient_accumulation_steps == expected_steps
    assert args.train.global_batch_size == 2 * expected_steps


def test_action_accumulation_overrides_global_batch_size(tmp_path, monkeypatch):
    from tasks.train_causal_cosmos3 import parse_causal_cosmos3_args

    monkeypatch.setenv("WORLD_SIZE", "2")
    path = tmp_path / "training.yaml"
    path.write_text(
        "model:\n  config_path: unused\ndata:\n  train_path: unused\n"
        "train:\n  grad_accum_iter: 2\n  global_batch_size: 3\n"
    )
    args = parse_causal_cosmos3_args([str(path)])
    assert args.train.global_batch_size == 4
    assert args.train.gradient_accumulation_steps == 2


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
    # The source YAML owns only dataset options. Training arguments own packing,
    # accumulation, RNG seeds and optimizer step counts.
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
        train=DiTTrainingArguments(micro_batch_size=1, max_steps=3, seed=19),
    )
    configure_gradient_accumulation(args, 2)
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
    assert trainer.base.validation_config is None
    assert trainer.base.validation_dataset is None
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
    import veomni.data.causal_cosmos3.datasets.dataset as module

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

    from veomni.data.causal_cosmos3.datasets.dataset import UniformSource
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
    from veomni.data.causal_cosmos3.transforms.layout import VIEW_DESCRIPTION, camera_layout, format_prompt

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
    from veomni.data.causal_cosmos3.transforms.layout import format_prompt

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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("caption", ["", " \t\n", None])
def test_json_prompt_rejects_missing_caption(caption):
    from veomni.data.causal_cosmos3.transforms.layout import format_prompt

    with pytest.raises(ValueError, match="nonempty.*caption"):
        format_prompt(dict(caption=caption))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("task_text", ["missing", "", " \t\n", None])
def test_missing_caption_skips_training_and_validation_but_keeps_next_sample(roots, task_text, monkeypatch, caplog):
    import veomni.data.causal_cosmos3.datasets.dataset as module

    # VeOmni's stdout logger does not propagate to pytest's root log handler.
    monkeypatch.setattr(module.logger, "handlers", [*module.logger.handlers, caplog.handler])
    root = roots[0]
    annotations = root / "meta/episodes.jsonl"
    record = json.loads(annotations.read_text())
    # The first window has no subtask caption; later windows still have "second".
    record["action_config"] = record["action_config"][1:]
    annotations.write_text(json.dumps(record) + "\n")
    tasks = root / "meta/tasks.parquet"
    if task_text == "missing":
        tasks.unlink()
    else:
        pq.write_table(pa.table(dict(task_index=[0], task=[task_text])), tasks)
    ds = dataset(root, mode="policy", candidates=CandidateConfig(max_actions=32))
    seen = []

    def record_tokenize(prompt):
        seen.append(prompt)
        return tokenize(prompt)

    ds.tokenize = record_tokenize
    assert ds[0] == []
    assert not seen
    assert "nonempty" in caplog.text and "candidate_index=0" in caplog.text
    following = ds[20]
    assert following["caption"] == "second" and following["prompt"].strip()
    assert len(seen) == 1
    validation = next(iter_validation_batches(ds))
    assert validation["weight"] == 0 and validation["batch"] == {"samples": []}
    assert len(seen) == 1


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_missing_subtask_caption_uses_nonempty_task_text(roots):
    ds = dataset(roots[0], candidates=CandidateConfig(use_subtask=False))
    sample = ds[0]
    assert sample["caption"] == "fallback"
    assert json.loads(sample["prompt"])["actions"] == [{"description": "fallback."}]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("prompt", ["", " \t\n"])
def test_empty_custom_prompt_skips_before_tokenization(roots, prompt):
    ds = dataset(roots[0], prompt_formatter=lambda sample: prompt)

    def reject_tokenization(_):
        pytest.fail("An empty prompt must not reach tokenization")

    ds.tokenize = reject_tokenization
    assert ds[0] == []


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
    import veomni.data.causal_cosmos3.readers.lerobot as reader

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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_source_episode_mapping_and_unknown_episode_rejection(roots, monkeypatch):
    from veomni.data.causal_cosmos3.datasets.pipeline import build_action_dataset

    patch_rank(monkeypatch, 0)
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["sources"][0]["roots"] = [str(root) for root in roots]
    cfg["sources"][0]["episodes"] = {str(root): [17] for root in roots}
    built = build_action_dataset(cfg, tokenize)
    assert len(built.datasets) == 2
    assert all(subset.selected_episodes == {17} for subset in built.datasets)
    cfg["sources"][0]["episodes"].pop(str(roots[1]))
    with pytest.raises(ValueError, match="cover exactly"):
        build_action_dataset(cfg, tokenize)
    cfg["sources"][0]["episodes"] = [999]
    with pytest.raises(ValueError, match="Unknown episodes"):
        split_action_config({**cfg, "validation": {"split_ratio": 0.1}})


def add_split_episodes(root, count):
    """Build independent metadata IDs sharing the small synthetic row fixture."""
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    row = pq.read_table(path).to_pylist()[0]
    length = row["dataset_to_index"] - row["dataset_from_index"]
    records = []
    for i in range(count):
        record = {
            **row,
            "episode_index": 100 + 3 * i,
            "dataset_from_index": 100 + length * i,
            "dataset_to_index": 100 + length * (i + 1),
            "data/file_from_offset": 0,
        }
        records.append(record)
    pq.write_table(pa.Table.from_pylist(records), path)
    return [record["episode_index"] for record in records]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("count,ratio,expected", [(1, 0.1, 0), (2, 0.01, 1), (3, 0.99, 2), (100, 0.07, 7)])
def test_auto_episode_split_small_sources_and_rounding(roots, count, ratio, expected):
    add_split_episodes(roots[0], count)
    add_split_episodes(roots[1], 2)
    cfg = config(roots, "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=ratio, seed=73)
    train, evaluation, split = split_action_config(cfg)
    record = next(record for record in split["roots"] if record["root"] == str(roots[0]))
    assert len(record["validation_episodes"]) == expected
    assert len(record["train_episodes"]) == count - expected
    if count == 1:
        assert len(train["sources"]) == 2 and len(evaluation["sources"]) == 1


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan"), float("inf"), True, "0.1"])
def test_auto_episode_split_rejects_invalid_ratio(roots, value):
    cfg = config(roots, "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=value)
    with pytest.raises(ValueError, match="split_ratio"):
        split_action_config(cfg)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_auto_episode_split_selection_metadata_refresh_and_duplicate_roots(roots):
    ids = add_split_episodes(roots[0], 10)
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=0.25, seed=73)
    cfg["sources"][0]["read"] = {"episodes": ids[::2], "decode_threads": 1}
    train, evaluation, split = split_action_config(cfg)
    assert set(split["roots"][0]["eligible_episodes"]) == set(ids[::2])
    assert "episodes" not in train["sources"][0]["read"]
    assert train["sources"][0]["read"]["decode_threads"] == 1
    assert evaluation["sources"][0]["read"]["decode_threads"] == 1
    cfg["sources"].append(copy.deepcopy(cfg["sources"][0]))
    cfg["sources"][-1]["name"] = "duplicate-root"
    with pytest.raises(ValueError, match="unique roots across sources"):
        split_action_config(cfg)
    cfg["sources"].pop()
    info = roots[0] / "meta/info.json"
    content = json.loads(info.read_text())
    content["fps"] = 31
    info.write_text(json.dumps(content))
    assert split_action_config(cfg)[2] == split


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_auto_episode_split_recomputes_current_inputs_without_persistence(roots, tmp_path):
    add_split_episodes(roots[0], 20)
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=0.25, seed=73)
    split = split_action_config(cfg)[2]
    assert split["validation_count"] == 5
    assert split_action_config(cfg)[2] == split
    cfg["validation"]["seed"] += 1
    changed = split_action_config(cfg)[2]
    assert changed["seed"] == 74 and changed["split_id"] != split["split_id"]
    cfg["validation"]["split_ratio"] = 0.5
    assert split_action_config(cfg)[2]["validation_count"] == 10
    add_split_episodes(roots[0], 25)
    refreshed = split_action_config(cfg)[2]
    assert refreshed["train_count"] == 12 and refreshed["validation_count"] == 13
    assert not list(tmp_path.rglob("data_split.json"))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "ratio,cap,expected",
    [(0.25, None, 5), (0.25, 2, 2), (0.25, 5, 5), (0.25, 10, 5), (0.01, 1, 1), (0.99, 100, 19)],
)
def test_source_split_episode_cap_uses_smaller_quota_and_retains_training(roots, ratio, cap, expected):
    complete = {(str(root), ep) for root in roots for ep in add_split_episodes(root, 10)}
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["sources"][0]["roots"] = [str(root) for root in roots]
    cfg["validation"] = dict(split_ratio=ratio, seed=73, max_episodes_per_source=cap)
    original = copy.deepcopy(cfg)
    train, evaluation, manifest = split_action_config(cfg)
    assert cfg == original
    assert split_action_config(cfg) == (train, evaluation, manifest)
    assert manifest["validation_count"] == expected and manifest["train_count"] == 20 - expected
    assert manifest["max_episodes_per_source"] == cap
    assert evaluation["validation"] == {"seed": 73}
    selected = {
        (root, ep) for source in evaluation["sources"] for root, ids in source["episodes"].items() for ep in ids
    }
    training = {(root, ep) for source in train["sources"] for root, ids in source["episodes"].items() for ep in ids}
    assert selected.isdisjoint(training) and selected | training == complete
    unlimited = copy.deepcopy(cfg)
    unlimited["validation"].pop("max_episodes_per_source")
    unlimited_eval = split_action_config(unlimited)[1]
    unlimited_selected = {
        (root, ep) for source in unlimited_eval["sources"] for root, ids in source["episodes"].items() for ep in ids
    }
    assert selected <= unlimited_selected


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("cap", [0, -1, True, False, 1.5, "10", float("nan"), float("inf"), [], {}])
def test_source_split_rejects_invalid_episode_cap(roots, cap):
    cfg = config(roots, "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=0.05, max_episodes_per_source=cap)
    with pytest.raises(ValueError, match="max_episodes_per_source must be a positive integer or null"):
        split_action_config(cfg)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_source_split_applies_cap_after_root_episode_prefilters(roots):
    selected = {str(root): add_split_episodes(root, 20)[::2] for root in roots}
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["sources"][0].update(roots=[str(root) for root in roots], episodes=selected)
    cfg["validation"] = dict(split_ratio=0.5, seed=73, max_episodes_per_source=3)
    train, evaluation, manifest = split_action_config(cfg)
    assert manifest["sources"][0]["eligible_count"] == 20
    assert manifest["validation_count"] == 3 and manifest["train_count"] == 17
    for record in manifest["roots"]:
        assert record["eligible_episodes"] == selected[record["root"]]
    for partition in (train, evaluation):
        for root, ids in partition["sources"][0]["episodes"].items():
            assert set(ids) <= set(selected[root])


def _split_subsets(dataset):
    if isinstance(dataset, ActionDataset):
        return [dataset]
    if hasattr(dataset, "dataset"):
        return _split_subsets(dataset.dataset)
    children = getattr(dataset, "datasets", getattr(dataset, "_datasets", ()))
    return [subset for child in children for subset in _split_subsets(child)]


def _split_episode_keys(dataset):
    return {(subset.root, episode) for subset in _split_subsets(dataset) for episode in subset.selected_episodes}


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("sampling", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("extra_options", [{}, {"description": "validation run metadata"}])
def test_auto_episode_split_disjoint_complete_deterministic_and_datasets(roots, monkeypatch, sampling, extra_options):
    import pickle
    import random

    patch_rank(monkeypatch, 0)
    complete = {(str(root), ep) for root, count in zip(roots, (10, 3)) for ep in add_split_episodes(root, count)}
    cfg = config(roots, sampling, 0)
    cfg["validation"] = dict(split_ratio=0.2, seed=73, **extra_options)
    original = copy.deepcopy(cfg)
    before = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state())
    train, evaluation, manifest = split_action_config(cfg)
    assert cfg == original
    assert random.getstate() == before[0] and pickle.dumps(np.random.get_state()) == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert manifest["train_count"] == 10 and manifest["validation_count"] == 3
    for record in manifest["roots"]:
        assert set(record["train_episodes"]).isdisjoint(record["validation_episodes"])
    training, heldout = (build_action_dataset(partition, tokenize) for partition in (train, evaluation))
    train_keys, eval_keys = _split_episode_keys(training), _split_episode_keys(heldout)
    assert train_keys.isdisjoint(eval_keys) and train_keys | eval_keys == complete
    assert len(train_keys) == 10 and len(eval_keys) == 3
    assert sum(len(subset) for subset in _split_subsets(heldout)) == 95
    items = list(iter_validation_batches(heldout, 0, 2, seed=73))
    assert len(items) == 144
    assert all(
        (item["record"]["identity"]["root"], item["record"]["identity"]["episode"]) not in train_keys for item in items
    )
    reordered = copy.deepcopy(cfg)
    reordered["sources"].reverse()
    patch_rank(monkeypatch, 1)
    assert split_action_config(reordered)[2] == manifest
    cfg["validation"]["seed"] += 1
    assert split_action_config(cfg)[2]["split_id"] != manifest["split_id"]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("stale_files", [False, True])
@pytest.mark.parametrize("cap,expected_count", [(None, 2), (1, 1), (3, 2)])
def test_trainer_auto_episode_split_uses_one_complete_source_config(
    roots, tmp_path, monkeypatch, resume, stale_files, cap, expected_count
):
    args = action_args(tmp_path, roots, monkeypatch, "sample_uniform")
    add_split_episodes(roots[0], 5)
    add_split_episodes(roots[1], 3)
    cfg = yaml.safe_load(Path(args.data.train_path).read_text())
    cfg["sources"] = [{**cfg["sources"][0], "roots": [str(root) for root in roots]}]
    cfg["validation"] = dict(split_ratio=0.2, seed=73, max_episodes_per_source=cap)
    Path(args.data.train_path).write_text(yaml.safe_dump(cfg))
    output, checkpoint = tmp_path / "split-output", tmp_path / "resume-checkpoint"
    args.train.checkpoint.output_dir = str(output)
    args.train.checkpoint.load_path = str(checkpoint) if resume else None
    if stale_files:
        for directory in (output, checkpoint):
            directory.mkdir()
            (directory / "data_split.json").write_text("old split must never be read or overwritten")
    parallel = SimpleNamespace(dp_rank=0, dp_size=2, dp_group=None, sp_size=1)
    trainer = CausalCosmos3Trainer.__new__(CausalCosmos3Trainer)
    trainer.base = SimpleNamespace(args=args, data_transform=tokenize, model=SimpleNamespace(parallel_state=parallel))
    trainer._build_dataset()
    manifest = trainer.base.episode_split_manifest
    assert manifest["stratify"] == "source" and manifest["max_episodes_per_source"] == cap
    assert manifest["train_count"] == 8 - expected_count and manifest["validation_count"] == expected_count
    assert trainer.base.validation_config["validation"] == {"seed": 73}
    fixed_dataset = trainer.base.validation_dataset
    assert hasattr(fixed_dataset, "datasets")
    assert all(isinstance(ds, ActionDataset) for ds in fixed_dataset.datasets)
    train_keys = _split_episode_keys(trainer.base.train_dataset)
    eval_keys = _split_episode_keys(build_action_dataset(trainer.base.validation_config, tokenize))
    assert train_keys.isdisjoint(eval_keys) and len(train_keys | eval_keys) == 8
    assert len(eval_keys) == expected_count
    assert {(ds.root, ep) for ds in action_subsets(fixed_dataset) for ep in ds.selected_episodes} == eval_keys
    if stale_files:
        for directory in (output, checkpoint):
            assert (directory / "data_split.json").read_text() == "old split must never be read or overwritten"
    else:
        assert not output.exists() and not checkpoint.exists()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("roots", [(75, 41, 66, 70, 50, 60)], indirect=True)
@pytest.mark.parametrize("sampling", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("ratio,cap,expected_per_source", [(0.05, None, 1), (0.5, 3, 3)])
def test_episode_split_pools_each_source_and_caps_each_independently(
    roots, monkeypatch, sampling, ratio, cap, expected_per_source
):
    patch_rank(monkeypatch, 0)
    complete = {
        (str(root), ep)
        for root, count in zip(roots, (10, 10, 9, 11, 1, 19), strict=True)
        for ep in add_split_episodes(root, count)
    }
    cfg = config(roots, sampling, 0)
    sources = []
    for index, name in enumerate(("agibot", "robotwin", "egosuite")):
        source = cfg["sources"][index * 2]
        source.update(name=name, roots=[str(root) for root in roots[index * 2 : index * 2 + 2]])
        sources.append(source)
    cfg["sources"] = sources
    cfg["validation"] = dict(split_ratio=ratio, seed=73, max_episodes_per_source=cap)
    train, evaluation, manifest = split_action_config(cfg)
    expected_count = expected_per_source * 3
    assert manifest["train_count"] == 60 - expected_count and manifest["validation_count"] == expected_count
    assert all(source["eligible_count"] == 20 for source in manifest["sources"])
    assert all(source["validation_count"] == expected_per_source for source in manifest["sources"])
    assert split_action_config_by_source(cfg) == (train, evaluation, manifest)
    reordered = copy.deepcopy(cfg)
    reordered["sources"].reverse()
    for source in reordered["sources"]:
        source["roots"].reverse()
        source["weight"] = 0 if source["name"] == "egosuite" else 10
    reordered["seed"] = 123456
    patch_rank(monkeypatch, 1)
    assert split_action_config(reordered)[2] == manifest
    renamed = copy.deepcopy(cfg)
    for source in renamed["sources"]:
        source["name"] += "-renamed"
    assert {record["root"]: record["validation_episodes"] for record in split_action_config(renamed)[2]["roots"]} == {
        record["root"]: record["validation_episodes"] for record in manifest["roots"]
    }
    training, heldout = (build_action_dataset(partition, tokenize) for partition in (train, evaluation))
    train_keys, eval_keys = _split_episode_keys(training), _split_episode_keys(heldout)
    assert train_keys.isdisjoint(eval_keys) and train_keys | eval_keys == complete
    assert len(train_keys) == 60 - expected_count and len(eval_keys) == expected_count
    assert {(ds.root, ep) for ds in action_subsets(heldout) for ep in ds.selected_episodes} == eval_keys
    duplicate = copy.deepcopy(cfg)
    duplicate["sources"][1]["name"] = duplicate["sources"][0]["name"]
    with pytest.raises(ValueError, match="unique source names"):
        split_action_config(duplicate)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_source_split_pools_singleton_roots_and_preserves_root_statistics(roots, monkeypatch):
    patch_rank(monkeypatch, 0)
    cfg = config(roots[:1], "sample_uniform", 0)
    cfg["sources"][0]["roots"] = [str(root) for root in roots]
    cfg["sources"][0]["statistics"] = "{root}/meta/stats.json"
    cfg["validation"] = dict(split_ratio=0.05, seed=73)
    for index, root in enumerate(roots):
        values = statistics()
        values["action"]["high"][0] += index + 1
        (root / "meta/stats.json").write_text(json.dumps(values))
    train, evaluation, manifest = split_action_config(cfg)
    assert manifest["train_count"] == manifest["validation_count"] == 1
    assert set(train["sources"][0]["roots"]).isdisjoint(evaluation["sources"][0]["roots"])
    for partition in (train, evaluation):
        assert partition["sources"][0]["statistics"] == cfg["sources"][0]["statistics"]
        for subset in _split_subsets(build_action_dataset(partition, tokenize)):
            expected = json.loads((Path(subset.root) / "meta/stats.json").read_text())
            np.testing.assert_array_equal(subset.processor.statistics["action"][1], expected["action"]["high"])
    cfg = config(roots, "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=0.05, seed=73)
    with pytest.raises(ValueError, match="at least one source with two eligible episodes"):
        split_action_config(cfg)
    add_split_episodes(roots[1], 2)
    train, evaluation, manifest = split_action_config(cfg)
    assert manifest["train_count"] == 2 and manifest["validation_count"] == 1
    assert train["sources"][0]["episodes"][str(roots[0])] == [17]
    assert len(evaluation["sources"]) == 1 and evaluation["sources"][0]["roots"] == [str(roots[1])]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5, "73", None])
def test_auto_episode_split_rejects_invalid_seed(roots, seed):
    cfg = config(roots, "sample_uniform", 0)
    cfg["validation"] = dict(split_ratio=0.05, seed=seed)
    with pytest.raises(ValueError, match="validation.seed"):
        split_action_config(cfg)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("sampling", ["sample_uniform", "source_weighted"])
@pytest.mark.parametrize("roots", [(75, 41), (75, 41, 38)], indirect=True)
def test_validation_fixed_windows_modes_padding_and_read_order(roots, monkeypatch, sampling):
    import pickle
    import random

    patch_rank(monkeypatch, 0)
    evaluation = config(roots[1:2], sampling, 0)
    evaluation["sources"][0]["roots"] = [str(root) for root in roots[1:]]
    evaluation["sample"].update(mode="joint", mode_weights=[1, 0, 0])
    evaluation["validation"] = dict(seed=73)
    fixed_dataset = build_action_dataset({**evaluation, "sampling": "sample_uniform"}, tokenize)
    window_count = len(fixed_dataset)
    assert window_count == (9 if len(roots) == 2 else 15)
    subsets = action_subsets(fixed_dataset)
    batches = iter_validation_batches(fixed_dataset, seed=73)
    from unittest.mock import patch

    with patch.object(subsets[0], "make_sample", wraps=subsets[0].make_sample) as make_sample:
        make_sample.assert_not_called()
        first = next(batches)
        make_sample.assert_called_once()
    expected_items = [first, *batches]
    assert len(expected_items) == 3 * window_count
    assert [item["record"]["identity"]["start"] for item in expected_items[:9]] == list(range(9))
    modes = ("forward_dynamics", "inverse_dynamics", "policy")
    assert [item["record"]["mode"] for item in expected_items] == [mode for mode in modes for _ in range(window_count)]
    expected = {(item["record"]["mode"], fingerprint(item["record"]["identity"])): item for item in expected_items}
    schedules = [list(iter_validation_batches(fixed_dataset, rank, 2, seed=73)) for rank in (0, 1)]
    assert len(schedules[0]) == len(schedules[1]) == 3 * ((window_count + 1) // 2)
    real = [
        (item["record"]["mode"], fingerprint(item["record"]["identity"]))
        for items in schedules
        for item in items
        if item["weight"]
    ]
    assert sorted(real) == sorted(expected)
    assert sum(item["weight"] == 0 for items in schedules for item in items) == 3
    for rank in range(5):
        assert len(list(iter_validation_batches(fixed_dataset, rank, 5, seed=73))) == 3 * ((window_count + 4) // 5)
    histories = {}
    for item in expected_items:
        record = item["record"]
        histories.setdefault(fingerprint(record["identity"]), set()).add(record["history_blocks"])
    assert all(len(values) == 1 for values in histories.values())
    # Rank assignment, root order, and previous validation passes cannot change a sample.
    patch_rank(monkeypatch, 1)
    evaluation["sources"][0]["roots"].reverse()
    other_dataset = build_action_dataset({**evaluation, "sampling": "sample_uniform"}, tokenize)
    for rank in (1, 0):
        for item in iter_validation_batches(other_dataset, rank, 2, seed=73):
            record = item["record"]
            reference_item = expected[(record["mode"], fingerprint(record["identity"]))]
            assert record == reference_item["record"]
            actual, reference = item["batch"]["samples"][0], reference_item["batch"]["samples"][0]
            for key in (
                "mode",
                "history_blocks",
                "prompt",
                "token_length",
                "condition_frame_indexes_vision",
                "condition_frame_indexes_action",
                "image_size",
            ):
                assert actual[key] == reference[key]
            torch.testing.assert_close(actual["text_token_ids"], reference["text_token_ids"], atol=0, rtol=0)
            torch.testing.assert_close(actual["action"], reference["action"], atol=0, rtol=0)
    assert all(ds.rng is None for ds in subsets + action_subsets(other_dataset))
    before = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state())
    repeated = list(iter_validation_batches(fixed_dataset, 0, 2, seed=73))
    assert [item["record"] for item in repeated] == [item["record"] for item in schedules[0]]
    assert random.getstate() == before[0] and pickle.dumps(np.random.get_state()) == before[1]
    assert torch.equal(before[2], torch.get_rng_state())


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("actions_per_block", [32, 64])
@pytest.mark.parametrize("full_history,expected", [(False, 3), (True, 5)])
def test_history_budget_counts_latents_for_training_and_validation(roots, actions_per_block, full_history, expected):
    from veomni.data.causal_cosmos3.transforms.processing import Geometry

    ds = dataset(
        roots[1],
        geometry=Geometry(actions_per_block=actions_per_block),
        candidates=CandidateConfig(max_actions=64),
        mode="policy",
        full_history=full_history,
        history_range=(3, 3),
    )
    training = ds[0]
    validation = next(iter_validation_batches(ds))
    assert training["history_blocks"] == expected
    assert validation["batch"]["samples"][0]["history_blocks"] == expected
    assert validation["record"]["history_blocks"] == expected
    assert training["window"].actions == 64
    assert ds.make_sample(0, mode="policy", history=1)["history_blocks"] == 1


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("mode", ["forward_dynamics", "inverse_dynamics", "policy"])
def test_validation_single_mode_short_episode_and_read_failure(roots, monkeypatch, mode):
    import veomni.data.causal_cosmos3.datasets.dataset as module

    patch_rank(monkeypatch, 0)
    cfg = config(roots[1:], "sample_uniform", 0)
    cfg["candidates"]["max_actions"] = 160
    cfg["sample"].update(mode=mode, full_history=True)
    fixed_dataset = build_action_dataset(cfg, tokenize)
    items = list(iter_validation_batches(fixed_dataset))
    assert len(items) == len(fixed_dataset) == 1
    item = items[0]["batch"]["samples"][0]
    assert item["mode"] == mode and item["history_blocks"] == 5
    assert item["real_action_steps"] == 40 and item["window"].actions == 64
    assert item["action_valid_steps"].sum() == 40
    padded = list(iter_validation_batches(fixed_dataset, 1, 2))
    assert len(padded) == 1 and padded[0]["weight"] == 0
    assert padded[0]["record"] == items[0]["record"]

    def failed_read(*args, **kwargs):
        raise OSError("validation read failure")

    monkeypatch.setattr(module, "read_window", failed_read)
    for rank in range(2):
        skipped = list(iter_validation_batches(fixed_dataset, rank, 2))
        assert len(skipped) == 1 and skipped[0]["weight"] == 0
        assert skipped[0]["batch"] == {"samples": []}
        assert skipped[0]["record"] == items[0]["record"]
    assert action_subsets(fixed_dataset)[0].rng is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("failed_start", [0, 1, 8])
def test_validation_read_failures_continue_without_changing_dp_rounds(roots, monkeypatch, failed_start):
    import veomni.data.causal_cosmos3.datasets.dataset as module

    patch_rank(monkeypatch, 0)
    fixed_dataset = build_action_dataset(config(roots[1:], "sample_uniform", 0), tokenize)
    original = module.read_window

    def read(window, **kwargs):
        if window.start == failed_start:
            raise OSError("validation read failure")
        return original(window, **kwargs)

    monkeypatch.setattr(module, "read_window", read)
    rounds = [list(iter_validation_batches(fixed_dataset, rank, 2)) for rank in range(2)]
    assert len(rounds[0]) == len(rounds[1]) == 15
    items = rounds[0] + rounds[1]
    failed = [item for item in items if not item["batch"]["samples"]]
    assert len(failed) == (6 if failed_start == 0 else 3)
    assert all(item["weight"] == 0 and item["record"]["identity"]["start"] == failed_start for item in failed)
    for mode in ("forward_dynamics", "inverse_dynamics", "policy"):
        counted = [item for item in items if item["weight"] and item["record"]["mode"] == mode]
        assert sorted(item["batch"]["samples"][0]["candidate_index"] for item in counted) == [
            index for index in range(9) if index != failed_start
        ]
    assert action_subsets(fixed_dataset)[0].rng is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_validation_seed_changes_preserve_candidate_identity(roots, monkeypatch):
    patch_rank(monkeypatch, 0)
    evaluation = config(roots[1:], "sample_uniform", 0)
    fixed_dataset = build_action_dataset(evaluation, tokenize)
    original = next(iter_validation_batches(fixed_dataset))["record"]
    assert original == next(iter_validation_batches(fixed_dataset, seed=42))["record"]
    changed = next(iter_validation_batches(fixed_dataset, seed=74))["record"]
    assert original["identity"] == changed["identity"]
    assert original["history_seed"] != changed["history_seed"]
    assert all(original["seeds"][stream] != changed["seeds"][stream] for stream in ("time", "video", "action"))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "candidates",
    [
        dict(mode="frame_window", start_stride=1),
        dict(mode="frame_window", start_stride=5),
        dict(mode="segmented", use_subtask=True),
        dict(mode="segmented", use_subtask=False, overlap=5),
        dict(mode="segmented", use_subtask=True, borrow_subtask_frames=True),
    ],
)
def test_validation_enumerates_all_training_candidates_and_preserves_samples(roots, monkeypatch, candidates):
    from dataclasses import asdict

    patch_rank(monkeypatch, 0)
    evaluation = config(roots[:1], "sample_uniform", 0)
    evaluation["sources"][0]["candidates"] = candidates
    evaluation["sample"]["mode"] = "forward_dynamics"
    fixed_dataset = build_action_dataset(evaluation, tokenize)
    expected = action_subsets(build_action_dataset(evaluation, tokenize))[0]
    items = list(iter_validation_batches(fixed_dataset))
    assert len(items) == len(expected)
    assert [item["record"]["candidate_index"] for item in items] == list(range(len(expected)))
    if candidates == dict(mode="frame_window", start_stride=1):
        assert len(expected) == 43
    for i, item in enumerate(items):
        entry = item["record"]
        window, caption, fps = expected.window(i)
        assert entry["window"] == asdict(window) and entry["caption"] == caption and entry["source_fps"] == fps
        reference = expected.make_sample(i, mode="forward_dynamics", history=entry["history_blocks"])
        actual = item["batch"]["samples"][0]
        for key in ("prompt", "token_length", "real_action_steps", "history_blocks", "condition_frame_indexes_vision"):
            assert actual[key] == reference[key]
        for key in ("action", "state", "text_token_ids", "action_valid_steps"):
            if key in reference:
                torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("roots", [(75, 33)], indirect=True)
def test_validation_preserves_subtasks_that_borrow_the_same_physical_window(roots, monkeypatch):
    patch_rank(monkeypatch, 0)
    evaluation = config(roots[1:], "sample_uniform", 0)
    evaluation["candidates"].update(mode="segmented", use_subtask=True, borrow_subtask_frames=True)
    fixed_dataset = build_action_dataset(evaluation, tokenize)
    items = list(iter_validation_batches(fixed_dataset))
    assert len(fixed_dataset) == 2 and len(items) == 6
    first, second = (item["record"] for item in items[:2])
    assert first["window"] == second["window"]
    assert (first["caption"], second["caption"]) == ("first", "second")
    assert first["identity"] != second["identity"]
    assert items[0]["batch"]["samples"][0]["prompt"] != items[1]["batch"]["samples"][0]["prompt"]


def _validation_gloo_worker(rank, evaluation, output_dir, endpoint):
    from contextlib import nullcontext
    from datetime import timedelta

    import torch.distributed as dist

    import veomni.data.causal_cosmos3.datasets.dataset as action_data
    import veomni.data.causal_cosmos3.datasets.pipeline as pipeline
    import veomni.trainer.causal_cosmos3 as validation

    dist.init_process_group("gloo", rank=rank, world_size=2, store=endpoint.connect(), timeout=timedelta(seconds=60))
    parallel = SimpleNamespace(dp_rank=rank, dp_size=2, dp_group=dist.group.WORLD, sp_size=1)
    pipeline.get_parallel_state = lambda: parallel
    validation.use_parallel_state = lambda state: nullcontext()
    validation.get_device_rng_state = torch.get_rng_state
    validation.set_device_rng_state = torch.set_rng_state

    class Condition(torch.nn.Module):
        config = SimpleNamespace(to_dict=lambda: {})
        fail = False

        def __init__(self):
            super().__init__()
            self.generator = torch.Generator().manual_seed(7)

        def rng_state_dict(self):
            return {"noise": self.generator.get_state()}

        def load_rng_state_dict(self, state):
            self.generator.set_state(state["noise"])

        def get_condition(self, samples):
            if self.fail and rank == 1:
                raise OSError("rank-local VAE failure")
            return {"samples": samples}

    class Runtime:
        calls = 0
        parallel_state = parallel

        def __init__(self):
            self.model = torch.nn.Linear(1, 1)
            self.model.config = SimpleNamespace(to_dict=lambda: {})
            self.condition_model = Condition()

        def __call__(self, sample, targets):
            # Exercise a collective in every forward to detect divergent skips.
            entered = torch.ones(1)
            dist.all_reduce(entered, group=parallel.dp_group)
            assert entered.item() == 2
            self.calls += 1
            value = (
                sample["window"].start + {"forward_dynamics": 10, "inverse_dynamics": 20, "policy": 30}[sample["mode"]]
            )
            return SimpleNamespace(loss={"video_loss": torch.tensor(float(value)), "action_loss": torch.tensor(0.0)})

    validation.fixed_validation_inputs = lambda condition, samples, seeds: dict(
        sample=samples[0], targets={"loss_masks": {"video_loss_mask": torch.tensor([True])}}
    )
    original_read = action_data.read_window
    try:
        for windows in (1, 3):
            config = copy.deepcopy(evaluation)
            config["candidates"]["start_stride"] = 16 if windows == 1 else 4
            config["validation"] = {"seed": 73}
            dataset = build_action_dataset(config, tokenize)
            runtime = Runtime()
            trainer = SimpleNamespace(
                model=runtime,
                device="cpu",
                model_fwd_context=nullcontext(),
                validation_config=config,
                args=SimpleNamespace(
                    model=SimpleNamespace(tokenizer_path=None),
                    train=SimpleNamespace(
                        global_rank=rank,
                        wandb=SimpleNamespace(enable=False),
                        checkpoint=SimpleNamespace(output_dir=output_dir),
                    ),
                ),
            )
            validator = validation.CausalCosmos3Validator(trainer, dataset)
            result = validator.evaluate(SimpleNamespace(global_step=windows, epoch=0))
            for tag, base in (("fd", 10), ("id", 20), ("policy", 30)):
                assert result[f"eval/{tag}/loss"] == base + (4 if windows == 3 else 0)
                assert result[f"eval/{tag}/count"] == windows
                assert result[f"eval/{tag}/padding_count"] == 1
                assert result[f"eval/{tag}/skipped_count"] == 0
            assert runtime.calls == (3 if windows == 1 else 6)
            before = runtime.calls
            runtime.condition_model.fail = True
            with pytest.raises((OSError, RuntimeError), match="VAE failure|another rank"):
                validator.evaluate(SimpleNamespace(global_step=9, epoch=0))
            assert runtime.calls == before
            assert runtime.model.training and runtime.condition_model.training
            runtime.condition_model.fail = False

            def failed_read(window, windows=windows, **kwargs):
                if rank == 1 and (windows == 1 or window.start == 4):
                    raise OSError("rank-local read failure")
                return original_read(window, **kwargs)

            action_data.read_window = failed_read
            if windows == 1:
                with pytest.raises(ValueError, match="no supervised windows"):
                    validator.evaluate(SimpleNamespace(global_step=10, epoch=0))
                assert runtime.calls == before
            else:
                skipped = validator.evaluate(SimpleNamespace(global_step=10, epoch=0))
                for tag, base in (("fd", 10), ("id", 20), ("policy", 30)):
                    assert skipped[f"eval/{tag}/loss"] == base + 8
                    assert skipped[f"eval/{tag}/count"] == 1
                    assert skipped[f"eval/{tag}/skipped_count"] == 2
                    assert skipped[f"eval/{tag}/padding_count"] == 1
                assert runtime.calls == before + 3
            assert runtime.model.training and runtime.condition_model.training
            action_data.read_window = original_read
    finally:
        action_data.read_window = original_read
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_validation_distributed_means_skips_and_prepare_failure(roots, tmp_path, monkeypatch):
    from tests.tools.launch_utils import spawn_with_rendezvous

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    evaluation = config(roots[1:], "sample_uniform", 0)
    spawn_with_rendezvous(_validation_gloo_worker, args=(evaluation, str(tmp_path / "gloo")), nprocs=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_inference_initial_reader_requests_only_one_observation(roots, monkeypatch):
    import veomni.data.causal_cosmos3.readers.observation as observation
    from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry

    source = dict(
        cameras=dict(head="cam"),
        viewpoint="ego_view",
        position_unit="mm",
        gripper_unit="m",
        columns=dict(
            state="action_unified",
            target="action_unified",
            state_mask="mask_action",
            target_mask="mask_action",
            target_offset=1,
        ),
    )
    calls = []
    real_read = observation.read_rows

    def checked(path, start, stop, columns):
        assert stop == start + 1
        assert set(columns) == {"action_unified", "mask_action", "timestamp", "task_index"}
        calls.append((start, stop))
        return real_read(path, start, stop, columns)

    monkeypatch.setattr(observation, "read_rows", checked)
    monkeypatch.setattr(observation, "read_window", lambda *a, **kw: pytest.fail("Future window must not be read"))

    def decode(path, timestamps, output, **kwargs):
        assert len(timestamps) == 1
        output[:] = 37

    monkeypatch.setattr(observation, "decode_into", decode)
    initial = observation.read_initial_observation(
        roots[0], 17, 5, 37, source=source, processor=ActionProcessor(statistics()), geometry=Geometry()
    )
    assert calls == [(5, 6)]
    assert initial.state[0] == 6
    assert initial.horizon == 37
    assert initial.caption == "fallback"
    assert initial.fps == 25
    assert initial.image.shape == (3, 1, 256, 256)
    assert initial.image.eq(37).all()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_inference_fps_prompt_and_payload_match_training_dataset(roots, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as reader
    from veomni.data.causal_cosmos3.readers.observation import read_teacher_window

    monkeypatch.setattr(reader, "decode_into", lambda path, timestamps, output, **kwargs: output.fill(13))
    training = dataset(roots[0], candidates=CandidateConfig(max_actions=64, use_subtask=False))
    training.video = True
    expected = training.make_sample(0, mode="policy", history=3)
    actual = read_teacher_window(
        roots[0],
        17,
        0,
        64,
        source=dict(cameras=dict(head="cam"), viewpoint="ego_view"),
        processor=training.processor,
        geometry=training.geometry,
        tokenize=tokenize,
        task="policy",
        history=3,
    )
    assert actual["conditioning_fps_action"] == expected["conditioning_fps_action"] == 25
    assert actual["conditioning_fps"] == expected["conditioning_fps"] == 6.25
    assert actual["prompt"] == expected["prompt"]
    for key in ("video", "action", "states", "anchors", "action_mask", "state_mask", "text_token_ids"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_inference_npz_uses_native_layout_and_preserves_physical_units(tmp_path):
    from veomni.data.causal_cosmos3.readers.observation import read_npz_observation
    from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry

    mask = np.zeros(55, dtype=bool)
    mask[0] = True
    state = np.zeros(55, dtype=np.float32)
    state[0] = 13.5
    path = tmp_path / "initial.npz"
    np.savez(
        path,
        cam_high=np.full((48, 64, 3), 12, dtype=np.uint8),
        cam_left_wrist=np.full((48, 64, 3), 31, dtype=np.uint8),
        cam_right_wrist=np.full((48, 64, 3), 79, dtype=np.uint8),
        state=state,
        state_mask=mask,
        action_mask=mask,
        prompt="move",
    )
    source = dict(
        cameras=dict(head="cam_high", left="cam_left_wrist", right="cam_right_wrist"),
        viewpoint="concat_view",
        position_unit="mm",
        gripper_unit="m",
        columns=dict(
            state="action_unified",
            target="action_unified",
            state_mask="mask_action",
            target_mask="mask_action",
            target_offset=1,
        ),
    )
    initial = read_npz_observation(
        path, source=source, processor=ActionProcessor(statistics()), geometry=Geometry(), fps=30, horizon=35
    )
    assert initial.state[0] == 13.5
    assert initial.image.shape == (3, 1, 384, 320)
    assert initial.image[:, :, 0:256].eq(12).all()
    assert initial.image[:, :, 256:, :160].eq(31).all()
    assert initial.image[:, :, 256:, 160:].eq(79).all()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_inference_exports_measured_gt_before_normalization_clipping(roots, monkeypatch):
    import veomni.data.causal_cosmos3.readers.lerobot as reader
    from veomni.data.causal_cosmos3.readers.observation import read_teacher_window
    from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor, Geometry

    stats = statistics()
    stats["action"]["low"][0], stats["action"]["high"][0] = -0.1, 0.1
    source = dict(cameras=dict(head="cam"), viewpoint="ego_view")
    monkeypatch.setattr(reader, "decode_into", lambda path, timestamps, output, **kwargs: output.fill(0))
    sample = read_teacher_window(
        roots[0],
        17,
        0,
        32,
        source=source,
        processor=ActionProcessor(stats),
        geometry=Geometry(),
        tokenize=tokenize,
        task="policy",
        history=1,
    )
    assert sample["action"][:, 0].eq(1.5).all()
    torch.testing.assert_close(sample["ground_truth_actions"][:, 0], torch.arange(1, 33, dtype=torch.float32))
