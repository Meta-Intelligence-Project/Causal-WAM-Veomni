"""Packed-episode opt-in, native singleton wiring and reader shutdown boundaries."""

from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.datasets import pipeline
from veomni.trainer import causal_cosmos3 as module
from veomni.trainer.base import BackgroundPrefetcher, VeOmniIter
from veomni.trainer.dit_trainer import DiTTrainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@dataclass
class WorkerOptions:
    type: str = "native"
    use_background_prefetcher: bool = True
    num_workers: int = 2
    in_order: bool = True
    persistent_workers: bool = False


def trainer_args(*, batch=2, accum=1, stage="action", tbptt=True, workers=2, global_batch=None):
    args = SimpleNamespace(
        model=SimpleNamespace(config_path="unused", model_config={}, condition_model_cfg={}),
        data=SimpleNamespace(max_seq_len=54000, dyn_bsz_buffer_size=8, dataloader=WorkerOptions(num_workers=workers)),
        train=SimpleNamespace(
            training_task="online_training",
            micro_batch_size=batch,
            gradient_accumulation_steps=accum,
            global_batch_size=batch * accum * 2 if global_batch is None else global_batch,
            global_rank=0,
            dataloader_batch_size=batch * accum,
            max_steps=30,
            dyn_bsz_runtime="main",
            seed=42,
            checkpoint=SimpleNamespace(save_steps=0, output_dir="unused"),
        ),
        train_steps=30,
    )
    config = SimpleNamespace(
        action_interface="shared" if stage == "action" else "none",
        action_dim=55 if stage == "action" else 0,
        state_dim=55 if stage == "action" else 0,
        enable_tbptt=tbptt,
        tbptt_segment_steps=256,
        tbptt_segment_latents=16,
    )
    trainer = module.CausalCosmos3Trainer.__new__(module.CausalCosmos3Trainer)
    trainer.tokenize = lambda text: [1, 2]
    trainer.dataset_config = dict(stage=stage, geometry={}, sample=dict(action_horizon=32))
    trainer.segment_memories = {}
    trainer._episode_batch_loader = None
    trainer.base = SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            model=SimpleNamespace(config=config), parallel_state=SimpleNamespace(dp_size=2, dp_group=None)
        ),
        train_dataset=SimpleNamespace(episodes=(0, 1, 2)),
    )
    return trainer, config


def mock_setup(monkeypatch, config):
    calls = []
    monkeypatch.setattr(module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(module, "build_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(sp_size=1, dp_size=2, dp_group=None))
    monkeypatch.setattr(DiTTrainer, "_setup", lambda self: calls.append(True))
    return calls


@pytest.mark.parametrize("batch,accum", [(1, 2), (2, 1), (3, 2)])
def test_action_tbptt_batch_size_opt_in_preserves_segment_accounting(monkeypatch, batch, accum):
    trainer, config = trainer_args(batch=batch, accum=accum, workers=4)
    calls = mock_setup(monkeypatch, config)
    trainer._setup()
    args = trainer.base.args
    assert calls == [True] and trainer.episode_training
    assert args.train.global_batch_size == batch * accum * 2
    assert args.train.dataloader_batch_size == (accum if batch == 1 else 1)
    assert not args.train.dyn_bsz
    assert args.model.model_config["history_encoding"] == "episode_stream"


@pytest.mark.parametrize("stage,tbptt", [("action", False), ("video", False), ("video", True)])
def test_other_training_paths_still_reject_multiple_packs(monkeypatch, stage, tbptt):
    trainer, config = trainer_args(stage=stage, tbptt=tbptt)
    calls = mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        trainer._setup()
    assert calls == []


@pytest.mark.parametrize("batch", [True, 0, -1, 1.5])
def test_episode_microbatch_requires_positive_integer(monkeypatch, batch):
    trainer, config = trainer_args(batch=batch)
    calls = mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="positive integer"):
        trainer._setup()
    assert calls == []


def test_global_batch_must_cover_complete_microbatches(monkeypatch):
    trainer, config = trainer_args(global_batch=6)
    mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="micro_batch_size.*dp_size"):
        trainer._setup()


@pytest.mark.parametrize("setting", [None, "0"])
def test_npu_episode_batch_rejects_matmul_k_shift_before_model_build(monkeypatch, setting):
    trainer, config = trainer_args()
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(module, "get_device_type", lambda: "npu")
    if setting is None:
        monkeypatch.delenv("CLOSE_MATMUL_K_SHIFT", raising=False)
    else:
        monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", setting)
    with pytest.raises(ValueError, match="CLOSE_MATMUL_K_SHIFT=1 before starting Python"):
        trainer._setup()


@pytest.mark.parametrize("batch,setting", [(1, None), (2, "1")])
def test_npu_single_stream_or_stable_packed_matmul_setup(monkeypatch, batch, setting):
    trainer, config = trainer_args(batch=batch)
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(module, "get_device_type", lambda: "npu")
    if setting is None:
        monkeypatch.delenv("CLOSE_MATMUL_K_SHIFT", raising=False)
    else:
        monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", setting)
    trainer._setup()
    assert trainer.episode_training


@pytest.mark.parametrize("batch,accum", [(1, 2), (2, 1)])
def test_worker_loader_reads_singletons_before_independent_lane_grouping(monkeypatch, batch, accum):
    trainer, _ = trainer_args(batch=batch, accum=accum)
    trainer.episode_training = True
    captured = {}
    source = SimpleNamespace(set_epoch=lambda epoch: None)

    def build(**kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr(module, "build_native_dataloader", build)
    trainer._build_dataloader()
    assert captured["micro_batch_size"] == 1 and not captured["dyn_bsz"]
    if batch == 1:
        assert captured["global_batch_size"] == 4 and captured["dataloader_batch_size"] == 2
        assert trainer._episode_batch_loader is None
        assert trainer.base.train_dataloader.dataloader is source
    else:
        assert captured["global_batch_size"] == 2 and captured["dataloader_batch_size"] == 1
        assert trainer._episode_batch_loader.batch_size == 2
        assert trainer._episode_batch_loader.num_micro_batches == 1
        assert trainer._episode_batch_loader.lane_ids == (0, 1)
        assert trainer.base.train_dataloader.dataloader is trainer._episode_batch_loader
    assert trainer.base.train_dataloader.state_dict() is None


@pytest.mark.parametrize("workers,episodes", [(0, 3), (1, 3), (4, 1)])
def test_insufficient_rank_lanes_fail_before_worker_iteration(monkeypatch, workers, episodes):
    trainer, config = trainer_args(workers=workers)
    trainer.episode_training = True
    trainer.base.data_transform = trainer.tokenize
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(
        pipeline, "build_episode_dataset", lambda *args, **kwargs: SimpleNamespace(episodes=tuple(range(episodes)))
    )
    with pytest.raises(ValueError, match="active lanes per rank"):
        trainer._build_dataset()


def test_epoch_end_stops_prefetch_before_closing_owned_workers(monkeypatch):
    trainer, _ = trainer_args()
    order = []
    trainer.base.data_iterator = SimpleNamespace(stop=lambda timeout=None: order.append(("stop", timeout)))
    trainer._episode_batch_loader = SimpleNamespace(close=lambda: order.append(("close", None)))
    monkeypatch.setattr(DiTTrainer, "on_epoch_end", lambda self: order.append(("callbacks", None)))
    trainer.on_epoch_end()
    assert order == [("callbacks", None), ("stop", None), ("stop", None), ("close", None)]


def test_slow_read_is_joined_and_late_rgb_update_drained_before_loader_close(monkeypatch):
    started, release, reading, closed = Event(), Event(), Event(), Event()

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            reading.set()
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test did not release its finite read")
            reading.clear()
            return {"raw_rgb_placeholder": object()}

    trainer, _ = trainer_args()
    iterator = VeOmniIter(Source(), use_background_prefetcher=True)
    assert started.wait(timeout=2)
    trainer.base.data_iterator = iterator
    timeouts, errors = [], []
    original_stop = BackgroundPrefetcher.stop

    def stop(self, timeout=5.0):
        timeouts.append(timeout)
        # Make a mistaken timed stop fail quickly without a five-second test.
        return original_stop(self, timeout=None if timeout is None else 0)

    def close():
        assert not reading.is_set()
        assert not iterator.iterator.thread.is_alive()
        assert iterator.iterator.queue.empty()
        closed.set()

    def shutdown():
        try:
            trainer._close_episode_loader()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(BackgroundPrefetcher, "stop", stop)
    trainer._episode_batch_loader = SimpleNamespace(close=close)
    thread = Thread(target=shutdown)
    thread.start()
    try:
        assert iterator.iterator.stop_event.wait(timeout=1)
        assert thread.is_alive() and not closed.is_set()
        release.set()
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert closed.is_set() and timeouts == [None, None]
    finally:
        release.set()
        thread.join(timeout=3)
        original_stop(iterator.iterator, timeout=None)
