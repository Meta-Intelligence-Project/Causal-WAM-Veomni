"""Cosmos3 restart recovery must not snapshot residual video buffers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.checkpoints.test_checkpoint_callback import _make_mock_trainer
from veomni.checkpoint import layout
from veomni.trainer.base import VeOmniIter
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.causal_cosmos3 import (
    CausalCosmos3GlobalStateCallback,
    CausalCosmos3RestartDataLoader,
    CausalCosmos3Trainer,
)
from veomni.trainer.dit_trainer import DiTTrainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class ItemsWithoutSnapshots:
    def __init__(self):
        self.epochs = []

    def __len__(self):
        return 2

    def __iter__(self):
        yield (1, {"video": 2})
        yield (3, 4)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def state_dict(self):
        raise AssertionError("Must not copy the original packing buffer")

    def load_state_dict(self, state):
        raise AssertionError("Must not restore the original data cursor")


@pytest.mark.parametrize("background", [False, True])
def test_restart_prefetch_preserves_batches_and_epoch_without_snapshots(background):
    original = ItemsWithoutSnapshots()
    loader = CausalCosmos3RestartDataLoader(original)
    loader.set_epoch(3)
    loader.load_state_dict({"old_video_buffer": torch.ones(4)})
    assert original.epochs == [3]
    assert len(loader) == 2
    for epoch in (3, 4):
        loader.set_epoch(epoch)
        iterator = VeOmniIter(loader, use_background_prefetcher=background)
        try:
            assert list(iterator) == [(1, {"video": 2}), (3, 4)]
            assert iterator.state_dict() is None
        finally:
            iterator.stop()


def test_cosmos3_installs_its_own_callback(monkeypatch):
    previous, other = object(), object()
    base = _make_mock_trainer()
    base.global_state_callback = previous
    base._callbacks = [other, previous]
    base.validation_dataset = None

    def init_parent(self, args):
        self.base = base

    monkeypatch.setattr(DiTTrainer, "__init__", init_parent)
    trainer = CausalCosmos3Trainer(object(), dataset_config={}, tokenize=lambda prompt: [1])
    assert isinstance(trainer.base.global_state_callback, CausalCosmos3GlobalStateCallback)
    assert base._callbacks == [other, base.global_state_callback]
    assert base.global_state_callback.trainer is base


@patch("veomni.trainer.callbacks.global_state_callback.dist")
class TestCosmos3RestartCheckpoint:
    @pytest.mark.parametrize("step", [17, 100])
    def test_roundtrip_keeps_progress_rng_and_meters(self, mock_dist, tmp_path, step):
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer(save_path=str(tmp_path))
        trainer.train_dataloader = CausalCosmos3RestartDataLoader(ItemsWithoutSnapshots())
        trainer.data_iterator = VeOmniIter(trainer.train_dataloader)
        trainer.environ_meter.state_dict.return_value = {"tokens": 42}
        callback = CausalCosmos3GlobalStateCallback(trainer)
        rng = torch.get_rng_state()
        callback.save_global_state(TrainerState(global_step=step))
        root = str(tmp_path / f"global_step_{step}")
        assert torch.load(layout.loader_path(root, 0), weights_only=False) == {"train_dataloader": None}

        resumed = _make_mock_trainer()
        resumed.args.train.checkpoint.load_path = root
        resumed.train_dataloader = CausalCosmos3RestartDataLoader(ItemsWithoutSnapshots())
        torch.manual_seed(999)
        state = CausalCosmos3GlobalStateCallback(resumed).load_global_state()
        assert state["data_resume_mode"] == "restart"
        assert state["train_dataloader"] is None
        assert resumed.state.global_step == step
        assert (resumed.start_epoch, resumed.start_step) == divmod(step, 100)
        assert torch.equal(torch.get_rng_state(), rng)
        resumed.environ_meter.load_state_dict.assert_called_once_with({"tokens": 42})

    @pytest.mark.parametrize("legacy", [False, True])
    def test_ignores_old_data_cursor(self, mock_dist, tmp_path, legacy):
        mock_dist.is_initialized.return_value = False
        trainer = _make_mock_trainer()
        trainer.args.train.checkpoint.load_path = str(tmp_path)
        payload = {"global_step": 17, "train_dataloader": {"residual_video": torch.ones(4)}}
        if legacy:
            path = tmp_path / "trainer_state_rank_0.pt"
        else:
            path = Path(layout.extra_state_path(str(tmp_path), 0))
            loader_path = Path(layout.loader_path(str(tmp_path), 0))
            loader_path.parent.mkdir(parents=True)
            loader_path.write_bytes(b"Old loader file must not be read")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        state = CausalCosmos3GlobalStateCallback(trainer).load_global_state()
        assert trainer.state.global_step == 17
        assert state["train_dataloader"] is None
        trainer.train_dataloader.load_state_dict.assert_not_called()

    def test_save_does_not_request_original_loader_state(self, mock_dist):
        trainer = _make_mock_trainer()
        original = MagicMock()
        trainer.train_dataloader = CausalCosmos3RestartDataLoader(original)
        # Before the first iterator exists, saving uses the loader directly.
        del trainer.data_iterator
        state = CausalCosmos3GlobalStateCallback(trainer).state_dict(TrainerState(global_step=17))
        assert state["train_dataloader"] is None
        original.state_dict.assert_not_called()
