import gc
import multiprocessing
import os
import types
from contextlib import contextmanager
from functools import partial
from typing import Literal

import pytest
import torch
from utils import DummyDataset, process_dummy_example

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data import build_dataloader, build_dataset
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _WorkerRuntimeStateDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        return [
            {
                "gc_enabled": gc.isenabled(),
                "threads": torch.get_num_threads(),
                "pid": os.getpid(),
                "worker": worker.id if worker else -1,
            }
        ]


@contextmanager
def _disabled_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()


def _build_worker_state_loader(monkeypatch, workers, threads, context=None):
    import veomni.data.data_loader as m_dl
    from veomni.data.data_collator import NoopDataCollator

    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: _fake_ps(sp_size=1))
    monkeypatch.setattr(m_dl, "get_device_type", lambda: "cpu")
    return build_dataloader(
        "native",
        dataset=_WorkerRuntimeStateDataset(),
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=1,
        train_steps=1,
        dyn_bsz=False,
        collate_fn=NoopDataCollator(),
        num_workers=workers,
        worker_num_threads=threads,
        pin_memory=False,
        prefetch_factor=2 if workers else None,
        multiprocessing_context=context,
    )


@pytest.mark.parametrize("context", ["fork", "spawn"])
@pytest.mark.parametrize("threads", [None, 1])
def test_native_workers_enable_gc_without_changing_parent(monkeypatch, context, threads):
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} is unavailable")
    # Fresh spawn workers import Torch before pytest's CPU fixtures can run.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "MLU_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "")
    with _disabled_gc():
        loader = _build_worker_state_loader(monkeypatch, workers=2, threads=threads, context=context)
        iterator = None
        try:
            iterator = iter(loader)
            rows = [batch[0][0] for batch in iterator]
            assert not gc.isenabled(), "Worker setup must preserve the trainer's GC policy"
            assert len(rows) == 4 and {row["worker"] for row in rows} == {0, 1}
            assert all(row["pid"] != os.getpid() and row["gc_enabled"] for row in rows)
            # PyTorch's worker default is one thread; None must preserve it.
            assert all(row["threads"] == 1 for row in rows)
        finally:
            if iterator is not None:
                iterator._shutdown_workers()


@pytest.mark.parametrize("threads", [None, 1])
def test_native_zero_workers_preserve_parent_gc_and_threads(monkeypatch, threads):
    original_threads = torch.get_num_threads()
    with _disabled_gc():
        loader = _build_worker_state_loader(monkeypatch, workers=0, threads=threads)
        rows = [batch[0][0] for batch in loader]
        assert not gc.isenabled()
        assert all(row["pid"] == os.getpid() and row["worker"] == -1 for row in rows)
        assert all(not row["gc_enabled"] and row["threads"] == original_threads for row in rows)


def _fake_ps(sp_size: int):
    sp_enabled = sp_size > 1
    return types.SimpleNamespace(
        dp_size=1,
        dp_rank=0,
        sp_enabled=sp_enabled,
        sp_size=sp_size,
        sp_rank=0,
    )


@pytest.fixture(scope="session")
def dummy_dataset_ci():
    dummy = DummyDataset(size=40, num_shard=1, dataset_name="ci_dyn_bsz_shared")
    yield dummy
    dummy.clean_cache()


@pytest.mark.parametrize("dataset_name", ["iterable", "mapping"])
@pytest.mark.parametrize("dyn_bsz", [True, False])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_sp_filling(
    monkeypatch,
    dummy_dataset_ci,
    dataset_name: str,
    dyn_bsz: bool,
    sp_size: int,
    dyn_bsz_runtime: Literal["main", "worker"],
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=sp_size)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    global_batch_size = 8
    micro_batch_size = 2
    max_seq_len = 100

    if dyn_bsz:
        if dyn_bsz_runtime == "main":
            dataloader_batch_size = 1
        else:
            dataloader_batch_size = global_batch_size // micro_batch_size
    else:
        dataloader_batch_size = global_batch_size

    transform = partial(process_dummy_example, max_seq_len=max_seq_len)

    dataset = build_dataset(
        dataset_name=dataset_name,
        train_path=dummy_dataset_ci.save_path,
        transform=transform,
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        max_seq_len=max_seq_len,
        train_steps=1,
        num_workers=0,
        dyn_bsz=dyn_bsz,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    micro_batches = next(iter(dl))

    if dyn_bsz:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert max_seq_len * (micro_batch_size - 1) <= sum(micro_batch["id"]) <= max_seq_len * micro_batch_size
    else:
        assert len(micro_batches) == global_batch_size // micro_batch_size
        for micro_batch in micro_batches:
            assert len(micro_batch["id"]) == micro_batch_size


@pytest.mark.parametrize("dyn_bsz_runtime", ["main", "worker"])
def test_build_dataloader_dyn_bsz_count_mode(
    monkeypatch, dummy_dataset_ci, dyn_bsz_runtime: Literal["main", "worker"]
):
    import veomni.data.data_collator as m_col
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_col, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1 if dyn_bsz_runtime == "main" else 2,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime=dyn_bsz_runtime,
        dyn_bsz_count_mode="effective",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    if dyn_bsz_runtime == "main":
        assert isinstance(dl, DynamicBatchSizeDataLoader)
        assert isinstance(dl.batching_strategy, TextBatchingStrategy)
        assert dl.batching_strategy.buffer._get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.batching_strategy.physical_token_cap == 48
        assert dl.batching_strategy.buffer._get_physical_length_fn is m_ds.get_length_fn
    else:
        assert isinstance(dl.dataset, m_ds.DynamicBatchingSizeDataset)
        assert dl.dataset.get_length_fn is m_ds.get_length_by_labels_fn
        assert dl.dataset.physical_token_cap == 48
        assert dl.dataset.get_physical_length_fn is m_ds.get_length_fn


def test_build_dataloader_dyn_bsz_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="effective",
        dyn_bsz_physical_overflow_ratio=1.25,
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        seed=0,
    )

    assert dl.batching_strategy.physical_token_cap == 40


def test_build_dataloader_suppresses_persistent_workers_with_zero_workers(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    dl = build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=0,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_buffer_size=1,
        drop_last=True,
        prefetch_factor=None,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert dl._dataloader.persistent_workers is False
    assert dl._dataloader.in_order is False


def test_build_dataloader_forwards_worker_scheduling_kwargs(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    captured = {}

    class FakeDistributedDataloader:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_dl, "DistributedDataloader", FakeDistributedDataloader)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    build_dataloader(
        "native",
        dataset=dataset,
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=1,
        max_seq_len=16,
        train_steps=1,
        num_workers=2,
        dyn_bsz=False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
        in_order=False,
        seed=0,
    )

    assert captured["persistent_workers"] is True
    assert captured["in_order"] is False


def test_build_dataloader_rejects_invalid_physical_overflow_ratio(monkeypatch, dummy_dataset_ci):
    import veomni.data.data_loader as m_dl
    import veomni.data.dataset as m_ds

    ps = _fake_ps(sp_size=1)
    monkeypatch.setattr(m_dl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(m_ds, "get_parallel_state", lambda: ps)

    dataset = build_dataset(
        dataset_name="iterable",
        train_path=dummy_dataset_ci.save_path,
        transform=partial(process_dummy_example, max_seq_len=16),
        seed=0,
    )
    with pytest.raises(ValueError, match="dyn_bsz_physical_overflow_ratio must be >= 1.0"):
        build_dataloader(
            "native",
            dataset=dataset,
            micro_batch_size=2,
            global_batch_size=4,
            dataloader_batch_size=1,
            max_seq_len=16,
            train_steps=1,
            num_workers=0,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            dyn_bsz_count_mode="effective",
            dyn_bsz_physical_overflow_ratio=0.5,
            dyn_bsz_buffer_size=1,
            drop_last=True,
            prefetch_factor=None,
            seed=0,
        )
