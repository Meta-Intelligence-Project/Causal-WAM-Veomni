"""Independent lane grouping, bounded reads and worker lifecycle on CPU."""

import gc
import weakref
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data import test_causal_cosmos3_episode as fixtures
from veomni.data.causal_cosmos3.datasets.dataset import StatefulConcatDataset
from veomni.data.causal_cosmos3.datasets.episode import collate_episode_segment
from veomni.data.causal_cosmos3.datasets.episode_batching import EpisodeBatchDataLoader, active_episode_lane_ids
from veomni.trainer.causal_cosmos3_episode import EpisodeLoaderItems


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def segment(lane, index, *, stream=None, start=None, final=False, decisions=2):
    start = index * 64 if start is None else start
    return dict(
        lane_id=lane,
        stream_id=(lane,) if stream is None else stream,
        segment_id=index,
        start_tick=start,
        end_tick=start + decisions * 32,
        reset=index == 0,
        final=final,
        decisions=[{"tick": start + offset * 32} for offset in range(decisions)],
    )


def singleton(raws):
    return [[{"segment": raw}] for raw in raws]


def batching(raws, *, batch_size=2, num_micro_batches=1, lane_ids=(0, 1), **kwargs):
    return EpisodeBatchDataLoader(
        singleton(raws), batch_size=batch_size, num_micro_batches=num_micro_batches, lane_ids=lane_ids, **kwargs
    )


def flatten(updates):
    return [raw for update in updates for micro in update for raw in micro["segments"]]


@pytest.mark.parametrize("num_workers,episodes,expected", [(0, 3, (0,)), (4, 3, (0, 1, 2)), (2, 5, (0, 1))])
def test_active_lanes_come_from_rank_owned_episodes(num_workers, episodes, expected):
    assert active_episode_lane_ids(SimpleNamespace(episodes=[None] * episodes), num_workers=num_workers) == expected


@pytest.mark.parametrize("num_workers", [-1, True, 1.5])
def test_invalid_worker_geometry_is_rejected(num_workers):
    with pytest.raises(ValueError, match="num_workers"):
        active_episode_lane_ids(SimpleNamespace(episodes=[None]), num_workers=num_workers)


def test_out_of_lane_arrival_is_grouped_without_dropping_or_reordering():
    raws = [segment(0, 0), segment(0, 1, final=True), segment(1, 0), segment(1, 1, final=True)]
    updates = list(batching(raws, num_micro_batches=2))
    assert len(updates) == 1 and len(updates[0]) == 2
    assert [[raw["lane_id"] for raw in micro["segments"]] for micro in updates[0]] == [[0, 1], [0, 1]]
    actual = flatten(updates)
    assert {id(raw) for raw in actual} == {id(raw) for raw in raws}
    for lane in (0, 1):
        assert [raw["segment_id"] for raw in actual if raw["lane_id"] == lane] == [0, 1]


def test_short_final_and_text_reset_segments_are_indivisible():
    raws = [
        segment(0, 0, final=True, decisions=1),
        segment(1, 0, decisions=8),
        segment(0, 0, stream=(0, "new text"), start=32, final=True, decisions=1),
        segment(1, 1, start=256, final=True, decisions=1),
    ]
    actual = flatten(list(batching(raws)))
    assert actual == raws
    assert [len(raw["decisions"]) for raw in actual] == [1, 8, 1, 1]
    assert actual[2]["reset"] and actual[2]["stream_id"] != actual[0]["stream_id"]


@pytest.mark.parametrize("lane_ids,batch_size", [((0,), 2), ((), 2), ((0, 0), 2), ((-1, 0), 2), ((True, 0), 2)])
def test_insufficient_or_invalid_lanes_fail_before_reading(lane_ids, batch_size):
    with pytest.raises(ValueError, match="lane|Lane"):
        EpisodeBatchDataLoader(None, batch_size=batch_size, num_micro_batches=1, lane_ids=lane_ids)


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("batch_size", True), ("num_micro_batches", 0)])
def test_invalid_batch_geometry_fails(field, value):
    kwargs = dict(batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        EpisodeBatchDataLoader([], **kwargs)


def test_unbalanced_source_does_not_silently_drop_tail():
    raws = [segment(0, 0), segment(1, 0, final=True), segment(0, 1, final=True)]
    iterator = iter(batching(raws))
    assert len(next(iterator)[0]["segments"]) == 2
    with pytest.raises(ValueError, match="exhausted with 1 unconsumed"):
        next(iterator)
    assert iterator.closed and iterator.buffered_segments == 0


def test_an_incomplete_accumulation_group_is_not_reported_as_an_update():
    with pytest.raises(ValueError, match="complete.*optimizer update"):
        next(iter(batching([segment(0, 0), segment(1, 0)], num_micro_batches=2)))


def test_skew_is_bounded_and_stops_reading_before_exceeding_cap():
    reads = []

    def source():
        for index in range(100):
            reads.append(index)
            yield [{"segment": segment(0, index)}]

    loader = EpisodeBatchDataLoader(
        source(), batch_size=2, num_micro_batches=1, lane_ids=(0, 1), max_buffered_segments=4
    )
    iterator = iter(loader)
    with pytest.raises(ValueError, match="lookahead reached max_buffered_segments"):
        next(iterator)
    assert reads == [0, 1, 2, 3]
    assert iterator.peak_buffered_segments == 4 and iterator.buffered_segments == 0


def test_buffer_cap_accounts_for_the_complete_update():
    with pytest.raises(ValueError, match="fit a complete optimizer update"):
        batching([], num_micro_batches=2, max_buffered_segments=3)
    iterator = iter(batching([segment(lane, i) for i in range(4) for lane in (0, 1)], num_micro_batches=4))
    assert len(next(iterator)) == 4
    assert iterator.peak_buffered_segments == 8 and iterator.buffered_segments == 0


@pytest.mark.parametrize(
    "raws,message",
    [
        ([segment(2, 0)], "unknown lane"),
        ([segment(0, 1)], "FIFO"),
        ([segment(0, 0), segment(0, 2)], "FIFO"),
        ([segment(0, 0), segment(0, 1, start=96)], "FIFO"),
        ([segment(0, 0), segment(0, 0, stream=("bad reset",))], "reset"),
        ([segment(0, 0, final=True), segment(0, 1)], "FIFO"),
    ],
)
def test_lane_stream_discontinuities_fail_before_model_execution(raws, message):
    with pytest.raises(ValueError, match=message):
        next(iter(batching(raws)))


@pytest.mark.parametrize("item", [[{"segment": segment(0, 0)}, {"segment": segment(1, 0)}], {"samples": []}])
def test_inner_loader_must_produce_singletons(item):
    loader = EpisodeBatchDataLoader([item], batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    with pytest.raises(ValueError, match="singleton"):
        next(iter(loader))


def test_close_epoch_and_restart_release_owned_iterator_without_snapshotting_payloads():
    class Source:
        def __init__(self):
            self.closed = False
            self.epochs = []

        def __iter__(self):
            self.closed = False
            try:
                yield from singleton([segment(0, 0), segment(1, 0)])
            finally:
                self.closed = True

        def state_dict(self):
            raise AssertionError("The raw loader's tensor snapshot must not be requested")

        def load_state_dict(self, state):
            raise AssertionError("The raw loader's cursor must not be restored")

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    source = Source()
    loader = EpisodeBatchDataLoader(source, batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    iterator = iter(loader)
    next(iterator)
    assert loader.state_dict() == {"data_resume_mode": "restart"} and not source.closed
    loader.load_state_dict({"raw_buffer": "must not restore"})
    assert source.closed and iterator.closed
    next(iter(loader))
    loader.set_epoch(3)
    assert source.closed and source.epochs == [3]
    loader.close()


def singleton_segment_batch(features):
    assert len(features) == 1
    return [{"segment": features[0]}]


@pytest.fixture
def roots(tmp_path, request):
    return fixtures.roots.__wrapped__(tmp_path, request)


@pytest.mark.parametrize("roots", [(129, 129)], indirect=True)
def test_real_cpu_workers_form_independent_lanes_and_release_processes(roots):
    dataset = fixtures.stream(StatefulConcatDataset([fixtures.action_dataset(root) for root in roots]))
    expected = {(raw["subset"], raw["episode"], raw["start_tick"], raw["end_tick"]) for raw in dataset}
    native = DataLoader(dataset, batch_size=1, collate_fn=singleton_segment_batch, num_workers=2)
    loader = EpisodeBatchDataLoader(
        native, batch_size=2, num_micro_batches=1, lane_ids=active_episode_lane_ids(dataset, num_workers=2)
    )
    iterator = iter(loader)
    workers = tuple(iterator.source._workers)
    updates = list(iterator)
    actual = flatten(updates)
    assert {(raw["subset"], raw["episode"], raw["start_tick"], raw["end_tick"]) for raw in actual} == expected
    by_lane = defaultdict(list)
    for update in updates:
        for micro in update:
            assert len({raw["lane_id"] for raw in micro["segments"]}) == 2
            for raw in micro["segments"]:
                by_lane[raw["lane_id"]].append(raw["segment_id"])
    assert dict(by_lane) == {0: [0, 1], 1: [0, 1]}
    assert iterator.peak_buffered_segments <= loader.max_buffered_segments
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("roots", [(129, 129)], indirect=True)
def test_early_close_stops_repeating_cpu_workers_and_allows_clean_restart(roots):
    dataset = fixtures.stream(StatefulConcatDataset([fixtures.action_dataset(root) for root in roots]), repeat=True)
    native = DataLoader(dataset, batch_size=1, collate_fn=singleton_segment_batch, num_workers=2)
    loader = EpisodeBatchDataLoader(native, batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    for _ in range(2):
        iterator = iter(loader)
        workers = tuple(iterator.source._workers)
        update = next(iterator)
        assert all(raw["reset"] for raw in update[0]["segments"])
        loader.close()
        assert iterator.closed and iterator.buffered_segments == 0
        assert iterator.source is None
        assert all(not worker.is_alive() for worker in workers)


def test_close_does_not_detach_a_newer_reader_cached_by_the_source():
    class CachedSource:
        def __iter__(self):
            self._iterator = iter(singleton([segment(0, 0), segment(1, 0)]))
            return self._iterator

    source = CachedSource()
    loader = EpisodeBatchDataLoader(source, batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    old = iter(loader)
    replacement = iter(source)
    loader.close()
    assert old.closed and source._iterator is replacement
    assert next(replacement)[0]["segment"]["reset"]


@pytest.mark.parametrize("roots", [(129, 129)], indirect=True)
def test_native_stateful_close_releases_owned_iterator_and_buffered_raw_payloads(roots, monkeypatch):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from veomni.data import data_loader as native_module

    dataset = fixtures.stream(StatefulConcatDataset([fixtures.action_dataset(root) for root in roots]), repeat=True)
    monkeypatch.setattr(native_module, "get_parallel_state", lambda: SimpleNamespace(dp_size=1, dp_rank=0, sp_size=1))
    monkeypatch.setattr(native_module, "get_device_type", lambda: "cpu")
    native = native_module.build_native_dataloader(
        dataset=EpisodeLoaderItems(dataset),
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=54000,
        train_steps=3,
        dyn_bsz=False,
        collate_fn=collate_episode_segment,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=False,
        in_order=True,
    )
    assert isinstance(native, StatefulDataLoader)
    loader = EpisodeBatchDataLoader(native, batch_size=2, num_micro_batches=1, lane_ids=(0, 1))
    for _ in range(2):
        iterator = iter(loader)
        owned = iterator.source
        workers = tuple(owned._workers)
        owned_ref = weakref.ref(owned)
        update = next(iterator)
        assert all(raw["reset"] and raw["segment_id"] == 0 for raw in update[0]["segments"])
        assert native._iterator is owned
        # Exercise the real iterator's retained out-of-order payload storage
        # deterministically, independent of worker scheduling speed.
        payload = torch.ones(8, 8)
        payload_ref = weakref.ref(payload)
        owned._task_info[max(owned._task_info, default=0) + 100] = (0, {"synthetic_raw_pixels": payload})
        del payload, owned
        try:
            loader.close()
            assert native._iterator is None
            assert iterator.source is None and iterator.closed
            assert all(not worker.is_alive() for worker in workers)
            gc.collect()
            assert owned_ref() is None and payload_ref() is None
        finally:
            loader.close()
