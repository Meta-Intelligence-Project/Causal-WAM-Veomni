"""CPU-only checks for event boundaries, units and rollout aggregation."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def load_profiler():
    clock = SimpleNamespace(host=0.0, device=0.0, synced=True, records=[])
    stream = object()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, selected):
            assert selected is stream
            self.timestamp = clock.device
            clock.synced = False
            clock.records.append(self)

        def elapsed_time(self, end):
            assert clock.synced
            return (end.timestamp - self.timestamp) * 1000

    def synchronize(device):
        assert device.type == "npu"
        clock.synced = True

    backend = SimpleNamespace(
        device=lambda device: nullcontext(),
        current_stream=lambda device: stream,
        Event=Event,
        synchronize=synchronize,
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 0,
        max_memory_reserved=lambda device: 0,
    )
    torch = SimpleNamespace(
        device=lambda device: SimpleNamespace(type=device),
        npu=backend,
        profiler=SimpleNamespace(record_function=lambda name: nullcontext()),
    )
    path = Path(__file__).parents[3] / "veomni/models/diffusers/causal_cosmos3/rollout_profiler.py"
    spec = importlib.util.spec_from_file_location("cosmos3_timing", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, torch=torch):
        spec.loader.exec_module(module)
    module.time = SimpleNamespace(perf_counter=lambda: clock.host)
    return module, clock


@pytest.mark.parametrize("traced", [False, True])
def test_denoise_timer_boundaries(traced):
    module, clock = load_profiler()
    timer = module.DenoiseTimer("npu", trace_factory=(lambda: nullcontext()) if traced else None)
    with timer.stage("noise_initialization", sampling=True):
        clock.host += 5
        clock.device += 5
    with timer.stage("denoising", sampling=True):
        clock.host += 0.03
        clock.device += 0.02
    with timer.stage("result_assembly", sampling=True):
        clock.host += 7
    assert timer.seconds == pytest.approx(0.03)
    assert timer.event_seconds == pytest.approx(0.02)
    assert len(clock.records) == 2


def test_rollout_resets_and_aggregates_events():
    module, clock = load_profiler()
    profiler = module.RolloutProfiler("npu")
    for index in range(2):
        profiler.start_chunk()
        with profiler.stage("vae_encode"):
            clock.host += 1
            clock.device += 1
        with profiler.stage("denoising", sampling=True):
            clock.host += 0.03
            clock.device += 0.02
        profiler.finish_chunk(index)
    assert len(clock.records) == 4
    assert profiler.summary()["total_denoise_seconds"] == pytest.approx(0.06)
    assert profiler.summary()["total_denoise_event_seconds"] == pytest.approx(0.04)
    assert [row["denoise_event_seconds"] for row in profiler.records] == pytest.approx([0.02, 0.02])


def test_cpu_event_is_unavailable():
    module, clock = load_profiler()
    timer = module.DenoiseTimer("cpu")
    with timer.stage("denoising", sampling=True):
        clock.host += 0.03
    assert timer.seconds == pytest.approx(0.03)
    assert timer.event_seconds is None
    assert not clock.records


@pytest.mark.parametrize("enabled", [False, True])
def test_details_are_opt_in_and_measure_both_clocks(enabled):
    module, clock = load_profiler()
    timer = module.DenoiseTimer("npu", sampling_details=enabled)
    with timer.stage("denoising", sampling=True):
        for step in range(2):
            with timer.detail_stage("model_forward", step=step, branch=True):
                clock.host += 0.03
                clock.device += 0.02
    assert timer.seconds == pytest.approx(0.06)
    assert timer.event_seconds == pytest.approx(0.04)
    if enabled:
        row = timer.detail_summary()["model_forward"]
        assert row["count"] == 2
        assert row["wall_seconds"] == pytest.approx(0.06)
        assert row["event_seconds"] == pytest.approx(0.04)
        assert row["host_call_seconds"] + row["final_wait_seconds"] == pytest.approx(row["wall_seconds"])
        assert [record["step"] for record in timer.detail_records] == [0, 1]
    else:
        assert timer.detail_summary() == {}
        assert len(clock.records) == 2


def test_sampling_timing_is_distinct_from_denoising():
    module, clock = load_profiler()
    profiler = module.RolloutProfiler("npu", sampling_details=True)
    profiler.start_chunk()
    with profiler.stage("sampling"):
        with profiler.stage("noise_initialization", sampling=True):
            clock.host += 0.1
            clock.device += 0.05
        with profiler.stage("denoising", sampling=True):
            with profiler.detail_stage("model_forward", step=0):
                clock.host += 0.3
                clock.device += 0.2
    record = profiler.finish_chunk(0)
    assert record["sampling_timing"]["sampling"]["wall_seconds"] == pytest.approx(0.4)
    assert record["sampling_timing"]["denoising"]["wall_seconds"] == pytest.approx(0.3)
    assert profiler.summary()["total_sampling_event_seconds"] == pytest.approx(0.25)
    assert record["sampling_details"]["model_forward"]["count"] == 1
    profiler.start_chunk()
    assert profiler.detail_records == []
    assert profiler.stage_timings == {}


def test_detail_separates_host_call_from_final_wait():
    module, clock = load_profiler()
    original = module.torch.npu.synchronize

    def synchronize(device):
        if not clock.synced:
            clock.host += 0.02
        original(device)

    module.torch.npu.synchronize = synchronize
    timer = module.DenoiseTimer("npu", sampling_details=True)
    with timer.detail_stage("model_forward", step=0):
        clock.host += 0.03
        clock.device += 0.02
    row = timer.detail_records[0]
    assert row["host_call_seconds"] == pytest.approx(0.03)
    assert row["final_wait_seconds"] == pytest.approx(0.02)
    assert row["wall_seconds"] == pytest.approx(0.05)
