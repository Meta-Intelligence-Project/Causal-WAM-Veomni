# SPDX-License-Identifier: OpenMDW-1.1
"""Wall-clock calibration must include device completion and ignore paused ticks."""

import json

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.causal_cosmos3.serving import rtc_calibration
from veomni.models.diffusers.causal_cosmos3.serving.rtc_calibration import measure_rtc_latency


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_synchronized_full_requests_use_slowest_sample_after_warmup():
    events = []
    ticks = 17  # Startup calibration does not run the controller clock.
    durations = iter([8.0, 0.125, 0.25])
    now = 0.0
    pending = None

    def synchronize():
        nonlocal now, pending
        events.append("sync")
        if pending is not None:
            now += pending
            pending = None

    def run():
        nonlocal pending
        events.append("run")
        pending = next(durations)
        return object()

    def clock():
        events.append("clock")
        return now

    result = measure_rtc_latency(run, controller_hz=8, synchronize=synchronize, clock=clock)
    assert result == {
        "controller_hz": 8.0,
        "warmup_seconds": [8.0],
        "sample_seconds": [0.125, 0.25],
        "max_seconds": 0.25,
        "margin_steps": 1,
        "delay_steps": 3,
    }
    assert ticks == 17
    assert events == ["sync", "clock", "run", "sync", "clock"] * 3
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("seconds,expected", [(0, 1), (0.25, 2), (0.2501, 3)])
def test_step_rounding_and_zero_latency_floor(seconds, expected):
    timestamps = iter([0.0, seconds])
    result = measure_rtc_latency(
        lambda: None,
        controller_hz=8,
        warmups=0,
        samples=1,
        margin_steps=0,
        clock=lambda: next(timestamps),
    )
    assert result["warmup_seconds"] == []
    assert result["delay_steps"] == expected


def test_default_clock_is_resolved_when_called(monkeypatch):
    timestamps = iter([0.0, 0.125])
    monkeypatch.setattr(rtc_calibration.time, "perf_counter", lambda: next(timestamps))
    result = measure_rtc_latency(lambda: None, controller_hz=8, warmups=0, samples=1)
    assert result["max_seconds"] == 0.125
    assert result["delay_steps"] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"controller_hz": 0},
        {"controller_hz": float("nan")},
        {"controller_hz": float("inf")},
        {"controller_hz": True},
        {"warmups": -1},
        {"warmups": True},
        {"samples": 0},
        {"samples": 1.5},
        {"margin_steps": -1},
        {"margin_steps": True},
        {"synchronize": 1},
        {"clock": 1},
    ],
)
def test_invalid_configuration_is_rejected_before_running(options):
    def unexpected():
        pytest.fail("Invalid calibration configuration must not launch inference")

    with pytest.raises(ValueError):
        measure_rtc_latency(unexpected, **({"controller_hz": 30} | options))


def test_non_callable_request_is_rejected():
    with pytest.raises(ValueError, match="run must be callable"):
        measure_rtc_latency(None, controller_hz=30)


def test_inference_error_propagates_and_device_is_synchronized():
    events = []
    failure = RuntimeError("request failed")

    def run():
        events.append("run")
        raise failure

    def synchronize():
        events.append("sync")

    with pytest.raises(RuntimeError) as exc:
        measure_rtc_latency(run, controller_hz=30, synchronize=synchronize)
    assert exc.value is failure
    assert events == ["sync", "run", "sync"]


def test_device_completion_failure_propagates():
    calls = 0

    def synchronize():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("device completion failed")

    with pytest.raises(RuntimeError, match="device completion failed"):
        measure_rtc_latency(lambda: None, controller_hz=30, synchronize=synchronize)


@pytest.mark.parametrize("timestamps", [[1.0, 0.0], [0.0, float("inf")], [False, 0.0]])
def test_invalid_clock_readings_do_not_produce_a_budget(timestamps):
    timestamps = iter(timestamps)
    with pytest.raises(ValueError, match="clock"):
        measure_rtc_latency(lambda: None, controller_hz=30, warmups=0, samples=1, clock=lambda: next(timestamps))
