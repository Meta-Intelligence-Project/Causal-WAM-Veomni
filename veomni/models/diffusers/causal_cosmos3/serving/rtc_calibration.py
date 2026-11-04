# SPDX-License-Identifier: OpenMDW-1.1
"""Measure complete RTC requests before starting the external control clock."""

import math
import time
from numbers import Real


def _count(value, name, *, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _timestamp(clock):
    value = clock()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("clock must return finite real timestamps")
    return float(value)


def measure_rtc_latency(
    run,
    *,
    controller_hz,
    warmups=1,
    samples=2,
    margin_steps=1,
    synchronize=None,
    clock=None,
):
    """Time full requests and reserve the slowest measured request plus a margin.

    ``run`` must build the observed batch, perform full guided inference and
    export the physical actions. Its result is discarded. Supply device
    synchronization when callbacks can launch asynchronous accelerator work.
    The control clock must not consume commands during startup calibration;
    its tick count is deliberately not used to estimate wall-clock latency.

    Warmup timings are retained for diagnostics but excluded from the budget.
    The returned dictionary contains only JSON-compatible Python values. This
    function measures a request; the caller checks whether its budget fits the
    available action horizon and decides whether RTC can be enabled.
    """
    if not callable(run):
        raise ValueError("run must be callable")
    if (
        isinstance(controller_hz, bool)
        or not isinstance(controller_hz, Real)
        or not math.isfinite(controller_hz)
        or controller_hz <= 0
    ):
        raise ValueError("controller_hz must be finite and positive")
    controller_hz = float(controller_hz)
    warmups = _count(warmups, "warmups", minimum=0)
    samples = _count(samples, "samples", minimum=1)
    margin_steps = _count(margin_steps, "margin_steps", minimum=0)
    if synchronize is not None and not callable(synchronize):
        raise ValueError("synchronize must be callable or None")
    if clock is not None and not callable(clock):
        raise ValueError("clock must be callable or None")
    # Resolve at call time so the actual timer can be replaced by test clocks.
    clock = time.perf_counter if clock is None else clock

    def measure():
        if synchronize is not None:
            synchronize()
        started = _timestamp(clock)
        try:
            run()
        finally:
            if synchronize is not None:
                synchronize()
        seconds = _timestamp(clock) - started
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("clock must measure a finite nonnegative request duration")
        return seconds

    warmup_seconds = [measure() for _ in range(warmups)]
    sample_seconds = [measure() for _ in range(samples)]
    max_seconds = max(sample_seconds)
    raw_steps = max_seconds * controller_hz
    if not math.isfinite(raw_steps):
        raise ValueError("Measured latency times controller_hz must be finite")
    return {
        "controller_hz": controller_hz,
        "warmup_seconds": warmup_seconds,
        "sample_seconds": sample_seconds,
        "max_seconds": max_seconds,
        "margin_steps": margin_steps,
        "delay_steps": max(1, math.ceil(raw_steps) + margin_steps),
    }
