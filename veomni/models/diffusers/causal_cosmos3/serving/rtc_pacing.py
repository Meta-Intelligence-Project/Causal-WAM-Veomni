# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only online latency fitting for ordinary and RTC execution pacing.

History is measured on the checkpoint's action-time grid, not in wall seconds.
The session supplies its bounded buffer of completed (history, latency) samples.
An online prediction is an estimate: runtime deadline checks remain necessary.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real


def _finite(name, value, *, minimum=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < minimum or (positive and value <= minimum):
        raise ValueError(f"{name} must be {'greater than' if positive else 'at least'} {minimum}")
    return value


def fit_latency(history_seconds, *, samples=(), observed_seconds=0.0):
    """Fit completed ordinary or RTC requests without retaining fit state.

    History is the encoded model context in model seconds, including padding,
    rather than episode age or execution wall time. With only one distinct
    history length, use a constant prediction instead of inventing a slope.
    ``observed_seconds`` may supply a pending request's elapsed lower bound;
    it does not add a completed sample or change ``maximum_seconds``.
    """
    history = _finite("history_seconds", history_seconds)
    observed = _finite("observed_seconds", observed_seconds)
    points = []
    try:
        records = iter(samples)
    except TypeError as exc:
        raise ValueError("samples must contain (history_seconds, wall_seconds) pairs") from exc
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise ValueError("samples must contain (history_seconds, wall_seconds) pairs")
        points.append((_finite("sample history", record[0]), _finite("sample latency", record[1])))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    maximum = max(ys, default=0.0)
    estimate = dict(
        fit_mode="constant" if max(maximum, observed) > 0 else "unavailable",
        sample_count=len(points),
        distinct_history_count=len(set(xs)),
        history_min_seconds=min(xs, default=None),
        history_max_seconds=max(xs, default=None),
        intercept_seconds=max(maximum, observed),
        slope_seconds_per_history_second=0.0,
        predicted_seconds=max(maximum, observed),
        residual_seconds=0.0,
        maximum_seconds=maximum,
    )
    if estimate["distinct_history_count"] > 1:
        # Translate before centering so large time indexes retain small gaps.
        origin = xs[0]
        offsets = [x - origin for x in xs]
        mean_x = math.fsum(offsets) / len(points)
        mean_y = math.fsum(ys) / len(points)
        centered_x = [x - mean_x for x in offsets]
        variance = math.fsum(x * x for x in centered_x)
        if variance > 0:
            covariance = math.fsum(x * (y - mean_y) for x, y in zip(centered_x, ys, strict=True))
            slope = max(0.0, covariance / variance)
            fitted = [mean_y + slope * x for x in centered_x]
            predicted = max(0.0, mean_y + slope * ((history - origin) - mean_x))
            residual = max(0.0, max(y - fit for y, fit in zip(ys, fitted, strict=True)))
            estimate.update(
                fit_mode="linear",
                intercept_seconds=mean_y - slope * (origin + mean_x),
                slope_seconds_per_history_second=slope,
                predicted_seconds=predicted,
                residual_seconds=residual,
            )
    return estimate


@dataclass(frozen=True)
class RTCExecutionAdvice:
    recommended_hz: float
    budget_seconds: float | None
    delay_steps: int | None
    feasible: bool
    history_seconds: float
    reason: str | None = None
    fit_mode: str = "unavailable"
    sample_count: int = 0
    distinct_history_count: int = 0
    history_min_seconds: float | None = None
    history_max_seconds: float | None = None
    intercept_seconds: float = 0.0
    slope_seconds_per_history_second: float = 0.0
    predicted_seconds: float = 0.0
    residual_seconds: float = 0.0
    maximum_seconds: float = 0.0


@dataclass(frozen=True)
class RTCExecutionPolicy:
    """Fit a local line from completed requests and only slow execution.

    Two distinct historical lengths permit CPU ordinary least squares fitting
    ``L(T) = intercept + slope*T``. The slope is clamped nonnegative. Until then,
    the largest measured latency is used as a constant prediction. There are no
    pretrained coefficients or assumptions about the device's inference speed.

    The budget is ``max(prediction + positive residual, sample maximum,
    observed_seconds) * latency_factor + extra_latency_seconds``. These margins
    are empirical headroom, not a proven worst-case execution time.
    """

    latency_factor: float = 1.2
    extra_latency_seconds: float = 0.0
    margin_steps: int = 1
    minimum_execution_hz: float = 0.0

    def __post_init__(self):
        _finite("latency_factor", self.latency_factor, minimum=1.0)
        _finite("extra_latency_seconds", self.extra_latency_seconds)
        _finite("minimum_execution_hz", self.minimum_execution_hz)
        if isinstance(self.margin_steps, bool) or not isinstance(self.margin_steps, Integral) or self.margin_steps < 1:
            raise ValueError("margin_steps must be a positive integer")
        object.__setattr__(self, "margin_steps", int(self.margin_steps))

    def recommend(self, history_seconds, *, horizon, execution_steps, current_hz, samples=(), observed_seconds=0.0):
        """Predict the next request from recent paired measurements on the CPU.

        ``execution_steps`` is s, the number of actions consumed before asking
        for a new chunk. Feasibility requires d <= min(s, H-s). Samples contain
        only completed (history_model_seconds, latency_wall_seconds) requests.
        The session owns the bounded buffer. No estimate is fabricated before
        the first positive wall-time sample.
        """
        history = _finite("history_seconds", history_seconds)
        current_hz = _finite("current_hz", current_hz, positive=True)
        observed = _finite("observed_seconds", observed_seconds)
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 2:
            raise ValueError("horizon must be an integer of at least two")
        if (
            isinstance(execution_steps, bool)
            or not isinstance(execution_steps, Integral)
            or not 0 < execution_steps < horizon
        ):
            raise ValueError("execution_steps must lie strictly between zero and horizon")
        estimate = fit_latency(history, samples=samples, observed_seconds=observed)
        maximum = estimate["maximum_seconds"]
        if max(maximum, observed) <= 0:
            return RTCExecutionAdvice(current_hz, None, None, False, history, "no_latency_samples", **estimate)
        budget = (
            max(estimate["predicted_seconds"] + estimate["residual_seconds"], maximum, observed) * self.latency_factor
            + self.extra_latency_seconds
        )
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("latency budget must be positive and finite")
        capacity = min(execution_steps, horizon - execution_steps)
        if capacity <= self.margin_steps:
            return RTCExecutionAdvice(current_hz, budget, None, False, history, "margin_exhausts_prefix", **estimate)
        limit = (capacity - self.margin_steps) / budget
        rate = min(current_hz, limit)
        if rate <= 0 or rate < self.minimum_execution_hz:
            return RTCExecutionAdvice(current_hz, budget, None, False, history, "minimum_execution_hz", **estimate)
        delay = math.ceil(budget * rate) + self.margin_steps
        # Keep the integer inequality valid even at floating-point boundaries.
        while delay > capacity:
            rate = math.nextafter(rate, 0.0)
            delay = math.ceil(budget * rate) + self.margin_steps
        if rate < self.minimum_execution_hz:
            return RTCExecutionAdvice(current_hz, budget, None, False, history, "minimum_execution_hz", **estimate)
        return RTCExecutionAdvice(rate, budget, delay, True, history, **estimate)

    def recommend_handoff(
        self,
        history_seconds,
        *,
        overlap_steps,
        current_hz,
        samples=(),
        observed_seconds=0.0,
        elapsed_seconds=0.0,
        remaining_steps=None,
    ):
        """Slow execution for a fixed ``D=R-S`` handoff without changing D.

        Before submission, reserve ``margin_steps`` out of D. During a pending
        request, pass the unconsumed steps to handoff and elapsed wall time.
        The initial margin is not subtracted again on every controller tick:
        even the last old command leaves one execution interval before R.

        Elapsed time is only a lower bound on this request's duration. Including
        it in the empirical latency budget can extend the estimate; it cannot
        establish that inference will finish before a future deadline. An
        exhausted estimate is reported explicitly for the caller to handle.
        A slower rate must be acknowledged by the actuator's execution adapter
        before the caller installs it. Model FPS and action indexes stay fixed.
        """
        history = _finite("history_seconds", history_seconds)
        current_hz = _finite("current_hz", current_hz, positive=True)
        observed = _finite("observed_seconds", observed_seconds)
        elapsed = _finite("elapsed_seconds", elapsed_seconds)
        if isinstance(overlap_steps, bool) or not isinstance(overlap_steps, Integral) or overlap_steps < 1:
            raise ValueError("overlap_steps must be a positive integer")
        overlap = int(overlap_steps)
        pending = remaining_steps is not None
        if pending:
            if (
                isinstance(remaining_steps, bool)
                or not isinstance(remaining_steps, Integral)
                or not 0 <= remaining_steps <= overlap
            ):
                raise ValueError("remaining_steps must be an integer between zero and overlap_steps")
            capacity = int(remaining_steps)
        else:
            if elapsed:
                raise ValueError("elapsed_seconds requires remaining_steps for a pending request")
            capacity = overlap - self.margin_steps

        observed = max(observed, elapsed)
        estimate = fit_latency(history, samples=samples, observed_seconds=observed)
        maximum = estimate["maximum_seconds"]
        if max(maximum, observed) <= 0:
            return RTCExecutionAdvice(current_hz, None, overlap, False, history, "no_latency_samples", **estimate)
        budget = (
            max(estimate["predicted_seconds"] + estimate["residual_seconds"], maximum, observed) * self.latency_factor
            + self.extra_latency_seconds
        )
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("latency budget must be positive and finite")
        if capacity <= 0:
            reason = "no_remaining_steps" if pending else "margin_exhausts_prefix"
            return RTCExecutionAdvice(current_hz, budget, overlap, False, history, reason, **estimate)
        remaining_seconds = budget - elapsed
        if remaining_seconds <= 0:
            return RTCExecutionAdvice(
                current_hz, budget, overlap, False, history, "latency_estimate_exhausted", **estimate
            )
        rate = min(current_hz, capacity / remaining_seconds)
        if rate <= 0 or rate < self.minimum_execution_hz:
            return RTCExecutionAdvice(current_hz, budget, overlap, False, history, "minimum_execution_hz", **estimate)
        # Preserve the deadline inequality at floating-point division boundaries.
        while remaining_seconds * rate > capacity:
            rate = math.nextafter(rate, 0.0)
        if rate <= 0 or rate < self.minimum_execution_hz:
            return RTCExecutionAdvice(current_hz, budget, overlap, False, history, "minimum_execution_hz", **estimate)
        return RTCExecutionAdvice(rate, budget, overlap, True, history, **estimate)
