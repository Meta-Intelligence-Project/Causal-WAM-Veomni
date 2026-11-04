# SPDX-License-Identifier: OpenMDW-1.1
"""Asynchronous, tick-aligned execution buffers for training-free RTC.

The caller owns the control clock and actuators. Each ``get_action`` call supplies
the measurement immediately before the next command is consumed. Planning runs
on one background worker while that clock continues consuming the previous plan.
Model adapters can measure startup latency and fall back to blocking ordinary
chunk inference. This pauses command delivery; it does not maintain RTC timing.
Commands are never replayed to conceal a deadline miss.
"""

import math
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from numbers import Real
from threading import RLock, get_ident
from time import perf_counter

import torch

from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix
from veomni.models.diffusers.causal_cosmos3.serving.rtc_calibration import measure_rtc_latency
from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy


class RTCError(RuntimeError):
    """A controller error that requires the execution adapter to stop or reset."""


class RTCBufferUnderrun(RTCError):
    """No unconsumed action remains in the current plan."""


class RTCDeadlineMiss(RTCError):
    """Inference consumed more ticks than the committed prefix reserves."""


class RTCPlannerError(RTCError):
    """The background planner failed or returned an invalid action chunk."""


class RTCFallbackWarning(RuntimeWarning):
    """RTC cannot fit its window; ordinary chunk inference will block instead."""


@dataclass(frozen=True)
class RTCObservation:
    """One real image sampled at an absolute controller tick within an episode."""

    tick: int
    image: torch.Tensor


@dataclass(frozen=True)
class _Request:
    revision: int
    tick: int
    chunk_start_tick: int
    delay_steps: int
    inputs: dict
    prefix: torch.Tensor
    action_mask: torch.Tensor
    submitted_at: float
    execution_hz: float
    history_seconds: float


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _frequency(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must be a single positive frequency")
        value = value.detach().cpu().double().item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a single positive frequency")
    return float(value)


def _validate_batch_frequency(batch, batch_builder, metadata, controller_hz):
    action_hz = _frequency(batch.get("conditioning_fps_action"), "conditioning_fps_action")
    if not math.isclose(action_hz, controller_hz, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("conditioning_fps_action must match RTC controller_hz")
    geometry = getattr(batch_builder, "geometry", None)
    stats = getattr(metadata, "statistics", None)
    provenance = getattr(stats, "provenance", {})
    statistics_geometry = provenance.get("geometry", {})
    candidates = (
        getattr(batch_builder, "video_stride", None),
        geometry.get("video_stride") if isinstance(geometry, dict) else getattr(geometry, "video_stride", None),
        batch.get("video_stride"),
        statistics_geometry.get("video_stride"),
    )
    strides = []
    for candidate in candidates:
        if candidate is not None:
            stride = _frequency(candidate, "video_stride")
            if not stride.is_integer():
                raise ValueError("video_stride must be a positive integer")
            strides.append(int(stride))
    if strides:
        if len(set(strides)) != 1:
            raise ValueError("RTC video_stride declarations disagree")
        video_hz = _frequency(batch.get("conditioning_fps"), "conditioning_fps")
        if not math.isclose(video_hz * strides[0], action_hz, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("conditioning_fps times video_stride must match conditioning_fps_action")


def make_model_planner(
    model, batch_builder, *, template, horizon, controller_hz=30.0, history_blocks=8, **sampling_options
):
    """Adapt the native decision builder to the local RTC session callable."""
    from .sampling import sample_decision

    if not math.isclose(batch_builder.controller_hz, controller_hz):
        raise ValueError("Builder and controller frequencies must match")
    options = dict(sampling_options)
    if "num_steps" in options:
        options["steps"] = options.pop("num_steps")

    def planner(**inputs):
        device = next(model.parameters()).device
        if device.type != "cpu":
            getattr(torch, device.type).set_device(device)
        values = dict(inputs)
        rtc = values.pop("rtc", None)
        sample = batch_builder(**values)
        _validate_batch_frequency(sample, batch_builder, sample["rtc_metadata"], controller_hz)
        if len(sample["action"]) != horizon:
            raise ValueError("Builder must preserve the prediction horizon")
        generated = sample_decision(model, sample, rtc=rtc, **options)
        metadata = sample["rtc_metadata"]
        physical = metadata.statistics.action.denormalize(generated["action"].cpu(), inputs["action_mask"])
        actions = template.decode_action_delta(physical, metadata.anchors[0], inputs["action_mask"])
        prepared = generated["prepared_prefix"]
        if prepared is not None and prepared.delay_steps:
            actions[: prepared.delay_steps] = prepared.absolute_prefix[: prepared.delay_steps].cpu()
        return actions.detach().float().cpu()

    return planner


class RTCSession:
    """Execute an absolute action buffer while asynchronously replanning.

    ``initial_actions`` is an already generated CPU ``[H, template.width]``
    bootstrap chunk. ``planner`` receives the fresh ``state``, ``state_mask``,
    ``image``, ``action_mask``, previous real-image ``history`` records, absolute
    ``tick``, episode ``revision`` and an ``rtc=RTCPrefix``. It returns the next
    absolute ``[H,D]`` chunk starting at that observation's tick.

    At each control period, call ``get_action`` once, then consume its command.
    Runtime delays count model waypoints consumed during inference. The model
    action-time grid remains ``controller_hz``; the client's wall-clock waypoint
    pace is ``execution_hz`` (equal by default). Startup calibration uses that
    execution rate with synchronized wall time. This class neither sleeps nor
    drives hardware. RTC requires ``d <= s <= H-d``.

    Optional ``execution_policy`` fits recent history/latency pairs on the CPU
    to predict the next request's wall-time budget and only slow consumption.
    Its callback must return True after the client accepts each new rate.
    A rate never changes
    while an RTC request is in flight; its committed prefix remains fixed.

    A result is published only when its actual delay does not exceed its reserved
    prefix. Prefix commands are restored exactly, and already consumed outputs
    are skipped using absolute ticks. When supplied, ``ordinary_planner`` permits
    warning and fallback to blocking non-RTC chunks. Without it, a deadline miss
    remains latched until reset. ``status`` exposes the execution mode and timing.
    """

    def __init__(
        self,
        planner,
        initial_actions,
        *,
        template,
        controller_hz=30.0,
        execution_hz=None,
        min_execution_steps=None,
        initial_delay_steps=1,
        delay_history_size=8,
        history_steps=256,
        beta=5.0,
        ordinary_planner=None,
        synchronize=None,
        calibration_warmups=1,
        calibration_samples=2,
        calibration_margin_steps=1,
        execution_policy=None,
        execution_rate_callback=None,
        on_observation=None,
        on_prepare_window=None,
        fixed_history_blocks=None,
        fixed_history_steps=None,
    ):
        if not callable(planner):
            raise ValueError("planner must be callable")
        self.template = template
        self._actions = self._validate_actions(initial_actions)
        self.horizon = len(self._actions)
        if self.horizon < 2:
            raise ValueError("RTC requires at least two actions per prediction")
        self.min_execution_steps = _positive_int(
            "min_execution_steps", self.horizon // 2 if min_execution_steps is None else min_execution_steps
        )
        self._configured_delay_steps = initial_delay_steps
        self.initial_delay_steps = (
            None if initial_delay_steps is None else _positive_int("initial_delay_steps", initial_delay_steps)
        )
        if self.initial_delay_steps is not None and execution_policy is None:
            self._check_geometry(self.initial_delay_steps, self.min_execution_steps)
        if ordinary_planner is not None and not callable(ordinary_planner):
            raise ValueError("ordinary_planner must be callable")
        if synchronize is not None and not callable(synchronize):
            raise ValueError("synchronize must be callable")
        self._calibration_options = dict(
            warmups=calibration_warmups, samples=calibration_samples, margin_steps=calibration_margin_steps
        )
        self.delay_history_size = _positive_int("delay_history_size", delay_history_size)
        self.history_steps = _positive_int("history_steps", history_steps)
        self.fixed_history_blocks = (
            None if fixed_history_blocks is None else _positive_int("fixed_history_blocks", fixed_history_blocks)
        )
        if fixed_history_steps is not None and (type(fixed_history_steps) is not int or fixed_history_steps < 0):
            raise ValueError("fixed_history_steps must be a nonnegative integer")
        self.fixed_history_steps = fixed_history_steps
        for callback in (on_observation, on_prepare_window):
            if callback is not None and not callable(callback):
                raise ValueError("Observation/window callbacks must be callable or None")
        self._on_observation, self._on_prepare_window = on_observation, on_prepare_window
        if isinstance(controller_hz, bool) or not math.isfinite(controller_hz) or controller_hz <= 0:
            raise ValueError("controller_hz must be positive and finite")
        if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        self.controller_hz, self.beta = float(controller_hz), float(beta)
        # Model time remains fixed at controller_hz. Only the external client's
        # waypoint consumption rate changes the wall-clock latency budget.
        self.execution_hz = _frequency(self.controller_hz if execution_hz is None else execution_hz, "execution_hz")
        if execution_policy is not None and not isinstance(execution_policy, RTCExecutionPolicy):
            raise ValueError("execution_policy must be an RTCExecutionPolicy")
        if execution_policy is not None and not callable(execution_rate_callback):
            raise ValueError("Adaptive execution requires an execution_rate_callback that acknowledges client pacing")
        if execution_rate_callback is not None and not callable(execution_rate_callback):
            raise ValueError("execution_rate_callback must be callable or None")
        self.execution_policy = execution_policy
        if execution_policy is not None:
            # An integer prefix length contains no wall-time observation for
            # online fitting. Adaptive sessions always measure startup requests.
            self._configured_delay_steps = self.initial_delay_steps = None
        self._execution_rate_callback = execution_rate_callback
        self._execution_advice = None
        self._execution_rate_changes = 0
        self._adaptive_delay_steps = None
        self.planner = planner
        self._ordinary_planner = ordinary_planner
        self._synchronize = synchronize
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-rtc")
        self._future = self._pending = None
        self._worker_thread = None
        self._closed = self._resetting = False
        self._revision = self._tick = self._chunk_start_tick = 0
        self._action_mask = None
        self._error = self._last_request = None
        self._published_requests = 0
        self._ordinary_requests = 0
        self._prepared = self.initial_delay_steps is not None
        self._preparing = self._blocking = False
        self._rtc_enabled = True
        self._ordinary_replan_needed = False
        self._fallback_reason = self._calibration = None
        self._delays = deque(
            [] if self.initial_delay_steps is None else [self.initial_delay_steps], maxlen=self.delay_history_size
        )
        self._history = deque(maxlen=self.history_steps)
        self._delay_seconds = deque(maxlen=self.delay_history_size)
        self._latency_samples = deque(maxlen=self.delay_history_size)

    def _validate_actions(self, actions):
        if (
            not isinstance(actions, torch.Tensor)
            or not actions.is_floating_point()
            or actions.ndim != 2
            or actions.shape[1] != self.template.width
            or not len(actions)
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Actions must be a finite floating-point [H,template.width] tensor")
        return actions.detach().float().cpu().clone()

    def _check_geometry(self, delay, execution_steps):
        if not delay <= execution_steps <= self.horizon - delay:
            raise ValueError(f"RTC requires d <= s <= H-d; got d={delay}, s={execution_steps}, H={self.horizon}")

    def _history_seconds(self, tick):
        if self.fixed_history_steps is not None:
            return self.fixed_history_steps / self.controller_hz
        blocks = self.fixed_history_blocks
        if blocks is None:
            blocks = min(tick // self.horizon, self.history_steps // self.horizon)
        return blocks * self.horizon / self.controller_hz

    def _observe(self, image):
        if self._on_observation is not None:
            self._on_observation(tick=self._tick, image=image, revision=self._revision)

    def _prepare_observation_window(self, execution_steps=None):
        if self._on_prepare_window is not None:
            steps = self.min_execution_steps if execution_steps is None else execution_steps
            anchor = max(self._tick, self._chunk_start_tick + steps)
            self._on_prepare_window(anchor_tick=anchor, revision=self._revision)

    def _update_execution_budget(self, *, history_tick, execution_steps):
        """Apply an acknowledged slower rate before submitting a new request.

        The caller holds the lock and there must be no request in flight. History
        follows complete action blocks on the model time grid. Runtime seconds
        remain valid when the waypoint rate changes; old tick delays do not.
        """
        if self.execution_policy is None or not self._rtc_enabled:
            return
        if self._pending is not None:
            raise RTCError("Cannot change execution pacing while an RTC request is in flight")
        advice = self.execution_policy.recommend(
            self._history_seconds(history_tick),
            horizon=self.horizon,
            execution_steps=execution_steps,
            current_hz=self.execution_hz,
            samples=tuple(self._latency_samples),
            observed_seconds=max(self._delay_seconds, default=0.0),
        )
        self._execution_advice = asdict(advice)
        if not advice.feasible:
            self._reject_execution_budget(f"Adaptive RTC execution budget is infeasible: {advice.reason}")
            return
        if advice.recommended_hz < self.execution_hz:
            # A rate in server metadata cannot pace a robot. The client must
            # commit its future waypoint schedule before this rate is installed.
            try:
                acknowledged = self._execution_rate_callback(advice.recommended_hz)
            except Exception as exc:
                self._reject_execution_budget(f"Client execution-rate acknowledgement failed: {exc}")
                return
            if acknowledged is not True:
                self._reject_execution_budget(
                    f"Client did not acknowledge the required slower waypoint rate {advice.recommended_hz:g} Hz"
                )
                return
            self.execution_hz = advice.recommended_hz
            self._execution_rate_changes += 1
        self._adaptive_delay_steps = advice.delay_steps
        self._delays.clear()
        self._delays.append(advice.delay_steps)

    def _reject_execution_budget(self, reason):
        if self._ordinary_planner is None:
            self._error = RTCDeadlineMiss(reason)
            raise self._error
        self._fallback(reason, replan=False)

    def _raise_state_error(self):
        if self._closed:
            raise RTCError("RTC session is closed")
        if self._resetting:
            raise RTCError("RTC session is resetting")
        if self._error is not None:
            raise self._error

    def _fallback(self, reason, *, replan):
        """Called with the state lock; never publish an expired RTC prediction."""
        if self._ordinary_planner is None:
            raise RTCDeadlineMiss(reason)
        if self._rtc_enabled:
            self._rtc_enabled = False
            self._fallback_reason = reason
            self._ordinary_replan_needed = replan
            warnings.warn(
                f"{reason}; falling back to blocking non-RTC chunk inference. "
                "Command delivery may pause during inference; continuous RTC control timing is not maintained.",
                RTCFallbackWarning,
                stacklevel=3,
            )

    def _planner_inputs(self, state, state_mask, image, action_mask, rtc):
        return dict(
            state=state,
            state_mask=state_mask,
            image=image,
            action_mask=action_mask.clone(),
            history=tuple(self._history),
            tick=self._tick,
            revision=self._revision,
            rtc=rtc,
        )

    def _validated_plan(self, planner, inputs):
        self._worker_thread = get_ident()
        actions = self._validate_actions(planner(**inputs))
        if len(actions) != self.horizon:
            raise ValueError("Planner must preserve the prediction horizon")
        self.template.sanitize(actions, inputs["action_mask"])
        return actions.masked_fill(~inputs["action_mask"], 0)

    def _measure_startup(self, inputs):
        self._worker_thread = get_ident()
        calibration = measure_rtc_latency(
            lambda: self._validated_plan(self.planner, inputs),
            controller_hz=self.execution_hz,
            synchronize=self._synchronize,
            **self._calibration_options,
        )
        calibration.update(
            controller_hz=self.controller_hz,
            execution_hz=self.execution_hz,
            playback_speed=self.execution_hz / self.controller_hz,
        )
        return calibration

    def prepare(self, *, state, state_mask, image, action_mask):
        """Measure full RTC requests before starting the external control clock.

        The first get_action also calls this lazily. Calibration outputs are
        discarded and no command tick advances. Fixed-rate sessions with an
        explicit integer delay retain the manual path. Adaptive sessions always
        measure startup, since an integer budget is not a latency observation.
        """
        with self._lock:
            self._raise_state_error()
            if get_ident() == self._worker_thread:
                raise RTCError("Cannot prepare or consume commands from the planner worker")
            if self._preparing or self._blocking:
                raise RTCError("Startup or blocking inference is already active")
            if self._prepared:
                return self.status()
            if self._tick or self._pending is not None:
                raise RTCError("Automatic calibration must precede command execution")
            state, sm, image, am = self._measurement(state, state_mask, image, action_mask)
            # These predictions start at the current observation. Do not shift
            # future bootstrap rows back in time to invent a later observation.
            rtc = RTCPrefix(self._actions.clone(), 1, self.beta)
            inputs = self._planner_inputs(state, sm, image, am, rtc)
            revision = self._revision
            self._preparing = True
            self._future = self._executor.submit(self._measure_startup, inputs)
            future = self._future
        try:
            calibration = future.result()
            with self._lock:
                self._raise_state_error()
                if revision != self._revision:
                    raise RTCError("Episode changed during startup calibration")
                self._calibration = calibration
                if self.execution_policy is not None:
                    for seconds in calibration["sample_seconds"]:
                        self._delay_seconds.append(seconds)
                        self._latency_samples.append((self._history_seconds(0), seconds))
                    self._update_execution_budget(history_tick=0, execution_steps=self.min_execution_steps)
                    if self._rtc_enabled:
                        calibration.update(
                            delay_steps=self._adaptive_delay_steps,
                            execution_hz=self.execution_hz,
                            playback_speed=self.execution_hz / self.controller_hz,
                            adaptive_execution=dict(self._execution_advice),
                        )
                delay = calibration["delay_steps"]
                self.initial_delay_steps = delay
                self._delays.append(delay)
                if self._rtc_enabled and not delay <= self.min_execution_steps <= self.horizon - delay:
                    self._fallback(
                        f"Measured RTC latency {calibration['max_seconds']:.3f}s needs {delay} control steps "
                        f"including margin at {self.execution_hz:g} waypoints/s; "
                        f"H={self.horizon}, s={self.min_execution_steps} cannot fit this delay",
                        replan=False,
                    )
                self._prepared = True
                self._preparing = False
                return self.status()
        finally:
            with self._lock:
                self._preparing = False

    def _measurement(self, state, state_mask, image, action_mask):
        if not isinstance(state, torch.Tensor) or state.shape != (self.template.width,):
            raise ValueError("state must be a current [template.width] measurement")
        sm = self.template.validate_valid_mask(state_mask).detach().cpu().clone()
        am = self.template.validate_valid_mask(action_mask).detach().cpu().clone()
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Current actions require valid measured anchor channels")
        # The model and physical action buffer use FP32. Canonicalize the fresh
        # measurement before both sanitization and batch-anchor comparison.
        state = self.template.sanitize(state.detach().float().cpu(), sm).clone()
        if not isinstance(image, torch.Tensor) or image.ndim < 2 or not torch.isfinite(image).all():
            raise ValueError("image must be a finite current observation tensor")
        if self._action_mask is None:
            self._action_mask = am.clone()
            # Validate active quaternion groups before any command is exposed.
            self.template.sanitize(self._actions, am)
            self._actions = self._actions.masked_fill(~am, 0)
        elif not torch.equal(am, self._action_mask):
            raise ValueError("Action validity cannot change during an RTC episode; reset first")
        return state, sm, image.detach().cpu().clone(), am

    def get_action(self, *, state, state_mask, image, action_mask):
        """Return a command; startup and ordinary fallback may block.

        The measurement belongs to ``status()['tick']`` immediately before this
        call. Calls must be serialized by the execution adapter. A new planning
        request is started at the first available observation after ``s_min``
        actions of the current plan, while its remaining actions keep executing.
        """
        measurement = dict(state=state, state_mask=state_mask, image=image, action_mask=action_mask)
        self.prepare(**measurement)
        with self._lock:
            self._raise_state_error()
            offset = self._tick - self._chunk_start_tick
            if self._rtc_enabled and not 0 <= offset < self.horizon:
                if self._ordinary_planner is None:
                    self._error = RTCBufferUnderrun("RTC action buffer exhausted before a new plan was available")
                    raise self._error
                self._fallback("RTC action buffer exhausted before the next plan was ready", replan=True)
            if self._rtc_enabled and self._pending is None:
                # Anticipate the next request's history before consuming more of
                # the old plan. The submitted request then keeps its own prefix
                # and rate until publication, even if history grows meanwhile.
                self._update_execution_budget(
                    history_tick=max(self._tick, self._chunk_start_tick + self.min_execution_steps),
                    execution_steps=max(offset, self.min_execution_steps),
                )
            if self._rtc_enabled:
                state, sm, current_image, am = self._measurement(state, state_mask, image, action_mask)
                self._observe(current_image)
                if self._pending is None:
                    self._prepare_observation_window()
                if self._pending is None and offset >= self.min_execution_steps:
                    delay = self._adaptive_delay_steps if self.execution_policy is not None else max(self._delays)
                    try:
                        self._check_geometry(delay, offset)
                    except ValueError as exc:
                        if self._ordinary_planner is None:
                            self._error = RTCDeadlineMiss("Insufficient remaining actions for the next RTC inference")
                            raise self._error from exc
                        self._fallback("Insufficient remaining actions for the next RTC inference", replan=True)
                    if self._rtc_enabled:
                        prefix = self._actions[offset:].clone()
                        inputs = self._planner_inputs(
                            state, sm, current_image, am, RTCPrefix(prefix.clone(), delay, self.beta)
                        )
                        request = _Request(
                            self._revision,
                            self._tick,
                            self._chunk_start_tick,
                            delay,
                            inputs,
                            prefix,
                            am.clone(),
                            perf_counter(),
                            self.execution_hz,
                            self._history_seconds(self._tick),
                        )
                        self._pending = request
                        self._future = self._executor.submit(self._run_request, request)
                if self._rtc_enabled:
                    action = self._actions[offset].clone()
                    self._history.append(RTCObservation(self._tick, current_image))
                    self._tick += 1
                    return action
        return self._get_ordinary_action(**measurement)

    def _get_ordinary_action(self, *, state, state_mask, image, action_mask):
        """Run ordinary chunks at boundaries; no RTC prefix or late output reuse."""
        with self._lock:
            self._raise_state_error()
            if self._blocking:
                raise RTCError("Concurrent blocking command requests are not supported")
            state, sm, image, am = self._measurement(state, state_mask, image, action_mask)
            self._observe(image)
            self._prepare_observation_window(0 if self._ordinary_replan_needed else self.horizon)
            offset = self._tick - self._chunk_start_tick
            if not self._ordinary_replan_needed and offset < self.horizon:
                action = self._actions[offset].clone()
                self._history.append(RTCObservation(self._tick, image))
                self._tick += 1
                return action
            self._blocking = True
            revision, tick = self._revision, self._tick
            previous_future = self._future
            inputs = self._planner_inputs(state, sm, image, am, None)
        try:
            # A late RTC worker may still hold temporary model patches. Finish
            # it before ordinary inference; its guided output is never installed.
            if previous_future is not None:
                previous_future.result()
            with self._lock:
                self._raise_state_error()
                self._future = self._executor.submit(self._validated_plan, self._ordinary_planner, inputs)
                future = self._future
            actions = future.result()
            with self._lock:
                self._raise_state_error()
                if revision != self._revision or tick != self._tick:
                    raise RTCError("Episode or command tick changed during blocking inference")
                self._actions = actions
                self._chunk_start_tick = tick
                self._ordinary_replan_needed = False
                self._ordinary_requests += 1
                self._prepare_observation_window(self.horizon)
                self._history.append(RTCObservation(tick, image))
                self._tick += 1
                return actions[0].clone()
        finally:
            with self._lock:
                self._blocking = False

    def _run_request(self, request):
        self._worker_thread = get_ident()
        started = request.submitted_at
        try:
            inputs = dict(request.inputs)
            inputs["history"] = tuple(RTCObservation(x.tick, x.image.clone()) for x in inputs["history"])
            actions = self._validate_actions(self.planner(**inputs))
            if len(actions) != self.horizon:
                raise ValueError("RTC planner must preserve the prediction horizon")
            self.template.sanitize(actions, request.action_mask)
            actions.masked_fill_(~request.action_mask, 0)
            # These commands are committed while inference runs. Preserve their
            # original absolute values, including normalization roundoff.
            actions[: request.delay_steps] = request.prefix[: request.delay_steps]
            with self._lock:
                if self._closed or request.revision != self._revision:
                    return
                if not self._rtc_enabled:
                    return
                actual_delay = self._tick - request.tick
                elapsed_seconds = perf_counter() - started
                self._last_request = dict(
                    observation_tick=request.tick,
                    previous_chunk_start_tick=request.chunk_start_tick,
                    publish_tick=self._tick,
                    estimated_delay_steps=request.delay_steps,
                    actual_delay_steps=actual_delay,
                    elapsed_seconds=elapsed_seconds,
                    execution_hz=request.execution_hz,
                    history_seconds=request.history_seconds,
                    accepted=False,
                )
                self._delays.append(actual_delay)
                self._delay_seconds.append(elapsed_seconds)
                self._latency_samples.append((request.history_seconds, elapsed_seconds))
                if self._error is not None:
                    return
                if actual_delay > request.delay_steps or actual_delay >= self.horizon:
                    reason = f"RTC inference consumed {actual_delay} actions, exceeding reserved prefix {request.delay_steps}"
                    if self._ordinary_planner is not None:
                        self._fallback(reason, replan=True)
                    else:
                        self._error = RTCDeadlineMiss(reason)
                    return
                if self._chunk_start_tick != request.chunk_start_tick:
                    self._error = RTCPlannerError("RTC plan origin changed while inference was running")
                    return
                self._actions = actions
                self._chunk_start_tick = request.tick
                self._published_requests += 1
                self._last_request["accepted"] = True
                # Publication and releasing this request must be atomic. At the
                # balanced d=s=H/2 boundary, the very next control tick already
                # needs to submit inference for the newly installed chunk.
                if self._pending is request:
                    self._pending = None
                self._prepare_observation_window()
        except Exception as exc:
            with self._lock:
                if not self._closed and request.revision == self._revision and self._error is None:
                    error = RTCPlannerError(f"RTC planner failed: {exc}")
                    error.__cause__ = exc
                    self._error = error
        finally:
            with self._lock:
                if self._pending is request:
                    self._pending = None

    def status(self):
        """Return host-only timing, buffer and lifecycle diagnostics."""
        with self._lock:
            return {
                "tick": self._tick,
                "chunk_start_tick": self._chunk_start_tick,
                "remaining_actions": max(0, self.horizon - (self._tick - self._chunk_start_tick)),
                "prediction_horizon": self.horizon,
                "minimum_execution_steps": self.min_execution_steps,
                "controller_hz": self.controller_hz,
                "execution_hz": self.execution_hz,
                "playback_speed": self.execution_hz / self.controller_hz,
                "adaptive_execution": self.execution_policy is not None,
                "execution_advice": None if self._execution_advice is None else dict(self._execution_advice),
                "execution_rate_changes": self._execution_rate_changes,
                "delay_history_seconds": list(self._delay_seconds),
                "latency_samples": [
                    {"history_seconds": history, "elapsed_seconds": seconds}
                    for history, seconds in self._latency_samples
                ],
                "delay_history_steps": list(self._delays),
                "estimated_delay_steps": max(self._delays) if self._delays else None,
                "rtc_enabled": self._rtc_enabled,
                "execution_mode": "rtc" if self._rtc_enabled else "blocking_non_rtc",
                "prepared": self._prepared,
                "preparing": self._preparing,
                "blocking_inference": self._blocking,
                "ordinary_requests": self._ordinary_requests,
                "fallback_reason": self._fallback_reason,
                "calibration": None if self._calibration is None else dict(self._calibration),
                "inference_pending": self._pending is not None,
                "published_requests": self._published_requests,
                "revision": self._revision,
                "closed": self._closed,
                "resetting": self._resetting,
                "error": None if self._error is None else str(self._error),
                "last_request": None if self._last_request is None else dict(self._last_request),
            }

    def wait_for_inference(self, timeout=None):
        """Join the currently submitted request; timeout does not cancel it."""
        with self._lock:
            if get_ident() == self._worker_thread:
                raise RTCError("Cannot join the RTC worker from its planner")
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        with self._lock:
            if self._error is not None:
                raise self._error
            return self.status()

    def join(self, timeout=None):
        return self.wait_for_inference(timeout)

    def reset(self, initial_actions):
        """Drain any old request and start a new episode from a fresh bootstrap.

        In-flight inference cannot be forcefully interrupted. Its episode version
        is invalidated before joining, so it cannot publish into the new episode.
        """
        actions = self._validate_actions(initial_actions)
        if len(actions) != self.horizon:
            raise ValueError("Reset must preserve the prediction horizon")
        with self._lock:
            if self._closed:
                raise RTCError("RTC session is closed")
            if self._resetting or self._preparing or self._blocking or get_ident() == self._worker_thread:
                raise RTCError("Cannot reset concurrently or from the RTC planner")
            self._resetting = True
            self._revision += 1
            future = self._future
        try:
            if future is not None:
                # The revision above invalidates this job. Wait for cleanup,
                # but do not rethrow a failed old episode while resetting it.
                future.exception()
            with self._lock:
                if self._closed:
                    raise RTCError("RTC session closed during reset")
                self._actions = actions
                self._future = self._pending = None
                self._tick = self._chunk_start_tick = self._published_requests = 0
                self._action_mask = self._error = self._last_request = None
                self.initial_delay_steps = self._configured_delay_steps
                self._prepared = self.initial_delay_steps is not None
                self._rtc_enabled = True
                self._ordinary_replan_needed = False
                self._fallback_reason = self._calibration = None
                self._execution_advice = None
                self._execution_rate_changes = 0
                self._adaptive_delay_steps = None
                self._ordinary_requests = 0
                self._history.clear()
                self._delays.clear()
                self._delay_seconds.clear()
                self._latency_samples.clear()
                if self.initial_delay_steps is not None:
                    self._delays.append(self.initial_delay_steps)
        finally:
            with self._lock:
                self._resetting = False

    def close(self, *, wait=True):
        """Reject new ticks, invalidate old results and release the worker."""
        with self._lock:
            if get_ident() == self._worker_thread:
                raise RTCError("Cannot close the RTC session from its planner")
            self._closed = True
            self._revision += 1
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        with self._lock:
            self._raise_state_error()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def for_model(
        cls,
        model,
        batch_builder,
        initial_actions,
        *,
        template,
        controller_hz=30.0,
        execution_hz=None,
        min_execution_steps=None,
        initial_delay_steps=None,
        delay_history_size=8,
        history_blocks=8,
        beta=5.0,
        calibration_warmups=1,
        calibration_samples=2,
        calibration_margin_steps=1,
        execution_policy=None,
        execution_rate_callback=None,
        **sampling_options,
    ):
        """Adapt the model to RTC without using future states or old anchors.

        ``batch_builder`` receives the planner inputs documented on this class.
        It must end its visual history at the current tick and reconstruct a
        training-compatible window whose current anchor is the fresh measured
        state. Requests can occur partway through the previous prediction, so an
        adapter must not reuse its old fixed block boundary or old state. History
        contains real images only, with tick indices for explicit resampling.
        Its action FPS must equal ``controller_hz``; when a video stride is
        declared, the visual FPS is checked against that same control clock.
        By default prepare/first get_action measures full RTC requests before
        command execution. Infeasible timing warns and selects blocking ordinary
        inference, with no RTC prefix in the model batch.
        ``execution_hz`` changes only latency budgeting; the external client must
        actually pace its waypoints at that rate. It does not change model FPS.
        An optional ``execution_policy`` can reduce that rate as history grows,
        provided ``execution_rate_callback`` acknowledges actual client pacing.
        Fit history seconds use the model clock; measured delays use wall time.
        """
        history_blocks = _positive_int("history_blocks", history_blocks)
        controller_hz = _frequency(controller_hz, "controller_hz")
        horizon = len(initial_actions)
        planner = make_model_planner(
            model,
            batch_builder,
            template=template,
            horizon=horizon,
            controller_hz=controller_hz,
            history_blocks=history_blocks,
            **sampling_options,
        )

        def synchronize():
            device = next(model.parameters()).device
            if device.type != "cpu":
                backend = getattr(torch, device.type)
                backend.set_device(device)
                backend.synchronize(device)

        return cls(
            planner,
            initial_actions,
            template=template,
            controller_hz=controller_hz,
            execution_hz=execution_hz,
            min_execution_steps=min_execution_steps,
            initial_delay_steps=initial_delay_steps,
            delay_history_size=delay_history_size,
            history_steps=max(1, batch_builder.history_steps)
            if getattr(batch_builder, "layout_mode", "block") == "decision_window"
            else history_blocks * horizon,
            fixed_history_steps=batch_builder.history_steps
            if getattr(batch_builder, "layout_mode", "block") == "decision_window"
            else None,
            beta=beta,
            ordinary_planner=planner,
            synchronize=synchronize,
            calibration_warmups=calibration_warmups,
            calibration_samples=calibration_samples,
            calibration_margin_steps=calibration_margin_steps,
            execution_policy=execution_policy,
            execution_rate_callback=execution_rate_callback,
        )
