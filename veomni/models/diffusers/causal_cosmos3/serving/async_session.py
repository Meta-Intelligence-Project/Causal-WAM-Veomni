# SPDX-License-Identifier: OpenMDW-1.1
"""Shared ordinary/RTC prediction with fixed tick-aligned buffer handoffs."""

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import RLock, get_ident
from time import perf_counter

import torch

from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import fit_latency


class AsyncSessionError(RuntimeError):
    """Command delivery must stop until the session is reset or closed."""


class AsyncPlannerError(AsyncSessionError):
    """Background planning, execution pacing or observation transport failed."""


def resolve_async_schedule(H, S, R=None):
    """Return trigger S and handoff R for one H-step prediction and one candidate."""
    if type(H) is not int or type(S) is not int or H < 1 or S < 1:
        raise ValueError("Async H and S must be positive integers")
    if R is None:
        R = min(H, S + max(1, S // 2))
    # H是预测长度，S是提交新推理的时刻，R是交接时刻；三者不可混成执行长度。
    # R<=2S保证交接时不会错过下一次提交（R=2S需要同tick先交接再提交）。
    if type(R) is not int or not 0 < S < R <= min(H, 2 * S):
        raise ValueError("Async schedule requires 0 < S < R <= min(H, 2*S)")
    return S, R


def _frequency(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class _Observation:
    tick: int
    image: torch.Tensor


@dataclass
class _Candidate:
    revision: int
    anchor: int
    handoff_tick: int
    inputs: dict
    history_seconds: float
    future: object = None
    actions: torch.Tensor | None = None
    late: bool = False
    submitted_at: float = 0.0
    prefix: torch.Tensor | None = None


class AsyncRemoteSession:
    """Execute old commands until R, then use the new plan at offset R-S.

    Planning uses a frozen measurement taken at S. Optional RTC guides the new
    solve with the old suffix, committing exactly R-S actions in either mode.
    A result arriving early waits for R. A late result pauses command delivery at
    R without advancing the control tick or replaying an expired command.
    """

    def __init__(
        self,
        client,
        initial_actions,
        *,
        execution_hz=None,
        enable_rtc=False,
        beta=5.0,
        execution_policy=None,
        execution_rate_callback=None,
        calibration_warmups=1,
        calibration_samples=2,
        calibration_margin_steps=1,
        latency_buffer_size=32,
        synchronize=None,
        clock=None,
    ):
        self.client, self.template = client, ActionTemplate55()
        self._actions = self._validate_actions(initial_actions)
        self.horizon = len(self._actions)
        advertised = client.metadata.get("action_horizon", client.metadata.get("actions_per_block", self.horizon))
        if advertised != self.horizon:
            raise ValueError("Initial actions must preserve the server prediction horizon")
        self.steps, self.handoff_steps = resolve_async_schedule(
            self.horizon, client.metadata["execute_steps"], client.metadata.get("handoff_steps")
        )
        self.controller_hz = _frequency(client.metadata["controller_hz"], "controller_hz")
        self.execution_hz = _frequency(self.controller_hz if execution_hz is None else execution_hz, "execution_hz")
        if type(enable_rtc) is not bool:
            raise ValueError("enable_rtc must be a boolean")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be nonnegative and finite")
        self.enable_rtc, self.beta = enable_rtc, float(beta)
        self.delay_steps = self.handoff_steps - self.steps
        if execution_policy is not None and not callable(execution_rate_callback):
            raise ValueError("Adaptive async execution requires execution_rate_callback")
        if execution_policy is not None and not callable(getattr(execution_policy, "recommend_handoff", None)):
            raise ValueError("Execution policy must support fixed handoff budgets")
        for value, name, minimum in (
            (calibration_warmups, "calibration_warmups", 0),
            (calibration_samples, "calibration_samples", 1),
            (calibration_margin_steps, "calibration_margin_steps", 0),
            (latency_buffer_size, "latency_buffer_size", 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if synchronize is not None and not callable(synchronize):
            raise ValueError("synchronize must be callable")
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable")
        self.execution_policy = execution_policy
        self._execution_rate_callback = execution_rate_callback
        self._calibration_options = dict(
            warmups=calibration_warmups,
            samples=calibration_samples,
            margin_steps=calibration_margin_steps,
            synchronize=synchronize,
        )
        self._clock = perf_counter if clock is None else clock
        history_steps = client.metadata.get("history_steps")
        if history_steps is None:
            history_steps = client.metadata.get("history_blocks", 32) * 32
        if type(history_steps) is not int or history_steps < 0:
            raise ValueError("history_steps must be a nonnegative integer")
        self.history_steps = history_steps
        for method in ("plan", "observe", "prepare_window"):
            if not callable(getattr(client, method, None)):
                raise ValueError(f"Async client requires {method}")
        self._history = deque(maxlen=max(1, history_steps))
        self._lock, self._callback_lock, self._prepare_lock = RLock(), RLock(), RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-async-policy")
        self._worker_thread = None
        self._rate_callback_thread = None
        self._pending = self._future = self._consumer = None
        self._tick = self._origin = self._revision = 0
        self._requests = self._published = self._late_handoffs = 0
        self._closed = self._resetting = self._prepared = self._waiting = False
        self._error = self._action_mask = self._state_mask = self._last_hint = None
        self._last_request = None
        # Both ordinary and guided requests retain completed measurements,
        # including when no execution-rate callback/policy is installed.
        self._latency_samples = deque(maxlen=latency_buffer_size)
        self._calibration = self._execution_advice = None
        self._execution_rate_changes = 0

    def _validate_actions(self, actions):
        if (
            not isinstance(actions, torch.Tensor)
            or not actions.is_floating_point()
            or actions.ndim != 2
            or actions.shape[1] != self.template.width
            or not len(actions)
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite floating-point [H,55] targets")
        return actions.detach().float().cpu().clone()

    def _check(self):
        if self._closed:
            raise AsyncSessionError("Async session is closed")
        if self._resetting:
            raise AsyncSessionError("Async session is resetting")
        if self._error is not None:
            raise self._error

    def _live(self, revision):
        return not self._closed and not self._resetting and revision == self._revision

    def _fail(self, exception, revision):
        error = (
            exception
            if isinstance(exception, AsyncPlannerError)
            else AsyncPlannerError(f"Async planning failed: {exception}")
        )
        if error is not exception:
            error.__cause__ = exception
        with self._lock:
            if self._live(revision) and self._error is None:
                self._error = error
        return error

    def _notify(self, method, revision, **kwargs):
        # 客户端方法只入发送队列；不持controller锁调用网络/用户回调。
        # 独立回调锁让reset/close能隔离旧episode的迟到hint。
        with self._callback_lock:
            with self._lock:
                if not self._live(revision):
                    return False
                if method == "prepare_window" and self._last_hint == (revision, kwargs["anchor_tick"]):
                    return True
            try:
                getattr(self.client, method)(revision=revision, **kwargs)
            except Exception as exc:
                error = self._fail(exc, revision)
                if error is exc:
                    raise
                raise error from exc
            with self._lock:
                if method == "prepare_window" and self._live(revision):
                    self._last_hint = revision, kwargs["anchor_tick"]
            return True

    def _measurement(self, *, state, state_mask, image, action_mask):
        if not isinstance(state, torch.Tensor) or state.shape != (self.template.width,):
            raise ValueError("state must be a current [55] measurement")
        sm = self.template.validate_valid_mask(state_mask).detach().cpu().clone()
        am = self.template.validate_valid_mask(action_mask).detach().cpu().clone()
        if not sm.any() or not am.any() or (self.template.required_anchor_mask(am) & ~sm).any():
            raise ValueError("Current actions require valid measured anchor channels")
        state = self.template.sanitize(state.detach().float().cpu(), sm).clone()
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or min(image.shape) < 1
            or not torch.isfinite(image).all()
        ):
            raise ValueError("image must be a finite real [3,H,W] observation")
        return state, sm, image.detach().cpu().clone(), am

    def _accept_masks(self, sm, am):
        if self._action_mask is None:
            self.template.sanitize(self._actions, am)
            self._actions = self._actions.masked_fill(~am, 0)
            self._action_mask, self._state_mask = am.clone(), sm.clone()
        elif not torch.equal(am, self._action_mask) or not torch.equal(sm, self._state_mask):
            raise ValueError("Robot validity masks cannot change within an async episode; reset first")

    def prepare(self, *, _expected_revision=None, **measurement):
        if get_ident() == self._rate_callback_thread:
            raise AsyncSessionError("Cannot prepare during execution-rate acknowledgement")
        with self._lock:
            self._check()
            # Calibration waits while holding _prepare_lock. Reject a worker
            # callback before it can wait for that lock and deadlock its caller.
            if get_ident() == self._worker_thread:
                raise AsyncSessionError("Cannot prepare from the planning worker")
            expected_revision = self._revision if _expected_revision is None else _expected_revision
        with self._prepare_lock:
            with self._lock:
                self._check()
                if expected_revision != self._revision:
                    raise AsyncSessionError("Episode changed before preparing async execution")
            values = self._measurement(**measurement) if measurement else None
            with self._lock:
                self._check()
                if expected_revision != self._revision:
                    raise AsyncSessionError("Episode changed while preparing the async measurement")
                if values is not None:
                    self._accept_masks(values[1], values[3])
                if self._prepared:
                    return self.status()
                revision, anchor = self._revision, self._origin + self.steps
                if self.execution_policy is not None:
                    if values is None:
                        raise ValueError("Adaptive async preparation needs a real current measurement")
                    self._future = future = self._executor.submit(self._calibrate, revision, values)
                else:
                    future = None
            if future is not None:
                future.result()
                self._update_pacing(revision)
            self._notify("prepare_window", revision, anchor_tick=anchor)
            with self._lock:
                self._check()
                if revision != self._revision:
                    raise AsyncSessionError("Episode changed while preparing async execution")
                self._prepared = True
                return self.status()

    def _history_seconds(self):
        # repeat_first pads from the first request to the full encoded window.
        # Episode age and real playback Hz do not change model input length.
        return self.history_steps / self.controller_hz

    def _rtc_prefix(self, actions):
        if not self.enable_rtc:
            return None
        from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix

        return RTCPrefix(actions.clone(), self.delay_steps, self.beta)

    def _calibrate(self, revision, values):
        from veomni.models.diffusers.causal_cosmos3.serving.rtc_calibration import measure_rtc_latency

        state, sm, image, am = values
        with self._lock:
            self._worker_thread = get_ident()
            prefix = self._actions[: self.horizon - self.steps].clone()
            history_seconds = self._history_seconds()

        def run():
            with self._lock:
                if not self._live(revision):
                    raise AsyncSessionError("Episode changed during startup calibration")
            actions = self.client.plan(
                state=state.clone(),
                state_mask=sm.clone(),
                image=image.clone(),
                action_mask=am.clone(),
                history=(),
                tick=0,
                revision=revision,
                rtc=self._rtc_prefix(prefix),
            )
            if len(self._validate_actions(actions)) != self.horizon:
                raise ValueError("Calibration planner must preserve the prediction horizon")

        try:
            measured = measure_rtc_latency(
                run, controller_hz=self.execution_hz, clock=self._clock, **self._calibration_options
            )
            with self._lock:
                if not self._live(revision):
                    raise AsyncSessionError("Episode changed during startup calibration")
                measured["measured_delay_steps"] = measured["delay_steps"]
                measured["delay_steps"] = self.delay_steps
                self._calibration = measured
                for seconds in measured["sample_seconds"]:
                    self._latency_samples.append((history_seconds, seconds))
        except Exception as exc:
            raise self._fail(exc, revision) from exc

    def _update_pacing(self, revision):
        if self.execution_policy is None:
            return
        # 与reset/close使用同一锁序；执行端回调期间不持session锁。
        with self._callback_lock:
            with self._lock:
                self._check()
                if not self._live(revision):
                    raise AsyncSessionError("Episode changed while updating execution pacing")
                offset = self._tick - self._origin
                if offset >= self.handoff_steps:
                    return  # 最后兜底仍在相同R等待，不关闭RTC或重放命令。
                candidate = self._pending
                if candidate is not None and candidate.actions is not None:
                    return
                elapsed, remaining = 0.0, None
                if candidate is not None:
                    elapsed = self._clock() - candidate.submitted_at
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise AsyncSessionError("Execution clock must advance monotonically")
                    remaining = candidate.handoff_tick - self._tick
                advice = self.execution_policy.recommend_handoff(
                    self._history_seconds() if candidate is None else candidate.history_seconds,
                    overlap_steps=self.delay_steps,
                    current_hz=self.execution_hz,
                    samples=tuple(self._latency_samples),
                    elapsed_seconds=elapsed,
                    remaining_steps=remaining,
                )
                self._execution_advice = asdict(advice)
                if not advice.feasible:
                    if candidate is not None and advice.reason == "latency_estimate_exhausted":
                        return  # 无法推断剩余时长；明确记录预算失效，保留R处等待。
                    error = AsyncSessionError(f"Adaptive async execution budget is infeasible: {advice.reason}")
                    self._error = error
                    raise error
                rate = advice.recommended_hz
                if rate >= self.execution_hz:
                    return
                self._rate_callback_thread = get_ident()
            try:
                acknowledged = self._execution_rate_callback(rate)
                if acknowledged is not True:
                    raise AsyncSessionError("Execution-rate slowdown was not acknowledged by the client")
            except Exception as exc:
                error = self._fail(exc, revision)
                raise error from exc
            finally:
                with self._lock:
                    self._rate_callback_thread = None
            with self._lock:
                self._check()
                if not self._live(revision):
                    raise AsyncSessionError("Episode changed during execution-rate acknowledgement")
                self.execution_hz = rate
                self._execution_rate_changes += 1

    def _submit(self, state, sm, image, am):
        prefix = self._actions[self._tick - self._origin :].clone()
        inputs = dict(
            state=state,
            state_mask=sm,
            action_mask=am,
            image=image,
            history=tuple(self._history),
            tick=self._tick,
            revision=self._revision,
            rtc=self._rtc_prefix(prefix),
        )
        candidate = _Candidate(
            self._revision,
            self._tick,
            self._origin + self.handoff_steps,
            inputs,
            history_seconds=self._history_seconds(),
            submitted_at=self._clock(),
            prefix=prefix if self.enable_rtc else None,
        )
        self._pending = candidate
        self._requests += 1
        self._last_request = dict(
            tick=candidate.anchor,
            observation_tick=candidate.anchor,
            handoff_tick=candidate.handoff_tick,
            estimated_delay_steps=self.delay_steps,
            history_seconds=candidate.history_seconds,
            accepted=False,
        )
        candidate.future = self._future = self._executor.submit(self._run_candidate, candidate)

    def _run_candidate(self, candidate):
        with self._lock:
            self._worker_thread = get_ident()
            if not self._live(candidate.revision):
                return
        try:
            actions = self._validate_actions(self.client.plan(**candidate.inputs))
            if len(actions) != self.horizon:
                raise ValueError("Planner must preserve the prediction horizon")
            mask = candidate.inputs["action_mask"]
            self.template.sanitize(actions, mask)
            actions = actions.masked_fill(~mask, 0)
            if candidate.prefix is not None:
                # 与实际执行相同的硬保留前缀；soft区仍使用新推理结果。
                actions[: self.delay_steps] = candidate.prefix[: self.delay_steps]
            # 推理一结束就提示下个anchor；提前量来自旧动作仍在执行，不必等R交接。
            if not self._notify("prepare_window", candidate.revision, anchor_tick=candidate.anchor + self.steps):
                return
            with self._lock:
                if self._live(candidate.revision) and self._pending is candidate:
                    # Count queue/RPC/validation and next-window hint waits up
                    # to the point where the controller can use this result.
                    elapsed = self._clock() - candidate.submitted_at
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("Execution clock must advance monotonically")
                    candidate.actions = actions
                    self._latency_samples.append((candidate.history_seconds, elapsed))
                    self._last_request.update(
                        elapsed_seconds=elapsed, actual_delay_steps=self._tick - candidate.anchor
                    )
        except Exception as exc:
            error = self._fail(exc, candidate.revision)
            if error is exc:
                raise
            raise error from exc

    def get_action(self, *, state, state_mask, image, action_mask):
        # Reset may run while observations are being copied or preparation is
        # waiting. An old caller must never consume the new episode's tick 0.
        with self._lock:
            self._check()
            expected_revision = self._revision
        self.prepare(
            _expected_revision=expected_revision,
            state=state,
            state_mask=state_mask,
            image=image,
            action_mask=action_mask,
        )
        values = self._measurement(state=state, state_mask=state_mask, image=image, action_mask=action_mask)
        state, sm, image, am = values
        consumer = object()
        with self._lock:
            self._check()
            if expected_revision != self._revision:
                raise AsyncSessionError("Episode changed while preparing the async command")
            if self._consumer is not None:
                raise AsyncSessionError("Concurrent command requests are not supported")
            self._accept_masks(sm, am)
            self._consumer = consumer
            revision, tick = self._revision, self._tick
        try:
            self._notify("observe", revision, tick=tick, image=image)
            while True:
                self._update_pacing(revision)
                with self._lock:
                    self._check()
                    if revision != self._revision or tick != self._tick:
                        raise AsyncSessionError("Episode or command tick changed while waiting for handoff")
                    offset = tick - self._origin
                    if offset >= self.handoff_steps:
                        candidate = self._pending
                        if candidate is None or candidate.handoff_tick != tick:
                            raise self._fail(
                                AsyncSessionError("No matching candidate for the fixed handoff"), revision
                            )
                        if candidate.actions is not None:
                            self._actions, self._origin = candidate.actions, candidate.anchor
                            self._pending = None
                            self._published += 1
                            self._last_request["accepted"] = True
                            self._last_request["publish_tick"] = tick
                            self._waiting = False
                            # R=2S时，新buffer的offset正好为S；必须继续循环再提交。
                            continue
                        if not candidate.late:
                            candidate.late = True
                            self._late_handoffs += 1
                        self._waiting = True
                        future = candidate.future
                    else:
                        if offset >= self.steps and self._pending is None:
                            self._submit(state, sm, image, am)
                        command = self._actions[offset].clone()
                        self._history.append(_Observation(tick, image))
                        self._tick += 1
                        return command
                # 迟到只能在R等待，不能重放旧命令；等待期间不持controller锁。
                future.result()
        finally:
            with self._lock:
                if self._consumer is consumer:
                    self._consumer = None
                    self._waiting = False

    def status(self):
        with self._lock:
            candidate = self._pending
            ready = candidate is not None and candidate.actions is not None and not self._closed
            return dict(
                tick=self._tick,
                chunk_start_tick=self._origin,
                remaining_actions=max(0, self.handoff_steps - (self._tick - self._origin)),
                prediction_horizon=self.horizon,
                minimum_execution_steps=self.steps,
                handoff_steps=self.handoff_steps,
                controller_hz=self.controller_hz,
                execution_hz=self.execution_hz,
                playback_speed=self.execution_hz / self.controller_hz,
                rtc_enabled=self.enable_rtc,
                execution_mode="async_rtc" if self.enable_rtc else "async_non_rtc",
                prepared=self._prepared,
                closed=self._closed,
                resetting=self._resetting,
                error=None if self._error is None else str(self._error),
                ordinary_requests=self._requests,
                published_requests=self._published,
                inference_ready=ready,
                inference_pending=candidate is not None and not ready and not self._closed,
                waiting_for_handoff=self._waiting,
                late_handoffs=self._late_handoffs,
                revision=self._revision,
                adaptive_execution=self.execution_policy is not None,
                estimated_delay_steps=self.delay_steps,
                execution_rate_changes=self._execution_rate_changes,
                execution_advice=None if self._execution_advice is None else dict(self._execution_advice),
                calibration=None if self._calibration is None else dict(self._calibration),
                latency_buffer_capacity=self._latency_samples.maxlen,
                latency_history_basis="encoded_window_model_seconds",
                latency_fit=fit_latency(self._history_seconds(), samples=tuple(self._latency_samples)),
                latency_samples=[
                    dict(history_seconds=history, elapsed_seconds=seconds)
                    for history, seconds in self._latency_samples
                ],
                last_request=None if self._last_request is None else dict(self._last_request),
            )

    def join(self, timeout=None):
        """Wait for the current worker without publishing before its fixed R tick."""
        with self._lock:
            if get_ident() in (self._worker_thread, self._rate_callback_thread):
                raise AsyncSessionError("Cannot join from the planning worker or execution-rate callback")
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        with self._lock:
            self._check()
            return self.status()

    wait_for_inference = join

    def reset(self, initial_actions):
        if get_ident() == self._rate_callback_thread:
            raise AsyncSessionError("Cannot reset during execution-rate acknowledgement")
        actions = self._validate_actions(initial_actions)
        if len(actions) != self.horizon:
            raise ValueError("Reset must preserve the prediction horizon")
        with self._callback_lock:
            with self._lock:
                if self._closed or self._resetting or get_ident() == self._worker_thread:
                    raise AsyncSessionError("Cannot reset this async session now")
                self._revision += 1
                self._resetting = True
                future = self._future
        try:
            if future is not None:
                future.exception()  # The invalidated episode cannot publish or send another hint.
            with self._lock:
                if self._closed:
                    raise AsyncSessionError("Async session closed during reset")
                self._actions = actions
                self._pending = self._future = self._consumer = None
                self._tick = self._origin = self._requests = self._published = self._late_handoffs = 0
                self._error = self._action_mask = self._state_mask = self._last_hint = self._last_request = None
                self._prepared = self._waiting = False
                self._latency_samples.clear()
                self._calibration = self._execution_advice = None
                self._execution_rate_changes = 0
                self._history.clear()
        finally:
            with self._lock:
                self._resetting = False

    def close(self, *, wait=True):
        if get_ident() == self._rate_callback_thread:
            raise AsyncSessionError("Cannot close during execution-rate acknowledgement")
        with self._callback_lock:
            with self._lock:
                if get_ident() == self._worker_thread:
                    raise AsyncSessionError("Cannot close from the planning worker")
                self._closed = True
                self._revision += 1
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        with self._lock:
            self._check()
        return self

    def __exit__(self, *args):
        self.close()
