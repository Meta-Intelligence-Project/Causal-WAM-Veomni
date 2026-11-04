# SPDX-License-Identifier: OpenMDW-1.1
"""Paced, offline client replay while an RTC session plans asynchronously.

This sends no robot or RPC commands. Recorded observations are indexed by model
action ticks, while waypoint consumption and virtual servo frames use wall time.
Interpolation visualizes targets; it is not a robot dynamics model.
"""

import math
import time
from numbers import Real

import torch


def _frequency(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive frequency")
    return float(value)


def _timestamp(clock):
    value = clock()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("clock must return finite timestamps")
    return float(value)


def interpolate_action(template, mask, start, end, alpha):
    """Virtual interpolation: linear scalars and shortest-sign quaternion nlerp.

    Normalized quaternion interpolation preserves unit rotations, including
    equivalent q/-q endpoints. Gripper/hand scalar interpolation is illustrative;
    a real hardware adapter must choose its own command interpolation policy.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    mask = template.validate_valid_mask(mask).detach().cpu()
    for label, value in (("start", start), ("end", end)):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.shape != mask.shape
            or not value.is_floating_point()
        ):
            raise ValueError(f"{label} must be one CPU floating-point template action")
    left = template.sanitize(start.detach().float(), mask)
    right = template.sanitize(end.detach().float(), mask)
    result = torch.lerp(left, right, float(alpha))
    for group in template.rotation_groups:
        indexes = list(group)
        if mask[indexes].all():
            qa, qb = left[indexes], right[indexes]
            if torch.dot(qa, qb) < 0:
                qb = -qb
            q = torch.lerp(qa, qb, float(alpha))
            result[indexes] = q / torch.linalg.vector_norm(q)
    return template.sanitize(result, mask)


def _active_status(session):
    status = session.status()
    if not status.get("prepared", False):
        raise RuntimeError("Prepare the RTC session before starting the paced client clock")
    if (
        not status.get("rtc_enabled", False)
        or status.get("execution_mode") != "rtc"
        or status.get("blocking_inference", False)
        or status.get("closed", False)
        or status.get("error") is not None
    ):
        raise RuntimeError("Paced RTC replay requires active RTC; blocking fallback is not a continuous replay")
    return status


def replay_rtc_client(
    session,
    observation_at,
    *,
    action_steps,
    servo_hz=30.0,
    interpolate,
    emit=None,
    clock=None,
    sleep=None,
):
    """Consume waypoints and emit virtual servo frames on absolute wall deadlines.

    ``observation_at(tick)`` returns the current recorded measurement dictionary
    accepted by ``session.get_action``. ``interpolate(start,end,alpha)`` returns
    one CPU virtual command. Optional ``emit(frame,event)`` observes each frame.
    No inference join or completion wait occurs in the timed loop. Every waypoint
    and servo event is retained if late, with lateness reported; no sample is
    skipped and late pacing is never labelled hard realtime.

    Both before and after each action request, blocking fallback is rejected.
    A concurrent fallback transition can make that last action call block before
    rejection; such a run raises instead of returning a continuous-RTC result.
    ``continuous_rtc`` means that RTC remained active throughout this replay;
    the timing traces must still be checked for scheduling gaps and lateness.
    A session may acknowledge a slower execution rate before returning a command.
    That rate controls the upcoming interval; elapsed intervals are never retimed.
    """
    if isinstance(action_steps, bool) or not isinstance(action_steps, int) or action_steps < 1:
        raise ValueError("action_steps must be a positive integer")
    for name, callback in (("observation_at", observation_at), ("interpolate", interpolate)):
        if not callable(callback):
            raise ValueError(f"{name} must be callable")
    for name, callback in (("emit", emit), ("clock", clock), ("sleep", sleep)):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable")
    controller_hz = _frequency(session.controller_hz, "controller_hz")
    execution_hz = _frequency(getattr(session, "execution_hz", controller_hz), "execution_hz")
    initial_execution_hz = execution_hz
    servo_hz = _frequency(servo_hz, "servo_hz")
    if servo_hz < execution_hz:
        raise ValueError("servo_hz must be at least execution_hz to represent every waypoint interval")
    initial_status = _active_status(session)
    first_tick = initial_status["tick"]
    clock = time.perf_counter if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    last_read = None

    def read_clock():
        nonlocal last_read
        now = _timestamp(clock)
        if last_read is not None and now < last_read:
            raise ValueError("clock must be monotonic")
        last_read = now
        return now

    started = read_clock()
    commands, frames, action_trace, servo_trace = [], [], [], []
    offset = 0.0
    servo_index = 0
    previous = target = None

    def wait_until(offset):
        deadline = started + offset
        now = read_clock()
        while now < deadline:
            sleep(deadline - now)
            next_time = read_clock()
            if next_time <= now:
                raise RuntimeError("sleep/clock did not advance while pacing the client")
            now = next_time
        return now

    def emit_frame(tick, offset, alpha):
        nonlocal servo_index
        wait_until(offset)
        status = _active_status(session)
        frame = interpolate(previous.clone(), target.clone(), alpha)
        if (
            not isinstance(frame, torch.Tensor)
            or frame.device.type != "cpu"
            or not frame.is_floating_point()
            or frame.shape != target.shape
            or not torch.isfinite(frame).all()
        ):
            raise ValueError("interpolate must return a finite CPU action matching the waypoint shape")
        actual = read_clock() - started
        event = dict(
            servo_index=servo_index,
            tick=tick,
            scheduled_seconds=offset,
            actual_seconds=actual,
            lateness_seconds=max(0.0, actual - offset),
            alpha=alpha,
            published_requests=status["published_requests"],
        )
        frames.append(frame.detach().clone())
        servo_trace.append(event)
        if emit is not None:
            emit(frame.detach().clone(), dict(event))
        servo_index += 1

    for step in range(action_steps):
        tick = first_tick + step
        wait_until(offset)
        observation = observation_at(tick)
        before = _active_status(session)
        if before["tick"] != tick:
            raise RuntimeError("RTC command tick changed outside the paced client")
        call_started = read_clock()
        command = session.get_action(**observation)
        call_completed = read_clock()
        after = _active_status(session)
        if after["tick"] != tick + 1:
            raise RuntimeError("One client command must advance exactly one RTC action tick")
        next_hz = _frequency(getattr(session, "execution_hz", execution_hz), "execution_hz")
        if next_hz > execution_hz:
            raise RuntimeError("The paced client permits execution-rate decreases only")
        execution_hz = next_hz
        period = 1 / execution_hz
        if (
            not isinstance(command, torch.Tensor)
            or command.device.type != "cpu"
            or command.ndim != 1
            or not command.is_floating_point()
            or not torch.isfinite(command).all()
            or (commands and command.shape != commands[0].shape)
        ):
            raise ValueError("RTC session must return one finite CPU floating-point action")
        previous = command.detach().clone() if target is None else target
        target = command.detach().clone()
        commands.append(target)
        action_trace.append(
            dict(
                tick=tick,
                execution_hz=execution_hz,
                scheduled_seconds=offset,
                started_seconds=call_started - started,
                completed_seconds=call_completed - started,
                call_seconds=call_completed - call_started,
                lateness_seconds=max(0.0, call_started - started - offset),
                published_requests_before=before["published_requests"],
                published_requests_after=after["published_requests"],
                inference_pending_before=before["inference_pending"],
                inference_pending_after=after["inference_pending"],
                last_request_after=after.get("last_request"),
            )
        )
        interval_end = offset + period
        while servo_index / servo_hz < interval_end - 1e-12:
            servo_offset = servo_index / servo_hz
            alpha = min(1.0, max(0.0, (servo_offset - offset) / period))
            emit_frame(tick, servo_offset, alpha)
        offset = interval_end

    # Include the last target itself, even if the endpoint lies off the servo grid.
    emit_frame(first_tick + action_steps - 1, offset, 1.0)
    ended = read_clock()
    return dict(
        controller_hz=controller_hz,
        execution_hz=execution_hz,
        initial_execution_hz=initial_execution_hz,
        servo_hz=servo_hz,
        speed_ratio=execution_hz / controller_hz,
        first_tick=first_tick,
        started_at=started,
        ended_at=ended,
        wall_seconds=ended - started,
        model_seconds=action_steps / controller_hz,
        action_commands=torch.stack(commands),
        servo_frames=torch.stack(frames),
        action_trace=action_trace,
        servo_trace=servo_trace,
        max_action_lateness_seconds=max(row["lateness_seconds"] for row in action_trace),
        max_servo_lateness_seconds=max(row["lateness_seconds"] for row in servo_trace),
        max_servo_gap_seconds=max(
            right["actual_seconds"] - left["actual_seconds"] for left, right in zip(servo_trace, servo_trace[1:])
        ),
        final_status=_active_status(session),
        continuous_rtc=True,
    )
