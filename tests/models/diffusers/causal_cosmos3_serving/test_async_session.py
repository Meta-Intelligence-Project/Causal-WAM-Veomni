# SPDX-License-Identifier: OpenMDW-1.1
"""Gated ordinary RPCs exercise asynchronous submission and fixed handoffs."""

from threading import Event, Thread
from time import monotonic

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.causal_cosmos3.serving.async_session import (
    AsyncPlannerError,
    AsyncRemoteSession,
    AsyncSessionError,
    resolve_async_schedule,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _actions(horizon=32, start=0):
    actions = torch.zeros(horizon, 55)
    actions[:, 0] = start + torch.arange(horizon)
    actions[:, 1] = 999  # Invalid slots must never reach an actuator.
    actions[:, 28] = 0.5
    return actions


def _measurement(tick):
    state = torch.zeros(55)
    state[0], state[28] = tick, float("nan")
    sm = torch.zeros(55, dtype=torch.bool)
    sm[0] = True
    am = sm.clone()
    am[28] = True
    return dict(state=state, state_mask=sm, action_mask=am, image=torch.full((3, 4, 4), tick, dtype=torch.uint8))


class _Client:
    def __init__(self, *, H=32, S=16, R=None, error=None):
        self.metadata = dict(action_horizon=H, execute_steps=S, handoff_steps=R, controller_hz=30, history_steps=1024)
        self.started, self.release = [Event() for _ in range(4)], [Event() for _ in range(4)]
        self.calls, self.observations, self.hints = [], [], []
        self.error = error

    def plan(self, **inputs):
        index = len(self.calls)
        self.calls.append(inputs)
        self.started[index].set()
        if not self.release[index].wait(5):
            raise TimeoutError("Test did not release the planner")
        if isinstance(self.error, Exception):
            raise self.error
        return self.error if self.error is not None else _actions(self.metadata["action_horizon"], 100 * (index + 1))

    def observe(self, *, tick, image, revision):
        self.observations.append((tick, image.clone(), revision))

    def prepare_window(self, *, anchor_tick, revision):
        self.hints.append((anchor_tick, revision))

    def unblock(self):
        for event in self.release:
            event.set()


def _consume(session, begin, end):
    commands = [session.get_action(**_measurement(tick)) for tick in range(begin, end)]
    assert all(command[1] == 0 for command in commands)
    return [command[0].item() for command in commands]


def _until(condition):
    deadline = monotonic() + 3
    while not condition():
        if monotonic() >= deadline:
            raise AssertionError("Concurrent session did not reach the expected state")
        Event().wait(0.001)


@pytest.mark.parametrize(
    "H,S,R,expected", [(32, 16, None, (16, 24)), (32, 16, 32, (16, 32)), (8, 3, None, (3, 4)), (2, 1, None, (1, 2))]
)
def test_schedule_resolves_independent_prediction_trigger_and_handoff(H, S, R, expected):
    assert resolve_async_schedule(H, S, R) == expected


@pytest.mark.parametrize(
    "H,S,R",
    [
        (0, 1, None),
        (32, 0, None),
        (32, 16, 16),
        (32, 16, 33),
        (32, 8, 20),
        (32, 32, None),
        (True, 1, 2),
        (32, 1.0, 2),
        (32, 16, True),
    ],
)
def test_invalid_single_candidate_schedules_are_rejected(H, S, R):
    with pytest.raises(ValueError):
        resolve_async_schedule(H, S, R)


def test_submit_at_16_handoff_at_24_skips_eight_and_early_ready_keeps_old_actions():
    client = _Client()
    session = AsyncRemoteSession(client, _actions(), execution_hz=15)
    try:
        assert session.prepare(**_measurement(0))["tick"] == 0
        assert not client.calls and client.hints == [(16, 0)]
        assert _consume(session, 0, 17) == list(range(17))
        assert client.started[0].wait(2)
        request = client.calls[0]
        assert request["tick"] == 16 and request["rtc"] is None
        assert request["state"][0] == 16 and request["image"][0, 0, 0] == 16
        assert [record.tick for record in request["history"]] == list(range(16))
        client.release[0].set()
        ready = session.join(2)
        assert ready["inference_ready"] and not ready["inference_pending"]
        assert ready["published_requests"] == 0 and ready["chunk_start_tick"] == 0
        assert client.hints == [(16, 0), (32, 0)]  # Pre-encode the next window before the fixed R.
        assert _consume(session, 17, 24) == list(range(17, 24))
        assert _consume(session, 24, 25) == [108]
        status = session.status()
        assert status["published_requests"] == 1 and status["chunk_start_tick"] == 16
        assert status["execution_mode"] == "async_non_rtc" and not status["rtc_enabled"]
        assert status["execution_hz"] == 15 and status["controller_hz"] == 30
        assert [tick for tick, _, _ in client.observations] == list(range(25))
        assert _consume(session, 25, 33) == list(range(109, 117))
        assert client.started[1].wait(2) and client.calls[1]["tick"] == 32
        assert all(call["rtc"] is None for call in client.calls)
    finally:
        client.unblock()
        session.close()


def test_late_plan_waits_at_r_without_consuming_or_replaying_old_commands():
    client = _Client()
    session = AsyncRemoteSession(client, _actions())
    outcome, returned = [], Event()

    def get_handoff():
        outcome.append(session.get_action(**_measurement(24))[0].item())
        returned.set()

    try:
        assert _consume(session, 0, 24) == list(range(24))
        assert client.started[0].wait(2)
        thread = Thread(target=get_handoff)
        thread.start()
        _until(lambda: session.status()["waiting_for_handoff"])
        status = session.status()
        assert status["tick"] == 24 and status["late_handoffs"] == 1 and not returned.is_set()
        assert client.observations[-1][0] == 24
        client.release[0].set()
        assert returned.wait(2)
        thread.join(2)
        assert outcome == [108] and session.status()["tick"] == 25
        assert session.status()["late_handoffs"] == 1
    finally:
        client.unblock()
        session.close()


def test_r_equals_two_s_commits_and_submits_on_the_same_control_tick():
    client = _Client(R=32)
    session = AsyncRemoteSession(client, _actions())
    try:
        assert _consume(session, 0, 17) == list(range(17))
        assert client.started[0].wait(2)
        client.release[0].set()
        session.join(2)
        assert _consume(session, 17, 32) == list(range(17, 32))
        assert _consume(session, 32, 33) == [116]
        assert client.started[1].wait(2)
        assert [call["tick"] for call in client.calls] == [16, 32]
        status = session.status()
        assert status["published_requests"] == 1 and status["ordinary_requests"] == 2
        assert status["inference_pending"] and status["chunk_start_tick"] == 16
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("error", [RuntimeError("remote failure"), _actions(31), torch.full((32, 55), float("nan"))])
def test_background_failure_stops_at_the_next_control_boundary(error):
    client = _Client(error=error)
    session = AsyncRemoteSession(client, _actions())
    try:
        _consume(session, 0, 17)
        assert client.started[0].wait(2)
        client.release[0].set()
        with pytest.raises(AsyncPlannerError):
            session.join(2)
        with pytest.raises(AsyncPlannerError):
            session.get_action(**_measurement(17))
        status = session.status()
        assert status["tick"] == 17 and status["error"] and not status["published_requests"]
        assert client.hints == [(16, 0)]
    finally:
        client.unblock()
        session.close()


def test_reset_invalidates_old_results_and_hints_before_waiting_for_the_worker():
    client = _Client()
    session = AsyncRemoteSession(client, _actions())
    finished, errors = Event(), []

    def reset():
        try:
            session.reset(_actions(start=500))
        except Exception as exc:
            errors.append(exc)
        finished.set()

    try:
        _consume(session, 0, 17)
        assert client.started[0].wait(2)
        thread = Thread(target=reset)
        thread.start()
        _until(lambda: session.status()["resetting"])
        assert not finished.is_set() and session.status()["revision"] == 1
        with pytest.raises(AsyncSessionError, match="resetting"):
            session.get_action(**_measurement(0))
        client.release[0].set()
        assert finished.wait(2)
        thread.join(2)
        assert not errors and session.status()["tick"] == 0
        assert not session.status()["published_requests"]
        assert client.hints == [(16, 0)]
        assert _consume(session, 0, 1) == [500]
        assert client.hints == [(16, 0), (16, 1)]
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("wait", [False, True])
def test_close_invalidates_inflight_requests_and_never_sends_a_late_hint(wait):
    client = _Client()
    session = AsyncRemoteSession(client, _actions())
    finished = Event()
    try:
        _consume(session, 0, 17)
        assert client.started[0].wait(2)
        thread = Thread(target=lambda: (session.close(wait=wait), finished.set()))
        thread.start()
        _until(lambda: session.status()["closed"])
        if not wait:
            assert finished.wait(2)
        else:
            assert not finished.is_set()
        with pytest.raises(AsyncSessionError, match="closed"):
            session.get_action(**_measurement(17))
        client.release[0].set()
        assert finished.wait(2)
        thread.join(2)
        session._future.result(timeout=2)
        assert client.hints == [(16, 0)]
        assert not session.status()["published_requests"]
    finally:
        client.unblock()
        session.close()


def test_close_can_invalidate_a_controller_waiting_at_the_handoff_without_lock_deadlock():
    client = _Client()
    session = AsyncRemoteSession(client, _actions())
    finished, errors = Event(), []

    def waiting_command():
        try:
            session.get_action(**_measurement(24))
        except Exception as exc:
            errors.append(exc)
        finished.set()

    try:
        _consume(session, 0, 24)
        thread = Thread(target=waiting_command)
        thread.start()
        _until(lambda: session.status()["waiting_for_handoff"])
        session.close(wait=False)
        assert session.status()["closed"] and session.status()["tick"] == 24
        client.release[0].set()
        assert finished.wait(2)
        thread.join(2)
        assert len(errors) == 1 and isinstance(errors[0], AsyncSessionError)
        assert session.status()["tick"] == 24
    finally:
        client.unblock()
        session.close()


def test_planning_inputs_remain_frozen_as_the_caller_and_history_advance():
    client = _Client()
    session = AsyncRemoteSession(client, _actions())
    try:
        _consume(session, 0, 16)
        measured = _measurement(16)
        session.get_action(**measured)
        assert client.started[0].wait(2)
        measured["state"].zero_()
        measured["image"].zero_()
        measured["action_mask"].zero_()
        _consume(session, 17, 20)
        request = client.calls[0]
        assert request["state"][0] == 16 and request["image"][0, 0, 0] == 16
        assert request["action_mask"][0] and [r.tick for r in request["history"]] == list(range(16))
        assert [tick for tick, _, _ in client.observations] == list(range(20))
    finally:
        client.unblock()
        session.close()
