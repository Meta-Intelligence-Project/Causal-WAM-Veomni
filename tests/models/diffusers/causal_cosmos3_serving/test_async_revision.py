# SPDX-License-Identifier: OpenMDW-1.1
"""Frozen admission revisions reject commands crossing reset/close boundaries."""

from threading import Event, Thread, current_thread, get_ident

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.causal_cosmos3_serving.test_async_session import _actions, _measurement
from tests.models.diffusers.causal_cosmos3_serving.test_unified_async import _CalibratedClient, _Clock
from veomni.models.diffusers.causal_cosmos3.serving.async_session import AsyncRemoteSession, AsyncSessionError
from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _Gate:
    def __init__(self):
        self.entered, self.release = Event(), Event()

    def pause(self):
        self.entered.set()
        if not self.release.wait(3):
            raise AssertionError("Test did not release the admission gate")


def _session(*, enable_rtc, adaptive):
    clock, rates = _Clock(), []
    client = _CalibratedClient(clock)

    def acknowledge(rate):
        rates.append(rate)
        return True

    session = AsyncRemoteSession(
        client,
        _actions(),
        enable_rtc=enable_rtc,
        execution_policy=RTCExecutionPolicy() if adaptive else None,
        execution_rate_callback=acknowledge if adaptive else None,
        calibration_warmups=0,
        calibration_samples=1,
        clock=clock,
    )
    return session, client, rates


def _notifications(client, rates):
    return (len(client.calibration_calls), len(client.calls), len(client.observations), len(client.hints), len(rates))


def _assert_clean_new_episode(session):
    status = session.status()
    assert status["revision"] == 1 and status["tick"] == status["chunk_start_tick"] == 0
    assert not status["prepared"] and not status["resetting"] and not status["closed"]
    assert not status["error"] and status["calibration"] is None and status["execution_advice"] is None
    assert status["latency_samples"] == [] and status["ordinary_requests"] == status["published_requests"] == 0
    assert session._consumer is None and session._pending is None and session._future is None
    assert session._state_mask is None and session._action_mask is None and session._last_hint is None
    assert not session._history


def _install_gate(monkeypatch, session, race, gate, thread_holder):
    original_measurement, original_prepare = session._measurement, session.prepare
    old_measurements = 0

    def is_old_call():
        return current_thread() is thread_holder[0]

    def measurement(**kwargs):
        nonlocal old_measurements
        if is_old_call():
            old_measurements += 1
            target = 2 if race == "get_action_after_prepare" else 1
            if race != "get_action_before_prepare" and old_measurements == target:
                gate.pause()
        return original_measurement(**kwargs)

    def prepare(**kwargs):
        if is_old_call() and race == "get_action_before_prepare":
            gate.pause()
        return original_prepare(**kwargs)

    monkeypatch.setattr(session, "_measurement", measurement)
    monkeypatch.setattr(session, "prepare", prepare)


@pytest.mark.parametrize("enable_rtc", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize(
    "race", ["get_action_after_prepare", "get_action_before_prepare", "direct_prepare_during_measurement"]
)
def test_reset_rejects_old_admission_without_consuming_or_polluting_the_new_episode(
    monkeypatch, enable_rtc, adaptive, race
):
    session, client, rates = _session(enable_rtc=enable_rtc, adaptive=adaptive)
    gate, finished, errors, returned = _Gate(), Event(), [], []
    thread_holder = [None]
    _install_gate(monkeypatch, session, race, gate, thread_holder)

    def stale_call():
        try:
            method = session.prepare if race == "direct_prepare_during_measurement" else session.get_action
            returned.append(method(**_measurement(7)))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = thread_holder[0] = Thread(target=stale_call, daemon=True)
    try:
        thread.start()
        assert gate.entered.wait(2), "Old admission did not reach the requested race boundary"
        session.reset(_actions(start=500))
        _assert_clean_new_episode(session)
        snapshot = _notifications(client, rates)
        gate.release.set()
        assert finished.wait(2), "Old admission did not terminate after reset"
        thread.join(2)
        assert len(errors) == 1 and isinstance(errors[0], AsyncSessionError)
        assert returned == []
        assert _notifications(client, rates) == snapshot
        _assert_clean_new_episode(session)
        assert session.get_action(**_measurement(0))[0].item() == 500
        assert session.status()["tick"] == 1 and session.status()["revision"] == 1
        assert client.observations[-1][0] == 0 and client.observations[-1][2] == 1
        assert client.observations[-1][1].eq(0).all()
        assert client.hints[-1] == (16, 1)
        if adaptive:
            assert len(client.calibration_calls) == snapshot[0] + 1
            assert client.calibration_calls[-1]["revision"] == 1
            assert client.calibration_calls[-1]["state"][0] == 0
    finally:
        gate.release.set()
        client.unblock()
        thread.join(2)
        session.close()


@pytest.mark.parametrize("enable_rtc", [False, True])
@pytest.mark.parametrize(
    "race", ["get_action_after_prepare", "get_action_before_prepare", "direct_prepare_during_measurement"]
)
def test_close_rejects_paused_admission_without_late_notifications(monkeypatch, enable_rtc, race):
    session, client, rates = _session(enable_rtc=enable_rtc, adaptive=False)
    gate, finished, errors, returned = _Gate(), Event(), [], []
    thread_holder = [None]
    _install_gate(monkeypatch, session, race, gate, thread_holder)

    def stale_call():
        try:
            method = session.prepare if race == "direct_prepare_during_measurement" else session.get_action
            returned.append(method(**_measurement(7)))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = thread_holder[0] = Thread(target=stale_call, daemon=True)
    try:
        thread.start()
        assert gate.entered.wait(2)
        session.close()
        snapshot = _notifications(client, rates)
        gate.release.set()
        assert finished.wait(2)
        thread.join(2)
        assert len(errors) == 1 and isinstance(errors[0], AsyncSessionError)
        assert returned == [] and _notifications(client, rates) == snapshot
        assert session.status()["closed"] and session.status()["tick"] == 0
        with pytest.raises(AsyncSessionError, match="closed"):
            session.get_action(**_measurement(0))
    finally:
        gate.release.set()
        client.unblock()
        thread.join(2)
        session.close()


def test_prepare_waiting_for_serial_admission_rechecks_the_episode(monkeypatch):
    session, client, rates = _session(enable_rtc=True, adaptive=True)
    waiting, finished, errors = Event(), Event(), []
    original_lock = session._prepare_lock

    class GatedLock:
        def __enter__(self):
            waiting.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    monkeypatch.setattr(session, "_prepare_lock", GatedLock())

    def prepare():
        try:
            session.prepare(**_measurement(7))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = Thread(target=prepare, daemon=True)
    original_lock.acquire()
    locked = True
    try:
        thread.start()
        assert waiting.wait(2)
        session.reset(_actions(start=500))
        snapshot = _notifications(client, rates)
        original_lock.release()
        locked = False
        assert finished.wait(2)
        thread.join(2)
        assert len(errors) == 1 and isinstance(errors[0], AsyncSessionError)
        assert _notifications(client, rates) == snapshot
        _assert_clean_new_episode(session)
        assert session.get_action(**_measurement(0))[0].item() == 500
    finally:
        if locked:
            original_lock.release()
        client.unblock()
        thread.join(2)
        session.close()


@pytest.mark.parametrize("operation", ["prepare", "get_action"])
def test_planning_worker_reentry_is_rejected_before_waiting_for_prepare_lock(monkeypatch, operation):
    session, client, rates = _session(enable_rtc=True, adaptive=True)
    original_lock, original_plan = session._prepare_lock, client.plan
    reentry_errors, worker_lock_attempts = [], []

    class WorkerDetectingLock:
        def __enter__(self):
            if get_ident() == session._worker_thread:
                # In the broken ordering this worker would wait forever for
                # the outer prepare, which is waiting for this same worker.
                # Fail before blocking so a regression is safe to clean up.
                worker_lock_attempts.append(get_ident())
                raise AssertionError("Planning worker reached the serial prepare lock before its guard")
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def plan(**inputs):
        try:
            getattr(session, operation)(**_measurement(7))
        except Exception as exc:
            reentry_errors.append(exc)
        return original_plan(**inputs)

    monkeypatch.setattr(session, "_prepare_lock", WorkerDetectingLock())
    monkeypatch.setattr(client, "plan", plan)
    try:
        status = session.prepare(**_measurement(0))
        assert len(reentry_errors) == 1 and isinstance(reentry_errors[0], AsyncSessionError)
        assert "planning worker" in str(reentry_errors[0])
        assert not worker_lock_attempts
        assert status["prepared"] and status["tick"] == 0 and status["error"] is None
        assert session._consumer is None
        assert len(client.calibration_calls) == 1 and not client.calls and not client.observations
        assert client.hints == [(16, 0)] and len(rates) == 1
    finally:
        client.unblock()
        session.close()
