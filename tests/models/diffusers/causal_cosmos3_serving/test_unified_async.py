# SPDX-License-Identifier: OpenMDW-1.1
"""RTC adds a fixed-prefix target to the shared asynchronous execution clock."""

from threading import Event, Thread

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.causal_cosmos3_serving.test_async_session import (
    _actions,
    _Client,
    _consume,
    _measurement,
)
from veomni.models.diffusers.causal_cosmos3.serving.async_session import AsyncRemoteSession, AsyncSessionError
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix, prefix_weights
from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_guidance_does_not_change_the_fixed_trigger_or_early_result_handoff(enable_rtc):
    client = _Client()
    client.metadata["rtc_enabled"] = True
    session = AsyncRemoteSession(client, _actions(), enable_rtc=enable_rtc, beta=3)
    try:
        session.prepare(**_measurement(0))
        assert _consume(session, 0, 17) == list(range(17))
        assert client.started[0].wait(2)
        request = client.calls[0]
        assert request["tick"] == 16 and request["state"][0] == 16
        if enable_rtc:
            prefix = request["rtc"]
            assert isinstance(prefix, RTCPrefix)
            assert prefix.delay_steps == 8 and prefix.beta == 3
            torch.testing.assert_close(prefix.previous_actions[:, 0], torch.arange(16, 32).float())
            weights = prefix_weights(prefix.delay_steps, len(prefix.previous_actions), 32)
            assert torch.equal(weights[:8], torch.ones(8))
            assert ((weights[8:16] > 0) & (weights[8:16] < 1)).all()
            assert (weights[8:15] > weights[9:16]).all()
            assert not weights[16:].any()
        else:
            assert request["rtc"] is None
        client.release[0].set()
        status = session.join(2)
        assert status["published_requests"] == 0 and status["inference_ready"]
        assert status["execution_mode"] == ("async_rtc" if enable_rtc else "async_non_rtc")
        assert status["rtc_enabled"] is enable_rtc
        assert _consume(session, 17, 24) == list(range(17, 24))
        # Even the guided path starts with B[8], rather than publishing at completion.
        assert _consume(session, 24, 25) == [108]
        assert session.status()["published_requests"] == 1
        assert session.status()["chunk_start_tick"] == 16
    finally:
        client.unblock()
        session.close()


def test_guided_result_preserves_only_the_committed_eight_rows():
    client = _Client()
    client.metadata["rtc_enabled"] = True
    session = AsyncRemoteSession(client, _actions(), enable_rtc=True)
    try:
        _consume(session, 0, 17)
        assert client.started[0].wait(2)
        client.release[0].set()
        session.join(2)
        candidate = session._pending.actions
        torch.testing.assert_close(candidate[:8, 0], torch.arange(16, 24).float())
        # The soft section is a generated target, not a second hard copy of A[24:32].
        torch.testing.assert_close(candidate[8:16, 0], torch.arange(108, 116).float())
        assert not candidate[:, 1].any()
    finally:
        client.unblock()
        session.close()


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _CalibratedClient(_Client):
    def __init__(self, clock):
        super().__init__()
        self.metadata["rtc_enabled"] = True
        self.clock, self.calibration_calls = clock, []

    def plan(self, **inputs):
        if inputs["tick"] == 0:
            self.calibration_calls.append(inputs)
            self.clock.advance(1.0)
            return _actions()
        return super().plan(**inputs)


def _paced_session(client, clock, callback, enable_rtc, *, latency_factor=1.2):
    return AsyncRemoteSession(
        client,
        _actions(),
        enable_rtc=enable_rtc,
        execution_hz=30,
        execution_policy=RTCExecutionPolicy(latency_factor=latency_factor, margin_steps=1),
        execution_rate_callback=callback,
        calibration_warmups=0,
        calibration_samples=1,
        clock=clock,
    )


def _consume_with_clock(session, clock, begin, end):
    for tick in range(begin, end):
        assert session.get_action(**_measurement(tick))[0] == tick
        clock.advance(1 / session.execution_hz)


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_calibration_and_inflight_slowdown_preserve_the_model_and_handoff_clocks(enable_rtc):
    clock, acknowledgements = _Clock(), []
    client = _CalibratedClient(clock)

    def accept(rate):
        acknowledgements.append(rate)
        return True

    session = _paced_session(client, clock, accept, enable_rtc)
    try:
        prepared = session.prepare(**_measurement(0))
        assert prepared["tick"] == prepared["published_requests"] == 0
        assert len(client.calibration_calls) == 1 and not client.calls
        assert acknowledgements and session.execution_hz == acknowledgements[-1] < 30
        if enable_rtc:
            assert client.calibration_calls[0]["rtc"].delay_steps == 8
        else:
            assert client.calibration_calls[0]["rtc"] is None
        _consume_with_clock(session, clock, 0, 17)
        assert client.started[0].wait(2)
        request = client.calls[0]
        rate = session.execution_hz
        prefix = request["rtc"].previous_actions.clone() if enable_rtc else None
        clock.advance(10.0)  # The revised remaining estimate exceeds the seven remaining execution intervals.
        assert session.get_action(**_measurement(17))[0] == 17
        assert session.execution_hz == acknowledgements[-1] < rate
        status = session.status()
        assert status["tick"] == 18 and status["published_requests"] == 0
        assert status["controller_hz"] == 30 and status["handoff_steps"] == 24
        assert request["tick"] == 16 and request["state"][0] == 16
        assert [record.tick for record in request["history"]] == list(range(16))
        if enable_rtc:
            assert request["rtc"].delay_steps == 8
            torch.testing.assert_close(request["rtc"].previous_actions, prefix)
        client.release[0].set()
        session.join(2)
        assert _consume(session, 18, 24) == list(range(18, 24))
        assert _consume(session, 24, 25) == [108]
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("acknowledgement", [False, RuntimeError("pacer failed")])
def test_rejected_inflight_slowdown_neither_installs_a_fake_rate_nor_changes_execution_mode(acknowledgement):
    clock = _Clock()
    client = _CalibratedClient(clock)
    calls = []

    def acknowledge(rate):
        calls.append(rate)
        if len(calls) == 1:
            return True
        if isinstance(acknowledgement, Exception):
            raise acknowledgement
        return acknowledgement

    session = _paced_session(client, clock, acknowledge, True)
    try:
        session.prepare(**_measurement(0))
        _consume_with_clock(session, clock, 0, 17)
        assert client.started[0].wait(2)
        accepted_rate = session.execution_hz
        clock.advance(10.0)
        with pytest.raises(AsyncSessionError):
            session.get_action(**_measurement(17))
        status = session.status()
        assert status["tick"] == 17 and session.execution_hz == accepted_rate
        assert status["rtc_enabled"] and status["execution_mode"] == "async_rtc"
        assert status["error"] and status["published_requests"] == 0
        assert client.calls[0]["rtc"].delay_steps == 8
    finally:
        client.unblock()
        session.close()


def test_exhausted_latency_estimate_is_reported_without_inventing_a_rate_or_disabling_rtc():
    clock, acknowledged = _Clock(), []
    client = _CalibratedClient(clock)

    def accept(rate):
        acknowledged.append(rate)
        return True

    session = _paced_session(client, clock, accept, True, latency_factor=1)
    try:
        session.prepare(**_measurement(0))
        _consume_with_clock(session, clock, 0, 17)
        assert client.started[0].wait(2)
        rate, changes = session.execution_hz, len(acknowledged)
        clock.advance(2)
        assert session.get_action(**_measurement(17))[0] == 17
        status = session.status()
        assert status["execution_advice"]["reason"] == "latency_estimate_exhausted"
        assert not status["error"] and status["rtc_enabled"]
        assert status["execution_mode"] == "async_rtc" and status["published_requests"] == 0
        assert session.execution_hz == rate and len(acknowledged) == changes
        client.release[0].set()
        session.join(2)
        assert _consume(session, 18, 24) == list(range(18, 24))
        assert _consume(session, 24, 25) == [108]
    finally:
        client.unblock()
        session.close()


@pytest.mark.parametrize("operation", ["reset", "close", "join", "prepare"])
def test_pacing_callback_cannot_reenter_session_lifecycle_or_wait_for_its_own_worker(operation):
    clock, rates = _Clock(), []
    client = _CalibratedClient(clock)

    def acknowledge(rate):
        rates.append(rate)
        if len(rates) == 1:
            return True
        if operation == "reset":
            session.reset(_actions())
        elif operation == "prepare":
            session.prepare(**_measurement(17))
        else:
            getattr(session, operation)()
        return True

    session = _paced_session(client, clock, acknowledge, True)
    completed, errors = Event(), []
    controller = None
    try:
        session.prepare(**_measurement(0))
        _consume_with_clock(session, clock, 0, 17)
        assert client.started[0].wait(2)
        accepted_rate = session.execution_hz
        clock.advance(10)

        def consume():
            try:
                session.get_action(**_measurement(17))
            except Exception as exc:
                errors.append(exc)
            finally:
                completed.set()

        controller = Thread(target=consume, daemon=True)
        controller.start()
        assert completed.wait(2), "A pacing callback must fail immediately instead of waiting on the planner"
        assert len(errors) == 1 and isinstance(errors[0], AsyncSessionError)
        assert "execution-rate" in str(errors[0])
        status = session.status()
        assert status["tick"] == 17 and status["revision"] == 0
        assert not status["closed"] and not status["resetting"]
        assert status["rtc_enabled"] and status["execution_mode"] == "async_rtc"
        assert status["published_requests"] == 0 and status["error"]
        assert session.execution_hz == accepted_rate and status["execution_rate_changes"] == 1
    finally:
        # If a lifecycle guard regresses, an unblocked successful planner would
        # itself need the callback lock. Fail it before releasing the gate so
        # its exception can break that cycle and the test process can clean up.
        client.error = RuntimeError("release the gated planner during test cleanup")
        client.unblock()
        if controller is not None:
            controller.join(2)
        session.close()
