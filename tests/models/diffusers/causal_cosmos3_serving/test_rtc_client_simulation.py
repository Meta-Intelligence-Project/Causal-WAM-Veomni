# SPDX-License-Identifier: OpenMDW-1.1
"""Separate model time, paced waypoint consumption and virtual servo sampling."""

import math

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.serving.rtc_client_simulation import interpolate_action, replay_rtc_client


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


class FakeSession:
    controller_hz = 30.0
    execution_hz = 2.0

    def __init__(self, clock):
        self.clock = clock
        self.tick = 0
        self.prepared = self.enabled = True
        self.published = 0
        self.ready_at = None
        self.calls = []

    def status(self):
        if self.ready_at is not None and self.clock() >= self.ready_at:
            self.published += 1
            self.ready_at = None
        return dict(
            prepared=self.prepared,
            rtc_enabled=self.enabled,
            execution_mode="rtc" if self.enabled else "blocking_non_rtc",
            blocking_inference=False,
            closed=False,
            error=None,
            tick=self.tick,
            published_requests=self.published,
            inference_pending=self.ready_at is not None,
            last_request=None,
        )

    def get_action(self, *, observed_tick):
        assert observed_tick == self.tick
        self.calls.append((self.tick, self.clock()))
        if self.tick == 1:
            self.ready_at = self.clock() + 0.1
        command = torch.tensor([float(self.tick)])
        self.tick += 1
        return command

    def join(self, *args, **kwargs):
        pytest.fail("The paced client must not wait for server completion")

    wait_for_inference = join


def replay(session, clock, observation=None, **kwargs):
    return replay_rtc_client(
        session,
        observation or (lambda tick: dict(observed_tick=tick)),
        action_steps=3,
        servo_hz=4,
        interpolate=torch.lerp,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_pacing_keeps_model_and_wall_time_separate_and_server_progresses_without_join():
    clock = FakeClock()
    session = FakeSession(clock)
    emitted = []
    result = replay(session, clock, emit=lambda frame, event: emitted.append((frame.clone(), event)))
    assert session.calls == [(0, 100.0), (1, 100.5), (2, 101.0)]
    assert result["wall_seconds"] == 1.5
    assert result["model_seconds"] == 0.1
    assert result["speed_ratio"] == 2 / 30
    torch.testing.assert_close(result["action_commands"], torch.tensor([[0.0], [1.0], [2.0]]))
    torch.testing.assert_close(result["servo_frames"].flatten(), torch.tensor([0, 0, 0, 0.5, 1, 1.5, 2]))
    assert [r["scheduled_seconds"] for r in result["servo_trace"]] == [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    assert [r["published_requests"] for r in result["servo_trace"]] == [0, 0, 0, 1, 1, 1, 1]
    assert all(row["lateness_seconds"] == 0 for row in result["action_trace"] + result["servo_trace"])
    assert len(emitted) == 7 and result["continuous_rtc"]


def test_slowdown_changes_only_future_intervals_and_preserves_the_servo_clock():
    clock = FakeClock()
    session = FakeSession(clock)
    original = session.get_action

    def adapt(**observation):
        if session.tick == 1:
            session.execution_hz = 1.0
        return original(**observation)

    session.get_action = adapt
    result = replay(session, clock)
    assert session.calls == [(0, 100.0), (1, 100.5), (2, 101.5)]
    assert [row["execution_hz"] for row in result["action_trace"]] == [2, 1, 1]
    assert result["initial_execution_hz"] == 2
    assert result["execution_hz"] == 1
    assert result["wall_seconds"] == 2.5
    assert result["model_seconds"] == 0.1
    assert [r["scheduled_seconds"] for r in result["servo_trace"]] == [i / 4 for i in range(11)]
    assert result["servo_frames"][-1].item() == 2
    assert all(row["lateness_seconds"] == 0 for row in result["action_trace"] + result["servo_trace"])


def test_unexpected_speed_increase_is_rejected():
    clock = FakeClock()
    session = FakeSession(clock)
    original = session.get_action

    def accelerate(**observation):
        session.execution_hz = 3.0
        return original(**observation)

    session.get_action = accelerate
    with pytest.raises(RuntimeError, match="decreases only"):
        replay(session, clock)


def test_lateness_is_recorded_without_dropping_or_replaying_waypoints_or_servo_events():
    clock = FakeClock()
    session = FakeSession(clock)

    def observe(tick):
        if tick == 1:
            clock.now += 0.6
        return dict(observed_tick=tick)

    result = replay(session, clock, observe)
    assert [tick for tick, _ in session.calls] == [0, 1, 2]
    assert len(result["servo_trace"]) == 7
    assert result["action_trace"][1]["lateness_seconds"] == pytest.approx(0.6)
    assert result["action_trace"][2]["lateness_seconds"] == pytest.approx(0.1)
    assert result["servo_trace"][2]["lateness_seconds"] == pytest.approx(0.6)
    assert result["wall_seconds"] == 1.5


def test_resume_uses_absolute_session_ticks_and_local_wall_deadlines():
    clock = FakeClock()
    session = FakeSession(clock)
    session.tick = 8
    observed = []

    def observe(tick):
        observed.append(tick)
        return dict(observed_tick=tick)

    result = replay(session, clock, observe)
    assert observed == [8, 9, 10]
    assert result["first_tick"] == 8
    assert result["action_trace"][0]["scheduled_seconds"] == 0
    assert result["final_status"]["tick"] == 11


@pytest.mark.parametrize("prepared,enabled", [(False, True), (True, False)])
def test_replay_rejects_unprepared_or_blocking_fallback_before_consuming_a_command(prepared, enabled):
    clock = FakeClock()
    session = FakeSession(clock)
    session.prepared, session.enabled = prepared, enabled
    with pytest.raises(RuntimeError, match="Prepare|active RTC"):
        replay(session, clock)
    assert not session.calls and not clock.sleeps


def test_fallback_during_action_call_is_rejected_instead_of_reporting_continuous_rtc():
    clock = FakeClock()
    session = FakeSession(clock)
    original = session.get_action

    def switch(**observation):
        command = original(**observation)
        session.enabled = False
        return command

    session.get_action = switch
    with pytest.raises(RuntimeError, match="blocking fallback"):
        replay(session, clock)
    assert len(session.calls) == 1


def test_fallback_during_servo_interval_is_rejected_before_next_waypoint():
    clock = FakeClock()
    session = FakeSession(clock)
    sleep = clock.sleep

    def switch(seconds):
        sleep(seconds)
        session.enabled = False

    clock.sleep = switch
    with pytest.raises(RuntimeError, match="blocking fallback"):
        replay(session, clock)
    assert len(session.calls) == 1


def test_final_target_is_emitted_when_duration_is_not_on_servo_grid():
    clock = FakeClock()
    session = FakeSession(clock)
    session.execution_hz = 3.0
    result = replay_rtc_client(
        session,
        lambda tick: dict(observed_tick=tick),
        action_steps=2,
        servo_hz=4,
        interpolate=torch.lerp,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result["wall_seconds"] == pytest.approx(2 / 3)
    assert result["servo_trace"][-1]["scheduled_seconds"] == 2 / 3
    assert result["servo_trace"][-1]["alpha"] == 1
    assert result["servo_frames"][-1].item() == 1


def test_backwards_clock_is_rejected_before_issuing_a_command():
    clock = FakeClock()
    session = FakeSession(clock)
    times = iter([100.0, 100.0, 99.0])
    with pytest.raises(ValueError, match="monotonic"):
        replay_rtc_client(
            session,
            lambda tick: dict(observed_tick=tick),
            action_steps=2,
            servo_hz=4,
            interpolate=torch.lerp,
            clock=lambda: next(times),
            sleep=clock.sleep,
        )
    assert not session.calls


@pytest.mark.parametrize("servo_hz", [0, -1, float("nan"), True, 1])
def test_invalid_or_insufficient_servo_frequency_is_rejected(servo_hz):
    clock = FakeClock()
    session = FakeSession(clock)
    with pytest.raises(ValueError, match="servo_hz"):
        replay_rtc_client(
            session,
            lambda tick: dict(observed_tick=tick),
            action_steps=3,
            servo_hz=servo_hz,
            interpolate=torch.lerp,
            clock=clock,
            sleep=clock.sleep,
        )
    assert not session.calls


def test_nonadvancing_injected_sleep_is_rejected_instead_of_hanging():
    clock = FakeClock()
    session = FakeSession(clock)
    with pytest.raises(RuntimeError, match="did not advance"):
        replay_rtc_client(
            session,
            lambda tick: dict(observed_tick=tick),
            action_steps=2,
            servo_hz=4,
            interpolate=torch.lerp,
            clock=clock,
            sleep=lambda seconds: None,
        )


def quaternion_endpoints():
    template = ActionTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[0] = True
    mask[17:21] = True
    start, end = torch.full((55,), float("nan")), torch.full((55,), float("nan"))
    angle = math.radians(170) / 2
    start[0], end[0] = 0, 10
    start[17:21] = torch.tensor([math.sin(angle), 0, 0, math.cos(angle)])
    end[17:21] = torch.tensor([-math.sin(angle), 0, 0, math.cos(angle)])
    return template, mask, start, end


def test_quaternion_interpolation_takes_shortest_rotation_path_and_preserves_masks():
    template, mask, start, end = quaternion_endpoints()
    left, right = start.clone(), end.clone()
    result = interpolate_action(template, mask, start, end, 0.5)
    assert result[0] == 5
    assert torch.linalg.vector_norm(result[17:21]) == pytest.approx(1)
    assert abs(result[17].item()) == pytest.approx(1)
    assert abs(result[20].item()) < 1e-6
    assert not result[~mask].any()
    torch.testing.assert_close(start, left, equal_nan=True)
    torch.testing.assert_close(end, right, equal_nan=True)


def test_quaternion_sign_equivalent_endpoints_do_not_collapse_to_zero():
    template, mask, start, end = quaternion_endpoints()
    end[17:21] = -start[17:21]
    result = interpolate_action(template, mask, start, end, 0.5)
    torch.testing.assert_close(result[17:21], template.sanitize(start, mask)[17:21])


@pytest.mark.parametrize("alpha", [-0.1, 1.1, True, float("nan")])
def test_interpolation_rejects_invalid_alpha(alpha):
    template, mask, start, end = quaternion_endpoints()
    with pytest.raises(ValueError, match="alpha"):
        interpolate_action(template, mask, start, end, alpha)
