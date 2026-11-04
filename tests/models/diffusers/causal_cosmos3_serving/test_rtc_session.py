# SPDX-License-Identifier: OpenMDW-1.1
"""Tick alignment, committed commands and asynchronous RTC lifecycle tests."""

from threading import Event, Thread, get_ident

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix
from veomni.models.diffusers.causal_cosmos3.serving.rtc_session import (
    RTCBufferUnderrun,
    RTCDeadlineMiss,
    RTCError,
    RTCPlannerError,
    RTCSession,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def actions(horizon=8, start=0):
    result = torch.zeros(horizon, 55)
    result[:, 0] = start + torch.arange(horizon)
    result[:, 28] = 0.5
    return result


def inputs(tick=0):
    state = torch.zeros(55)
    state[0] = tick
    state[28] = float("nan")  # Invalid absolute-field state is allowed.
    state_mask = torch.zeros(55, dtype=torch.bool)
    state_mask[0] = True
    action_mask = state_mask.clone()
    action_mask[28] = True
    return dict(state=state, state_mask=state_mask, action_mask=action_mask, image=torch.full((3, 4, 4), tick))


class GatePlanner:
    def __init__(self, horizon=8, requests=1):
        self.horizon = horizon
        self.started = [Event() for _ in range(requests)]
        self.release = [Event() for _ in range(requests)]
        self.calls = []

    def __call__(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        self.started[index].set()
        if not self.release[index].wait(10):
            raise TimeoutError("Test did not release the planner")
        return actions(self.horizon, start=100 * (index + 1))

    def unblock(self):
        for event in self.release:
            event.set()


def session(planner, **kwargs):
    return RTCSession(
        planner,
        actions(),
        template=ActionTemplate55(),
        min_execution_steps=3,
        initial_delay_steps=2,
        **kwargs,
    )


def consume(controller, begin, end):
    return [controller.get_action(**inputs(i))[0].item() for i in range(begin, end)]


def test_partial_execution_replans_and_skips_already_consumed_prefix():
    planner = GatePlanner()
    controller = session(planner)
    try:
        assert consume(controller, 0, 4) == [0, 1, 2, 3]
        assert planner.started[0].wait(2)
        request = planner.calls[0]
        assert request["tick"] == 3
        assert request["state"][0].item() == 3
        assert torch.isfinite(request["state"]).all()
        assert request["image"].eq(3).all()
        assert [r.tick for r in request["history"]] == [0, 1, 2]
        assert [r.image[0, 0, 0].item() for r in request["history"]] == [0, 1, 2]
        assert not hasattr(request["history"][0], "state")
        assert isinstance(request["rtc"], RTCPrefix)
        torch.testing.assert_close(request["rtc"].previous_actions[:, 0], torch.arange(3, 8).float())
        assert request["rtc"].delay_steps == 2
        assert consume(controller, 4, 5) == [4]
        planner.release[0].set()
        status = controller.wait_for_inference(2)
        assert status["chunk_start_tick"] == 3
        assert status["tick"] == 5
        assert status["last_request"]["actual_delay_steps"] == 2
        assert status["published_requests"] == 1
        # The new plan began at tick3. Its outputs 0 and 1 are already consumed.
        assert consume(controller, 5, 6) == [102]
    finally:
        planner.unblock()
        controller.close()


def test_fast_result_keeps_unconsumed_committed_commands_exactly():
    planner = GatePlanner()
    controller = session(planner)
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        controller.join(2)
        # Only action3 was consumed while inference ran. Reserved action4 must
        # remain old action4, even if the planner returned an arbitrary value.
        assert consume(controller, 4, 6) == [4, 102]
        assert controller.status()["last_request"]["actual_delay_steps"] == 1
    finally:
        planner.unblock()
        controller.close()


def test_delay_estimate_updates_and_second_request_uses_new_buffer_origin():
    planner = GatePlanner(requests=2)
    controller = session(planner, delay_history_size=1)
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        controller.join(2)
        assert controller.status()["estimated_delay_steps"] == 1
        assert consume(controller, 4, 7) == [4, 102, 103]
        assert planner.started[1].wait(2)
        second = planner.calls[1]
        assert second["tick"] == 6
        assert second["rtc"].delay_steps == 1
        torch.testing.assert_close(second["rtc"].previous_actions[:, 0], torch.arange(103, 108).float())
        assert second["state"][0] == 6
        planner.release[1].set()
        status = controller.join(2)
        assert status["chunk_start_tick"] == 6
        assert status["published_requests"] == 2
        assert consume(controller, 7, 8) == [201]
    finally:
        planner.unblock()
        controller.close()


def test_balanced_deadline_publication_does_not_skip_next_planning_tick():
    planner = GatePlanner(requests=2)
    controller = RTCSession(
        planner, actions(), template=ActionTemplate55(), min_execution_steps=4, initial_delay_steps=4
    )
    published, release_publication = Event(), Event()

    class PublishBarrier:
        """Pause after the publish critical section, before worker cleanup."""

        def __init__(self, lock):
            self.lock = lock
            self.seen = False

        def __enter__(self):
            return self.lock.__enter__()

        def __exit__(self, *args):
            pause = not self.seen and get_ident() == controller._worker_thread and controller._published_requests == 1
            result = self.lock.__exit__(*args)
            if pause:
                self.seen = True
                published.set()
                if not release_publication.wait(10):
                    raise TimeoutError("Test did not release publication barrier")
            return result

    controller._lock = PublishBarrier(controller._lock)
    try:
        assert consume(controller, 0, 8) == list(range(8))
        assert planner.started[0].wait(2)
        planner.release[0].set()
        assert published.wait(2)
        # The new plan starts at tick4, and four actions have elapsed. Inference
        # for its successor must start at tick8, even while old cleanup is paused.
        assert consume(controller, 8, 9) == [104]
        assert controller._pending is not None and controller._pending.tick == 8
        release_publication.set()
        assert planner.started[1].wait(2)
        assert planner.calls[1]["tick"] == 8
        torch.testing.assert_close(planner.calls[1]["rtc"].previous_actions[:, 0], torch.arange(104, 108).float())
        planner.release[1].set()
        assert controller.join(2)["published_requests"] == 2
        assert consume(controller, 9, 10) == [105]
    finally:
        release_publication.set()
        planner.unblock()
        controller.close()


def test_late_plan_is_rejected_instead_of_installing_uncommitted_actions():
    planner = GatePlanner()
    controller = RTCSession(
        planner, actions(), template=ActionTemplate55(), min_execution_steps=3, initial_delay_steps=1
    )
    try:
        consume(controller, 0, 5)
        assert planner.started[0].wait(2)
        planner.release[0].set()
        with pytest.raises(RTCDeadlineMiss, match="consumed 2.*reserved prefix 1"):
            controller.join(2)
        status = controller.status()
        assert status["published_requests"] == 0
        assert status["chunk_start_tick"] == 0
        assert not status["last_request"]["accepted"]
        with pytest.raises(RTCDeadlineMiss):
            controller.get_action(**inputs(5))
    finally:
        planner.unblock()
        controller.close()


def test_buffer_underrun_raises_without_repeating_the_last_action():
    planner = GatePlanner()
    controller = session(planner)
    try:
        assert consume(controller, 0, 8) == list(range(8))
        with pytest.raises(RTCBufferUnderrun, match="exhausted"):
            controller.get_action(**inputs(8))
        assert controller.status()["tick"] == 8
        assert controller.status()["remaining_actions"] == 0
        planner.release[0].set()
        with pytest.raises(RTCBufferUnderrun):
            controller.join(2)
    finally:
        planner.unblock()
        controller.close()


@pytest.mark.parametrize(
    "result", [RuntimeError("broken planner"), torch.ones(7, 55), torch.full((8, 55), float("nan"))]
)
def test_background_failures_are_reported_at_control_boundary(result):
    def planner(**kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    with session(planner) as controller:
        consume(controller, 0, 4)
        with pytest.raises(RTCPlannerError):
            controller.wait_for_inference(2)
        with pytest.raises(RTCPlannerError):
            controller.get_action(**inputs(4))


def test_reset_invalidates_running_old_episode_before_joining():
    planner = GatePlanner()
    controller = session(planner)
    finished = Event()
    thread = None
    try:
        consume(controller, 0, 4)
        assert planner.started[0].wait(2)

        def reset():
            controller.reset(actions(start=500))
            finished.set()

        invalidated = Event()
        original_lock = controller._lock

        class ResetBarrier:
            def __enter__(self):
                return original_lock.__enter__()

            def __exit__(self, *args):
                resetting = controller._resetting
                result = original_lock.__exit__(*args)
                if resetting:
                    invalidated.set()
                return result

        # Observe episode invalidation, independently of how the old Future is
        # drained (including failed jobs whose exception must not block reset).
        controller._lock = ResetBarrier()
        thread = Thread(target=reset)
        thread.start()
        # Acquire/release is deterministic: reset changes the version while the
        # planner remains gated, so it cannot publish into the new episode.
        assert invalidated.wait(2)
        interim = controller.status()
        assert interim["resetting"]
        assert interim["revision"] == 1
        assert not finished.is_set()
        with pytest.raises(RTCError, match="resetting"):
            controller.get_action(**inputs(4))
        planner.release[0].set()
        thread.join(2)
        assert finished.is_set()
        status = controller.status()
        assert status["revision"] == 1
        assert status["published_requests"] == 0
        assert status["tick"] == 0 and status["chunk_start_tick"] == 0
        assert status["delay_history_steps"] == [2]
        assert consume(controller, 0, 1) == [500]
    finally:
        planner.unblock()
        if thread is not None:
            thread.join(2)
        controller.close()


def test_close_blocks_ticks_and_does_not_publish_old_inference():
    planner = GatePlanner()
    controller = session(planner)
    consume(controller, 0, 4)
    assert planner.started[0].wait(2)
    controller.close(wait=False)
    with pytest.raises(RTCError, match="closed"):
        controller.get_action(**inputs(4))
    planner.release[0].set()
    controller.close(wait=True)
    assert controller.status()["published_requests"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_delay_steps": 0},
        {"initial_delay_steps": True},
        {"initial_delay_steps": 5},
        {"min_execution_steps": 8},
        {"min_execution_steps": 0},
        {"delay_history_size": 0},
        {"history_steps": 0},
        {"controller_hz": 0},
        {"controller_hz": float("nan")},
        {"controller_hz": True},
        {"beta": 0},
    ],
)
def test_invalid_controller_geometry_is_rejected_before_starting_a_worker(kwargs):
    with pytest.raises(ValueError):
        RTCSession(lambda **_: actions(), actions(), template=ActionTemplate55(), **kwargs)


def test_action_mask_changes_require_reset_and_invalid_slots_are_zero():
    with session(lambda **_: actions()) as controller:
        first = controller.get_action(**inputs(0))
        assert first[28] == 0.5 and first[1:28].eq(0).all()
        changed = inputs(1)
        changed["action_mask"][1] = True
        changed["state_mask"][1] = True
        with pytest.raises(ValueError, match="validity cannot change"):
            controller.get_action(**changed)
        assert controller.status()["tick"] == 1


def test_planner_inputs_are_copied_before_control_caller_mutates_them():
    planner = GatePlanner()
    controller = session(planner)
    try:
        consume(controller, 0, 3)
        data = inputs(3)
        controller.get_action(**data)
        data["state"][0] = 999
        data["image"].fill_(999)
        data["action_mask"].fill_(False)
        assert planner.started[0].wait(2)
        assert planner.calls[0]["state"][0] == 3
        assert planner.calls[0]["image"].eq(3).all()
        assert planner.calls[0]["action_mask"].any()
    finally:
        planner.unblock()
        controller.close()
