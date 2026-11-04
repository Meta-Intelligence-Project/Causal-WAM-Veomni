# SPDX-License-Identifier: OpenMDW-1.1
"""Real localhost transport with a fake policy, separate from model-quality tests."""

import asyncio
import copy
import threading
import time
from contextlib import contextmanager
from queue import Queue

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.causal_cosmos3.serving.client import CausalPolicyClient, RemotePolicyError
from veomni.models.diffusers.causal_cosmos3.serving.server import PolicyConnection, serve
from veomni.models.diffusers.causal_cosmos3.serving.wire import MAX_MESSAGE_BYTES, PROTOCOL, packb, unpackb


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
pytest.importorskip("websockets", minversion="15.0", reason="requires the serving extra")


def measurement(tick=0, *, image_size=False):
    state = torch.zeros(55)
    state[0] = tick
    mask = torch.zeros(55, dtype=torch.bool)
    mask[0] = True
    result = dict(
        state=state,
        state_mask=mask,
        action_mask=mask.clone(),
        image=torch.full((3, 16, 16), tick % 255, dtype=torch.uint8),
    )
    if image_size:
        result["image_size"] = torch.tensor([16, 16, 16, 16])
    return result


class FakePolicy:
    def __init__(self, *, delay=0.0, rtc_enabled=True):
        mask = np.zeros(55, dtype=bool)
        mask[0] = True
        self.metadata = dict(
            actions_per_block=32,
            execute_steps=16,
            controller_hz=30.0,
            history_blocks=32,
            video_stride=4,
            resolution="256",
            rtc_enabled=rtc_enabled,
            state_mask=mask,
            action_mask=mask.copy(),
            available_control_modes=["joint"],
        )
        self.delay = delay
        self.requests = []
        self.clear_count = 0
        self.gate = None
        self.started = threading.Event()

    def infer(self, request):
        self.requests.append(copy.deepcopy(request))
        if request["operation"] == "predict":
            self.started.set()
            if self.gate is not None and not self.gate.wait(5):
                raise TimeoutError("Test policy was never released")
            time.sleep(self.delay)
        action = np.zeros((32, 55), dtype=np.float32)
        action[:, 0] = np.arange(32) + request["tick"]
        return dict(actions=action)

    def clear(self):
        self.clear_count += 1


@contextmanager
def live_server(policy):
    startup = Queue()
    errors = []

    async def run():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        def ready(server):
            startup.put((loop, stop, server.sockets[0].getsockname()[1]))

        await serve(policy, host="127.0.0.1", port=0, ready=ready, stop=stop)

    def worker():
        try:
            asyncio.run(run())
        except BaseException as error:
            errors.append(error)
            startup.put(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        address = startup.get(timeout=10)
        if isinstance(address, BaseException):
            raise address
        loop, stop, port = address
        yield f"ws://127.0.0.1:{port}"
    finally:
        if policy.gate is not None:
            policy.gate.set()
        if "loop" in locals() and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        thread.join(10)
        assert not thread.is_alive(), "Policy server did not release its model worker"
        assert not errors, errors


def test_numpy_wire_preserves_images_actions_and_scalar_types():
    payload = dict(
        image=np.arange(24, dtype=np.uint8).reshape(2, 4, 3), action=np.eye(3, dtype=np.float32), number=np.int64(3)
    )
    result = unpackb(packb(payload))
    for key in ("image", "action"):
        np.testing.assert_array_equal(result[key], payload[key])
        assert result[key].dtype == payload[key].dtype
    assert result["number"] == 3
    with pytest.raises(ValueError, match="dtype"):
        packb(dict(unsafe=np.array([object()], dtype=object)))


def test_message_limit_covers_the_full_cold_34_second_history_recipe():
    history_blocks, actions_per_block, video_stride = 32, 32, 4
    frame_count = history_blocks * actions_per_block // video_stride + 1
    raw_frame_bytes = frame_count * 3 * 640 * 640 * np.dtype(np.uint8).itemsize
    prefix_bytes = actions_per_block * 55 * np.dtype(np.float32).itemsize
    framing_and_metadata_allowance = 1024 * 1024
    assert frame_count == 257
    assert raw_frame_bytes > 256 * 1024 * 1024  # The previous transport limit rejected a valid request.
    assert raw_frame_bytes + prefix_bytes + framing_and_metadata_allowance < MAX_MESSAGE_BYTES
    assert MAX_MESSAGE_BYTES == 512 * 1024 * 1024


def test_ordinary_remote_session_preserves_predict32_execute16_baseline():
    policy = FakePolicy(rtc_enabled=False)
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(measurement(image_size=True), prompt="pick", enable_rtc=False)
        assert session.prepare(**measurement())["rtc_enabled"] is False
        actual = [session.get_action(**measurement(t))[0].item() for t in range(40)]
        assert actual == list(range(40))
        assert [request["tick"] for request in policy.requests] == [0, 16, 32]
        assert all(request.get("rtc") is None for request in policy.requests)
        assert [r["tick"] for r in policy.requests[-1]["history"]] == list(range(0, 32, 4))
        assert session.status()["ordinary_requests"] == 2
        assert client.metadata["protocol"] == PROTOCOL
        assert len(client.request_timings) == 3


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_startup_measures_rpc_latency_and_client_keeps_executing_during_request(monkeypatch, enable_rtc):
    policy = FakePolicy(delay=0.025)
    acknowledged = []

    def accept(rate):
        acknowledged.append(rate)
        return True

    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        upload_delay = 0.01
        original_send = client.connection.send

        def delayed_send(payload):
            time.sleep(upload_delay)
            return original_send(payload)

        monkeypatch.setattr(client.connection, "send", delayed_send)
        session = client.create_session(
            measurement(image_size=True),
            prompt="pick",
            enable_rtc=enable_rtc,
            execution_mode="async",
            execution_hz=1000,
            execution_rate_callback=accept,
        )
        status = session.prepare(**measurement())
        assert status["tick"] == status["published_requests"] == 0
        assert len(policy.requests) == 4  # Ordinary bootstrap, warmup, two matching planner measurements.
        assert policy.requests[0]["rtc"] is None
        assert all((request["rtc"] is not None) == enable_rtc for request in policy.requests[1:])
        assert len(status["latency_samples"]) == 2
        assert all(row["elapsed_seconds"] >= policy.delay + upload_delay for row in status["latency_samples"])
        assert client.last_timing["round_trip_seconds"] >= client.last_timing["infer_seconds"] + upload_delay
        assert acknowledged and status["execution_hz"] < 1000
        assert status["estimated_delay_steps"] == 8

        policy.gate = threading.Event()
        policy.started.clear()
        for tick in range(17):
            assert session.get_action(**measurement(tick))[0].item() == tick
        assert policy.started.wait(2)
        rate = session.execution_hz
        for tick in range(17, 21):
            assert session.get_action(**measurement(tick))[0].item() == tick
        assert session.status()["inference_pending"]
        assert session.execution_hz <= rate
        request = policy.requests[-1]
        assert request["tick"] == 16
        if enable_rtc:
            assert request["rtc"]["delay_steps"] == 8
            np.testing.assert_array_equal(request["rtc"]["previous_actions"][:, 0], np.arange(16, 32))
        else:
            assert request["rtc"] is None
        policy.gate.set()
        session.join(3)
        status = session.status()
        assert not status["last_request"]["accepted"]
        assert status["published_requests"] == 0 and status["inference_ready"]
        assert status["latency_samples"][-1]["elapsed_seconds"] >= policy.delay
        assert [session.get_action(**measurement(t))[0].item() for t in range(21, 25)] == list(range(21, 25))
        assert session.status()["last_request"]["accepted"]
        assert session.status()["published_requests"] == 1


def test_reset_drains_old_work_and_new_episode_cannot_receive_old_outputs():
    policy = FakePolicy(delay=0.005)
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        old = client.create_session(
            measurement(image_size=True),
            prompt="old",
            enable_rtc=True,
            execution_hz=1000,
            execution_rate_callback=lambda rate: True,
        )
        old.prepare(**measurement())
        first_episode = client._episode_id
        policy.gate = threading.Event()
        policy.started.clear()
        for tick in range(17):
            old.get_action(**measurement(tick))
        assert policy.started.wait(2)
        returned, errors = [], []

        def reset():
            try:
                returned.append(client.create_session(measurement(image_size=True), prompt="new", enable_rtc=False))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=reset)
        thread.start()
        deadline = time.monotonic() + 2
        while not old.status()["closed"] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert old.status()["closed"]
        assert not returned
        policy.gate.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        new = returned[0]
        assert old.status()["published_requests"] == 0
        assert client._episode_id != first_episode
        assert new.get_action(**measurement())[0].item() == 0
        ids = [r["request_id"] for r in policy.requests]
        assert ids == sorted(set(ids))
        assert policy.requests[-1]["episode_id"] != policy.requests[0]["episode_id"]


def test_timeout_closes_connection_so_late_reply_cannot_be_reused():
    policy = FakePolicy(delay=0.15)
    with live_server(policy) as uri, CausalPolicyClient(uri, request_timeout=0.03) as client:
        client.reset(prompt="pick", measurement=measurement(image_size=True))
        with pytest.raises(TimeoutError):
            client.infer(dict(tick=16))
        with pytest.raises(RuntimeError, match="closed"):
            client.infer(dict(tick=32))


def test_mismatched_response_id_closes_connection(monkeypatch):
    original = PolicyConnection.infer

    def corrupted(self, request):
        response = original(self, request)
        response["request_id"] += 1
        return response

    monkeypatch.setattr(PolicyConnection, "infer", corrupted)
    with live_server(FakePolicy()) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="identity"):
            client.reset(prompt="pick", measurement=measurement(image_size=True))
        with pytest.raises(RuntimeError, match="closed"):
            client.infer(dict(tick=0))


def test_disabled_server_rejects_rtc_before_bootstrap():
    policy = FakePolicy(rtc_enabled=False)
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="Enable RTC"):
            client.create_session(measurement(image_size=True), prompt="pick", enable_rtc=True)
        assert not policy.requests


def test_second_connection_is_rejected_without_concurrent_policy_use():
    from websockets.exceptions import ConnectionClosed

    policy = FakePolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri):
        with pytest.raises(ConnectionClosed):
            CausalPolicyClient(uri)
        assert not policy.requests


def test_stale_episode_or_duplicate_request_requires_reset():
    policy = FakePolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        client.reset(prompt="pick", measurement=measurement(image_size=True))
        with pytest.raises(RemotePolicyError, match="nonnegative"):
            client.infer(dict(tick=1, revision=-1))
        with pytest.raises(RemotePolicyError, match="Reset"):
            client.infer(dict(tick=2))
        client.reset(prompt="new", measurement=measurement(image_size=True))
        assert client.infer(dict(tick=0))["tick"] == 0


def test_history_references_and_full_payload_produce_identical_model_inputs():
    observed = []
    traces = []
    for references in (False, True):
        policy = FakePolicy(rtc_enabled=False)
        with live_server(policy) as uri, CausalPolicyClient(uri) as client:
            if not references:
                client.metadata.pop("history_transport")
            session = client.create_session(measurement(image_size=True), prompt="pick", enable_rtc=False)
            result = [session.get_action(**measurement(tick))[0].item() for tick in range(81)]
            assert result == list(range(81))
            observed.append(policy.requests)
            traces.append(copy.deepcopy(client.request_timings))
    assert [r["tick"] for r in observed[0]] == [r["tick"] for r in observed[1]]
    for full, cached in zip(*observed, strict=True):
        for key in ("state", "image"):
            np.testing.assert_array_equal(full[key], cached[key])
        assert [r["tick"] for r in full["history"]] == [r["tick"] for r in cached["history"]]
        for a, b in zip(full["history"], cached["history"], strict=True):
            np.testing.assert_array_equal(a["image"], b["image"])
    assert [t["history_frames_sent"] for t in traces[0]] == [0, 0, 8, 8, 16, 16]
    assert [t["history_frames_sent"] for t in traces[1]] == [0, 0, 8, 4, 4, 4]
    assert traces[1][-1]["request_bytes"] < traces[0][-1]["request_bytes"]


def test_current_images_are_never_committed_to_the_history_transport_cache():
    policy = FakePolicy()
    lifecycle = PolicyConnection(policy)
    first = np.full((3, 4, 4), 1, np.uint8)
    lifecycle.infer(dict(operation="reset", episode_id="a", request_id=0, tick=0, image=first))
    lifecycle.infer(dict(operation="predict", episode_id="a", request_id=1, tick=0, image=first + 1))
    assert not lifecycle.history_images
    actual_history = first + 9
    response = lifecycle.infer(
        dict(
            operation="predict",
            episode_id="a",
            request_id=2,
            tick=32,
            image=first + 2,
            history=[dict(tick=0, image=actual_history)],
        )
    )
    assert response["cached_history_ticks"] == [0]
    np.testing.assert_array_equal(lifecycle.history_images[0], actual_history)
    assert 32 not in lifecycle.history_images


def test_missing_or_conflicting_history_references_require_reset_and_do_not_commit():
    policy = FakePolicy()
    lifecycle = PolicyConnection(policy)
    image = np.ones((3, 4, 4), np.uint8)
    lifecycle.infer(dict(operation="reset", episode_id="a", request_id=0, tick=0))
    with pytest.raises(ValueError, match="Unknown historical"):
        lifecycle.infer(dict(operation="predict", episode_id="a", request_id=1, tick=16, history=[dict(tick=0)]))
    assert lifecycle.needs_reset and not lifecycle.history_images
    lifecycle.infer(dict(operation="reset", episode_id="b", request_id=2, tick=0))
    lifecycle.infer(
        dict(operation="predict", episode_id="b", request_id=3, tick=16, history=[dict(tick=0, image=image)])
    )
    with pytest.raises(ValueError, match="overwritten"):
        lifecycle.infer(
            dict(operation="predict", episode_id="b", request_id=4, tick=32, history=[dict(tick=0, image=image + 1)])
        )
    np.testing.assert_array_equal(lifecycle.history_images[0], image)
    assert lifecycle.needs_reset
    lifecycle.infer(dict(operation="reset", episode_id="c", request_id=5, tick=0))
    assert not lifecycle.history_images
    with pytest.raises(ValueError, match="different episode"):
        lifecycle.infer(dict(operation="predict", episode_id="b", request_id=6, tick=16))


def test_history_transport_cache_is_pruned_at_fixed_window_capacity():
    policy = FakePolicy(rtc_enabled=False)
    policy.metadata["history_blocks"] = 1
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(measurement(image_size=True), prompt="pick", enable_rtc=False)
        for tick in range(81):
            session.get_action(**measurement(tick))
        assert client._cached_history_ticks == set(range(48, 80, 4))


def test_failed_model_request_does_not_commit_staged_history():
    class FailingPolicy(FakePolicy):
        def infer(self, request):
            if request["operation"] == "predict":
                raise RuntimeError("model failed")
            return super().infer(request)

    lifecycle = PolicyConnection(FailingPolicy())
    lifecycle.infer(dict(operation="reset", episode_id="a", request_id=0, tick=0))
    with pytest.raises(RuntimeError, match="model failed"):
        lifecycle.infer(
            dict(
                operation="predict",
                episode_id="a",
                request_id=1,
                tick=32,
                history=[dict(tick=0, image=np.zeros((3, 4, 4), np.uint8))],
            )
        )
    assert lifecycle.needs_reset and not lifecycle.history_images


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_adaptive_false_cannot_silently_enable_an_explicit_policy(enable_rtc):
    from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy

    policy = FakePolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="conflicts"):
            client.create_session(
                measurement(image_size=True),
                prompt="pick",
                enable_rtc=enable_rtc,
                adaptive_execution=False,
                execution_policy=RTCExecutionPolicy(),
            )
        assert not policy.requests


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_explicit_adaptive_execution_requires_a_real_client_pacing_callback_before_bootstrap(enable_rtc):
    with live_server(FakePolicy()) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="callback|acknowledg"):
            client.create_session(
                measurement(image_size=True),
                prompt="pick",
                enable_rtc=enable_rtc,
                execution_mode="async",
                adaptive_execution=True,
            )


class StreamingPolicy(FakePolicy):
    def __init__(self):
        super().__init__()
        self.metadata["streaming_vae"] = True
        self.metadata["history_padding"] = "repeat_first"
        self.events = []

    def infer(self, request):
        self.events.append(("infer", request["tick"], threading.get_ident()))
        return super().infer(request)

    def observe(self, request):
        self.events.append(("observe", request["tick"], threading.get_ident()))
        return {}

    def prepare_window(self, request):
        self.events.append(("prepare", request["anchor_tick"], threading.get_ident()))
        return {}


def test_streaming_observations_are_sent_during_inflight_inference_without_gpu_races(monkeypatch):
    policy = StreamingPolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        client.reset(prompt="pick", measurement=measurement(image_size=True))
        policy.gate = threading.Event()
        policy.started.clear()
        sent = threading.Event()
        original_send = client.connection.send

        def send(payload):
            if unpackb(payload).get("operation") == "observe":
                sent.set()
            return original_send(payload)

        monkeypatch.setattr(client.connection, "send", send)
        result, errors = [], []

        def predict():
            try:
                result.append(client.infer(dict(tick=16)))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=predict)
        thread.start()
        assert policy.started.wait(2)
        for tick in range(21):
            client.observe(tick=tick, image=measurement(tick)["image"])
        client.prepare_window(anchor_tick=32)
        assert sent.wait(2), "A prediction waiting for its result must not lock the observation sender"
        assert thread.is_alive() and not any(event[0] == "observe" for event in policy.events)
        policy.gate.set()
        thread.join(3)
        assert not errors and result[0]["tick"] == 16
        client.infer(dict(tick=32))  # FIFO barrier after all observe/prepare jobs.
        assert [event[1] for event in policy.events if event[0] == "observe"] == list(range(21))
        assert ("prepare", 32) in [event[:2] for event in policy.events]
        assert len({event[2] for event in policy.events}) == 1


def test_streaming_ordinary_execution_remains_independent_of_rtc():
    policy = StreamingPolicy()
    policy.metadata["rtc_enabled"] = False
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(
            measurement(image_size=True), prompt="pick", enable_rtc=False, execution_mode="blocking"
        )
        assert [session.get_action(**measurement(t))[0].item() for t in range(34)] == list(range(34))
        client.infer(dict(tick=34))
        assert [event[1] for event in policy.events if event[0] == "observe"] == list(range(34))
        assert [event[1] for event in policy.events if event[0] == "prepare"] == [16, 32, 48]
        assert not any(request.get("rtc") for request in policy.requests)


def test_streaming_callbacks_share_the_rtc_clock_and_preserve_guided_requests():
    policy = StreamingPolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(
            measurement(image_size=True),
            prompt="pick",
            enable_rtc=True,
            adaptive_execution=False,
            calibration_warmups=0,
            calibration_samples=1,
        )
        assert session.prepare(**measurement())["rtc_enabled"]
        for tick in range(17):
            assert session.get_action(**measurement(tick))[0].item() == tick
        session.join(3)
        client.infer(dict(tick=17))
        assert [event[1] for event in policy.events if event[0] == "observe"] == list(range(17))
        guided = [request for request in policy.requests if request.get("rtc") is not None]
        assert guided[-1]["tick"] == 16 and guided[-1]["rtc"]["delay_steps"] == 8
        assert session.status()["published_requests"] == 0
        assert [session.get_action(**measurement(t))[0].item() for t in range(17, 25)] == list(range(17, 25))
        assert session.status()["published_requests"] == 1


class AsyncPolicy(StreamingPolicy):
    """Distinct future values expose premature handoff or expired-row replay."""

    def infer(self, request):
        result = super().infer(request)
        if request["operation"] == "predict":
            result["actions"][:, 0] += 1000
        return result


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("ready_before_handoff", [False, True])
@pytest.mark.parametrize("enable_rtc", [False, True])
def test_async_rpc_keeps_old_actions_until_fixed_handoff_and_skips_expired_rows(
    streaming, ready_before_handoff, enable_rtc
):
    policy = AsyncPolicy()
    policy.metadata["streaming_vae"] = streaming
    policy.metadata["rtc_enabled"] = enable_rtc
    policy.metadata["handoff_steps"] = 24
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        # streaming默认走async；无streaming也可显式选择同一执行器。
        mode = "auto" if streaming else "async"
        session = client.create_session(
            measurement(image_size=True), prompt="pick", execution_mode=mode, enable_rtc=enable_rtc
        )
        assert session.status()["execution_mode"] == ("async_rtc" if enable_rtc else "async_non_rtc")
        policy.gate = threading.Event()
        policy.started.clear()
        assert [session.get_action(**measurement(t))[0].item() for t in range(17)] == list(range(17))
        assert policy.started.wait(2)
        assert session.status()["inference_pending"]
        assert policy.requests[-1]["tick"] == 16
        if enable_rtc:
            assert policy.requests[-1]["rtc"]["delay_steps"] == 8
            np.testing.assert_array_equal(policy.requests[-1]["rtc"]["previous_actions"][:, 0], np.arange(16, 32))
        else:
            assert policy.requests[-1]["rtc"] is None
        if ready_before_handoff:
            policy.gate.set()
            session.join(3)
            assert session.status()["inference_ready"]
        assert [session.get_action(**measurement(t))[0].item() for t in range(17, 24)] == list(range(17, 24))
        assert session.status()["published_requests"] == 0
        if not ready_before_handoff:
            returned, errors = [], []

            def consume_handoff():
                try:
                    returned.append(session.get_action(**measurement(24))[0].item())
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=consume_handoff)
            thread.start()
            thread.join(0.05)
            assert thread.is_alive(), "A late candidate must wait at R, not extend the old chunk"
            policy.gate.set()
            thread.join(3)
            assert not thread.is_alive() and not errors
            assert returned == [1024]  # candidate tick16 + expired8 + marker1000.
        else:
            assert session.get_action(**measurement(24))[0].item() == 1024
        assert session.status()["chunk_start_tick"] == 16
        assert session.status()["published_requests"] == 1
        assert all((request.get("rtc") is not None) == enable_rtc for request in policy.requests[1:])


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_async_rpc_reset_and_close_drain_the_old_inflight_request(operation):
    policy = AsyncPolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        old = client.create_session(measurement(image_size=True), prompt="old")
        policy.gate = threading.Event()
        policy.started.clear()
        for tick in range(17):
            old.get_action(**measurement(tick))
        assert policy.started.wait(2)
        returned, errors = [], []

        def drain():
            try:
                returned.append(
                    client.create_session(measurement(image_size=True), prompt="new")
                    if operation == "reset"
                    else old.close()
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=drain)
        thread.start()
        thread.join(0.05)
        assert thread.is_alive(), "Reset/close must drain the previous model request"
        policy.gate.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert old.status()["closed"] and old.status()["published_requests"] == 0
        if operation == "reset":
            fresh = returned[0]
            assert fresh.status()["tick"] == 0
            assert fresh.get_action(**measurement())[0].item() == 0


@pytest.mark.parametrize("mode", ["blocking", "invalid"])
def test_execution_mode_conflicts_are_rejected_before_bootstrap(mode):
    policy = FakePolicy()
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="execution_mode"):
            client.create_session(measurement(image_size=True), prompt="pick", enable_rtc=True, execution_mode=mode)
        assert not policy.requests


def test_async_only_validates_schedule_when_the_client_selects_it():
    policy = StreamingPolicy()
    policy.metadata.update(actions_per_block=16, action_horizon=16, execute_steps=16, handoff_steps=None)
    with live_server(policy) as uri, CausalPolicyClient(uri) as client:
        with pytest.raises(ValueError, match="S|trigger|execute|horizon"):
            client.create_session(measurement(image_size=True), prompt="pick")
        assert not policy.requests


@pytest.mark.parametrize("past,horizon", [(48, 32), (1024, 32), (48, 48), (0, 16)])
def test_compact_serving_loopback_has_only_real_prediction_actions_and_one_state(past, horizon):
    from tests.models.diffusers.causal_cosmos3_serving.test_serving_model import policy

    service = policy(
        enable_rtc=False,
        history_steps=past,
        action_horizon=horizon,
        history_padding="repeat_first",
    )
    # live_server's cleanup hooks are shared with the older FakePolicy fixture.
    service.gate = None
    captured = []
    service.model.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)

    def observed(tick):
        result = measurement(tick, image_size=True)
        result["state_mask"] = service.adapter.state_mask.clone()
        result["action_mask"] = service.adapter.action_mask.clone()
        return result

    with live_server(service) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(observed(0), prompt="pick", enable_rtc=False, execution_mode="blocking")
        for tick in range(34):
            value = observed(tick)
            value.pop("image_size")
            assert session.get_action(**value).shape == (55,)
        assert client.metadata["history_steps"] == past
        assert client.metadata["action_horizon"] == horizon
        assert captured
        for inputs in captured:
            assert inputs["action_tokens"].shape == (horizon, 55)
            assert inputs["state_tokens"].shape == (1, 55)
            assert inputs["layout"].decision_windows == (1 + past // 16,)


def test_default_serving_uses_streaming_and_selects_async_without_rtc():
    from tests.models.diffusers.causal_cosmos3_serving.test_serving_model import policy

    service = policy(enable_rtc=False)
    service.gate = None
    observed = measurement(image_size=True)
    observed["state_mask"] = service.adapter.state_mask.clone()
    observed["action_mask"] = service.adapter.action_mask.clone()
    with live_server(service) as uri, CausalPolicyClient(uri) as client:
        session = client.create_session(observed, prompt="pick", enable_rtc=False)
        assert client.metadata["streaming_vae"] is True
        assert session.status()["execution_mode"] == "async_non_rtc"
        assert service.builder is not None
        assert service.window_encoder.state is None  # Finished bootstrap released its VAE caches.
        session.close()
