"""Piper protocol regression tests: no robot, cameras or model weights."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.client_server.client.piper.protocol import CosmosProtocol
from tasks.client_server.client.runtime import InferenceClient
from tasks.client_server.wire import packb, unpackb


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Connection:
    def __init__(self):
        self.requests = []
        self.fail = False

    def send(self, payload):
        self.requests.append(unpackb(payload))

    def recv(self):
        return packb({"error": "failed"} if self.fail else {"execute_steps": 32})


def protocol(streaming=False):
    client = object.__new__(InferenceClient)
    client.connection = Connection()
    client.control_mode = "joint"
    client.metadata = dict(actions_per_block=32, execute_steps=32, fps=30, video_stride=4)
    if streaming:
        client.metadata.update(observation_mode="streaming", observation_streaming=True, observation_chunk_steps=16)
    return CosmosProtocol(client, SimpleNamespace(model_fps=30, prompt="pick box", control_mode="movej"))


def frame(index):
    return {
        key: np.full((2, 2, 3), index, dtype=np.uint8) for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }


def test_piper_full_block_feedback_and_reset():
    p = protocol()
    p.reset()
    initial = dict(images=frame(0), state={"measured": 0})
    p.infer(initial)
    first = p.client.connection.requests[-1]
    assert first["observation_step"] == 0
    assert "observation_stride" not in first
    for step in (32, 64):
        feedback = [dict(images=frame(i)) for i in range(1, 33)]
        final = dict(images=frame(32), state={"measured": step})
        p.finish_block(feedback, {}, final, True)
        p.infer(final)
        request = p.client.connection.requests[-1]
        assert request["observation_step"] == step
        assert request["executed_steps"] == 32
        assert request["state"] == {"measured": step}
        assert request["observation_stride"] == 4
        assert [int(f["cam_high"][0, 0, 0]) for f in request["obs"]] == list(range(4, 33, 4))
    p.reset()
    p.infer(initial)
    assert p.client.connection.requests[-1]["observation_step"] == 0


def test_piper_rejects_partial_feedback_and_server_errors():
    p = protocol()
    with pytest.raises(ValueError, match="one feedback frame"):
        p.finish_block([dict(images=frame(1))], {}, {}, True)
    assert p.observation_step == 0
    p.client.connection.fail = True
    with pytest.raises(RuntimeError, match="failed"):
        p.infer(dict(images=frame(0), state={}))
    assert p.predicted_chunks == 0


def test_piper_rejects_partial_horizon():
    p = protocol()
    p.client.metadata["execute_steps"] = 16
    with pytest.raises(ValueError, match="full-block"):
        CosmosProtocol(p.client, p.cfg)


class StreamingConnection(Connection):
    """A blocking RPC double: concurrent socket access fails immediately."""

    def __init__(self):
        import threading

        super().__init__()
        self.active = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.bad_ack = False

    def send(self, payload):
        assert not self.active, "Concurrent requests on the inference socket"
        self.active = True
        super().send(payload)

    def recv(self):
        request = self.requests[-1]
        try:
            if request.get("observe_only"):
                self.started.set()
                assert self.release.wait(5), "Action execution blocked on the observation upload"
                if self.fail:
                    return packb(dict(error="observation failed"))
                return packb(dict(observed=True, observation_step=-1 if self.bad_ack else request["observation_step"]))
            return packb(dict(execute_steps=32, action=dict(joint=np.zeros((32, 14)))))
        finally:
            self.active = False


@pytest.mark.parametrize("duplicate_mapping", [False, True])
def test_piper_loop_uploads_during_motion_and_preserves_feedback(tmp_path, monkeypatch, duplicate_mapping):
    import sys

    from tasks.client_server.client.piper import protocol as module

    p = protocol(streaming=True)
    p.client.connection = wire = StreamingConnection()
    cfg = p.cfg
    cfg.__dict__.update(
        max_chunks=2,
        model_family="causal-cosmos",
        report_dir=str(tmp_path),
        wait_tol_deg=1,
        action_dt=0.03,
        left_gripper_scale=1,
        right_gripper_scale=1,
        gripper_close_threshold=0,
    )
    report = module.RunReport(cfg)
    logger = SimpleNamespace(
        save_dir=tmp_path,
        log_key_frame=lambda *args: None,
    )
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 3
    camera = SimpleNamespace(get_frames=lambda: frames)
    captured = []

    def observe(*args, feedback_only=False):
        if feedback_only:
            captured.append(len(captured) + 1)
            return dict(images=frame(captured[-1]))
        return dict(images=frame(0), state=dict(measured=len(captured)))

    p.observe = observe
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "wait_robot_target", lambda *args: dict(reached=True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=lambda *args: True))
    commands = []
    factor = 2 if duplicate_mapping else 1

    def send(*args):
        commands.append(True)
        if len(commands) == 16 // factor + 1:
            assert wire.started.wait(5)
            assert wire.active  # A later motion command ran before the upload completed.
            wire.release.set()

    smoother = (
        SimpleNamespace(smooth_full_action=lambda raw: (raw[::2], np.arange(32) // 2)) if duplicate_mapping else None
    )
    try:
        total = module.run_loop(None, camera, p, cfg, logger, smoother, send, lambda: False, report)
    finally:
        wire.release.set()
        p.close()
    assert total == len(commands) == 64 // factor
    uploads = [r for r in wire.requests if r.get("observe_only")]
    assert [r["observation_step"] for r in uploads] == [16, 32, 48, 64]
    actual = [int(f["cam_high"][0, 0, 0]) for r in uploads for f in r["obs"]]
    expected = np.repeat(np.arange(1, 64 // factor + 1), factor)[3::4].tolist()
    assert actual == expected
    inference = [r for r in wire.requests if "state" in r]
    assert "obs" in inference[0] and "obs" not in inference[1]
    assert inference[1]["state"] == dict(measured=32 // factor)


@pytest.mark.parametrize("group", [None, True, 0, 3, 64])
def test_piper_rejects_invalid_streaming_metadata(group):
    p = protocol(streaming=True)
    p.client.metadata["observation_chunk_steps"] = group
    with pytest.raises(ValueError, match="complete streamed observation groups"):
        CosmosProtocol(p.client, p.cfg)


@pytest.mark.parametrize("bad_ack", [False, True])
def test_piper_upload_error_prevents_later_requests(bad_ack):
    p = protocol(streaming=True)
    p.client.connection = wire = StreamingConnection()
    wire.bad_ack = bad_ack
    wire.fail = not bad_ack
    p.reset()
    p.infer(dict(images=frame(0), state={}))
    feedback = [dict(images=frame(i)) for i in range(32)]
    try:
        p.submit_feedback(feedback)
        assert wire.started.wait(5)
        with pytest.raises(ValueError, match="Finish the observation block"):
            p.infer(dict(images=frame(0), state={}))
        wire.release.set()
        with pytest.raises((RuntimeError, ValueError), match="observation"):
            p.finish_block(feedback, {}, {}, True)
        with pytest.raises((RuntimeError, ValueError), match="observation"):
            p.check_feedback()
    finally:
        wire.release.set()
        with pytest.raises((RuntimeError, ValueError), match="observation"):
            p.close()
    assert [r["observation_step"] for r in wire.requests if r.get("observe_only")] == [16]


@pytest.mark.parametrize("operation", ["close", "reset"])
def test_piper_stop_or_reset_drains_active_upload_and_cancels_queue(operation):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    p = protocol(streaming=True)
    p.client.connection = wire = StreamingConnection()
    p.reset()
    p.infer(dict(images=frame(0), state={}))
    p.submit_feedback([dict(images=frame(i)) for i in range(32)])
    assert wire.started.wait(5)
    # Observe cancellation directly so the release is deterministic, without sleeps.
    cancelled = threading.Event()
    p._uploads[-1].add_done_callback(lambda future: cancelled.set() if future.cancelled() else None)
    with ThreadPoolExecutor(max_workers=1) as cleanup:
        closing = cleanup.submit(getattr(p, operation))
        try:
            assert cancelled.wait(5)
            assert not closing.done()
        finally:
            wire.release.set()
        closing.result(timeout=5)
    assert [r["observation_step"] for r in wire.requests if r.get("observe_only")] == [16]
    if operation == "reset":
        assert wire.requests[-1]["reset"] is True
        p.infer(dict(images=frame(0), state={}))
        assert wire.requests[-1]["observation_step"] == 0
    p.close()


def test_incompatible_server_never_enables_hardware(tmp_path, monkeypatch):
    from tasks.client_server.client.piper import __main__ as entry

    class RejectedClient:
        def __init__(self, *args, **kwargs):
            raise ValueError("incompatible server")

    calls = []
    hardware = SimpleNamespace(
        InferenceLogger=lambda *args: None,
        PiperController=lambda **kwargs: calls.append("hardware"),
    )
    cfg = SimpleNamespace(
        report_dir=str(tmp_path),
        save_dir=str(tmp_path),
        prompt="pick box",
        model_family="causal-cosmos",
        control_mode="movej",
        server_host="localhost",
        server_port=8000,
    )
    monkeypatch.setattr(entry, "PiperPolicyClient", RejectedClient)
    assert entry.run(cfg, hardware, tmp_path / "stop.json") == 1
    assert calls == []


def test_piper_connection_interrupt_closes_partial_socket(monkeypatch):
    from tasks.client_server.client.piper import __main__ as entry

    closed = []

    def connect(self, *args, **kwargs):
        self.connection = SimpleNamespace(close=lambda: closed.append(True))
        raise KeyboardInterrupt("Stop requested")

    monkeypatch.setattr(entry.CausalPolicyClient, "__init__", connect)
    with pytest.raises(KeyboardInterrupt, match="Stop requested"):
        entry.PiperPolicyClient("ws://unused")
    assert closed == [True]


def test_piper_stop_during_connection_retry(monkeypatch):
    from threading import Event

    from tasks.client_server.client.piper import __main__ as entry

    stopped = Event()
    calls = []

    def connect(*args, **kwargs):
        calls.append(True)
        raise ConnectionRefusedError("offline")

    monkeypatch.setattr(entry.CausalPolicyClient, "__init__", connect)
    monkeypatch.setattr(entry.time, "sleep", lambda _: stopped.set())
    with pytest.raises(KeyboardInterrupt, match="Stop requested"):
        entry.PiperPolicyClient("ws://unused", stop_check=stopped.is_set)
    assert len(calls) == 1


def test_piper_stop_while_waiting_for_server_metadata():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Thread

    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    from tasks.client_server.client.piper.__main__ import PiperPolicyClient

    entered, stopped, closed = Event(), Event(), Event()

    def handler(socket):
        entered.set()
        try:
            socket.recv()
        except ConnectionClosed:
            closed.set()

    with serve(handler, "127.0.0.1", 0) as server:
        listener = Thread(target=server.serve_forever)
        listener.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                future = worker.submit(
                    PiperPolicyClient,
                    f"ws://127.0.0.1:{server.socket.getsockname()[1]}",
                    stop_check=stopped.is_set,
                )
                try:
                    assert entered.wait(2)
                    stopped.set()
                    with pytest.raises(KeyboardInterrupt, match="Stop requested"):
                        future.result(timeout=2)
                    assert closed.wait(2)
                finally:
                    stopped.set()
                    server.shutdown()
        finally:
            server.shutdown()
            listener.join(timeout=5)


def test_piper_failed_video_export_does_not_poison_cleanup():
    p = protocol()

    def export(path):
        raise RuntimeError("video encoder failed")

    p.client.save_prediction_video = export
    with pytest.raises(RuntimeError, match="video encoder failed"):
        p.save_prediction_video("unused.mp4", stop_check=lambda: False)
    p.close()
    assert p._request is None and p._sender is None


def test_piper_stop_interrupts_video_export(tmp_path, monkeypatch):
    from threading import Event, Thread

    from tasks.client_server.client.piper import __main__ as entry

    entered, released = Event(), Event()
    stop_file = tmp_path / "stop.json"

    def export(*args):
        entered.set()
        assert released.wait(5), "Video request must be aborted before waiting for a response"
        raise RuntimeError("transport closed")

    client = SimpleNamespace(
        metadata=dict(actions_per_block=32, execute_steps=32, fps=30, video_stride=4, left_arm_dof=6, right_arm_dof=6),
        save_prediction_video=export,
        close=released.set,
    )
    robot = SimpleNamespace(set_speed_percent=lambda _: None)
    piper = SimpleNamespace(
        connect=lambda: None, enable=lambda: None, close=lambda: None, robot_l=robot, robot_r=robot
    )
    hardware = SimpleNamespace(
        InferenceLogger=lambda *args: SimpleNamespace(start=lambda _: None, finish=lambda *args: None),
        PiperController=lambda **kwargs: piper,
        CameraManager=lambda **kwargs: SimpleNamespace(start=lambda: None, stop=lambda: None),
        restore_movej_mode=lambda _: None,
    )
    cfg = SimpleNamespace(
        report_dir=str(tmp_path),
        prompt="Move.",
        model_family="causal-cosmos",
        control_mode="movej",
        server_host="unused",
        server_port=8000,
        model_fps=30,
        save_prediction_video=True,
        can_left="left",
        can_right="right",
        movej_speed_percent=0.06,
        smooth_method="none",
        cam_high="head",
        cam_left="left",
        cam_right="right",
        cam_width=6,
        cam_height=4,
        cam_fps=30,
    )
    monkeypatch.setattr(entry, "PiperPolicyClient", lambda *args, **kwargs: client)
    import sys

    monkeypatch.setitem(sys.modules, "rospy", SimpleNamespace(core=SimpleNamespace(is_initialized=lambda: True)))

    def loop(piper, camera, protocol, cfg, logger, smoother, send, check_stop, report):
        protocol.predicted_chunks = 1
        report.update(status="complete")

    hardware.send_action = lambda *args: None

    def stop_export():
        if entered.wait(5):
            stop_file.write_text('{"emergency_stop": true}')

    stopper = Thread(target=stop_export)
    stopper.start()
    try:
        assert entry.run(cfg, hardware, stop_file, loop=loop) == 130
        assert released.is_set()
    finally:
        released.set()
        stopper.join(timeout=5)


@pytest.mark.parametrize("delay", [12, 14, 16])
def test_rtc_decision_offsets_and_executed_ticks(delay):
    p = protocol(streaming=True)
    p.cfg.__dict__.update(smooth_method="none", save_prediction_video=False)
    p.client.metadata.update(rtc_async=True, async_infer=dict(rtc_enabled=True, rtc_delay=delay))
    p = CosmosProtocol(p.client, p.cfg)
    executed = []
    for index in range(5):
        ticks = np.arange(32) + 16 * index
        result = dict(action=dict(joint=np.tile(ticks[:, None], (1, 14))), execute_steps=32)
        raw, skip, keep, feedback, units = p.plan(result, index == 0, 5 - index)
        executed.extend(raw[skip:keep, 0])
        assert feedback == list(range(skip, keep)) and units == 1
    np.testing.assert_array_equal(executed, np.arange(96))


@pytest.mark.parametrize(
    "horizon,delay,chunks", [(32, 1, 4), (32, 14, 4), (32, 16, 4), (64, 16, 4), (64, 32, 4), (32, 16, 1)]
)
def test_rtc_production_loop_overlaps_requests_without_replaying_commands(
    tmp_path, monkeypatch, horizon, delay, chunks
):
    import sys
    import threading

    from tasks.client_server.client.piper import protocol as module

    p = protocol(streaming=True)
    cfg = p.cfg
    cfg.__dict__.update(
        smooth_method="none",
        save_prediction_video=False,
        max_chunks=chunks,
        model_family="causal-cosmos",
        report_dir=str(tmp_path),
        wait_tol_deg=1,
        action_dt=0.01,
        eef_position_scale=1,
        left_gripper_scale=1,
        right_gripper_scale=1,
        gripper_close_threshold=0,
    )
    p.client.metadata.update(rtc_async=True, async_infer=dict(rtc_enabled=True, rtc_delay=delay))
    p.client.metadata.update(actions_per_block=horizon, execute_steps=horizon)
    stride = horizon // 2
    anchors = list(range(stride, chunks * stride, stride))
    p = CosmosProtocol(p.client, cfg)
    started, release = threading.Event(), threading.Event()
    commands, requests = [], []
    waited = []
    lock = threading.Lock()

    def infer(request):
        assert lock.acquire(blocking=False), "Concurrent socket use"
        try:
            if request.get("reset"):
                return dict(reset=True)
            step = request["observation_step"]
            assert not {"rtc_enabled", "rtc_delay", "async_infer"} & request.keys()
            requests.append(request)
            if step:
                started.set()
                assert release.wait(5), "Motion did not continue while planning"
                release.clear()
                started.clear()
            actions = np.zeros((horizon, 14))
            actions[:, 0] = np.arange(step, step + horizon)
            return dict(observation_step=step, action=dict(joint=actions), execute_steps=horizon)
        finally:
            lock.release()

    p.client.infer = infer
    p.observe = lambda *args, **kwargs: dict(images=frame(len(commands)), state=dict(measured=len(commands)))
    monkeypatch.setattr(module, "cosmos_state", lambda *args: dict(measured=len(commands)))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def wait(*args):
        waited.append(len(commands))
        return dict(reached=True)

    monkeypatch.setattr(module, "wait_robot_target", wait)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=lambda *args: True))
    logger = SimpleNamespace(
        save_dir=tmp_path,
        log_key_frame=lambda *args: None,
    )
    camera = SimpleNamespace(get_frames=lambda: [np.zeros((2, 2, 3))] * 3)

    def send(piper, left, *args):
        commands.append(left[0])
        count = len(commands)
        if count in [anchor + 1 for anchor in anchors]:
            assert started.wait(5)
            assert lock.locked()
        if count in [anchor + delay for anchor in anchors]:
            release.set()

    try:
        total = module.run_loop(None, camera, p, cfg, logger, None, send, lambda: False, module.RunReport(cfg))
    finally:
        release.set()
        p.close()
    assert total == horizon + (chunks - 1) * stride
    assert len(waited) == len(set(waited)), "Do not wait twice on the same final target"
    assert {total, *(stride + delay + index * stride for index in range(chunks - 1))} <= set(waited)
    np.testing.assert_array_equal(commands, np.arange(total))
    assert [request["observation_step"] for request in requests] == [0, *anchors]
    for request in requests[1:]:
        step = request["observation_step"]
        assert request["state"]["measured"] == step
        assert [int(image["cam_high"][0, 0, 0]) for image in request["obs"]] == list(
            range(step - stride + 1, step + 1)
        )


def test_rtc_stop_while_waiting_retains_future_for_safe_drain():
    from concurrent.futures import Future

    p = protocol(streaming=True)
    p.rtc_stride = 16
    future = p.rtc_future = Future()
    with pytest.raises(KeyboardInterrupt, match="Stop"):
        p.infer({}, stop_check=lambda: True)
    assert p.rtc_future is future
    future.set_result({})
    p.close()
    assert p.rtc_future is None


@pytest.mark.parametrize("operation", ["reset", "first_rtc", "serial", "feedback"])
def test_piper_stop_interrupts_every_request_wait(operation):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from threading import Event

    entered, release, stopped = Event(), Event(), Event()
    p = protocol(streaming=True)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("transport closed")

    p.client.infer = p.client.reset = p.client.observe = blocked
    p.client.close = release.set
    initial = dict(images=frame(0), state={})
    if operation == "reset":
        invoke = partial(p.reset, stop_check=stopped.is_set)
    elif operation == "feedback":
        feedback = [dict(images=frame(i)) for i in range(32)]
        p.predicted_chunks = 1
        p.submit_feedback(feedback)
        invoke = partial(p.finish_block, feedback, {}, {}, True, stop_check=stopped.is_set)
    else:
        p.rtc_stride = 16 if operation == "first_rtc" else 0
        invoke = partial(p.infer, initial, stop_check=stopped.is_set)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(invoke)
        try:
            assert entered.wait(2), "Request never reached the transport"
            stopped.set()
            with pytest.raises(KeyboardInterrupt, match="Stop requested"):
                future.result(timeout=2)
            assert not release.is_set(), "Stop must interrupt the wait before the server responds"
        finally:
            p.close(abort=True)
    assert p._sender is None


@pytest.mark.parametrize("horizon", [32, 64])
def test_rtc_defaults_and_derived_half_horizon(horizon):
    p = protocol(streaming=True)
    p.client.metadata.update(
        rtc_async=True, async_infer=dict(rtc_enabled=True), actions_per_block=horizon, execute_steps=horizon
    )
    actual = CosmosProtocol(p.client, p.cfg)
    assert actual.rtc_enabled and actual.rtc_delay == 16 and actual.rtc_stride == horizon // 2
    assert not actual.stream_observations


@pytest.mark.parametrize(
    "change",
    [
        dict(rtc_enabled="false"),
        dict(rtc_enabled=1),
        dict(rtc_delay=True),
        dict(rtc_delay=0),
        dict(rtc_delay=-1),
        dict(rtc_delay=17),
        dict(rtc_delay=1.5),
        dict(smooth_method="spline"),
        dict(save_prediction_video=True),
        dict(control_mode="eef"),
    ],
)
def test_rtc_rejects_invalid_client_configuration(change):
    p = protocol(streaming=True)
    p.client.metadata.update(rtc_async=True, async_infer=dict(rtc_enabled=True))
    if set(change) & {"rtc_enabled", "rtc_delay"}:
        p.client.metadata["async_infer"].update(change)
    else:
        p.cfg.__dict__.update(change)
    with pytest.raises(ValueError):
        CosmosProtocol(p.client, p.cfg)


@pytest.mark.parametrize(
    "change",
    [
        dict(rtc_async=False),
        dict(actions_per_block=33, execute_steps=33),
        dict(actions_per_block=48, execute_steps=48),
        dict(observation_chunk_steps=0),
    ],
)
def test_rtc_rejects_incompatible_server_geometry(change):
    p = protocol(streaming=True)
    p.client.metadata.update(rtc_async=True, async_infer=dict(rtc_enabled=True))
    p.client.metadata.update(change)
    with pytest.raises(ValueError):
        CosmosProtocol(p.client, p.cfg)


def test_rtc_waits_for_late_reply():
    from concurrent.futures import Future, ThreadPoolExecutor
    from threading import Event

    entered = Event()

    class Pending(Future):
        def result(self, *args, **kwargs):
            entered.set()
            return super().result(*args, **kwargs)

    p = protocol(streaming=True)
    p.rtc_stride = 16
    p.rtc_future = pending = Pending()
    with ThreadPoolExecutor(max_workers=1) as worker:
        response = worker.submit(p.infer, {})
        assert entered.wait(5)
        pending.set_result(dict(observation_step=16))
        result = response.result(timeout=5)
    assert result == dict(observation_step=16)
    assert p.rtc_future is None and p.predicted_chunks == 1


@pytest.mark.parametrize("flags", [[], ["--rtc"], ["--no-rtc"], ["--rtc-delay", "14"]])
def test_piper_cli_cannot_override_server_async_settings(monkeypatch, flags):
    import sys

    from tasks.client_server.client.piper import __main__ as entry

    hardware = SimpleNamespace(
        CONFIG_MAP={
            "movej": lambda: SimpleNamespace(
                can_left="left",
                can_right="right",
                wait_tol_deg=3,
                smooth_method="spline",
                action_dt=0.02,
                max_chunks=1,
            )
        }
    )
    monkeypatch.setattr(entry, "load_lingbot", lambda _: hardware)
    monkeypatch.setattr(sys, "argv", ["piper", "--prompt", "pick box", *flags])
    captured = []
    if flags:
        with pytest.raises(SystemExit) as error:
            entry.main(runner=lambda cfg, *_: captured.append(cfg) or 0)
        assert error.value.code == 2 and not captured
        return
    assert entry.main(runner=lambda cfg, *_: captured.append(cfg) or 0) == 0
    cfg = captured[0]
    assert not hasattr(cfg, "rtc_enabled") and not hasattr(cfg, "rtc_delay")
    assert cfg.smooth_method == "none" and cfg.action_dt == 1 / 30


def test_rtc_abort_unblocks_active_websocket_worker():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from websockets.sync.server import serve

    entered, release = threading.Event(), threading.Event()
    metadata = protocol(streaming=True).client.metadata
    metadata.update(
        protocol="cosmos-causal-action-v2",
        quaternion_order="xyzw",
        eef_pose_frame="dataset",
        available_control_modes=["joint"],
        rtc_async=True,
        async_infer=dict(rtc_enabled=True, rtc_delay=16),
    )

    def handler(socket):
        socket.send(packb(metadata))
        socket.recv()
        entered.set()
        release.wait(10)  # Deliberately never sends an inference reply.

    with serve(handler, "127.0.0.1", 0) as server:
        serving = threading.Thread(target=server.serve_forever)
        serving.start()
        client = InferenceClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
        p = CosmosProtocol(client, SimpleNamespace(model_fps=30, control_mode="movej"))
        p._sender = ThreadPoolExecutor(max_workers=1)
        future = p.rtc_future = p._sender.submit(p._rtc_request, dict(state={}), frame(0), 0)
        errors = []

        def close():
            try:
                p.close(abort=True)
            except Exception as exc:
                errors.append(exc)

        closing = threading.Thread(target=close)
        try:
            assert entered.wait(5)
            with pytest.raises(KeyboardInterrupt):
                p.infer({}, stop_check=lambda: True)
            closing.start()
            closing.join(2)
            assert not closing.is_alive(), "Stop must close transport before waiting for the blocked RPC"
            assert not errors
            assert future.done() and p.rtc_future is None and p._sender is None
        finally:
            release.set()
            client.close()
            if closing.ident is not None:
                closing.join(5)
            p.close(abort=True)
            server.shutdown()
            serving.join(5)


@pytest.mark.parametrize("key", ["rtc_enabled", "rtc_delay", "async_infer"])
def test_piper_rejects_obsolete_client_rtc_config(tmp_path, key):
    import json

    from tasks.client_server.client.piper.__main__ import configuration

    path = tmp_path / "client.json"
    path.write_text(json.dumps(dict(client={key: False})))
    with pytest.raises(ValueError, match="server YAML async_infer"):
        configuration(SimpleNamespace(config_file=str(path)), None)


def test_piper_old_server_without_async_metadata_uses_serial_loop():
    p = protocol(streaming=True)
    p.client.metadata["rtc_async"] = True  # Capability alone must not enable execution.
    actual = CosmosProtocol(p.client, p.cfg)
    assert not actual.rtc_enabled and actual.rtc_stride == 0 and actual.stream_observations
