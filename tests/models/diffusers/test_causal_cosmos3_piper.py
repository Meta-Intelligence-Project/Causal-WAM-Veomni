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
