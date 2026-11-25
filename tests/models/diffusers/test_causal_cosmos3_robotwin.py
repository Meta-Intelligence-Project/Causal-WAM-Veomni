"""RoboTwin measured feedback, native sampling and real local wire transport."""

import copy
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


pytest.importorskip("msgpack")

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.client_server.server.adapter import Unified55Adapter
from tasks.client_server.server.session import CausalInferenceSession, build_adapter
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_inference import ConstantVelocity
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class ObservedVAE:
    def __init__(self):
        self.inputs = []

    def encode_pack(self, samples):
        results = []
        for sample in samples:
            assert set(sample) == {"video"}
            pixels = sample["video"]
            self.inputs.append(pixels.clone())
            results.append(dict(video_latents=pixels[:2, ::4, ::16, ::16].float() / 255))
        return results

    def decode(self, latent):
        return torch.zeros((3, 1 + (latent.shape[1] - 1) * 4, 16, 16), dtype=torch.uint8)


class ObservedVelocity(ConstantVelocity):
    def forward(self, **inputs):
        video = inputs["video_tokens"]
        return SimpleNamespace(
            video=torch.ones_like(video),
            action=torch.ones_like(inputs.get("action_tokens", video.new_empty((0, 55)))),
        )


def policy_fixture(history=2, *, task="policy", structured=True, real_model=False):
    mask = torch.zeros(55, dtype=torch.bool)
    mask[list(range(6)) + list(range(7, 13)) + [28, 29]] = True
    statistics = dict(
        state=dict(low=[-10.0] * 55, high=[10.0] * 55, valid=mask.tolist()),
        action=dict(low=[-1.0] * 55, high=[1.0] * 55, valid=mask.tolist()),
        provenance=dict(sources=[dict(state_mask=mask.tolist(), action_mask=mask.tolist())]),
    )
    config = dict(
        source=dict(viewpoint="concat_view", resolution="256", structured_prompt=structured),
        statistics=statistics,
        sampling=dict(history=history, steps=2, seed=42, guidance=1),
    )
    model = ObservedVelocity()
    if real_model:
        torch.manual_seed(63)
        with ops_config_scope(eager_ops_config()):
            model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    pipe = CausalCosmos3Pipeline(model, ObservedVAE(), ActionProcessor(statistics), lambda text: [1, 2], config)
    session = CausalInferenceSession(pipe, fps=30, task=task)
    compose = session._compose
    session._compose = lambda cameras: compose(cameras)[:, :, ::12, ::14].contiguous()
    return session


@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
@pytest.mark.parametrize("structured", [True, False])
def test_server_matches_native_session_with_real_model_and_measured_feedback(monkeypatch, task, structured):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import packing

    def forbidden(*args, **kwargs):
        raise AssertionError("Online inference must not construct training samples")

    monkeypatch.setattr(packing, "pack_model_inputs", forbidden)
    server = policy_fixture(task=task, structured=structured, real_model=True)
    reference = copy.deepcopy(server.pipe).session(
        caption="Move the bottle.",
        fps=30,
        state_mask=server.adapter.state_mask,
        action_mask=server.adapter.action_mask,
        task=task,
    )
    server.infer(dict(reset=True, prompt="Move the bottle.", record_video=True))
    calls = []
    handle = server.pipe.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["layout"]), with_kwargs=True
    )
    try:
        for step in (0, 32, 64):
            request = observation(step)
            cameras, _ = server._cameras(request["obs"], 32 if step else 1)
            if step:
                cameras = {key: value[3::4] for key, value in cameras.items()}
            pixels = server._compose(cameras)
            state = torch.from_numpy(server.adapter.assemble_state(request["state"]))
            reference.observe(pixels, state, timestamp=step / 30)
            expected = reference.predict()
            actual = server.infer(request)
            split = server.adapter.split_actions(expected["actions"].numpy())
            np.testing.assert_array_equal(actual["action"]["joint"], split["joint"])
            np.testing.assert_array_equal(actual["action_timestamps"], expected["action_timestamps"].numpy())
            torch.testing.assert_close(server.video_latents[-1], expected["video_latents"][:, -2:], atol=0, rtol=0)
            assert actual["seeds"] == expected["seeds"]
            assert actual["revision"] == step // 32 + 1
            if step == 0:
                assert actual["prompt"] == expected["prompt"]
                assert ("structured_prompt" in actual) is structured
                if not structured:
                    assert actual["prompt"] == "Move the bottle."
    finally:
        handle.remove()
    assert [pixels.shape[1] for pixels in server.pipe.vae.inputs] == [1, 9, 9]
    torch.testing.assert_close(server.observations._images[0, :, 0, 0], torch.arange(32, 65, 4, dtype=torch.uint8))
    if task == "inverse_dynamics":
        assert all(layout.modes == ("inverse_dynamics",) for layout in calls)
        assert any(2 not in layout.modalities and 3 not in layout.modalities for layout in calls)
        assert any(
            2 in layout.modalities
            and all(role != 1 for role, modality in zip(layout.roles, layout.modalities) if modality == 1)
            for layout in calls
        )


@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
def test_decode_uses_latest_measured_anchor_once(monkeypatch, task):
    from veomni.models.diffusers.causal_cosmos3 import observation_session as module

    def zero_delta(model, context, *args, **kwargs):
        return dict(action=torch.zeros(context.action_shape), video_latents=torch.zeros(context.video_shape), seeds={})

    monkeypatch.setattr(module, "generate_block", zero_delta)
    monkeypatch.setattr(module, "generate_action_block", zero_delta)
    server = policy_fixture(task=task)
    server.infer(dict(reset=True, prompt="Move."))
    server.infer(observation())
    actual = server.infer(observation(32))["action"]["joint"]
    np.testing.assert_allclose(actual[:, :6], 3.2)
    np.testing.assert_allclose(actual[:, 6:], 0)


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(observation_step=0), "observation_step=32"),
        (dict(observation_step=True), "observation_step=32"),
        (dict(executed_steps=16), "exactly 32"),
        (dict(prompt="Another task"), "requires reset"),
        (dict(control_mode="eef"), "Unsupported control_mode"),
        (dict(action_mask=[1] * 55), "Masks"),
        (dict(obs=[]), "32 RGB frames"),
    ],
)
def test_invalid_feedback_does_not_advance_observation_session(change, match):
    server = policy_fixture()
    server.infer(dict(reset=True, prompt="Move."))
    server.infer(observation())
    before = server.observations._images.clone()
    with pytest.raises(ValueError, match=match):
        server.infer(dict(observation(32), **change))
    assert server.revision == 1 and not server.needs_reset
    torch.testing.assert_close(server.observations._images, before)
    assert server.infer(observation(32))["revision"] == 2


def test_idm_action_failure_requires_reset_without_committing_predicted_history(monkeypatch):
    from veomni.models.diffusers.causal_cosmos3 import observation_session as module

    server = policy_fixture(task="inverse_dynamics")
    server.infer(dict(reset=True, prompt="Move."))
    original = module.generate_action_block

    def fail(*args, **kwargs):
        raise RuntimeError("action stage failed")

    monkeypatch.setattr(module, "generate_action_block", fail)
    with pytest.raises(RuntimeError, match="action stage failed"):
        server.infer(observation())
    assert server.needs_reset and server.observations is None and not server.video_latents
    with pytest.raises(ValueError, match="send reset"):
        server.infer(observation())
    monkeypatch.setattr(module, "generate_action_block", original)
    server.infer(dict(reset=True, prompt="Move."))
    assert server.infer(observation())["revision"] == 1


@pytest.mark.parametrize("sampling", [dict(kv_cache=True), dict(observation_mode="streaming")])
def test_incompatible_pr_cache_is_rejected_before_loading_weights(tmp_path, monkeypatch, sampling):
    import yaml

    from tasks.client_server.server import server as module

    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(dict(device="cpu", fps=30, sampling=sampling)))
    monkeypatch.setattr(sys, "argv", ["server", "--config", str(path), "--checkpoint", "unused"])
    with pytest.raises(ValueError, match="kv_cache=false"):
        module.main()


def test_plain_prompt_is_recorded_without_json_parsing(tmp_path):
    from tasks.client_server.client.runtime import save_structured_prompt

    save_structured_prompt(dict(prompt="Move the bottle."), tmp_path / "structured_prompt.json")
    assert (tmp_path / "prompt.txt").read_text() == "Move the bottle."
    assert not (tmp_path / "structured_prompt.json").exists()


def test_idm_closed_loop_over_real_websocket_with_robotwin_device():
    pytest.importorskip("websockets.sync.client")
    from tasks.client_server.client.robotwin.simulator import rollout
    from tasks.client_server.client.runtime import InferenceClient
    from tasks.client_server.server.server import create_server

    class Env:
        step_lim = 65
        take_action_cnt = 0
        eval_success = False

        def __init__(self):
            self.command = np.zeros(14, dtype=np.float32)

        def get_instruction(self):
            return "Move the bottle."

        def get_obs(self):
            frame = np.full((4, 6, 3), self.take_action_cnt, dtype=np.uint8)
            return dict(
                observation={key: dict(rgb=frame) for key in ("head_camera", "left_camera", "right_camera")},
                joint_action=dict(
                    left_arm=self.command[:6],
                    left_gripper=self.command[6],
                    right_arm=self.command[7:13],
                    right_gripper=self.command[13],
                ),
                endpose={},
            )

        def take_action(self, action, *, action_type):
            assert action_type == "qpos" and action.shape == (14,) and np.isfinite(action).all()
            self.command = action.copy()
            self.take_action_cnt += 1

    policy = policy_fixture(task="inverse_dynamics", structured=False, real_model=True)
    with create_server(policy, host="127.0.0.1", port=0) as server:
        listener = threading.Thread(target=server.serve_forever)
        listener.start()
        try:
            with InferenceClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}") as client:
                result = rollout(Env(), client)
                assert result["executed_steps"] == 65
                assert [r["observation_step"] for r in result["requests"]] == [0, 32, 64]
                assert policy.revision == 3
                torch.testing.assert_close(
                    policy.observations._images[0, :, 0, 0], torch.arange(32, 65, 4, dtype=torch.uint8)
                )
        finally:
            server.shutdown()
            listener.join(timeout=10)
    assert not listener.is_alive()


@pytest.mark.parametrize("fps", [None, 0, -30, float("nan"), float("inf"), True, "30"])
def test_server_rejects_missing_or_invalid_fps_before_loading_weights(tmp_path, monkeypatch, fps):
    import yaml

    from tasks.client_server.server import server as server_module

    config = dict(device="cpu")
    if fps is not None:
        config["fps"] = fps
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(CausalCosmos3Pipeline, "from_pretrained", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["server", "--config", str(path), "--checkpoint", "unused"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match=r"fps.*--fps"):
        server_module.main()
    assert calls == []


@pytest.mark.parametrize("fps_override", [None, 60])
@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
def test_server_startup_uses_yaml_fps_or_cli_override(monkeypatch, caplog, fps_override, task):
    import logging

    from tasks.client_server.server import server as server_module

    config = Path(__file__).resolve().parents[3] / "configs/server/causal_cosmos3_server.yaml"
    args = ["server", "--config", str(config), "--checkpoint", "unused", "--device", "cpu"]
    args.extend(["--task", task])
    expected_fps = 30 if fps_override is None else fps_override
    if fps_override is not None:
        args.extend(["--fps", str(fps_override)])
    pipe = policy_fixture().pipe
    loaded, served = [], []

    def load(checkpoint, options):
        loaded.append(options)
        return pipe

    @contextmanager
    def server(policy, **kwargs):
        served.append(policy)
        yield SimpleNamespace(
            socket=SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8000)), serve_forever=lambda: None
        )

    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(CausalCosmos3Pipeline, "from_pretrained", load)
    monkeypatch.setattr(server_module, "create_server", server)
    with caplog.at_level(logging.INFO, logger=server_module.__name__):
        server_module.main()
    assert loaded[0]["fps"] == expected_fps
    assert served[0].task == task
    assert loaded[0]["task"] == task
    assert served[0].fps == expected_fps
    assert str(config.resolve()) in caplog.text
    assert f"fps={expected_fps}" in caplog.text


def observation(step=0):
    keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    frames = [
        {key: np.full((4, 6, 3), index, dtype=np.uint8) for key in keys}
        for index in range(max(1, step - 31), step + 1)
    ]
    return dict(
        obs=frames if step else {key: np.zeros((4, 6, 3), dtype=np.uint8) for key in keys},
        state=dict(
            joint_action=dict(left_arm=[step / 10] * 6, right_arm=[0] * 6, left_gripper=0.1, right_gripper=0.2)
        ),
        observation_step=step,
        executed_steps=32 if step else 0,
    )


def test_masks_come_from_exact_source_and_statistics_geometry_is_checked():
    policy = policy_fixture()
    pipe = policy.pipe
    pipe.config["statistics"]["provenance"]["sources"].append(dict(state_mask=[0] * 55, action_mask=[0] * 55))
    with pytest.raises(ValueError, match="identical source masks"):
        build_adapter(pipe)
    pipe.config["adapter"] = dict(
        state_mask=policy.adapter.state_mask.tolist(), action_mask=policy.adapter.action_mask.tolist()
    )
    assert build_adapter(pipe).available_control_modes == ["joint"]
    pipe.processor.statistics_geometry["actions_per_block"] = 16
    with pytest.raises(ValueError, match="actions_per_block"):
        build_adapter(pipe)


def test_eef_adapter_preserves_dataset_xyzw_and_interleaved_grippers():
    policy = policy_fixture()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[14:30] = True
    adapter = Unified55Adapter(policy.pipe.processor.template, state_mask=mask, action_mask=mask)
    assert adapter.available_control_modes == ["eef"]
    state = adapter.assemble_state(
        dict(
            endpose=dict(
                left_endpose=[1, 2, 3, 0, 0, 0, 1],
                right_endpose=[4, 5, 6, 0, 0, 1, 0],
                left_gripper=0.1,
                right_gripper=0.2,
            )
        )
    )
    result = adapter.split_actions(np.repeat(state[None], 32, axis=0))
    assert result["joint"] is None
    np.testing.assert_allclose(result["eef"][0], [1, 2, 3, 0, 0, 0, 1, 0.1, 4, 5, 6, 0, 0, 1, 0, 0.2])


def test_prediction_video_is_explicit_and_decoded_once(tmp_path):
    policy = policy_fixture()
    policy.infer(dict(reset=True, prompt="Move.", record_video=True))
    for step in (0, 32):
        policy.infer(observation(step))
    result = policy.infer(dict(finish_video=True))
    assert result["video_frames"] == 17 and result["video_fps"] == 7.5
    assert b"ftyp" in result["video_mp4"][:32]
    with pytest.raises(ValueError, match="No prediction video"):
        policy.infer(dict(finish_video=True))


@pytest.mark.parametrize("task", ["policy", "inverse_dynamics"])
def test_real_websocket_metadata_single_client_errors_and_reconnect(task):
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets.sync.client")
    from tasks.client_server.client.runtime import InferenceClient
    from tasks.client_server.server.server import create_server
    from tasks.client_server.wire import packb, unpackb

    policy = policy_fixture(task=task)
    with create_server(policy, host="127.0.0.1", port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        try:
            with InferenceClient(uri) as client:
                assert client.metadata["execute_steps"] == 32
                assert client.metadata["mode"] == task
                with pytest.raises(Exception, match="1013"):
                    InferenceClient(uri)
                client.reset(prompt="Move.")
                result = client.infer(observation())
                assert client.actions(result).shape == (32, 14)
                with pytest.raises(RuntimeError, match="observation_step=32"):
                    client.infer(observation())
                assert client.infer(observation(32))["revision"] == 2
            with InferenceClient(uri) as client:
                client.reset(prompt="New episode.")
                assert client.infer(observation())["revision"] == 1
            array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
            restored = unpackb(packb(dict(value=array)))["value"]
            np.testing.assert_array_equal(restored, array)
            assert restored.dtype == array.dtype
        finally:
            server.shutdown()
            thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_simulator_help_does_not_import_torch_or_veomni(tmp_path, entrypoint):
    pytest.importorskip("msgpack")
    root = Path(__file__).resolve().parents[3]
    target = root / "tasks/client_server/client/robotwin/__main__.py"
    invocation = (
        f"runpy.run_path({str(target)!r}, run_name='__main__')"
        if entrypoint == "script"
        else "runpy.run_module('tasks.client_server.client.robotwin', run_name='__main__')"
    )
    script = (
        f"import runpy,sys; sys.argv=[{str(target)!r},'--help']; "
        f"\ntry: {invocation}"
        "\nexcept SystemExit as e: assert e.code == 0"
        "\nassert 'torch' not in sys.modules and 'veomni' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path if entrypoint == "script" else root,
        env=dict(os.environ, PYTHONPATH=""),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--robotwin-root" in result.stdout


def test_server_shutdown_closes_connected_idle_client_and_resets_session():
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets.sync.client")
    from websockets.protocol import State

    from tasks.client_server.client.runtime import InferenceClient
    from tasks.client_server.server.server import create_server

    policy = policy_fixture()
    server = create_server(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = InferenceClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
    try:
        client.reset(prompt="Move.")
        server.shutdown()
        thread.join(timeout=10)
        assert client.connection.state is State.CLOSED
        assert policy.prompt is None
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=10)


def test_server_shutdown_waits_for_inflight_inference(monkeypatch):
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets.sync.client")
    from websockets.exceptions import ConnectionClosed

    from tasks.client_server.client.runtime import InferenceClient
    from tasks.client_server.server.server import create_server

    policy = policy_fixture()
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original = policy.infer

    def blocked(request):
        entered.set()
        assert release.wait(10)
        return original(request)

    server = create_server(policy, host="127.0.0.1", port=0)
    listener = threading.Thread(target=server.serve_forever)
    listener.start()
    client = InferenceClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
    client.reset(prompt="Move.")
    monkeypatch.setattr(policy, "infer", blocked)

    def request():
        try:
            client.infer(observation())
        except ConnectionClosed as error:
            errors.append(error)

    def stop():
        server.shutdown()
        stopped.set()

    worker = threading.Thread(target=request)
    shutdown = threading.Thread(target=stop)
    try:
        worker.start()
        assert entered.wait(10)
        shutdown.start()
        # Closing the socket releases the client while model cleanup must wait.
        worker.join(timeout=10)
        assert not worker.is_alive() and len(errors) == 1
        assert not stopped.is_set()
    finally:
        release.set()
        worker.join(timeout=10)
        if shutdown.ident is not None:
            shutdown.join(timeout=10)
        client.close()
        server.shutdown()
        listener.join(timeout=10)
    assert stopped.is_set() and policy.observations is None


def test_simulator_rollout_returns_real_post_action_frames_and_stops_mid_block():
    pytest.importorskip("msgpack")
    from tasks.client_server.client.robotwin import simulator

    class Env:
        step_lim = 35
        take_action_cnt = 0
        eval_success = False

        def get_instruction(self):
            return "Move."

        def get_obs(self):
            frame = np.full((4, 6, 3), self.take_action_cnt, dtype=np.uint8)
            return dict(
                observation={key: dict(rgb=frame) for key in ("head_camera", "left_camera", "right_camera")},
                joint_action=dict(left_arm=[self.take_action_cnt] * 6),
                endpose={},
            )

        def take_action(self, action, *, action_type):
            assert action_type == "qpos"
            self.take_action_cnt += 1

    class Client:
        control_mode = "joint"
        metadata = dict(actions_per_block=32, execute_steps=32, left_arm_dof=6, right_arm_dof=6)

        def __init__(self):
            self.requests = []

        def reset(self, *, prompt):
            assert prompt == "Move."

        def infer(self, request):
            self.requests.append(copy.deepcopy(request))
            return dict(action=dict(joint=np.zeros((32, 14))), execute_steps=32)

        def actions(self, response):
            return response["action"]["joint"]

    env, client = Env(), Client()
    result = simulator.rollout(env, client)
    assert result["executed_steps"] == 35
    assert [r["observation_step"] for r in client.requests] == [0, 32]
    feedback = client.requests[1]
    assert [int(frame["cam_high"][0, 0, 0]) for frame in feedback["obs"]] == list(range(1, 33))
    assert feedback["state"]["joint_action"]["left_arm"] == [32] * 6


def test_simulator_eef_conversion_round_trips_translated_rotated_base():
    pytest.importorskip("msgpack")
    from tasks.client_server.client.robotwin.simulator import RobotwinPoseConverter

    converter = RobotwinPoseConverter(
        eef_pose_frame="base",
        T_world_base=dict(position=[1, 2, 3], quaternion_wxyz=[2**-0.5, 0, 0, 2**-0.5]),
    )
    world = np.array([1, 3, 3, 2**-0.5, 0, 0, 2**-0.5], dtype=np.float32)
    dataset = converter.to_dataset(world)
    np.testing.assert_allclose(dataset, [1, 0, 0, 0, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(converter.to_world(dataset), world, atol=1e-6)
    actions = np.tile(np.r_[dataset, 0.2, dataset, 0.3], (32, 1))
    native = converter.actions_to_world(actions)
    np.testing.assert_allclose(native[0], np.r_[world, 0.2, world, 0.3], atol=1e-6)
