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

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_inference import ConstantVelocity
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline
from veomni.models.diffusers.causal_cosmos3.robotwin_adapter import RobotwinAdapter
from veomni.models.diffusers.causal_cosmos3.robotwin_policy import RobotwinPolicy, build_adapter


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("fps", [None, 0, -30, float("nan"), float("inf"), True, "30"])
def test_server_rejects_missing_or_invalid_fps_before_loading_weights(tmp_path, monkeypatch, fps):
    import yaml

    from tasks.client_server.server import serve_robotwin

    config = dict(device="cpu")
    if fps is not None:
        config["fps"] = fps
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(CausalCosmos3Pipeline, "from_pretrained", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["serve_robotwin", "--config", str(path), "--checkpoint", "unused"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match=r"fps.*--fps"):
        serve_robotwin.main()
    assert calls == []


@pytest.mark.parametrize(
    "config_name", ["causal_cosmos3_robotwin.yaml", "causal_cosmos3_edge_action_policy.yaml", None]
)
def test_server_startup_uses_yaml_fps_or_cli_override(tmp_path, monkeypatch, caplog, config_name):
    import logging

    from tasks.client_server.server import serve_robotwin

    config = Path(__file__).resolve().parents[3] / "configs/infer" / (config_name or "offline.yaml")
    args = ["serve_robotwin", "--config", str(config), "--checkpoint", "unused", "--device", "cpu"]
    expected_fps = 30
    if config_name is None:
        config = tmp_path / "offline.yaml"
        config.write_text("device: cpu\n")
        args[2] = str(config)
        args.extend(["--fps", "60"])
        expected_fps = 60
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
    monkeypatch.setattr(serve_robotwin, "create_server", server)
    with caplog.at_level(logging.INFO, logger=serve_robotwin.__name__):
        serve_robotwin.main()
    assert loaded[0]["fps"] == expected_fps
    assert served[0].fps == expected_fps
    assert str(config.resolve()) in caplog.text
    assert f"fps={expected_fps}" in caplog.text


class ObservedVAE:
    def __init__(self):
        self.inputs = []

    def encode_pack(self, samples):
        results = []
        for sample in samples:
            self.inputs.append(sample["video"].clone())
            latent = sample["video"][:, ::4, ::16, ::16].float()
            results.append(dict(sample, video_latents=latent))
        return results

    def stream_encoder(self):
        owner = self

        class Stream:
            def __init__(self):
                self.initial = True

            def encode(self, video):
                owner.inputs.append(video.clone())
                selected = video if self.initial else video[:, 3::4]
                self.initial = False
                return selected[:2, :, ::16, ::16].float() / 255

        return Stream()

    def decode(self, latent):
        return torch.zeros((3, 1 + (latent.shape[1] - 1) * 4, 16, 16), dtype=torch.uint8)


def policy_fixture(history=2, *, cached=False, guidance=1):
    mask = torch.zeros(55, dtype=torch.bool)
    mask[list(range(6)) + list(range(7, 13)) + [28, 29]] = True
    statistics = dict(
        state=dict(low=[-10.0] * 55, high=[10.0] * 55, valid=mask.tolist()),
        action=dict(low=[-1.0] * 55, high=[1.0] * 55, valid=mask.tolist()),
        provenance=dict(sources=[dict(state_mask=mask.tolist(), action_mask=mask.tolist())]),
    )
    config = dict(
        source=dict(viewpoint="concat_view", resolution="256"),
        statistics=statistics,
        sampling=dict(
            kv_cache=cached,
            history=history,
            steps=1,
            seed=42,
            guidance=guidance,
            fixed_step_sampler_config=None if cached else dict(t_list=[0.5]),
        ),
    )
    model = ConstantVelocity()
    if cached:
        torch.manual_seed(63)
        with ops_config_scope(eager_ops_config()):
            model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55))
    pipe = CausalCosmos3Pipeline(
        model, ObservedVAE(), ActionProcessor(statistics), lambda text: [1, 2] if text else [3], config
    )
    policy = RobotwinPolicy(pipe, fps=30)
    if cached:
        compose = policy._compose
        policy._compose = lambda cameras: compose(cameras)[:, :, ::12, ::14].contiguous()
    return policy


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


def test_cached_server_reuses_text_and_measured_history_across_requests_and_reset():
    policy = policy_fixture(history=3, cached=True, guidance=2)
    assert policy.metadata["kv_cache"] is True
    text_calls = []
    handle = policy.pipe.model.embed_tokens.register_forward_pre_hook(
        lambda module, args: text_calls.append(len(args[0]))
    )
    try:
        policy.infer(dict(reset=True, prompt="Move.", control_mode="joint", record_video=True))
        first = policy.infer(observation())
        caches = dict(policy.kv_caches)
        for step in (32, 64, 96):
            result = policy.infer(observation(step))
            assert result["action"]["joint"].shape == (32, 14)
            assert all(policy.kv_caches[name] is cache for name, cache in caches.items())
        assert text_calls == [2, 1]  # One text prefill per CFG branch for the entire episode.
        assert [pixels.shape[1] for pixels in policy.pipe.vae.inputs] == [1, 8, 8, 8]
        for index, pixels in enumerate(policy.pipe.vae.inputs[1:], 1):
            torch.testing.assert_close(
                pixels[0, :, 0, 0], torch.arange((index - 1) * 32 + 4, index * 32 + 1, 4, dtype=torch.uint8)
            )
        assert all(cache.frame_ids == (4, 5, 6) and cache.next_frame == 7 for cache in caches.values())
        assert policy.history.shape[1] == 8
        policy.infer(dict(finish_video=True))
        policy.infer(observation(128))
        assert text_calls == [2, 1]
        policy.reset()
        assert not policy.kv_caches and policy.history is None
        policy.infer(dict(reset=True, prompt="Move.", control_mode="joint"))
        repeated = policy.infer(observation())
        np.testing.assert_array_equal(repeated["action"]["joint"], first["action"]["joint"])
    finally:
        handle.remove()


def test_native_sampler_uses_measured_history_latest_anchor_and_bounded_context():
    policy = policy_fixture()
    policy.infer(dict(reset=True, prompt="Move the bottle.", control_mode="joint"))
    first = policy.infer(observation())
    assert first["action"]["joint"].shape == (32, 14)
    assert first["action"]["eef"] is None
    assert first["structured_prompt"]["fps"] == 7.5
    assert first["structured_prompt"]["resolution"] == dict(H=384, W=448)
    for step in (32, 64, 96):
        result = policy.infer(observation(step))
        assert result["observation_step"] == step
        assert result["revision"] == step // 32 + 1
        assert "structured_prompt" not in result
        # The same per-block seed is independent of context length, and the
        # decoder must add the latest measured joint anchor, not a past prediction.
        assert np.isfinite(result["action"]["joint"]).all()
    assert policy.history.shape == (3, 9, 384, 448)
    torch.testing.assert_close(policy.history[0, :, 0, 0], torch.arange(64, 97, 4, dtype=torch.uint8))
    assert [x.shape[1] for x in policy.pipe.vae.inputs] == [9, 17, 17, 17]
    # Future placeholders are black; previous generated frames never enter real history.
    for encoded in policy.pipe.vae.inputs:
        assert not encoded[:, -8:].any()
    policy.reset()
    policy.infer(dict(reset=True, prompt="Move the bottle.", control_mode="joint"))
    repeated = policy.infer(observation())
    np.testing.assert_array_equal(first["action"]["joint"], repeated["action"]["joint"])


def test_decode_reanchors_joints_and_preserves_absolute_grippers(monkeypatch):
    import veomni.models.diffusers.causal_cosmos3.robotwin_policy as module

    policy = policy_fixture()

    def zero_delta(model, encoded, block, **kwargs):
        return dict(action=torch.zeros_like(encoded["action"]), video_latents=encoded["video_latents"])

    monkeypatch.setattr(module, "sample_block", zero_delta)
    policy.infer(dict(reset=True, prompt="Move.", control_mode="joint"))
    policy.infer(observation())
    result = policy.infer(observation(32))["action"]["joint"]
    np.testing.assert_allclose(result[:, :6], 3.2)
    np.testing.assert_allclose(result[:, 6], 0)
    np.testing.assert_allclose(result[:, 7:], 0)


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
def test_invalid_feedback_is_rejected_without_advancing_session(change, match):
    policy = policy_fixture()
    policy.infer(dict(reset=True, prompt="Move.", control_mode="joint"))
    policy.infer(observation())
    before = policy.history.clone()
    with pytest.raises(ValueError, match=match):
        policy.infer(dict(observation(32), **change))
    assert policy.revision == 1 and policy.observation_step == 0
    assert not policy.needs_reset
    torch.testing.assert_close(policy.history, before)
    assert policy.infer(observation(32))["revision"] == 2


@pytest.mark.parametrize("cached", [False, True])
def test_sampling_error_requires_reset_and_drops_history(monkeypatch, cached):
    import veomni.models.diffusers.causal_cosmos3.robotwin_policy as module

    policy = policy_fixture(cached=cached)
    original = module.sample_block
    monkeypatch.setattr(module, "sample_block", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken")))
    policy.infer(dict(reset=True, prompt="Move.", control_mode="joint"))
    with pytest.raises(RuntimeError, match="broken"):
        policy.infer(observation())
    assert policy.history is None and policy.needs_reset
    assert not policy.kv_caches
    with pytest.raises(ValueError, match="send reset"):
        policy.infer(observation())
    monkeypatch.setattr(module, "sample_block", original)
    policy.infer(dict(reset=True, prompt="Move.", control_mode="joint"))
    assert policy.infer(observation())["revision"] == 1


def test_masks_come_from_exact_source_and_statistics_geometry_is_checked():
    policy = policy_fixture()
    pipe = policy.pipe
    pipe.config["statistics"]["provenance"]["sources"].append(dict(state_mask=[0] * 55, action_mask=[0] * 55))
    with pytest.raises(ValueError, match="identical source masks"):
        build_adapter(pipe)
    pipe.config["robotwin"] = dict(
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
    adapter = RobotwinAdapter(policy.pipe.processor.template, state_mask=mask, action_mask=mask)
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


def test_real_websocket_metadata_single_client_errors_and_reconnect():
    pytest.importorskip("msgpack")
    pytest.importorskip("websockets.sync.client")
    from tasks.client_server.client.policy import CausalPolicyClient
    from tasks.client_server.server.serve_robotwin import create_server
    from tasks.client_server.wire import packb, unpackb

    policy = policy_fixture(cached=True)
    with create_server(policy, host="127.0.0.1", port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        uri = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        try:
            with CausalPolicyClient(uri) as client:
                assert client.metadata["execute_steps"] == 32
                with pytest.raises(Exception, match="1013"):
                    CausalPolicyClient(uri)
                client.reset(prompt="Move.")
                result = client.infer(observation())
                assert client.actions(result).shape == (32, 14)
                with pytest.raises(RuntimeError, match="observation_step=32"):
                    client.infer(observation())
                assert client.infer(observation(32))["revision"] == 2
            with CausalPolicyClient(uri) as client:
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

    from tasks.client_server.client.policy import CausalPolicyClient
    from tasks.client_server.server.serve_robotwin import create_server

    policy = policy_fixture()
    server = create_server(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = CausalPolicyClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
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

    from tasks.client_server.client.policy import CausalPolicyClient
    from tasks.client_server.server.serve_robotwin import create_server

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
    client = CausalPolicyClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
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
    assert stopped.is_set() and policy.history is None


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
