# SPDX-License-Identifier: OpenMDW-1.1
"""Local RoboTwin execution timing with fakes; no simulator or model checkpoint."""

import sys
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.serving import robotwin_client as runner
from veomni.models.diffusers.causal_cosmos3.serving.async_session import AsyncRemoteSession
from veomni.models.diffusers.causal_cosmos3.serving.robotwin import RobotwinAdapter
from veomni.models.diffusers.causal_cosmos3.serving.rtc_session import RTCSession


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


class FakeEnv:
    def __init__(self, clock, *, limit=1000, success_at=None, action_seconds=0.0):
        self.clock, self.action_seconds = clock, action_seconds
        self.step_lim, self.success_at = limit, success_at
        self.take_action_cnt, self.eval_success = 0, False
        self.executed = []
        self.buffer = np.zeros((4, 6, 3), dtype=np.uint8)
        self.joints = np.zeros(14, dtype=np.float32)
        self.eef = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], np.float32)
        self.after_action = None
        self.robot = SimpleNamespace(
            left_arm_joints=[None] * 6,
            right_arm_joints=[None] * 6,
            get_left_arm_real_jointState=lambda: self.joints[:7].copy(),
            get_right_arm_real_jointState=lambda: self.joints[7:].copy(),
        )

    def get_instruction(self):
        return "Move the bottle."

    def get_obs(self):
        self.buffer.fill(self.take_action_cnt % 255)
        return dict(
            observation={key: dict(rgb=self.buffer) for key in ("head_camera", "left_camera", "right_camera")},
            joint_action=dict(left_arm=[9] * 6, right_arm=[8] * 6, left_gripper=0.1, right_gripper=0.2),
            endpose=dict(left_endpose=self.eef[:7].copy(), right_endpose=self.eef[7:].copy()),
        )

    def take_action(self, action, *, action_type):
        assert not runner.episode_finished(self)
        self.executed.append((self.clock(), action.copy(), action_type))
        if action_type == "qpos":
            self.joints[:] = action
        else:
            self.eef[:7] = action[:7]
            self.eef[7:] = action[8:15]
            self.joints[6], self.joints[13] = action[7], action[15]
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at
        self.clock.now += self.action_seconds
        if self.after_action:
            self.after_action()


def adapter():
    template = ActionTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[:6] = mask[7:13] = mask[14:30] = True
    return RobotwinAdapter(template, state_mask=mask, action_mask=mask, eef_pose_frame="world")


def prepare_image(images, resolution):
    image = torch.from_numpy(images["cam_high"].copy()).permute(2, 0, 1).contiguous()
    return image, torch.tensor([4, 6, 4, 6], dtype=torch.float32)


def command(tick):
    result = torch.zeros(55)
    result[0] = tick + 1
    result[14] = tick + 1
    result[20] = result[27] = 1
    return result


class FakeSession:
    def __init__(self, env, options, *, change_at=None, slow_to=None, prepare_seconds=0.0, prepare_rate=None):
        self.env, self.options = env, options
        self.controller_hz = 30.0
        self.execution_hz = options["execution_hz"]
        self.change_at, self.slow_to, self.prepare_seconds = change_at, slow_to, prepare_seconds
        self.prepare_rate = prepare_rate
        self.tick, self.closed, self.prepared = 0, False, False
        self.observations = []

    def prepare(self, **measurement):
        assert not self.env.executed
        assert "image_size" not in measurement
        self.env.clock.now += self.prepare_seconds
        if self.prepare_rate is not None:
            assert self.options["execution_rate_callback"](self.prepare_rate) is True
            self.execution_hz = self.prepare_rate
        self.prepared = True
        return self.status()

    def status(self):
        enabled = self.options["enable_rtc"]
        asynchronous = self.options["execution_mode"] != "blocking"
        return dict(
            tick=self.tick,
            rtc_enabled=enabled,
            execution_mode=("async_rtc" if enabled else "async_non_rtc") if asynchronous else "blocking_non_rtc",
            adaptive_execution=bool(asynchronous and self.options["adaptive_execution"]),
            inference_pending=False,
            error=None,
            published_requests=0,
            prepared=self.prepared,
        )

    def get_action(self, **measurement):
        assert self.prepared and "image_size" not in measurement
        self.observations.append(measurement)
        if self.tick == self.change_at:
            assert self.options["execution_rate_callback"](self.slow_to) is True
            self.execution_hz = self.slow_to
        result = command(self.tick)
        self.tick += 1
        return result

    def close(self, *, wait=True):
        assert wait
        self.closed = True


class FakeClient:
    def __init__(self, env, *, control_mode="joint", **session_options):
        self.env, self.control_mode = env, control_mode
        self.session_options = session_options
        self.metadata = dict(controller_hz=30.0, resolution="480")
        self.session = None
        self.request_timings = []

    def create_session(self, first_measurement, *, prompt, **options):
        assert not self.env.executed
        assert "image_size" in first_measurement and prompt == self.env.get_instruction()
        self.session = FakeSession(self.env, options, **self.session_options)
        return self.session


def run(env, client, **options):
    return runner.rollout(
        env,
        client,
        adapter=adapter(),
        prepare_image=prepare_image,
        clock=env.clock,
        sleep=env.clock.sleep,
        execution_hz=options.pop("execution_hz", 2.0),
        servo_hz=options.pop("servo_hz", 4.0),
        max_waypoints=options.pop("max_waypoints", 3),
        **options,
    )


def test_bootstrap_and_calibration_keep_environment_stationary_then_interpolate_real_targets(tmp_path):
    clock = FakeClock()
    env = FakeEnv(clock)
    client = FakeClient(env, prepare_seconds=5)
    result = run(env, client, output=tmp_path)
    assert client.session.closed
    assert result["executed_waypoints"] == result["issued_waypoints"] == 3
    assert result["servo_commands"] == 6
    assert result["wall_seconds"] == 1.5
    assert result["startup_calibration_seconds"] == 5
    np.testing.assert_allclose([row[0] for row in env.executed], [105.25, 105.5, 105.75, 106, 106.25, 106.5])
    np.testing.assert_allclose([row[1][0] for row in env.executed], [0.5, 1, 1.5, 2, 2.5, 3])
    assert {row[2] for row in env.executed} == {"qpos"}
    assert [int(obs["image"][0, 0, 0]) for obs in client.session.observations] == [0, 2, 4]
    assert np.load(tmp_path / "waypoints_unified.npy").shape == (3, 55)


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_changing_live_camera_reuses_bootstrap_tick_zero_then_streams_fresh_measurements(enable_rtc):
    from tests.models.diffusers.causal_cosmos3_serving.test_serving_model import policy

    class ChangingCameraEnv(FakeEnv):
        def __init__(self, clock):
            super().__init__(clock)
            self.captures = 0

        def get_obs(self):
            observation = super().get_obs()
            self.captures += 1
            self.buffer.fill(self.captures)
            return observation

    env = ChangingCameraEnv(FakeClock())
    service = policy(history_steps=32, enable_rtc=enable_rtc)

    class StreamingClient(FakeClient):
        def __init__(self):
            super().__init__(env)
            self.metadata = service.metadata
            self.observations = []

        def create_session(self, first_measurement, *, prompt, **options):
            bootstrap = service.infer(
                dict(
                    operation="reset",
                    state=first_measurement["state"],
                    image=first_measurement["image"],
                    image_size=first_measurement["image_size"],
                    prompt=prompt,
                    tick=0,
                    revision=0,
                )
            )
            self.session = AsyncRemoteSession(
                self,
                torch.from_numpy(bootstrap["actions"]),
                execution_hz=options["execution_hz"],
                enable_rtc=enable_rtc,
            )
            return self.session

        def observe(self, **request):
            service.observe(request)
            self.observations.append((request["tick"], request["image"].clone()))

        def prepare_window(self, **request):
            service.prepare_window(request)

        def plan(self, **request):
            pytest.fail("This three-waypoint regression must not reach the planning trigger")

    client = StreamingClient()
    try:
        result = runner.rollout(
            env,
            client,
            enable_rtc=enable_rtc,
            adaptive_execution=False,
            execution_hz=2,
            servo_hz=4,
            max_waypoints=3,
            adapter=service.adapter,
            prepare_image=prepare_image,
            clock=env.clock,
            sleep=env.clock.sleep,
        )
        assert result["executed_waypoints"] == env.captures == 3
        assert [tick for tick, _ in client.observations] == [0, 1, 2]
        assert [int(image[0, 0, 0]) for _, image in client.observations] == [1, 2, 3]
        assert int(service.first_image[0, 0, 0]) == 1
        assert client.session.status()["closed"]
        # The actual serving policy must still reject any conflicting tick-zero
        # image; the runner fixes capture ownership instead of weakening history.
        with pytest.raises(ValueError, match="cannot be overwritten"):
            service.observe(dict(tick=0, image=torch.zeros_like(service.first_image)))
    finally:
        service.clear()


@pytest.mark.parametrize("execution_mode", ["auto", "async", "blocking"])
def test_rollout_passes_execution_mode_to_the_session_factory(execution_mode):
    env = FakeEnv(FakeClock())
    client = FakeClient(env)
    result = run(env, client, execution_mode=execution_mode, max_waypoints=1)
    assert client.session.options["execution_mode"] == execution_mode
    assert client.session.options["enable_rtc"] is False
    expected_mode = "blocking_non_rtc" if execution_mode == "blocking" else "async_non_rtc"
    assert result["execution_mode"] == result["final_status"]["execution_mode"] == expected_mode
    assert result["adaptive_execution"] is (execution_mode != "blocking")


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_shared_async_planning_keeps_servo_execution_running_and_hands_off_at_24(enable_rtc):
    env = FakeEnv(FakeClock())
    started, release = Event(), Event()

    class AsyncClient(FakeClient):
        def __init__(self, env):
            super().__init__(env)
            self.metadata.update(
                action_horizon=32, execute_steps=16, handoff_steps=24, history_steps=1024, rtc_enabled=True
            )
            self.requests, self.hints = [], []

        def create_session(self, first_measurement, *, prompt, **options):
            assert options["execution_mode"] == "async" and options["enable_rtc"] is enable_rtc
            self.session = AsyncRemoteSession(
                self,
                torch.stack([command(i) for i in range(32)]),
                execution_hz=options["execution_hz"],
                enable_rtc=enable_rtc,
            )
            return self.session

        def plan(self, **request):
            self.requests.append(request)
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release ordinary background prediction")
            predicted = torch.stack([command(request["tick"] + i) for i in range(32)])
            predicted[:, 0] += 100
            return predicted

        def observe(self, **measurement):
            pass

        def prepare_window(self, *, anchor_tick, revision):
            self.hints.append(anchor_tick)

    client = AsyncClient(env)
    pending_commands = []

    def progress():
        if client.session.status()["inference_pending"]:
            assert started.wait(2)
            pending_commands.append(len(env.executed))
        if len(env.executed) == 35:
            release.set()
            client.session.join(timeout=2)

    env.after_action = progress
    try:
        result = run(env, client, execution_mode="async", enable_rtc=enable_rtc, max_waypoints=27)
        assert result["execution_mode"] == ("async_rtc" if enable_rtc else "async_non_rtc")
        assert result["continuous_rtc"] is enable_rtc
        # This direct session fixture supplies no rate policy, even though the runner requested adaptation.
        assert not result["adaptive_execution"]
        assert result["executed_waypoints"] == 27 and result["servo_commands"] == 54
        assert pending_commands[:3] == [33, 34, 35]
        assert result["final_status"]["published_requests"] == 1
        assert [request["tick"] for request in client.requests] == [16]
        prefix = client.requests[0]["rtc"]
        if enable_rtc:
            assert prefix.delay_steps == 8
            assert prefix.previous_actions.shape == (16, 55)
            torch.testing.assert_close(prefix.previous_actions[:, 0], torch.arange(17, 33, dtype=torch.float32))
        else:
            assert prefix is None
        assert client.hints == [16, 32]
        np.testing.assert_allclose(
            [row[1][0] for row in env.executed[1::2]], np.concatenate([np.arange(1, 25), np.arange(125, 128)])
        )
    finally:
        release.set()


def test_late_async_handoff_starts_a_fresh_servo_interval_without_catching_up():
    env = FakeEnv(FakeClock())
    client = FakeClient(env)
    create = client.create_session

    def create_late_async(*args, **kwargs):
        session = create(*args, **kwargs)
        original_action, original_status = session.get_action, session.status

        def status():
            value = original_status()
            value.update(execution_mode="async_non_rtc", published_requests=int(session.tick > 1))
            return value

        def get_action(**measurement):
            if session.tick == 1:
                # 模拟固定交接点等待候选结果；等待期间没有新的模型tick。
                env.clock.now += 0.75
            return original_action(**measurement)

        session.status, session.get_action = status, get_action
        return session

    client.create_session = create_late_async
    result = run(env, client, execution_mode="async")
    assert result["execution_mode"] == "async_non_rtc" and not result["continuous_rtc"]
    assert [row["tick"] for row in result["action_trace"]] == [0, 1, 2]
    assert [row["call_seconds"] for row in result["action_trace"]] == [0, 0.75, 0]
    assert result["executed_waypoints"] == 3 and result["servo_commands"] == 6
    assert result["wall_seconds"] == 2.25 and result["skipped_servo_ticks"] == 0
    np.testing.assert_allclose([row[0] for row in env.executed], [100.25, 100.5, 101.5, 101.75, 102, 102.25])
    np.testing.assert_allclose([row[1][0] for row in env.executed], [0.5, 1, 1.5, 2, 2.5, 3])


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_acknowledged_slowdown_is_applied_to_future_interpolation_intervals(enable_rtc):
    clock = FakeClock()
    env = FakeEnv(clock)
    client = FakeClient(env, change_at=1, slow_to=1)
    result = run(env, client, enable_rtc=enable_rtc)
    assert result["continuous_rtc"] is enable_rtc
    assert result["adaptive_execution"]
    assert result["controller_hz"] == 30
    assert [row["execution_hz"] for row in result["action_trace"]] == [2, 1, 1]
    assert result["wall_seconds"] == 2.5
    assert result["servo_commands"] == 10
    np.testing.assert_allclose([row[1][0] for row in env.executed], [0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])


@pytest.mark.parametrize("enable_rtc", [False, True])
def test_startup_slowdown_is_acknowledged_before_motion_in_both_async_modes(enable_rtc):
    env = FakeEnv(FakeClock())
    client = FakeClient(env, prepare_seconds=2, prepare_rate=1)
    result = run(env, client, enable_rtc=enable_rtc, max_waypoints=1)
    assert result["requested_execution_hz"] == 2
    assert result["initial_execution_hz"] == result["final_execution_hz"] == 1
    assert result["startup_calibration_seconds"] == 2
    assert result["adaptive_execution"] and result["controller_hz"] == 30
    assert result["wall_seconds"] == 1 and result["model_seconds"] == 1 / 30
    assert result["rate_changes"] == [dict(previous_hz=2, execution_hz=1)]
    np.testing.assert_allclose([row[0] for row in env.executed], [102.25, 102.5, 102.75, 103])


def test_eef_execution_uses_world_wxyz_and_stops_on_mid_interval_success():
    clock = FakeClock()
    env = FakeEnv(clock, success_at=3)
    client = FakeClient(env, control_mode="eef")
    result = run(env, client)
    assert result["success"] and result["servo_commands"] == 3
    assert result["issued_waypoints"] == 2 and result["executed_waypoints"] == 1
    assert {row[2] for row in env.executed} == {"ee"}
    assert all(row[1].shape == (16,) and row[1][3] == 1 for row in env.executed)
    assert client.session.closed


def test_slow_simulator_skips_expired_servo_ticks_without_command_bursts():
    clock = FakeClock()
    env = FakeEnv(clock, action_seconds=0.4)
    client = FakeClient(env)
    result = run(env, client, execution_hz=1, servo_hz=10, max_waypoints=1)
    assert result["executed_waypoints"] == 1
    assert result["skipped_servo_ticks"] >= 4
    assert result["max_actuator_call_seconds"] == pytest.approx(0.4)
    assert result["max_servo_interval_seconds"] >= 0.39
    times = np.array([row[0] for row in env.executed])
    assert np.diff(times).min() >= 0.39
    assert env.executed[-1][1][0] == 1


def test_prediction_error_closes_session_and_does_not_replay_commands():
    clock = FakeClock()
    env = FakeEnv(clock)
    client = FakeClient(env)
    create = client.create_session

    def create_failing(*args, **kwargs):
        session = create(*args, **kwargs)
        original = session.get_action

        def get_action(**measurement):
            if session.tick == 1:
                raise RuntimeError("RPC disconnected")
            return original(**measurement)

        session.get_action = get_action
        return session

    client.create_session = create_failing
    with pytest.raises(RuntimeError, match="RPC disconnected"):
        run(env, client, enable_rtc=True)
    assert len(env.executed) == 2
    assert client.session.closed


def test_changed_content_region_requires_reset_before_the_next_waypoint():
    clock = FakeClock()
    env = FakeEnv(clock)
    client = FakeClient(env)

    def changing_geometry(images, resolution):
        image, size = prepare_image(images, resolution)
        if env.take_action_cnt >= 2:
            size[2] -= 1  # Same padded canvas, different real content region.
        return image, size

    with pytest.raises(ValueError, match="geometry changed"):
        runner.rollout(
            env,
            client,
            execution_hz=2,
            servo_hz=4,
            max_waypoints=3,
            adapter=adapter(),
            prepare_image=changing_geometry,
            clock=clock,
            sleep=clock.sleep,
        )
    assert len(env.executed) == 2
    assert client.session.closed


def test_native_measurements_and_camera_copy_do_not_mutate_renderer_buffers():
    clock = FakeClock()
    env = FakeEnv(clock)
    observation = env.get_obs()
    images = runner.camera_images(observation)
    command_state = runner.native_state(observation, env, state_source="command")
    real_state = runner.native_state(observation, env, state_source="measured")
    env.buffer.fill(99)
    assert images["cam_high"][0, 0, 0] == 0
    assert command_state["joint_action"]["left_arm"] == [9] * 6
    np.testing.assert_array_equal(real_state["joint_action"]["left_arm"], np.zeros(6))
    assert observation["joint_action"]["left_arm"] == [9] * 6


def test_background_planner_does_not_stop_native_command_execution():
    clock = FakeClock()
    env = FakeEnv(clock)
    started, release = Event(), Event()
    native_adapter = adapter()
    native_adapter.action_mask[:] = False
    native_adapter.action_mask[0] = True
    native_adapter.available_control_modes = ["joint"]

    class NetworkLikeClient(FakeClient):
        def create_session(self, first_measurement, *, prompt, **options):
            def planner(**request):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release simulated RPC")
                result = torch.stack([command(request["tick"] + i) for i in range(32)])
                return result

            self.session = RTCSession(
                planner,
                torch.stack([command(i) for i in range(32)]),
                template=native_adapter.template,
                controller_hz=30,
                execution_hz=2,
                min_execution_steps=16,
                initial_delay_steps=2,
            )
            return self.session

    client = NetworkLikeClient(env)
    pending_commands = []

    def progress():
        if len(env.executed) == 32:
            return
        if client.session.status()["inference_pending"]:
            assert started.wait(2)
            pending_commands.append(len(env.executed))
        if len(env.executed) == 35:
            release.set()
            deadline = time.monotonic() + 2
            while client.session.status()["inference_pending"] and time.monotonic() < deadline:
                time.sleep(0.001)

    env.after_action = progress
    try:
        result = runner.rollout(
            env,
            client,
            enable_rtc=True,
            adaptive_execution=False,
            execution_hz=2,
            servo_hz=4,
            max_waypoints=20,
            adapter=native_adapter,
            prepare_image=prepare_image,
            clock=clock,
            sleep=clock.sleep,
        )
        assert result["continuous_rtc"]
        assert result["executed_waypoints"] == 20
        assert pending_commands[:3] == [33, 34, 35]
        assert result["final_status"]["published_requests"] == 1
    finally:
        release.set()


def test_cli_requires_explicit_robotwin_root_and_rtc_is_opt_in():
    with pytest.raises(SystemExit):
        runner.parser().parse_args([])
    args = runner.parser().parse_args(["--robotwin-root", "/tmp/robotwin"])
    assert args.enable_rtc is False
    assert args.execution_hz is None and args.state_source == "measured"
    args = runner.parser().parse_args(["--robotwin-root", "/tmp/robotwin", "--enable-rtc", "--no-adaptive-execution"])
    assert args.enable_rtc and not args.adaptive_execution


@pytest.mark.parametrize("instruction", [None, "Use this instruction."])
def test_main_reuses_robotwin_setup_and_closes_each_episode(monkeypatch, tmp_path, instruction):
    from veomni.models.diffusers.causal_cosmos3.serving import client as client_module

    environments, closed, rollouts = [], [], []

    class Environment:
        def __init__(self):
            self.robot = SimpleNamespace(left_arm_joints=[None] * 6, right_arm_joints=[None] * 6)
            self.step_lim = 1000
            self.setups = []

        def setup_demo(self, **kwargs):
            self.setups.append(kwargs)

        def play_once(self):
            return dict(task="bottle")

        def set_instruction(self, *, instruction):
            self.instruction = instruction

    class Client:
        metadata = dict(adapter=dict(left_arm_dof=6, right_arm_dof=6))

        def __init__(self, uri, **kwargs):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    def create_env(task):
        env = Environment()
        environments.append(env)
        return env

    def execute(env, client, **kwargs):
        rollouts.append((env.instruction, kwargs))
        return dict(success=True, executed_waypoints=2)

    helpers = SimpleNamespace(
        load_task_args=lambda args: (dict(camera={}, data_type={}), None),
        resolve_eval_instruction=lambda args, task: "seen",
        class_decorator=create_env,
        build_instruction=lambda args, info, kind, count, seed: "Generated bottle instruction.",
        safe_close_env=lambda env: closed.append(env),
        UnStableError=type("UnStableError", (RuntimeError,), {}),
    )
    monkeypatch.setattr(client_module, "CausalPolicyClient", Client)
    monkeypatch.setattr(runner, "load_robotwin", lambda root: helpers)
    monkeypatch.setattr(runner, "rollout", execute)
    arguments = [
        "robotwin_client",
        "--robotwin-root",
        "/test/robotwin",
        "--episodes",
        "2",
        "--max-waypoints",
        "2",
        "--enable_rtc",
        "--output",
        str(tmp_path),
    ]
    if instruction is not None:
        arguments += ["--instruction", instruction]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    assert len(environments) == len(rollouts) == 2
    assert all(env in closed for env in environments)
    assert [len(env.setups) for env in environments] == ([2, 2] if instruction is None else [1, 1])
    assert [row[0] for row in rollouts] == [instruction or "Generated bottle instruction."] * 2
    assert all(options["enable_rtc"] and options["max_waypoints"] == 2 for _, options in rollouts)
