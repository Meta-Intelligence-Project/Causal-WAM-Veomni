# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboTwin execution adapter for the causal policy WebSocket service.

The client owns action time and interpolates slower model waypoints into native
simulator commands. RoboTwin imports are deferred until main(). The simulator's
take_action may itself block; timings report this instead of claiming realtime.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from numbers import Real
from pathlib import Path

import numpy as np


def camera_images(observation):
    cameras = observation["observation"]
    return {
        target: np.asarray(cameras[source]["rgb"]).copy()
        for source, target in (
            ("head_camera", "cam_high"),
            ("left_camera", "cam_left_wrist"),
            ("right_camera", "cam_right_wrist"),
        )
    }


def native_state(observation, env, *, state_source="measured"):
    joint = dict(observation["joint_action"])
    if state_source == "measured":
        left = list(env.robot.get_left_arm_real_jointState())
        right = list(env.robot.get_right_arm_real_jointState())
        joint.update(
            left_arm=left[:-1],
            right_arm=right[:-1],
            left_gripper=left[-1],
            right_gripper=right[-1],
            vector=np.asarray(left + right),
        )
    elif state_source != "command":
        raise ValueError(f"Unknown state_source: {state_source}")
    return dict(joint_action=joint, endpose=observation["endpose"])


def episode_finished(env):
    return bool(env.eval_success or env.take_action_cnt >= env.step_lim)


def _frequency(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _measurement(env, adapter, resolution, state_source, prepare_image):
    import torch

    observation = env.get_obs()
    image, image_size = prepare_image(camera_images(observation), resolution)
    state = torch.from_numpy(adapter.assemble_state(native_state(observation, env, state_source=state_source)))
    return dict(
        state=state,
        state_mask=adapter.state_mask.detach().cpu().clone(),
        action_mask=adapter.action_mask.detach().cpu().clone(),
        image=image,
        image_size=image_size,
    )


def _control_measurement(measurement):
    return {key: measurement[key] for key in ("state", "state_mask", "action_mask", "image")}


def rollout(
    env,
    client,
    *,
    enable_rtc=False,
    execution_mode="auto",
    execution_hz=None,
    servo_hz=30.0,
    adaptive_execution=True,
    execution_policy=None,
    state_source="measured",
    max_waypoints=None,
    output=None,
    adapter=None,
    prepare_image=None,
    clock=None,
    sleep=None,
):
    """Execute one episode with local timing and a background network planner.

    Each model waypoint advances the session once. Servo interpolation does not
    create observation ticks. The bootstrap and prepare calls run before motion.
    Both asynchronous modes calibrate complete RPCs when adaptive execution is
    enabled, and share the same trigger/handoff schedule. RTC additionally guides
    the new prediction with the old action prefix. Acknowledged rate reductions
    stretch future interpolation intervals without changing the model time grid.
    A late handoff may wait; missed servo deadlines are skipped, rather than
    emitted as a burst of old commands.
    The final target is sent unless the episode ends before it can be executed.

    This adapter implements a RoboTwin interpolation policy. Hardware clients
    must implement their own actuator limits, interpolation and stop behavior.
    """
    import torch

    from veomni.models.diffusers.causal_cosmos3.serving.rtc_client_simulation import interpolate_action
    from veomni.models.diffusers.causal_cosmos3.serving.serving_model import (
        adapter_from_metadata,
        prepare_camera_observation,
    )

    if state_source not in ("measured", "command"):
        raise ValueError("state_source must be measured or command")
    if max_waypoints is not None and (type(max_waypoints) is not int or max_waypoints < 1):
        raise ValueError("max_waypoints must be a positive integer")
    if client.control_mode not in ("joint", "eef"):
        raise ValueError("control_mode must be joint or eef")
    model_hz = _frequency(client.metadata.get("controller_hz", client.metadata.get("fps", 30.0)), "controller_hz")
    requested_hz = _frequency(model_hz if execution_hz is None else execution_hz, "execution_hz")
    servo_hz = _frequency(servo_hz, "servo_hz")
    clock = time.perf_counter if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    if not callable(clock) or not callable(sleep):
        raise ValueError("clock and sleep must be callable")
    adapter = adapter_from_metadata(client.metadata) if adapter is None else adapter
    prepare_image = prepare_camera_observation if prepare_image is None else prepare_image
    resolution = client.metadata.get("resolution", "480")
    if client.control_mode not in adapter.available_control_modes:
        raise ValueError(f"The action mask does not support {client.control_mode} execution")
    pacer = dict(hz=requested_hz)
    changes, action_trace, servo_trace, waypoints = [], [], [], []
    last_clock = None

    def now():
        nonlocal last_clock
        value = clock()
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("clock must return finite timestamps")
        if last_clock is not None and value < last_clock:
            raise ValueError("clock must be monotonic")
        last_clock = float(value)
        return last_clock

    def wait_until(deadline):
        value = now()
        while value < deadline:
            sleep(deadline - value)
            following = now()
            if following <= value:
                raise RuntimeError("sleep/clock did not advance")
            value = following
        return value

    def acknowledge_rate(rate):
        rate = _frequency(rate, "execution_hz")
        if rate > pacer["hz"]:
            raise ValueError("The execution adapter accepts slower rates only")
        # This loop owns the future schedule; updating it is the actual local ACK.
        changes.append(dict(previous_hz=pacer["hz"], execution_hz=rate))
        pacer["hz"] = rate
        return True

    if episode_finished(env):
        return dict(success=bool(env.eval_success), executed_steps=0, executed_waypoints=0, issued_waypoints=0)
    first = _measurement(env, adapter, resolution, state_source, prepare_image)
    image_size = torch.as_tensor(first["image_size"]).detach().cpu().clone()
    first_request_index = len(getattr(client, "request_timings", []))
    session = client.create_session(
        first,
        prompt=env.get_instruction(),
        enable_rtc=enable_rtc,
        execution_mode=execution_mode,
        execution_hz=requested_hz,
        execution_policy=execution_policy if adaptive_execution else None,
        execution_rate_callback=acknowledge_rate,
        adaptive_execution=adaptive_execution,
    )
    prepared_at = now()
    issued = completed = 0
    skipped_servo_ticks = 0
    try:
        session.prepare(**_control_measurement(first))
        status = session.status()
        initial_hz = _frequency(session.execution_hz, "execution_hz")
        if not math.isclose(initial_hz, pacer["hz"], rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("Session execution rate was not acknowledged by the local pacer")
        if servo_hz < initial_hz:
            raise ValueError("servo_hz must be at least the prepared waypoint execution rate")
        started = now()
        deadline = started
        previous = None
        action_type = "qpos" if client.control_mode == "joint" else "ee"
        rtc_active_throughout = bool(status.get("rtc_enabled", False))
        while not episode_finished(env) and (max_waypoints is None or issued < max_waypoints):
            wait_until(deadline)
            # Bootstrap already owns tick zero. A live camera can change while
            # startup inference runs, so capture a fresh image only after tick 0.
            measurement = first if issued == 0 else _measurement(env, adapter, resolution, state_source, prepare_image)
            if not torch.equal(torch.as_tensor(measurement["image_size"]).detach().cpu(), image_size):
                raise ValueError("Camera layout/content geometry changed within the episode; reset is required")
            before = session.status()
            tick = before["tick"]
            call_started = now()
            command = session.get_action(**_control_measurement(measurement))
            returned_at = now()
            status = session.status()
            if status["tick"] != tick + 1:
                raise RuntimeError("One waypoint must advance exactly one session tick")
            rate = _frequency(session.execution_hz, "execution_hz")
            if not math.isclose(rate, pacer["hz"], rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("Session rate changed without a corresponding pacing acknowledgement")
            command = torch.as_tensor(command).detach().float().cpu()
            if command.shape != (adapter.template.width,) or not torch.isfinite(command).all():
                raise ValueError("Session returned an invalid unified action")
            command = adapter.template.sanitize(command, adapter.action_mask)
            if previous is None:
                previous = torch.where(measurement["state_mask"], measurement["state"].float(), command).clone()
            period = 1 / rate
            # 普通blocking或异步计划在交接点迟到都会产生真实等待。
            # 从返回时刻开启新周期，绝不补发已过期的servo指令。
            interval_start = max(deadline, returned_at)
            interval_end = interval_start + period
            rtc_active_throughout &= bool(status.get("rtc_enabled", False))
            issued += 1
            waypoints.append(command.clone())
            action_trace.append(
                dict(
                    tick=tick,
                    scheduled_seconds=deadline - started,
                    started_seconds=call_started - started,
                    returned_seconds=returned_at - started,
                    call_seconds=returned_at - call_started,
                    lateness_seconds=max(0.0, returned_at - deadline),
                    execution_hz=rate,
                    inference_pending=bool(status.get("inference_pending", False)),
                    published_requests=status.get("published_requests", 0),
                    execution_mode=status.get("execution_mode"),
                )
            )
            servo_tick = 1
            while not episode_finished(env):
                servo_deadline = min(interval_end, interval_start + servo_tick / servo_hz)
                at = wait_until(servo_deadline)
                status = session.status()
                if status.get("error"):
                    raise RuntimeError(f"Remote planning failed: {status['error']}")
                rtc_active_throughout &= bool(status.get("rtc_enabled", False))
                alpha = 1.0 if at >= interval_end else min(1.0, max(0.0, (at - interval_start) / period))
                target = interpolate_action(adapter.template, adapter.action_mask, previous, command, alpha)
                native = adapter.split_actions(target[None].numpy())[client.control_mode]
                if native is None or native.ndim != 2 or native.shape[0] != 1 or not np.isfinite(native).all():
                    raise ValueError("Adapter returned an invalid native action")
                env.take_action(native[0].copy(), action_type=action_type)
                finished_at = now()
                servo_trace.append(
                    dict(
                        tick=tick,
                        alpha=alpha,
                        scheduled_seconds=servo_deadline - started,
                        started_seconds=at - started,
                        completed_seconds=finished_at - started,
                        lateness_seconds=max(0.0, at - servo_deadline),
                    )
                )
                if alpha >= 1:
                    completed += 1
                    break
                following_tick = max(servo_tick + 1, math.floor((finished_at - interval_start) * servo_hz) + 1)
                skipped_servo_ticks += following_tick - servo_tick - 1
                servo_tick = following_tick
            previous = command
            deadline = max(interval_end, now())
        elapsed = now() - started
        final_status = session.status()
        if final_status.get("error"):
            raise RuntimeError(f"Remote planning failed: {final_status['error']}")
        result = dict(
            success=bool(env.eval_success),
            executed_steps=completed,
            executed_waypoints=completed,
            issued_waypoints=issued,
            servo_commands=len(servo_trace),
            simulator_action_count=env.take_action_cnt,
            model_seconds=completed / model_hz,
            wall_seconds=elapsed,
            startup_calibration_seconds=started - prepared_at,
            controller_hz=model_hz,
            requested_execution_hz=requested_hz,
            initial_execution_hz=initial_hz,
            final_execution_hz=pacer["hz"],
            servo_hz=servo_hz,
            adaptive_execution=bool(final_status.get("adaptive_execution", False)),
            continuous_rtc=bool(enable_rtc and rtc_active_throughout),
            execution_mode=final_status.get("execution_mode"),
            state_source=state_source,
            rate_changes=changes,
            skipped_servo_ticks=skipped_servo_ticks,
            max_servo_lateness_seconds=max((row["lateness_seconds"] for row in servo_trace), default=0.0),
            max_waypoint_lateness_seconds=max((row["lateness_seconds"] for row in action_trace), default=0.0),
            max_actuator_call_seconds=max(
                (row["completed_seconds"] - row["started_seconds"] for row in servo_trace), default=0.0
            ),
            max_servo_interval_seconds=max(
                (
                    right["started_seconds"] - left["started_seconds"]
                    for left, right in zip(servo_trace, servo_trace[1:])
                ),
                default=0.0,
            ),
            final_status=final_status,
            requests=list(getattr(client, "request_timings", []))[first_request_index:],
            action_trace=action_trace,
            servo_trace=servo_trace,
            qualification="RoboTwin execution adapter; simulator wall timing is measured, not a hard realtime guarantee.",
        )
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            np.save(
                directory / "waypoints_unified.npy", torch.stack(waypoints).numpy() if waypoints else np.empty((0, 55))
            )
            (directory / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        # Finish or discard the old episode's RPC before the caller resets the
        # simulator or starts a new episode on this connection.
        session.close(wait=True)


def load_robotwin(root):
    """Reuse a RoboTwin checkout's task/config helpers without its RPC loop."""
    root = Path(root).expanduser().resolve()
    helper = root / "scripts/eval_policy_xpolicylab.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Expected RoboTwin task/config helpers: {helper}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    name = "_cosmos_robotwin_eval_helpers"
    spec = importlib.util.spec_from_file_location(name, helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return parsed


def positive_frequency(value):
    try:
        return _frequency(float(value), "frequency")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robotwin-root", required=True, help="Path to the RoboTwin simulator repository")
    p.add_argument("--server-uri", default="ws://127.0.0.1:8000")
    p.add_argument("--task-name", default="adjust_bottle")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--instruction", help="Override the RoboTwin-generated instruction")
    p.add_argument("--instruction-type", choices=["seen", "unseen"])
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--episodes", type=positive_int, default=1)
    p.add_argument("--control-mode", choices=["joint", "eef"], default="joint")
    p.add_argument("--state-source", choices=["command", "measured"], default="measured")
    p.add_argument("--max-steps", type=positive_int, help="Cap native simulator take_action calls")
    p.add_argument(
        "--max-waypoints", type=positive_int, help="Cap model waypoint requests independently of servo calls"
    )
    p.add_argument("--render-freq", type=int, default=0)
    p.add_argument("--enable_rtc", "--enable-rtc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--execution-mode",
        choices=("auto", "blocking", "async"),
        default="auto",
        help="auto: asynchronous execution with optional RTC guidance; blocking is an explicit non-RTC baseline",
    )
    p.add_argument("--adaptive-execution", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--initial-execution-hz",
        "--execution-hz",
        dest="execution_hz",
        type=positive_frequency,
        metavar="HZ",
        help="Initial waypoint rate in Hz (defaults to model rate); adaptive execution may reduce it before and during motion",
    )
    p.add_argument("--servo-hz", type=positive_frequency, default=30.0)
    p.add_argument("--request-timeout", type=positive_frequency, default=120.0)
    p.add_argument("--latency-factor", type=positive_frequency, default=1.2)
    p.add_argument("--margin-steps", type=positive_int, default=1)
    p.add_argument("--extra-latency-seconds", type=float, default=0.0)
    p.add_argument("--minimum-execution-hz", type=float, default=0.0)
    p.add_argument("--output", default="outputs/causal_action/robotwin_sim")
    return p


def main():
    from veomni.models.diffusers.causal_cosmos3.serving.client import CausalPolicyClient
    from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy

    args = parser().parse_args()
    if args.render_freq < 0:
        raise ValueError("render-freq must be nonnegative")
    if args.instruction is not None and not args.instruction.strip():
        raise ValueError("instruction override must not be empty")
    policy = RTCExecutionPolicy(
        latency_factor=args.latency_factor,
        margin_steps=args.margin_steps,
        extra_latency_seconds=args.extra_latency_seconds,
        minimum_execution_hz=args.minimum_execution_hz,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        task=args.task_name, task_config=args.task_config, server_uri=args.server_uri, episodes=[], skipped_seeds=[]
    )
    with CausalPolicyClient(
        args.server_uri, control_mode=args.control_mode, request_timeout=args.request_timeout
    ) as client:
        helpers = load_robotwin(args.robotwin_root)
        task_args, _ = helpers.load_task_args(
            dict(
                task_name=args.task_name,
                task_config=args.task_config,
                policy_name="cosmos_causal",
                ckpt_setting="causal_action",
            )
        )
        task_args.update(eval_mode=True, eval_video_log=False, eval_video_save_dir=None, render_freq=args.render_freq)
        task_args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
        task_args["data_type"].update(rgb=True, qpos=True, endpose=True)
        instruction_type = (
            helpers.resolve_eval_instruction(dict(instruction_type=args.instruction_type), task_args)
            if args.instruction is None
            else None
        )
        episode, seed = 0, args.seed
        while episode < args.episodes:
            env = helpers.class_decorator(args.task_name)
            env.suc, env.test_num = 0, episode
            record = dict(episode=episode, seed=seed)
            try:
                setup_args = dict(now_ep_num=episode, seed=seed, is_test=True, **task_args)
                env.setup_demo(**(dict(setup_args, render_freq=0) if args.instruction is None else setup_args))
                adapter_meta = client.metadata.get("adapter", {})
                for side in ("left", "right"):
                    expected = adapter_meta.get(f"{side}_arm_dof", client.metadata.get(f"{side}_arm_dof"))
                    if len(getattr(env.robot, f"{side}_arm_joints")) != expected:
                        raise ValueError(f"{side} arm DOF differs between RoboTwin and server")
                instruction = args.instruction
                if instruction is None:
                    info = env.play_once()
                    instruction = helpers.build_instruction(task_args, info, instruction_type, args.episodes, seed)
                    if not isinstance(instruction, str) or not instruction.strip() or instruction == args.task_name:
                        raise ValueError("RoboTwin instruction generation failed; supply --instruction")
                    helpers.safe_close_env(env)
                    env.setup_demo(**setup_args)
                if args.max_steps is not None:
                    env.step_lim = min(env.step_lim, args.max_steps)
                env.set_instruction(instruction=instruction)
                record.update(
                    rollout(
                        env,
                        client,
                        enable_rtc=args.enable_rtc,
                        execution_mode=args.execution_mode,
                        execution_hz=args.execution_hz,
                        servo_hz=args.servo_hz,
                        adaptive_execution=args.adaptive_execution,
                        execution_policy=policy,
                        state_source=args.state_source,
                        max_waypoints=args.max_waypoints,
                        output=output / f"seed{seed}",
                    )
                )
                record.update(status="finished", instruction=instruction)
            except helpers.UnStableError as error:
                record.update(status="skipped_unstable", error=f"{type(error).__name__}: {error}")
                continue
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                helpers.safe_close_env(env)
                summary["skipped_seeds" if record.get("status") == "skipped_unstable" else "episodes"].append(record)
                (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
                seed += 1
            episode += 1


if __name__ == "__main__":
    main()
