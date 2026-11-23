"""Cosmos protocol adapter for the LingBot-based Piper execution loop.

No hardware is connected on import. Executor actions use [L6, Lgrip, R6, Rgrip],
radians and metres for joints or XYZ+RPY flange poses. Observation indices
refer to model steps, not interpolated controller steps.
"""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import numpy as np

from tasks.client_server.client.piper.eef import eef_actions_to_pose, wait_eef_target
from tasks.client_server.client.piper.timing import PiperTimings
from tasks.client_server.client.runtime import save_structured_prompt
from tasks.common.causal_cosmos3_args import parse_async_infer


def validate_actions(actions):
    """Check the Cosmos wire shape; execution limits come from Lingbot hardware."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected joint actions [N,14]")
    return actions


def wait_request(future, stop_check=None):
    """Keep the control thread interruptible while one worker owns the socket."""
    while True:
        if stop_check is not None and stop_check():
            raise KeyboardInterrupt("Stop requested")
        try:
            return future.result(timeout=0.05)
        except TimeoutError:
            if future.done():
                return future.result()  # Propagate an RPC timeout instead of treating it as a polling timeout.


def wait_robot_target(piper, left, right, tolerance, stop_check, timeout=3.0):
    """Use Lingbot's arrival checks and continue after its timeout warning."""
    if stop_check():
        raise KeyboardInterrupt("Stop requested")
    reached = piper.wait_both_motion_done(
        left_joints=left, right_joints=right, timeout=timeout, poll_interval=0.01, tol_rad=tolerance
    )
    if stop_check():
        raise KeyboardInterrupt("Stop requested")
    feedback = dict(reached=bool(reached))
    # Diagnostic feedback must not add another arrival/freshness requirement.
    try:
        target = np.asarray(list(left) + list(right), dtype=float)
        measured = np.asarray(piper.get_joint_angles(), dtype=float)
        feedback.update(
            target_deg=np.degrees(target).tolist(),
            measured_deg=np.degrees(measured).tolist(),
            max_error_deg=float(np.degrees(np.abs(target - measured).max())),
        )
    except Exception as exc:
        print("[Piper] 到位反馈日志读取失败：" + str(exc), flush=True)
    return feedback


class RunReport:
    def __init__(self, cfg):
        self.path = Path(cfg.report_dir) / "report.json" if cfg.report_dir else None
        self.data = dict(
            status="starting",
            robot_commands_sent=0,
            completed_chunks=0,
            round=0,
            prompt=cfg.prompt,
            model_family=cfg.model_family,
            action_space="eef" if cfg.control_mode == "eef" else "joint",
        )
        self.update()

    def update(self, **values):
        self.data.update(values)
        print(
            "CLIENT_STATUS "
            + json.dumps(
                {key: self.data[key] for key in ("status", "robot_commands_sent", "completed_chunks", "round")}
            ),
            flush=True,
        )
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.data, indent=2) + "\n")
            temp.replace(self.path)


def cosmos_state(piper, position_scale=1000.0):
    """Read Lingbot joint feedback and the EEF/gripper fields Cosmos requires."""
    from scipy.spatial.transform import Rotation

    joints = np.asarray(piper.get_joint_angles(), dtype=float)
    state = dict(joint_action={}, endpose={})
    for index, (side, arm, gripper) in enumerate(
        (("left", piper.robot_l, piper.gripper_l), ("right", piper.robot_r, piper.gripper_r))
    ):
        pose, grip = arm.get_flange_pose(), gripper.get_gripper_status()
        if pose is None or grip is None:
            raise ValueError("Cosmos observations require EEF and gripper feedback")
        if grip.msg.mode != "width":
            raise ValueError("Cosmos gripper feedback must use width in metres")
        xyzrpy = np.asarray(pose.msg)
        state["joint_action"][side + "_arm"] = joints[index * 6 : (index + 1) * 6].tolist()
        state["joint_action"][side + "_gripper"] = float(grip.msg.value)
        # Old Piper SDK EEF positions were mm; pyAgxArm reports metres.
        state["endpose"][side + "_endpose"] = (xyzrpy[:3] * position_scale).tolist() + Rotation.from_euler(
            "xyz", xyzrpy[3:]
        ).as_quat().tolist()
    return state


class CosmosProtocol:
    def __init__(self, client, cfg, format_obs=None):
        self.client, self.cfg = client, cfg
        self.timings = PiperTimings()
        meta = client.metadata
        self.block = meta["actions_per_block"]
        if (
            type(self.block) is not int
            or self.block <= 0
            or type(meta["execute_steps"]) is not int
            or not 0 < meta["execute_steps"] <= self.block
            or meta["fps"] != cfg.model_fps
        ):
            raise ValueError("Server horizon/frequency differs from saved configuration")
        self.execute_steps = meta["execute_steps"]
        if self.execute_steps != self.block:
            raise ValueError("VeOmni Piper requires full-block execution")
        self.video_stride = meta.get("video_stride", 4)
        if type(self.video_stride) is not int or self.video_stride <= 0:
            raise ValueError("Invalid Cosmos video stride")
        self.stream_observations = meta.get("observation_mode") == "streaming"
        async_infer = parse_async_infer(meta.get("async_infer", {}))
        self.rtc_enabled, self.rtc_delay = async_infer["rtc_enabled"], async_infer["rtc_delay"]
        self.rtc_stride = self.block // 2 if self.rtc_enabled else 0
        self.observation_group = meta.get("observation_chunk_steps")
        if self.rtc_enabled:
            if not meta.get("rtc_async"):
                raise ValueError(
                    "Server does not support pure asynchronous RTC; set async_infer.rtc_enabled=false in server YAML"
                )
            if self.rtc_delay > self.block // 2:
                raise ValueError("rtc_delay must not exceed actions_per_block / 2")
            if (
                self.block % 2
                or type(self.observation_group) is not int
                or self.observation_group <= 0
                or self.rtc_stride % self.observation_group
                or self.observation_group % self.video_stride
                or getattr(cfg, "smooth_method", "none") != "none"
                or getattr(cfg, "save_prediction_video", False)
                or cfg.control_mode == "eef"
            ):
                raise ValueError(
                    "RTC requires a latent-aligned half-horizon, joint controls, smooth_method=none and video recording off"
                )
            self.stream_observations = False
        self.rtc_future = None
        self.rtc_images = []
        if self.stream_observations and (
            not meta.get("observation_streaming", False)
            or type(self.observation_group) is not int
            or self.observation_group <= 0
            or self.execute_steps % self.observation_group
            or self.observation_group % self.video_stride
        ):
            raise ValueError("Server does not support complete streamed observation groups")
        self._sender = None
        self._request = None
        self._uploads = []
        self._submitted_steps = 0
        self.observation_step = 0
        self.feedback = None
        self.predicted_chunks = 0

    def _submit(self, function, *args, **kwargs):
        if self._sender is None:
            self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-rpc")
        return self._sender.submit(function, *args, **kwargs)

    def _call(self, function, *args, stop_check=None, **kwargs):
        if stop_check is None:
            return function(*args, **kwargs)
        self._request = self._submit(function, *args, **kwargs)
        try:
            return wait_request(self._request, stop_check)
        finally:
            if self._request.done():
                self._request = None

    def save_prediction_video(self, path, *, stop_check=None):
        return self._call(self.client.save_prediction_video, path, stop_check=stop_check)

    def reset(self, stop_check=None):
        # Finish any active RPC and cancel queued groups before reusing the socket.
        self.close()
        self.observation_step = 0
        self._submitted_steps = 0
        self.feedback = None
        self.predicted_chunks = 0
        self.rtc_images.clear()
        self._call(
            self.client.reset,
            prompt=self.cfg.prompt,
            record_video=getattr(self.cfg, "save_prediction_video", False),
            stop_check=stop_check,
        )

    def observe(self, piper, frames, feedback_only=False):
        import cv2

        images = {
            key: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for key, frame in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), frames)
        }
        if feedback_only:
            return dict(images=images)
        return dict(images=images, state=cosmos_state(piper, self.cfg.eef_position_scale))

    def _timed_infer(self, request):
        started = self.timings.now()
        step = request["observation_step"]
        try:
            result = self.client.infer(request)
        except BaseException as exc:
            self.timings.record("request_failed", observation_step=step, error=type(exc).__name__)
            raise
        self.timings.record(
            "request_completed",
            observation_step=step,
            client_round_trip_ms=(self.timings.now() - started) * 1000,
            server_infer_ms=result.get("server_timing", {}).get("infer_ms"),
        )
        return result

    def _rtc_request(self, observation, images, step):
        result = self._timed_infer(
            dict(
                state=observation["state"],
                obs=images,
                observation_step=step,
            )
        )
        if result.get("observation_step") != step:
            raise ValueError("RTC result belongs to a different decision")
        return result

    def prefetch(self, observation):
        if len(self.rtc_images) != self.rtc_stride or self.rtc_future is not None:
            raise RuntimeError("RTC snapshot must advance by exactly one stride")
        self.observation_step += self.rtc_stride
        self.timings.record("request_submitted", observation_step=self.observation_step)
        self.rtc_future = self._submit(self._rtc_request, observation, self.rtc_images, self.observation_step)
        self.rtc_images = []

    def infer(self, observation, stop_check=None):
        if self.rtc_stride:
            if self.rtc_future is None:
                self.timings.record("request_submitted", observation_step=0)
                self.rtc_future = self._submit(self._rtc_request, observation, observation["images"], 0)
            waited = self.timings.now()
            result = wait_request(self.rtc_future, stop_check)
            self.timings.record(
                "handoff_received",
                observation_step=self.observation_step,
                wait_ms=(self.timings.now() - waited) * 1000,
            )
            self.rtc_future = None
            self.predicted_chunks += 1
            return result
        if self._uploads:
            raise ValueError("Finish the observation block before inference")
        request = dict(
            state=observation["state"],
            executed_steps=self.execute_steps,
            observation_step=self.observation_step,
        )
        if not self.stream_observations or self.observation_step == 0:
            request["obs"] = self.feedback if self.feedback is not None else observation["images"]
        self.timings.record("request_submitted", observation_step=self.observation_step)
        result = self._call(self._timed_infer, request, stop_check=stop_check)
        self.predicted_chunks += 1
        return result

    def check_feedback(self):
        """Surface background failures before sending another robot command."""
        if self.rtc_future is not None and self.rtc_future.done():
            self.rtc_future.result()
        for upload in self._uploads:
            if upload.done() and not upload.cancelled():
                upload.result()

    def submit_feedback(self, observations):
        """Queue complete model-step groups without waiting in the motion loop.

        The list is the current block's cumulative feedback, including repeated
        samples from the smoother mapping. RGB arrays are owned by observe() and
        remain immutable. At most one block is queued; one worker owns the socket.
        """
        if not self.stream_observations:
            return
        self.check_feedback()
        if not self.predicted_chunks or not self._submitted_steps <= len(observations) <= self.execute_steps:
            raise ValueError("Feedback must advance within the current inferred block")
        while self._submitted_steps + self.observation_group <= len(observations):
            end = self._submitted_steps + self.observation_group
            images = [item["images"] for item in observations[self._submitted_steps : end]]
            previous = self._uploads[-1] if self._uploads else None
            self._uploads.append(
                self._submit(self._upload_observations, previous, images, self.observation_step + end)
            )
            self._submitted_steps = end

    def _upload_observations(self, previous, images, step):
        # A failed group must prevent all later queued requests on this session.
        if previous is not None:
            previous.result()
        response = self.client.observe(images, observation_step=step)
        if response.get("observed") is not True or response.get("observation_step") != step:
            raise ValueError("Server acknowledged an unexpected observation step")

    def close(self, *, abort=False):
        """Drain before reuse; on abort close the transport to unblock a pending RPC."""
        if abort:
            self.client.close()
        if self._sender is not None:
            self._sender.shutdown(wait=True, cancel_futures=True)
            self._sender = None
        request, self._request = self._request, None
        if not abort and request is not None and not request.cancelled():
            request.result()
        future, self.rtc_future = self.rtc_future, None
        if not abort and future is not None and not future.cancelled():
            future.result()
        uploads, self._uploads = self._uploads, []
        for upload in uploads:
            if not abort and not upload.cancelled():
                upload.result()

    def plan(self, result, first, remaining):
        if self.cfg.control_mode == "eef":
            actions = eef_actions_to_pose(result["action"].get("eef"), self.cfg.eef_position_scale)
        else:
            actions = validate_actions(result["action"]["joint"])
        if len(actions) != self.block or result.get("execute_steps") != self.execute_steps:
            raise ValueError("Server returned unexpected Cosmos horizon")
        # Match Lingbot: clip grippers at execution, after trajectory smoothing.
        keep = self.execute_steps
        if self.rtc_stride:
            skip = 0 if first else self.rtc_delay
            keep = self.block if remaining == 1 else self.rtc_stride + self.rtc_delay
            return actions, skip, keep, list(range(skip, keep)), 1
        return actions, 0, keep, list(range(keep)), 1

    def finish_block(self, observations, result, final_observation, full, *, stop_check=None):
        if full:
            if len(observations) != self.execute_steps:
                raise ValueError("Cosmos requires one feedback frame per model step")
            if self.stream_observations:
                if self._submitted_steps != self.execute_steps:
                    raise ValueError("All observation groups must be submitted during execution")
                for upload in self._uploads:
                    wait_request(upload, stop_check)
                self._uploads = []
                self._submitted_steps = 0
                self.feedback = None
            else:
                self.feedback = [item["images"] for item in observations]
            self.observation_step += self.execute_steps
        return final_observation


def run_loop(piper, camera, protocol, cfg, logger, smoother, send_action, stop_check, report, before_round=None):
    """One executor for both protocols, using LingBot controls and smoothing."""
    import cv2

    def check_stop():
        if stop_check():
            raise KeyboardInterrupt("Stop requested")
        if isinstance(protocol, CosmosProtocol):
            protocol.check_feedback()

    def observe(feedback_only=False):
        frames = camera.get_frames()
        return protocol.observe(piper, frames, feedback_only=feedback_only), frames

    def wait_target(left, right):
        check_stop()
        wait = wait_eef_target if cfg.control_mode == "eef" else wait_robot_target
        started = time.perf_counter()
        feedback = wait(piper, left, right, np.radians(cfg.wait_tol_deg), stop_check)
        if isinstance(protocol, CosmosProtocol):
            protocol.timings.record(
                "arrival_wait",
                command_step=total,
                duration_ms=(time.perf_counter() - started) * 1000,
                reached=feedback.get("reached"),
            )
        report.data["last_motion_feedback"] = feedback
        with (logger.save_dir / "motion_feedback.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(command_step=total, **feedback)) + "\n")
        check_stop()

    rtc = getattr(protocol, "rtc_stride", 0)
    if rtc and (before_round is not None or smoother is not None):
        raise ValueError("RTC needs continuous execution and one command per model step")
    check_stop()
    protocol.reset(stop_check=stop_check) if isinstance(protocol, CosmosProtocol) else protocol.reset()
    time.sleep(3.0)
    while camera.get_frames()[0] is None:
        check_stop()
        time.sleep(0.5)
    initial, _ = observe()
    completed = rounds = total = 0
    first = True
    while not cfg.max_chunks or completed < cfg.max_chunks:
        check_stop()
        if before_round is not None:
            command = before_round(completed)
            check_stop()
            if command == "quit":
                break
            if command == "reset":
                protocol.reset(stop_check=stop_check) if isinstance(protocol, CosmosProtocol) else protocol.reset()
                initial, _ = observe()
                first = True
                continue
            if command != "next":
                raise ValueError("Unknown round command")
            # Refresh measured state after a potentially long operator pause.
            initial, _ = observe()
        report.update(status="inferring", completed_chunks=completed, round=rounds)
        result = (
            protocol.infer(initial, stop_check=stop_check)
            if isinstance(protocol, CosmosProtocol)
            else protocol.infer(initial)
        )
        save_structured_prompt(result, logger.save_dir / "structured_prompt.json")
        remaining = cfg.max_chunks - completed if cfg.max_chunks else None
        raw, skip, keep, feedback_indices, units = protocol.plan(result, first, remaining)
        if cfg.control_mode == "eef" and smoother is not None:
            raise ValueError("Joint smoothing cannot be applied to Cartesian EEF poses")
        actions, mapping = smoother.smooth_full_action(raw) if smoother else (raw, np.arange(len(raw)))
        actions = validate_actions(actions)
        mapping = np.asarray(mapping)
        start_idx, end_idx = int(mapping[skip]), int(mapping[keep - 1]) + 1
        # Duplicate nearest indices are valid: preserve one observation for each
        # original model step even when multiple indices map to the same sample.
        feedback_at = Counter(int(mapping[i]) for i in feedback_indices)
        wait_at = {int(mapping[i]) for i in range(skip, keep) if (i + 1) % protocol.video_stride == 0}
        action_log = (
            {"raw_eef_pose": raw, "executed_eef_pose": actions[start_idx:end_idx]}
            if cfg.control_mode == "eef"
            else {"raw_joint": raw, "executed_joint": actions[start_idx:end_idx]}
        )
        np.savez_compressed(
            logger.save_dir / ("actions-%04d.npz" % rounds),
            **(result["action"] if isinstance(result["action"], dict) else {"joint": raw}),
            action_fps=cfg.model_fps if cfg.model_family == "causal-cosmos" else 1 / cfg.action_dt,
            **action_log,
            original_indices=mapping,
            retained_steps=keep - skip,
        )
        report.update(status="executing", planned_steps=end_idx - start_idx)
        observations = []
        if rtc and skip == rtc and remaining != 1:
            protocol.prefetch(initial)
        if isinstance(protocol, CosmosProtocol):
            protocol.timings.record("execution_start", round=rounds, command_step=total)
        for step_idx in range(start_idx, end_idx):
            check_stop()
            step_start = time.monotonic()
            action = actions[step_idx]
            left, right = action[:6].tolist(), action[7:13].tolist()
            gl, gr = np.clip(action[[6, 13]], 0, 0.1)
            gl *= cfg.left_gripper_scale
            gr *= cfg.right_gripper_scale
            gl = 0 if gl <= cfg.gripper_close_threshold else gl
            gr = 0 if gr <= cfg.gripper_close_threshold else gr
            # A partially sent command can move one arm even if sending raises.
            report.data["motion_started"] = True
            if isinstance(protocol, CosmosProtocol):
                protocol.timings.record("command", round=rounds, command_step=total + 1)
            send_action(piper, left, right, gl, gr, cfg.control_mode, first)
            total += 1
            report.data["robot_commands_sent"] = total
            if step_idx in wait_at:
                wait_target(left, right)
            if step_idx in feedback_at:
                # Cosmos needs one image per model action, even between keyframes.
                observation, frames = observe(feedback_only=True)
                observations.extend([observation] * feedback_at[step_idx])
                check_stop()
                if rtc:
                    protocol.rtc_images.append(observation["images"])
                    if step_idx + 1 == rtc and remaining != 1:
                        snapshot = dict(state=cosmos_state(piper, cfg.eef_position_scale))
                        protocol.prefetch(snapshot)
                elif isinstance(protocol, CosmosProtocol):
                    protocol.submit_feedback(observations)
                if step_idx in wait_at:
                    logger.log_key_frame(rounds, total, *frames, action)
                    frame_dir = logger.save_dir / "key_frames"
                    frame_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(frame_dir / ("step_%04d.jpg" % total)), np.hstack(frames))
            if rtc or step_idx not in wait_at:
                time.sleep(max(0, cfg.action_dt - (time.monotonic() - step_start)))
        # A keyframe already waited on this target inside the action loop.
        if end_idx - 1 not in wait_at:
            wait_target(left, right)
        if isinstance(protocol, CosmosProtocol):
            protocol.timings.record("execution_end", round=rounds, command_step=total)
            if protocol.rtc_future is not None:
                protocol.timings.record("handoff_due", observation_step=protocol.observation_step)
        final, _ = observe()
        report.update(status="updating_cache")
        initial = (
            final
            if rtc
            else protocol.finish_block(
                observations,
                result,
                final,
                keep == protocol.execute_steps,
                **({"stop_check": stop_check} if isinstance(protocol, CosmosProtocol) else {}),
            )
        )
        first = False
        completed += units
        rounds += 1
        report.update(completed_chunks=completed, round=rounds, robot_commands_sent=total)
        print("CHUNK_COMPLETE", completed, "steps", total, flush=True)
    report.update(status="complete")
    return total
