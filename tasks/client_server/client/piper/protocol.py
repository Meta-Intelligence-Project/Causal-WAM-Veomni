"""Cosmos protocol adapter for the LingBot-based Piper execution loop.

No hardware is connected on import. Executor actions use [L6, Lgrip, R6, Rgrip],
radians and metres for joints or XYZ+RPY flange poses. Observation indices
refer to model steps, not interpolated controller steps.
"""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tasks.client_server.client.piper.eef import eef_actions_to_pose, wait_eef_target
from tasks.client_server.client.runtime import save_structured_prompt


def validate_actions(actions):
    """Check the Cosmos wire shape; execution limits come from Lingbot hardware."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected joint actions [N,14]")
    return actions


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
        self.observation_group = meta.get("observation_chunk_steps")
        if self.stream_observations and (
            not meta.get("observation_streaming", False)
            or type(self.observation_group) is not int
            or self.observation_group <= 0
            or self.execute_steps % self.observation_group
            or self.observation_group % self.video_stride
        ):
            raise ValueError("Server does not support complete streamed observation groups")
        self._sender = None
        self._uploads = []
        self._submitted_steps = 0
        self.observation_step = 0
        self.feedback = None
        self.predicted_chunks = 0

    def reset(self):
        # Finish any active RPC and cancel queued groups before reusing the socket.
        self.close()
        self.observation_step = 0
        self._submitted_steps = 0
        self.feedback = None
        self.predicted_chunks = 0
        self.client.reset(prompt=self.cfg.prompt, record_video=getattr(self.cfg, "save_prediction_video", False))

    def observe(self, piper, frames, feedback_only=False):
        import cv2

        images = {
            key: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for key, frame in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), frames)
        }
        if feedback_only:
            return dict(images=images)
        return dict(images=images, state=cosmos_state(piper, self.cfg.eef_position_scale))

    def infer(self, observation):
        if self._uploads:
            raise ValueError("Finish the observation block before inference")
        request = dict(
            state=observation["state"],
            executed_steps=self.execute_steps,
            observation_step=self.observation_step,
        )
        if not self.stream_observations or self.observation_step == 0:
            request["obs"] = self.feedback if self.feedback is not None else observation["images"]
        result = self.client.infer(request)
        self.predicted_chunks += 1
        return result

    def check_feedback(self):
        """Surface background failures before sending another robot command."""
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
            if self._sender is None:
                self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-observation")
            previous = self._uploads[-1] if self._uploads else None
            self._uploads.append(
                self._sender.submit(self._upload_observations, previous, images, self.observation_step + end)
            )
            self._submitted_steps = end

    def _upload_observations(self, previous, images, step):
        # A failed group must prevent all later queued requests on this session.
        if previous is not None:
            previous.result()
        response = self.client.observe(images, observation_step=step)
        if response.get("observed") is not True or response.get("observation_step") != step:
            raise ValueError("Server acknowledged an unexpected observation step")

    def close(self):
        """Cancel unsent groups and drain the active RPC before video export/close."""
        if self._sender is not None:
            self._sender.shutdown(wait=True, cancel_futures=True)
            self._sender = None
        uploads, self._uploads = self._uploads, []
        for upload in uploads:
            if not upload.cancelled():
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
        return actions, 0, keep, list(range(keep)), 1

    def finish_block(self, observations, result, final_observation, full):
        if full:
            if len(observations) != self.execute_steps:
                raise ValueError("Cosmos requires one feedback frame per model step")
            if self.stream_observations:
                if self._submitted_steps != self.execute_steps:
                    raise ValueError("All observation groups must be submitted during execution")
                for upload in self._uploads:
                    upload.result()
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
        feedback = wait(piper, left, right, np.radians(cfg.wait_tol_deg), stop_check)
        report.data["last_motion_feedback"] = feedback
        with (logger.save_dir / "motion_feedback.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(command_step=total, **feedback)) + "\n")
        check_stop()

    check_stop()
    protocol.reset()
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
                protocol.reset()
                initial, _ = observe()
                first = True
                continue
            if command != "next":
                raise ValueError("Unknown round command")
            # Refresh measured state after a potentially long operator pause.
            initial, _ = observe()
        report.update(status="inferring", completed_chunks=completed, round=rounds)
        result = protocol.infer(initial)
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
                if isinstance(protocol, CosmosProtocol):
                    protocol.submit_feedback(observations)
                if step_idx in wait_at:
                    logger.log_key_frame(rounds, total, *frames, action)
                    frame_dir = logger.save_dir / "key_frames"
                    frame_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(frame_dir / ("step_%04d.jpg" % total)), np.hstack(frames))
            if step_idx not in wait_at:
                time.sleep(max(0, cfg.action_dt - (time.monotonic() - step_start)))
        wait_target(left, right)
        final, _ = observe()
        report.update(status="updating_cache")
        initial = protocol.finish_block(observations, result, final, keep == protocol.execute_steps)
        first = False
        completed += units
        rounds += 1
        report.update(completed_chunks=completed, round=rounds, robot_commands_sent=total)
        print("CHUNK_COMPLETE", completed, "steps", total, flush=True)
    report.update(status="complete")
    return total
