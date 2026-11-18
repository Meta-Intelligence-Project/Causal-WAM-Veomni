"""Piper MOVE_P execution for Cosmos dataset-frame XYZ+XYZW EEF trajectories.

Dataset positions use cfg.eef_position_scale units per metre, matching the
inverse conversion used by cosmos_state(). pyAgxArm takes metres and XYZ RPY
radians for flange poses. No hardware connects on import.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation


def eef_actions_to_pose(actions, position_scale):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected finite EEF actions [N,16]: L xyz+xyzw+grip, R xyz+xyzw+grip")
    if not np.isfinite(position_scale) or position_scale <= 0:
        raise ValueError("EEF position scale must be finite and positive")
    parts = []
    for offset in (0, 8):
        quaternions = actions[:, offset + 3 : offset + 7]
        if (np.linalg.norm(quaternions, axis=1) < 1e-8).any():
            raise ValueError("EEF quaternion must be nonzero")
        parts.extend(
            (
                actions[:, offset : offset + 3] / position_scale,
                Rotation.from_quat(quaternions).as_euler("xyz"),
                actions[:, offset + 7 : offset + 8],
            )
        )
    return np.concatenate(parts, axis=1)


def send_eef_action(piper, left, right, left_gripper, right_gripper, control_mode, first):
    """Send flange targets through pyAgxArm's Cartesian point-to-point API."""
    if control_mode != "eef":
        raise ValueError("EEF commands require EEF control mode")
    for pose in (left, right):
        if np.asarray(pose).shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite XYZ+RPY flange target")
    piper.robot_l.move_p(list(left))
    piper.robot_r.move_p(list(right))
    piper.gripper_l.move_gripper_m(float(np.clip(left_gripper, 0, 0.08)), force=1.0)
    piper.gripper_r.move_gripper_m(float(np.clip(right_gripper, 0, 0.08)), force=1.0)


def wait_eef_target(piper, left, right, tolerance, stop_check, timeout=3.0, position_tolerance=0.005):
    """Wait on Cartesian feedback (5 mm and configured angular tolerance).

    Match the joint executor: a motion timeout is logged and returned as a
    warning, while a user stop interrupts the loop.
    """
    targets = [np.asarray(left, dtype=float), np.asarray(right, dtype=float)]
    rotations = [Rotation.from_euler("xyz", pose[3:]) for pose in targets]
    deadline = time.monotonic() + timeout
    errors = {}
    while True:
        if stop_check():
            raise KeyboardInterrupt("Stop requested")
        reached = True
        for name, arm, target, rotation in zip(("left", "right"), (piper.robot_l, piper.robot_r), targets, rotations):
            feedback = arm.get_flange_pose()
            measured = np.asarray(feedback.msg, dtype=float) if feedback is not None else np.array([])
            if measured.shape != (6,) or not np.isfinite(measured).all():
                reached = False
                errors[name] = dict(feedback_available=False)
                continue
            distance = float(np.linalg.norm(target[:3] - measured[:3]))
            angle = float((rotation.inv() * Rotation.from_euler("xyz", measured[3:])).magnitude())
            errors[name] = dict(position_error_mm=distance * 1000, rotation_error_deg=float(np.degrees(angle)))
            reached = reached and distance <= position_tolerance and angle <= tolerance
        if reached or time.monotonic() >= deadline:
            if stop_check():
                raise KeyboardInterrupt("Stop requested")
            if not reached:
                print("[Piper EEF] 到位超时，记录警告并继续：" + str(errors), flush=True)
            return dict(reached=bool(reached), action_space="eef", **errors)
        time.sleep(0.01)
