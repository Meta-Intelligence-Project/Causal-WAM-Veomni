"""Synthetic RoboTwin observations using the native device adapter and episode loop."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tasks.client_server.client.robotwin.simulator import RoboTwinDevice, RobotwinPoseConverter
from tasks.client_server.client.runtime import InferenceClient, run_episode


def main():
    parser = argparse.ArgumentParser(
        description="Run the native RoboTwin episode loop with synthetic observations, without SAPIEN."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--dataset-info", type=Path, help="Optional LeRobot info.json for camera shapes and FPS")
    args = parser.parse_args()
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "wire_requests.jsonl").exists():
        parser.error("Choose a fresh output directory; existing requests must be preserved")
    INFO = args.dataset_info
    info = (
        json.loads(INFO.read_text())
        if INFO
        else {
            "fps": 30,
            "features": {
                "observation.images." + k: {"shape": [480, 640, 3]}
                for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            },
        }
    )
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    converter = RobotwinPoseConverter()
    shapes = {
        k: info["features"]["observation.images." + k]["shape"]
        for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    (ROOT / "mock_config.json").write_text(
        json.dumps(
            dict(
                task="adjust_bottle",
                task_config="demo_clean",
                camera_shape_source=str(INFO) if INFO else "RoboTwin adjust_bottle clean defaults",
                camera_shapes=shapes,
                control_mode="joint",
                state_source="command",
                stream_observations=True,
                control_fps=info["fps"],
                steps=96,
                T_world_base=converter.T_world_base,
            ),
            indent=2,
        )
    )

    class Env:
        def __init__(self):
            self.take_action_cnt = 0
            self.step_lim = 96
            self.eval_success = False
            self.q = np.zeros(14, np.float32)
            self.q[[6, 13]] = 0.5
            self.actions = []
            rng = np.random.default_rng(100000)
            self.images = {k: rng.integers(0, 256, tuple(v), dtype=np.uint8) for k, v in shapes.items()}

        def get_obs(self):
            cams = {
                name: dict(rgb=np.roll(self.images[key], self.take_action_cnt, axis=1))
                for name, key in [
                    ("head_camera", "cam_high"),
                    ("left_camera", "cam_left_wrist"),
                    ("right_camera", "cam_right_wrist"),
                ]
            }
            joint = dict(
                left_arm=self.q[:6].copy(),
                left_gripper=float(self.q[6]),
                right_arm=self.q[7:13].copy(),
                right_gripper=float(self.q[13]),
                vector=self.q.copy(),
            )
            eef = {
                side + "_endpose": converter.to_world(np.array([0.3, sign * 0.2, 0.3, 0, 0, 0, 1], np.float32))
                for side, sign in [("left", 1), ("right", -1)]
            }
            return dict(observation=cams, joint_action=joint, endpose=eef)

        def take_action(self, action, action_type):
            assert action_type == "qpos"
            self.q = np.asarray(action, np.float32).copy()
            assert self.q.shape == (14,) and np.isfinite(self.q).all()
            self.actions.append(self.q.copy())
            self.take_action_cnt += 1

    class Client(InferenceClient):
        def infer(self, req):
            obs = req.get("obs")
            shapes = None
            if isinstance(obs, list):
                shapes = {k: [len(obs), *obs[0][k].shape] for k in obs[0]}
            elif isinstance(obs, dict):
                shapes = {k: list(v.shape) for k, v in obs.items()}
            row = dict(
                kind="reset" if req.get("reset") else "observe" if req.get("observe_only") else "infer",
                observation_step=req.get("observation_step"),
                obs_shapes=shapes,
                state="state" in req,
            )
            result = super().infer(req)
            row.update(
                server_timing=result.get("server_timing"),
                revision=result.get("revision"),
                pending_steps=result.get("pending_steps"),
            )
            with (ROOT / "wire_requests.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            return result

    with Client(args.server_uri, control_mode="joint") as client:
        if client.metadata["execute_steps"] != 32 or client.metadata["observation_chunk_steps"] != 16:
            raise ValueError("This mock workload requires 32 actions/block and 16-step observation groups")
        env = Env()
        device = RoboTwinDevice(env, control_mode="joint", state_source="command", pose_converter=converter)
        result = run_episode(device, client, prompt="Adjust the bottle.", output=None, stream_observations=True)
        np.save(ROOT / "executed_actions.npy", np.stack(env.actions))
        (ROOT / "episode_results.json").write_text(json.dumps(result, indent=2))
        assert result["executed_steps"] == 96 and len(result["requests"]) == 3
        print("MOCK EPISODE PASSED", flush=True)


if __name__ == "__main__":
    main()
