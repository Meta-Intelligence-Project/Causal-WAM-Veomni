# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Policy transport, device-independent episode execution and recording; no VeOmni/Torch dependency."""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from tasks.client_server.wire import packb, unpackb


class H264VideoWriter:
    """Stream RGB frames to FFmpeg's libx264 encoder without buffering a video."""

    def __init__(self, path, *, fps):
        from pathlib import Path

        import numpy as np

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        self.path = Path(path) if path is not None else None
        self.fps = float(fps)
        self.frames = 0
        self.writer = self.errors = self.shape = None

    def __enter__(self):
        return self

    def append(self, frame):
        if self.path is None:
            return
        import subprocess
        import tempfile

        import numpy as np

        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 2:
            raise ValueError("Video recording requires uint8 RGB [H,W,3] frames")
        height, width = frame.shape[:2]
        frame = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.shape = frame.shape
            self.errors = tempfile.TemporaryFile()
            try:
                self.writer = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{frame.shape[1]}x{frame.shape[0]}",
                        "-r",
                        str(self.fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "yuv420p",
                        "-tag:v",
                        "avc1",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=self.errors,
                )
            except OSError as error:
                self.errors.close()
                raise RuntimeError("H.264 video saving requires ffmpeg with the libx264 encoder on PATH") from error
        if frame.shape != self.shape:
            raise ValueError("Frame dimensions changed during video recording")
        try:
            self.writer.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.close()
            raise RuntimeError(f"FFmpeg stopped accepting frames: {self.path}") from None
        self.frames += 1

    def close(self):
        if self.writer is None:
            return
        writer, self.writer = self.writer, None
        try:
            writer.stdin.close()
        except BrokenPipeError:
            pass
        status = writer.wait()
        self.errors.seek(0)
        detail = self.errors.read().decode(errors="replace").strip()
        self.errors.close()
        if status:
            raise RuntimeError(f"H.264 encoding failed for {self.path}: {detail}")

    def __exit__(self, *args):
        self.close()


def save_structured_prompt(response, path):
    """Save the server's full prompt when included in the first inference response."""
    if "structured_prompt" not in response:
        return
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response["structured_prompt"], ensure_ascii=False, indent=2))


def save_prediction_video(response, path):
    """Save one server RGB preview as H.264/AVC in an MP4 container."""
    if "video" not in response:
        return {}  # Allow clients to keep working with earlier v2 servers.
    import numpy as np

    frames = response["video"]
    fps = response["video_fps"]
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[0] < 1
        or frames.shape[-1] != 3
        or min(frames.shape[1:3]) < 2
    ):
        raise ValueError("Server video must be RGB uint8 [T,H,W,3]")
    with H264VideoWriter(path, fps=fps) as writer:
        for frame in frames:
            writer.append(frame)
    return dict(prediction_video_path=str(path), prediction_video_frames=len(frames), prediction_video_fps=float(fps))


def append_prediction_video(writer, response, *, action_steps, video_stride):
    """Join the executed horizon of each prediction without overlapping time."""
    if "video" not in response or writer.path is None:
        return
    if response["video_fps"] != writer.fps:
        raise ValueError("Prediction video FPS differs from connection metadata")
    for frame in response["video"][: action_steps // video_stride]:
        writer.append(frame)


class CausalPolicyClient:
    def __init__(self, uri="ws://localhost:8000", *, control_mode="joint"):
        from websockets.sync.client import connect

        self.connection = connect(uri, compression=None, max_size=None, ping_interval=None, close_timeout=5)
        self.metadata = unpackb(self.connection.recv())
        if self.metadata.get("protocol") != "cosmos-causal-action-v2":
            self.close()
            raise ValueError("Expected a cosmos-causal-action-v2 server")
        if self.metadata.get("quaternion_order") != "xyzw" or self.metadata.get("eef_pose_frame") != "dataset":
            self.close()
            raise ValueError("Expected dataset-frame xyz+xyzw EEF poses; update the server and clients together")
        if control_mode not in self.metadata["available_control_modes"]:
            self.close()
            raise ValueError(f"Server does not support control_mode={control_mode!r}")
        self.control_mode = control_mode

    def infer(self, observation):
        self.connection.send(packb(observation))
        response = unpackb(self.connection.recv())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def reset(self, *, prompt, record_video=False):
        return self.infer(dict(reset=True, prompt=prompt, control_mode=self.control_mode, record_video=record_video))

    def save_prediction_video(self, path):
        """Request one whole-sequence VAE decode and save the server-encoded MP4."""
        from pathlib import Path

        result = self.infer(dict(finish_video=True))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result["video_mp4"])
        return dict(
            prediction_video_path=str(path),
            prediction_video_frames=result["video_frames"],
            prediction_video_fps=result["video_fps"],
        )

    def actions(self, response):
        """Select packed joint or dataset-frame xyz+xyzw EEF actions."""
        actions = response["action"][self.control_mode]
        if actions is None:
            raise ValueError(f"Server returned no valid {self.control_mode} actions")
        return actions

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PolicyDevice(Protocol):
    """Device boundary for protocol-format observations and absolute action targets."""

    def observe(self) -> tuple[dict, dict]:
        """Return copied RGB images and a state in the server's protocol coordinates."""
        ...

    def execute(self, action: np.ndarray) -> None:
        """Convert and execute one absolute joint or dataset-frame EEF action."""
        ...

    def is_done(self) -> bool:
        """Return whether execution should stop, including success or a step limit."""
        ...

    def is_success(self) -> bool:
        """Return the device/task's success result for episode reporting."""
        ...


class VideoRecorder(H264VideoWriter):
    """Stream copied RGB observations to an MP4 without retaining an episode in RAM."""

    def __init__(self, output, *, fps=30):
        super().__init__(Path(output) / "video.mp4" if output is not None else None, fps=fps)

    def append(self, images):
        if self.path is None:
            return
        import cv2

        head, left, right = (images[name] for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"))
        for image in (head, left, right):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 2:
                raise ValueError("Video recording requires uint8 RGB camera images")
        # Match VideoViewConfig's concat_view geometry without importing Torch.
        height, width = head.shape[:2]
        left = cv2.resize(left, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (width - width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        frame = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        super().append(frame)


@contextmanager
def record_actions(path):
    """Save complete prediction blocks in request order, including on rollout errors."""
    blocks = []
    try:
        yield blocks
    finally:
        if path is not None and blocks:
            np.save(path, np.concatenate(blocks, axis=0))


def validate_action_chunk(client, response):
    """Validate a prediction against protocol metadata and return actions plus execution count."""
    actions = np.asarray(client.actions(response), dtype=np.float32)
    count = response["execute_steps"]
    if type(count) is not int or count < 1 or count != client.metadata["execute_steps"]:
        raise ValueError("Server execute_steps differs from connection metadata")
    width = (
        client.metadata["left_arm_dof"] + client.metadata["right_arm_dof"] + 2
        if client.control_mode == "joint"
        else 16
    )
    if (
        actions.shape != (client.metadata["actions_per_block"], width)
        or len(actions) < count
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Server returned an invalid native action chunk")
    return actions, count


def detect_eef_low_motion(actions, *, position_threshold_m=0.01, rotation_threshold_deg=3.0, gripper_threshold=0.1):
    """Return per-arm deviations when both arms stay below all three thresholds."""
    poses = actions.astype(np.float64).reshape(-1, 2, 8)
    position = np.linalg.norm(poses[..., :3] - poses[0, :, :3], axis=-1).max(axis=0)
    quaternion = poses[..., 3:7]
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = np.divide(quaternion, norms, out=np.zeros_like(quaternion), where=norms > 0)
    cosine = np.abs((quaternion * quaternion[0]).sum(axis=-1))
    rotation = np.rad2deg(2 * np.arccos(np.clip(cosine, 0, 1))).max(axis=0)
    gripper = np.abs(poses[..., 7] - poses[0, :, 7]).max(axis=0)
    if (
        (position < position_threshold_m).all()
        and (rotation < rotation_threshold_deg).all()
        and (gripper < gripper_threshold).all()
    ):
        return dict(
            max_position_deviation_m=position.tolist(),
            max_rotation_deviation_deg=rotation.tolist(),
            max_gripper_deviation=gripper.tolist(),
        )
    return None


def run_episode(device: PolicyDevice, client, *, prompt, output=None, video_fps=30, enable_active_failure=False):
    """One episode: initial frame, then exactly one frame per executed target.

    Predicted rows after execute_steps are discarded. If the episode finishes
    during a chunk, no partial feedback or padded observations are submitted.
    """
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executed = 0
    requests = []
    active_failure = None
    whole_video = client.metadata.get("video_decode_mode") == "whole_sequence"
    if whole_video:
        client.reset(prompt=prompt, record_video=output is not None)
    else:
        client.reset(prompt=prompt)

    video_stride = client.metadata.get("video_stride", 4)
    prediction_fps = client.metadata.get("video_fps", client.metadata.get("fps", 30) / video_stride)
    action_path = output / f"action_{client.control_mode}.npy" if output is not None else None
    with (
        record_actions(action_path) as action_blocks,
        VideoRecorder(output, fps=video_fps) as video,
        H264VideoWriter(
            output / "prediction.mp4" if output is not None and not whole_video else None, fps=prediction_fps
        ) as prediction,
    ):
        if not device.is_done():
            images, state = device.observe()
            video.append(images)
            request = dict(
                obs=images,
                state=state,
                observation_step=0,
            )
            while True:
                request_started = time.perf_counter()
                response = client.infer(request)
                client_round_trip_ms = (time.perf_counter() - request_started) * 1000
                if output is not None:
                    save_structured_prompt(response, output / "structured_prompt.json")
                actions, count = validate_action_chunk(client, response)
                requests.append(
                    {
                        "observation_step": executed,
                        **response.get("server_timing", {}),
                        "client_round_trip_ms": client_round_trip_ms,
                    }
                )
                if output is not None:
                    action_blocks.append(actions.copy())
                    requests[-1].update(save_prediction_video(response, output / f"step{executed:06d}_prediction.mp4"))
                if client.control_mode == "eef" and enable_active_failure:
                    failure = detect_eef_low_motion(actions)
                    if failure is not None:
                        active_failure = dict(observation_step=executed, **failure)
                        break

                observations = []
                block_start = executed
                try:
                    for action in actions[:count]:
                        device.execute(action)
                        images, state = device.observe()
                        video.append(images)
                        observations.append(images)
                        executed += 1
                        if device.is_done():
                            break
                finally:
                    append_prediction_video(
                        prediction, response, action_steps=executed - block_start, video_stride=video_stride
                    )
                if device.is_done():
                    break
                request = dict(
                    obs=observations,
                    state=state,
                    executed_steps=count,
                    observation_step=executed,
                )

    prediction_metrics = dict(
        prediction_video_path=str(prediction.path) if prediction.frames else None,
        prediction_video_frames=prediction.frames,
        prediction_video_fps=prediction.fps,
    )
    if whole_video and output is not None and requests:
        prediction_metrics = client.save_prediction_video(output / "prediction.mp4")
    seconds = time.perf_counter() - started
    requests_path = output / "requests.json" if output is not None else None
    if requests_path is not None:
        requests_path.write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    return dict(
        success=bool(device.is_success()) and active_failure is None,
        active_failure=active_failure is not None,
        failure_reason="eef_low_motion" if active_failure is not None else None,
        failure_details=active_failure,
        executed_steps=executed,
        seconds=seconds,
        requests=requests,
        requests_path=str(requests_path) if requests_path is not None else None,
        action_path=str(action_path) if action_path is not None and action_blocks else None,
        video_path=str(video.path) if video.path is not None else None,
        video_frames=video.frames,
        video_fps=video_fps,
        **prediction_metrics,
    )
