# SPDX-License-Identifier: OpenMDW-1.1
"""Measured-observation RoboTwin policy with session-owned VAE and KV caches."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from veomni.data.causal_cosmos3.readers.observation import InitialObservation
from veomni.data.causal_cosmos3.transforms.layout import camera_layout

from .cosmos3_core.inference_cache import OnlinePolicyKVCache
from .inference import rollout_sample, sample_block, write_video
from .robotwin_adapter import RobotwinAdapter


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def build_adapter(pipe):
    """Resolve exact source masks; group-statistics validity is never a robot mask."""
    config = dict(pipe.config.get("robotwin", {}))
    if ("state_mask" in config) != ("action_mask" in config):
        raise ValueError("robotwin must specify both state_mask and action_mask")
    if "state_mask" not in config:
        statistics = pipe.config["statistics"]
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text())
        sources = statistics.get("provenance", {}).get("sources", [])
        pairs = {(tuple(s["state_mask"]), tuple(s["action_mask"])) for s in sources}
        if len(pairs) != 1:
            raise ValueError("Statistics need identical source masks, or configure robotwin.state_mask/action_mask")
        sm, am = pairs.pop()
        config.update(state_mask=sm, action_mask=am)
    adapter = RobotwinAdapter(pipe.processor.template, **config)
    if pipe.processor.statistics is None:
        raise ValueError("Serving requires measured training statistics")
    for name, mask in (("state", adapter.state_mask), ("action", adapter.action_mask)):
        if (mask & ~pipe.processor.statistics[name][2]).any():
            raise ValueError(f"{name} statistics do not cover the robot mask")
    for key, expected in (
        ("actions_per_block", pipe.geometry.actions_per_block),
        ("video_stride", pipe.geometry.video_stride),
        ("temporal_compression_factor", pipe.geometry.temporal_compression),
    ):
        if pipe.processor.statistics_geometry.get(key, expected) != expected:
            raise ValueError(f"Statistics geometry differs for {key}")
    return adapter


class RobotwinPolicy:
    """One session per connection, with full-block real camera/state feedback."""

    def __init__(self, pipe, *, fps, adapter=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Action/control FPS must be finite and positive")
        if pipe.source.get("viewpoint", "concat_view") != "concat_view":
            raise ValueError("RoboTwin serving requires concat_view")
        self.pipe, self.fps = pipe, float(fps)
        self.adapter = adapter or build_adapter(pipe)
        self.size = pipe.geometry.actions_per_block
        latent_block = self.size // pipe.geometry.actions_per_latent
        self.history_limit = 1 + math.ceil(pipe.history / latent_block) * self.size // pipe.geometry.video_stride
        self.reset()

    @property
    def metadata(self):
        return dict(
            protocol="cosmos-causal-action-v2",
            implementation="veomni-online-policy-kv" if self.pipe.kv_cache else "veomni-measured-rgb-history",
            kv_cache=self.pipe.kv_cache,
            mode="policy",
            action_dim=55,
            actions_per_block=self.size,
            execute_steps=self.size,
            steps=self.pipe.sampling_options["steps"],
            seed=self.pipe.sampling_options["seed"],
            fps=self.fps,
            video_stride=self.pipe.geometry.video_stride,
            video_fps=self.fps / self.pipe.geometry.video_stride,
            video_decode_mode="whole_sequence",
            prediction_video_fps=self.fps / self.pipe.geometry.video_stride,
            camera_keys=list(CAMERA_KEYS),
            history_latents=self.pipe.history,
            **self.adapter.metadata,
        )

    def reset(self):
        self.kv_caches = {}
        self.encoder = self.pipe.vae.stream_encoder() if self.pipe.kv_cache else None
        self.prompt = self.control_mode = self.history = self.camera_shapes = None
        self.revision = self.observation_step = 0
        self.needs_reset = self.record_video = False
        self.video_latents = []

    def _sample_cached(self, initial, state, pixels):
        """Encode only new measured pixels and reuse all branches across requests."""
        sample, _ = rollout_sample(
            initial,
            state,
            pixels[:, :1],
            source=self.pipe.source,
            processor=self.pipe.processor,
            geometry=self.pipe.geometry,
            tokenize=self.pipe.tokenize,
            history_blocks=self.pipe.history,
            valid=self.size,
        )
        observed = self.encoder.encode(pixels).to(self.pipe.device)
        block = self.size // self.pipe.geometry.actions_per_latent
        expected = block if self.revision else 1
        if observed.shape[1] != expected:
            raise ValueError(f"Expected {expected} measured latent frames, got {observed.shape[1]}")
        video = observed.new_zeros((observed.shape[0], 1 + block, *observed.shape[2:]))
        if not self.revision:
            video[:, :1] = observed
        encoded = {key: value for key, value in sample.items() if key != "video"}
        encoded["video_latents"] = video
        for key in ("action", "states", "action_mask", "state_mask"):
            encoded[key] = encoded[key].to(self.pipe.device)
        options = self.pipe.sampling_options
        branches = {"cond": None}
        if options["guidance"] != 1:
            branches["uncond"] = options["negative_text_ids"]
        for name, text in branches.items():
            if not self.revision:
                self.kv_caches[name] = OnlinePolicyKVCache(
                    self.pipe.model,
                    encoded,
                    attention_backend=options["attention_backend"],
                    position_options=options["position_options"],
                    text_ids=text,
                )
            else:
                self.kv_caches[name].update(encoded, observed, revision=self.revision)
        generated = sample_block(
            self.pipe.model, encoded, 0, seed_block=self.revision, kv_cache=self.kv_caches, **options
        )
        return sample, encoded, generated

    def _cameras(self, obs, count):
        if isinstance(obs, dict):
            cameras = {key: np.asarray(obs[key]) for key in CAMERA_KEYS}
            cameras = {key: value[None] if value.ndim == 3 else value for key, value in cameras.items()}
        elif isinstance(obs, list) and len(obs) == count and all(isinstance(frame, dict) for frame in obs):
            cameras = {key: np.stack([frame[key] for frame in obs]) for key in CAMERA_KEYS}
        else:
            raise ValueError(f"obs must contain {count} RGB frames for each camera")
        for key, value in cameras.items():
            if (
                value.dtype != np.uint8
                or value.ndim != 4
                or value.shape[0] != count
                or value.shape[-1] != 3
                or min(value.shape[1:3]) < 2
            ):
                raise ValueError(f"{key} must be RGB uint8 [{count},H,W,3]")
        shapes = {key: value.shape[1:] for key, value in cameras.items()}
        if self.camera_shapes is not None and shapes != self.camera_shapes:
            raise ValueError("Camera shapes changed; reset before changing camera geometry")
        return cameras, shapes

    def _compose(self, cameras):
        """Match the training bucket/rectangles and PyAV BICUBIC camera resize."""
        import av
        from av.video.reformatter import Interpolation

        head = cameras[CAMERA_KEYS[0]]
        roles = dict(zip(("head", "left", "right"), CAMERA_KEYS, strict=True))
        info = dict(features={CAMERA_KEYS[0]: dict(shape=head.shape[1:], names=["height", "width", "channels"])})
        layout = camera_layout(info, roles, self.pipe.source.get("resolution", "256"), "concat_view")
        height = max(c.top + c.height for c in layout)
        width = max(c.left + c.width for c in layout)
        result = np.zeros((3, len(head), height, width), dtype=np.uint8)
        for camera in layout:
            for index, pixels in enumerate(cameras[camera.key]):
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
                resized = frame.reformat(
                    height=camera.height, width=camera.width, format="rgb24", interpolation=Interpolation.BICUBIC
                ).to_ndarray()
                result[:, index, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width] = (
                    resized.transpose(2, 0, 1)
                )
        return torch.from_numpy(result)

    def finish_video(self):
        if not self.record_video or len(self.video_latents) < 2:
            raise ValueError("No prediction video recorded; reset with record_video=True before inference")
        pixels = self.pipe.vae.decode(torch.cat(self.video_latents, dim=1))
        fps = self.fps / self.pipe.geometry.video_stride
        with TemporaryDirectory(prefix="veomni-robotwin-") as directory:
            path = Path(directory) / "prediction.mp4"
            write_video(path, pixels.cpu(), fps)
            result = dict(video_mp4=path.read_bytes(), video_frames=pixels.shape[1], video_fps=fps)
        self.video_latents.clear()
        self.record_video = False
        return result

    @torch.no_grad()
    def infer(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        if "state_mask" in request or "action_mask" in request:
            raise ValueError("Masks are configured on the server")
        if request.get("reset", False):
            self.reset()
            self.prompt = request.get("prompt")
            self.control_mode = request.get("control_mode")
            self.record_video = bool(request.get("record_video", False))
            if "obs" not in request:
                return dict(reset=True, observation_step=0)
        if self.needs_reset:
            raise ValueError("Previous inference failed; send reset before continuing")
        if request.get("finish_video", False):
            return self.finish_video()
        prompt = request.get("prompt", self.prompt)
        mode = request.get("control_mode", self.control_mode)
        if mode is not None and mode not in self.adapter.available_control_modes:
            raise ValueError(f"Unsupported control_mode={mode!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        if self.revision and (prompt != self.prompt or mode != self.control_mode):
            raise ValueError("Changing prompt/control_mode requires reset")
        expected = self.observation_step + (self.size if self.revision else 0)
        step = request.get("observation_step")
        if type(step) is not int or step != expected:
            raise ValueError(f"Expected observation_step={expected}; duplicate/out-of-order feedback")
        if self.revision and (
            type(request.get("executed_steps")) is not int or request["executed_steps"] != self.size
        ):
            raise ValueError(f"Feedback requires exactly {self.size} executed actions")
        cameras, shapes = self._cameras(request.get("obs"), self.size if self.revision else 1)
        state = torch.from_numpy(self.adapter.assemble_state(request.get("state")))
        state = self.pipe.processor.template.sanitize(state, self.adapter.state_mask)
        # Select the initial frame, then measured endpoints stride, 2*stride, ...
        if self.revision:
            cameras = {
                key: value[self.pipe.geometry.video_stride - 1 :: self.pipe.geometry.video_stride]
                for key, value in cameras.items()
            }
        pixels = self._compose(cameras)
        history = pixels if self.history is None else torch.cat((self.history, pixels), dim=1)
        history = history[:, -self.history_limit :].contiguous()
        initial = InitialObservation(
            history[:, :1],
            state,
            self.adapter.state_mask,
            self.adapter.action_mask,
            prompt,
            self.fps,
            self.size,
            0.0,
            {},
        )
        try:
            if self.pipe.kv_cache:
                sample, encoded, generated = self._sample_cached(initial, state, pixels)
                history = pixels  # Retain only this request's RGB, never an episode-long buffer.
            else:
                sample, context = rollout_sample(
                    initial,
                    state,
                    history,
                    source=self.pipe.source,
                    processor=self.pipe.processor,
                    geometry=self.pipe.geometry,
                    tokenize=self.pipe.tokenize,
                    history_blocks=self.pipe.history,
                    valid=self.size,
                )
                encoded = self.pipe.vae.encode_pack([sample])[0]
                for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
                    encoded[key] = encoded[key].to(self.pipe.device)
                generated = sample_block(
                    self.pipe.model, encoded, context, seed_block=self.revision, **self.pipe.sampling_options
                )
            action = self.pipe.processor.decode(
                generated["action"][-self.size :].cpu(), state.expand(self.size, -1), self.adapter.action_mask
            ).numpy()
            split = self.adapter.split_actions(action)
            if self.record_video:
                latent = generated["video_latents"].cpu()
                if not self.video_latents:
                    self.video_latents.append(encoded["video_latents"][:, :1].cpu().clone())
                keep = self.size // self.pipe.geometry.actions_per_latent
                self.video_latents.append(latent[:, -keep:].clone())
        except Exception:
            self.reset()
            self.needs_reset = True
            raise
        self.history, self.camera_shapes = history, shapes
        self.prompt, self.control_mode, self.observation_step = prompt, mode, step
        self.revision += 1
        response = dict(
            action=split,
            available_control_modes=self.adapter.available_control_modes,
            execute_steps=self.size,
            observation_step=step,
            revision=self.revision,
        )
        if self.revision == 1:
            response["structured_prompt"] = json.loads(sample["prompt"])
        return response
