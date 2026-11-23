# SPDX-License-Identifier: OpenMDW-1.1
"""Causal Cosmos3 Policy/IDM inference session with unified55 actions and session-owned VAE/KV caches."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tasks.common.causal_cosmos3_args import parse_async_infer
from veomni.data.causal_cosmos3.readers.observation import InitialObservation
from veomni.data.causal_cosmos3.transforms.layout import camera_layout
from veomni.models.diffusers.causal_cosmos3.inference import (
    ObservationSequence,
    compact_prediction_sample,
    create_observation_caches,
    rollout_sample,
    sample_block,
    sample_rollout_block,
    sample_window_block,
    validate_rollout_task,
    write_video,
)
from veomni.utils.device import is_nvidia_cuda_device

from .adapter import Unified55Adapter


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def build_adapter(pipe):
    """Resolve exact source masks; group-statistics validity is not an execution mask."""
    config = dict(pipe.config.get("adapter", {}))
    if ("state_mask" in config) != ("action_mask" in config):
        raise ValueError("adapter must specify both state_mask and action_mask")
    if "state_mask" not in config:
        statistics = pipe.config["statistics"]
        if isinstance(statistics, (str, Path)):
            statistics = json.loads(Path(statistics).read_text())
        sources = statistics.get("provenance", {}).get("sources", [])
        pairs = {(tuple(s["state_mask"]), tuple(s["action_mask"])) for s in sources}
        if len(pairs) != 1:
            raise ValueError("Statistics need identical source masks, or configure adapter.state_mask/action_mask")
        sm, am = pairs.pop()
        config.update(state_mask=sm, action_mask=am)
    adapter = Unified55Adapter(pipe.processor.template, **config)
    if pipe.processor.statistics is None:
        raise ValueError("Serving requires measured training statistics")
    for name, mask in (("state", adapter.state_mask), ("action", adapter.action_mask)):
        if (mask & ~pipe.processor.statistics[name][2]).any():
            raise ValueError(f"{name} statistics do not cover the adapter mask")
    for key, expected in (
        ("actions_per_block", pipe.geometry.actions_per_block),
        ("video_stride", pipe.geometry.video_stride),
        ("temporal_compression_factor", pipe.geometry.temporal_compression),
    ):
        if pipe.processor.statistics_geometry.get(key, expected) != expected:
            raise ValueError(f"Statistics geometry differs for {key}")
    return adapter


class CausalInferenceSession:
    """One session per connection, with full-block real camera/state feedback."""

    def __init__(self, pipe, *, fps, adapter=None, task="policy"):
        async_infer = parse_async_infer(pipe.config.get("async_infer", {}))
        self.rtc_enabled, self.rtc_delay = async_infer["rtc_enabled"], async_infer["rtc_delay"]
        validate_rollout_task(task)
        if pipe.sampling_options.get("npu_graph", False) and task != "policy":
            raise ValueError("NPU graph currently supports Policy sessions only")
        self.task = task
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Action/control FPS must be finite and positive")
        if pipe.source.get("viewpoint", "concat_view") != "concat_view":
            raise ValueError("Online inference serving requires concat_view")
        self.pipe, self.fps = pipe, float(fps)
        self.adapter = adapter or build_adapter(pipe)
        self.size = pipe.geometry.actions_per_block
        if self.rtc_enabled and self.rtc_delay > self.size // 2:
            raise ValueError("async_infer.rtc_delay must not exceed actions_per_block / 2")
        latent_block = self.size // pipe.geometry.actions_per_latent
        self.history_limit = 1 + math.ceil(pipe.history / latent_block) * self.size // pipe.geometry.video_stride
        self.prediction_recordings = []
        self.reset()
        if self.rtc_enabled and not self.metadata["rtc_async"]:
            raise ValueError(
                "async_infer.rtc_enabled requires a Policy streaming-KV session on CPU/CUDA, "
                "a latent-aligned half-horizon, grouped/dense attention and no NPU graph"
            )

    @property
    def metadata(self):
        return dict(
            protocol="cosmos-causal-action-v2",
            implementation="veomni-observation-kv" if self.pipe.kv_cache else "veomni-full-pack",
            observation_mode=self.pipe.observation_mode,
            observation_streaming=self.observations is not None,
            observation_chunk_steps=self.pipe.geometry.actions_per_latent,
            kv_cache=self.pipe.kv_cache,
            npu_graph=self.pipe.sampling_options.get("npu_graph", False),
            async_infer=dict(rtc_enabled=self.rtc_enabled, rtc_delay=self.rtc_delay),
            rtc_async=self.task == "policy"
            and self.pipe.kv_cache
            and self.pipe.observation_mode == "streaming"
            and self.size % (2 * self.pipe.geometry.actions_per_latent) == 0
            and (self.pipe.device.type == "cpu" or is_nvidia_cuda_device(self.pipe.device))
            and not self.pipe.sampling_options.get("npu_graph", False)
            and self.pipe.sampling_options.get("attention_backend", "grouped") in ("grouped", "dense"),
            mode=self.task,
            action_dim=55,
            actions_per_block=self.size,
            execute_steps=self.size,
            steps=self.pipe.sampling_options["steps"],
            seed=self.pipe.sampling_options["seed"],
            fps=self.fps,
            video_stride=self.pipe.geometry.video_stride,
            video_fps=self.fps / self.pipe.geometry.video_stride,
            video_decode_mode="whole_sequence",
            rtc_prediction_video="per_request_on_disconnect",
            rtc_history="single_measured_timeline" if self.rtc_enabled else None,
            prediction_video_fps=self.fps / self.pipe.geometry.video_stride,
            camera_keys=list(CAMERA_KEYS),
            history_latents=self.pipe.history,
            **self.adapter.metadata,
        )

    def reset(self):
        self.observations = ObservationSequence(self.pipe) if self.pipe.observation_mode == "streaming" else None
        self.kv_caches = self.observations.caches if self.observations is not None else {}
        self.prompt = self.control_mode = self.history = self.camera_shapes = None
        self.revision = self.observation_step = 0
        self.needs_reset = self.record_video = False
        self.video_latents = []
        self.prediction_recording = None
        self.pending_frame = None
        self.pending_latents = []
        self.pending_steps = 0
        self.rtc_caches = {}

    def _observe(self, request):
        """Preencode arrived camera groups while the client executes the current block."""
        if self.observations is None or not self.revision:
            raise ValueError("Observation streaming requires streaming mode and an initial inference")
        step = request.get("observation_step")
        previous = self.observation_step + self.pending_steps
        group = self.pipe.geometry.actions_per_latent
        if type(step) is not int or not previous < step <= self.observation_step + self.size:
            raise ValueError("Observations must advance within the current action block")
        count = step - previous
        if count % group:
            raise ValueError(f"Observations require complete groups of {group} control steps")
        cameras, shapes = self._observation_cameras(request, count)
        pixels = self._compose(cameras)
        try:
            latent = self.observations.encoder.encode(pixels)
        except Exception:
            self.reset()
            self.needs_reset = True
            raise
        if self.pending_frame is None:
            # Own only one frame for sample geometry; a view would retain the whole RGB group.
            self.pending_frame = pixels[:, :1].clone()
        self.pending_latents.append(latent)
        self.pending_steps += count
        self.camera_shapes = shapes
        return dict(observed=True, observation_step=step, pending_steps=self.pending_steps)

    def _observation_cameras(self, request, count, *, initial=False):
        """Accept full-rate feedback or explicitly sampled control-step endpoints."""
        stride = request.get("observation_stride", 1)
        video_stride = self.pipe.geometry.video_stride
        if type(stride) is not int or stride not in (1, video_stride) or (initial and stride != 1):
            raise ValueError("observation_stride must be 1 for the initial frame, otherwise 1 or video_stride")
        if count % stride:
            raise ValueError("Observations must contain complete observation_stride groups")
        cameras, shapes = self._cameras(request.get("obs"), count // stride)
        if not initial:
            remaining_stride = video_stride // stride
            cameras = {key: value[remaining_stride - 1 :: remaining_stride] for key, value in cameras.items()}
        return cameras, shapes

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
        with TemporaryDirectory(prefix="veomni-inference-") as directory:
            path = Path(directory) / "prediction.mp4"
            write_video(path, pixels.cpu(), fps)
            result = dict(video_mp4=path.read_bytes(), video_frames=pixels.shape[1], video_fps=fps)
        self.video_latents.clear()
        self.record_video = False
        return result

    def finish_prediction_recordings(self):
        """Called under the server reservation after the robot client disconnects."""
        try:
            for recording in self.prediction_recordings:
                recording.finish(self.pipe.vae)
        finally:
            self.prediction_recordings.clear()

    def _rtc(self, request, prompt, mode):
        """Append each measured half-block to one timeline rooted at the initial observation."""
        if (
            not self.metadata["rtc_async"]
            or mode != "joint"
            or self.pending_steps
            or request.get("observe_only", False)
        ):
            raise ValueError(
                "RTC requires a fresh joint Policy streaming-KV session on CPU/CUDA and a latent-aligned half-horizon"
            )
        step = request.get("observation_step")
        stride = self.size // 2
        if type(step) is not int or step != (self.observation_step + stride if self.revision else 0):
            raise ValueError("Out-of-order RTC decision")
        cameras, shapes = self._observation_cameras(request, stride if self.revision else 1, initial=not self.revision)
        pixels = self._compose(cameras)
        window = pixels[:, -1:]
        state = torch.from_numpy(self.adapter.assemble_state(request["state"]))
        state = self.pipe.processor.template.sanitize(state, self.adapter.state_mask)
        mask = self.adapter.action_mask
        initial = InitialObservation(
            window[:, :1], state, self.adapter.state_mask, mask, prompt, self.fps, self.size, 0.0, {}
        )
        sample, block = rollout_sample(
            initial,
            state,
            window,
            source=self.pipe.source,
            processor=self.pipe.processor,
            geometry=self.pipe.geometry,
            tokenize=self.pipe.tokenize,
            history_blocks=self.pipe.history,
            valid=self.size,
        )
        # VAE and Transformer history share the same physical origin. Append
        # only new real latents; predictions never enter observation history.
        measured = self.observations.encoder.encode(pixels).to(self.pipe.device)
        expected_latents = stride // self.pipe.geometry.actions_per_latent if self.revision else 1
        if measured.shape[1] != expected_latents:
            raise ValueError("RTC VAE output does not match the measured control interval")
        latent_block = self.size // self.pipe.geometry.actions_per_latent
        revision = self.revision
        if not self.rtc_caches:
            self.rtc_caches = create_observation_caches(
                self.pipe,
                measured,
                text_ids=sample["text_token_ids"],
                conditioning_fps=sample["conditioning_fps"],
                conditioning_fps_action=sample["conditioning_fps_action"],
            )
        else:
            for cache in self.rtc_caches.values():
                cache.update(measured, revision=revision, half_block=True)
        encoded = compact_prediction_sample(
            sample, measured[:, -1:], 0, actions_per_block=self.size, patch_size=self.pipe.model.config.patch_size
        )
        encoded["cache_revision"] = revision
        options = self.pipe.sampling_options
        generated = sample_block(
            self.pipe.model,
            encoded,
            block,
            seed_block=self.revision,
            kv_cache=self.rtc_caches,
            **options,
        )
        action = self.pipe.processor.decode(generated["action"][-self.size :].cpu(), state.expand(self.size, -1), mask)
        split = self.adapter.split_actions(action.numpy())
        if self.prediction_recording is not None:
            self.prediction_recording.capture(
                step=step,
                latents=torch.cat((measured[:, -1:], generated["video_latents"][:, -latent_block:]), dim=1),
                pixels=pixels,
                state=state,
                actions=split,
            )
        self.camera_shapes = shapes
        self.prompt, self.control_mode, self.observation_step = prompt, mode, step
        self.revision += 1
        response = dict(
            action=split,
            execute_steps=self.size,
            observation_step=step,
            revision=self.revision,
            available_control_modes=self.adapter.available_control_modes,
        )

        if self.revision == 1:
            response["structured_prompt"] = json.loads(sample["prompt"])
        return response

    @torch.no_grad()
    def infer(self, request):
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        if "state_mask" in request or "action_mask" in request:
            raise ValueError("Masks are configured on the server")
        if {"async_infer", "rtc_enabled", "rtc_delay", "rtc_stride", "rtc_mode"} & request.keys():
            raise ValueError("RTC is configured by the server YAML async_infer; client overrides are not supported")
        if request.get("reset", False):
            self.reset()
            self.prompt = request.get("prompt")
            self.control_mode = request.get("control_mode")
            self.record_video = bool(request.get("record_video", False))
            if self.record_video and self.rtc_enabled:
                from .prediction_video import PredictionRecording

                self.prediction_recording = PredictionRecording(
                    self.pipe.config.get("prediction_debug_root", "outputs/predictions"),
                    request.get("prediction_run_id"),
                    fps=self.fps,
                    video_stride=self.pipe.geometry.video_stride,
                    horizon=self.size,
                    delay=self.rtc_delay,
                    prompt=self.prompt,
                )
                self.prediction_recordings.append(self.prediction_recording)
            if "obs" not in request:
                response = dict(reset=True, observation_step=0)
                if self.prediction_recording is not None:
                    response["prediction_debug_dir"] = str(self.prediction_recording.directory)
                return response
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
        if self.rtc_enabled:
            try:
                return self._rtc(request, prompt, mode)
            except Exception:
                self.reset()
                self.needs_reset = True
                raise
        if request.get("observe_only", False):
            return self._observe(request)
        expected = self.observation_step + (self.size if self.revision else 0)
        step = request.get("observation_step")
        if type(step) is not int or step != expected:
            raise ValueError(f"Expected observation_step={expected}; duplicate/out-of-order feedback")
        if self.revision and (
            type(request.get("executed_steps")) is not int or request["executed_steps"] != self.size
        ):
            raise ValueError(f"Feedback requires exactly {self.size} executed actions")
        if self.pending_steps:
            if "obs" in request:
                raise ValueError("Do not resend obs after streaming observations")
            if self.pending_steps != self.size:
                raise ValueError("Complete the streamed observation block before inference")
            shapes = self.camera_shapes
            pixels = self.pending_frame
        else:
            cameras, shapes = self._observation_cameras(
                request, self.size if self.revision else 1, initial=not self.revision
            )
            pixels = self._compose(cameras)
        state = torch.from_numpy(self.adapter.assemble_state(request.get("state")))
        state = self.pipe.processor.template.sanitize(state, self.adapter.state_mask)
        # Streaming keeps history in VAE/KV; one RGB frame suffices to build the sample.
        history = pixels[:, :1]
        if self.observations is None:
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
                task=self.task,
            )
            if self.observations is not None:
                observed = torch.cat(self.pending_latents, dim=1) if self.pending_latents else None
                encoded, block = self.observations.prepare(sample, pixels, self.revision, observed=observed)
                generated = sample_rollout_block(
                    self.pipe,
                    encoded,
                    block,
                    revision=self.revision,
                    task=self.task,
                    kv_cache=self.kv_caches if self.pipe.kv_cache else None,
                )
            else:
                encoded = self.pipe.vae.encode_pack([sample])[0]
                for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
                    encoded[key] = encoded[key].to(self.pipe.device)
                generated = sample_window_block(
                    self.pipe, encoded, context, revision=self.revision, rollout_task=self.task
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
        self.history = history if self.observations is None else None
        self.camera_shapes = shapes
        self.prompt, self.control_mode, self.observation_step = prompt, mode, step
        self.revision += 1
        self.pending_frame = None
        self.pending_latents.clear()
        self.pending_steps = 0
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


CausalPolicy = CausalInferenceSession
