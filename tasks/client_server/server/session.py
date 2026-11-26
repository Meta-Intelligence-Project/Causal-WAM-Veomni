# SPDX-License-Identifier: OpenMDW-1.1
"""Causal Cosmos3 online inference session for Triple-Expert IDM and Legacy Policy."""

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from veomni.data.causal_cosmos3.readers.observation import InitialObservation
from veomni.data.causal_cosmos3.transforms.layout import camera_layout
from veomni.models.diffusers.causal_cosmos3.inference import (
    _triple_rollout_sample,
    rollout_sample,
    sample_block,
    write_video,
)

from .adapter import Unified55Adapter


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def validate_serving_task(task, *, is_triple):
    if task not in ("policy", "inverse_dynamics"):
        raise ValueError("Online serving supports policy or inverse_dynamics")
    if is_triple and task != "inverse_dynamics":
        raise ValueError("Triple-expert online serving supports inverse_dynamics; policy is disabled")
    if not is_triple and task != "policy":
        raise ValueError("Legacy dual-expert online serving supports policy")


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
    """One session per connection, with full-block real camera/state feedback.

    In Triple-Expert IDM mode, each request updates history strictly from real
    camera observations (via StreamingVideoEncoder in streaming mode, or bounded
    window VAE encoding in window mode). Within each block:
      1. Video denoises 1.0 -> stop_sigma (e.g. 0.9) and extracts video_cache;
      2. Action Expert denoises 20 UniPC steps to predict the 32 block actions;
      3. When record_video=True (enabled automatically by InferenceClient under
         video_decode_mode='whole_sequence'), the same 0.9 video latent continues
         denoising 0.9 -> 0.0 so finish_video can decode prediction.mp4 once.
    Predicted video latents are NEVER fed back into self.history_latents.
    """

    def __init__(self, pipe, *, fps, adapter=None, task=None):
        self.is_triple = getattr(pipe.model.config, "expert_layout", "dual") == "triple"
        resolved_task = task or pipe.config.get("task") or ("inverse_dynamics" if self.is_triple else "policy")
        validate_serving_task(resolved_task, is_triple=self.is_triple)
        self.task = resolved_task
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Action/control FPS must be finite and positive")
        if pipe.source.get("viewpoint", "concat_view") != "concat_view":
            raise ValueError("Online inference serving requires concat_view")
        options = pipe.config.get("sampling", {})
        observation_mode = options.get("observation_mode", "streaming")
        if observation_mode not in ("streaming", "window"):
            raise ValueError("sampling.observation_mode must be 'streaming' or 'window'")
        self.observation_mode = observation_mode
        default_cache = options.get("kv_cache", True)
        self.use_text_cache = bool(options.get("text_cache", default_cache))
        self.pipe, self.fps = pipe, float(fps)
        self.adapter = adapter or build_adapter(pipe)
        self.size = pipe.geometry.actions_per_block
        self.latent_block = self.size // pipe.geometry.actions_per_latent
        context_limit = math.ceil(pipe.history / self.latent_block)
        self.history_latent_limit = 1 + context_limit * self.latent_block
        self.history_pixel_limit = 1 + context_limit * self.size // pipe.geometry.video_stride
        self.reset()

    @property
    def metadata(self):
        steps = (
            self.pipe.sampling_options["action_steps"]
            if self.is_triple
            else self.pipe.sampling_options["steps"]
        )
        info = dict(
            protocol="cosmos-causal-action-v2",
            implementation="veomni-triple-idm" if self.is_triple else "veomni-full-pack",
            expert_layout="triple" if self.is_triple else "dual",
            observation_mode=self.observation_mode,
            kv_cache=self.use_text_cache,
            text_cache=self.use_text_cache,
            mode=self.task,
            action_dim=55,
            actions_per_block=self.size,
            execute_steps=self.size,
            steps=steps,
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
        if self.is_triple:
            info["idm_video_mode"] = self.pipe.sampling_options["idm_video_mode"]
        return info

    def reset(self):
        use_streaming_vae = self.observation_mode == "streaming" and self.is_triple
        self.stream_encoder = self.pipe.vae.stream_encoder() if use_streaming_vae else None
        self.history_latents = None
        self.history_pixels = None
        self.text_caches = []
        self.prompt = self.control_mode = self.camera_shapes = None
        self.revision = self.observation_step = 0
        self.needs_reset = self.record_video = False
        self.video_latents = []

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

    def _encode_triple_history(self, pixels):
        """Update clean observation latents strictly from real camera frames."""
        begin = time.perf_counter()
        if self.stream_encoder is not None:
            observed = self.stream_encoder.encode(pixels).to(device=self.pipe.device, dtype=torch.float32)
            expected = self.latent_block if self.revision else 1
            if observed.shape[1] != expected:
                raise ValueError(f"Expected {expected} observed latent frames, got {observed.shape[1]}")
            history_latents = (
                observed
                if self.history_latents is None
                else torch.cat((self.history_latents, observed), dim=1)
            )
            history_latents = history_latents[:, -self.history_latent_limit :].contiguous()
            history_pixels = pixels
        else:
            history_pixels = (
                pixels if self.history_pixels is None else torch.cat((self.history_pixels, pixels), dim=1)
            )
            history_pixels = history_pixels[:, -self.history_pixel_limit :].contiguous()
            encoded = self.pipe.vae.encode_pack([{"video": history_pixels}])[0]["video_latents"]
            history_latents = encoded.to(device=self.pipe.device, dtype=torch.float32)
        vae_encode_ms = (time.perf_counter() - begin) * 1000
        return history_latents, history_pixels, vae_encode_ms

    def _sample_triple(self, initial, state, pixels):
        history_latents, history_pixels, vae_encode_ms = self._encode_triple_history(pixels)
        sample, context = _triple_rollout_sample(
            self.pipe,
            initial,
            state,
            history_latents,
            task="inverse_dynamics",
            valid=self.size,
        )
        generated = sample_block(
            self.pipe.model,
            sample,
            context,
            seed_block=self.revision,
            complete_video=self.record_video,
            external_text_caches=self.text_caches if self.use_text_cache else None,
            **self.pipe.sampling_options,
        )
        if self.record_video and (not generated["video_is_clean"] or generated["video_sigma"] != 0):
            raise ValueError("Prediction video recording requires a completed clean video block")
        timing_extra = dict(
            vae_encode_ms=vae_encode_ms,
            expert_calls=generated.get("expert_calls"),
            stage_seconds=generated.get("stage_seconds"),
            idm_video_mode=generated.get("idm_video_mode"),
            action_condition_video_sigma=generated.get("action_condition_video_sigma"),
        )
        return sample, history_latents, history_pixels, generated, timing_extra

    def _sample_legacy_policy(self, initial, state, pixels):
        begin = time.perf_counter()
        history_pixels = (
            pixels if self.history_pixels is None else torch.cat((self.history_pixels, pixels), dim=1)
        )
        history_pixels = history_pixels[:, -self.history_pixel_limit :].contiguous()
        sample, context = rollout_sample(
            initial,
            state,
            history_pixels,
            source=self.pipe.source,
            processor=self.pipe.processor,
            geometry=self.pipe.geometry,
            tokenize=self.pipe.tokenize,
            history_blocks=self.pipe.history,
            valid=self.size,
        )
        encoded = self.pipe.vae.encode_pack([sample])[0]
        vae_encode_ms = (time.perf_counter() - begin) * 1000
        for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
            encoded[key] = encoded[key].to(self.pipe.device)
        generated = sample_block(
            self.pipe.model,
            encoded,
            context,
            seed_block=self.revision,
            **self.pipe.sampling_options,
        )
        return sample, encoded["video_latents"], history_pixels, generated, dict(vae_encode_ms=vae_encode_ms)

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
        # Select the initial frame on step 0, then measured endpoints stride, 2*stride, ...
        if self.revision:
            cameras = {
                key: value[self.pipe.geometry.video_stride - 1 :: self.pipe.geometry.video_stride]
                for key, value in cameras.items()
            }
        pixels = self._compose(cameras)
        initial = InitialObservation(
            pixels[:, :1],
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
            if self.is_triple:
                sample, observed_latents, history_pixels, generated, timing_extra = self._sample_triple(
                    initial, state, pixels
                )
            else:
                sample, observed_latents, history_pixels, generated, timing_extra = self._sample_legacy_policy(
                    initial, state, pixels
                )
            action = self.pipe.processor.decode(
                generated["action"][-self.size :].cpu(), state.expand(self.size, -1), self.adapter.action_mask
            ).numpy()
            split = self.adapter.split_actions(action)
            if self.record_video:
                latent = generated["video_latents"].cpu()
                if not self.video_latents:
                    self.video_latents.append(observed_latents[:, :1].cpu().clone())
                self.video_latents.append(latent[:, -self.latent_block :].clone())
            if self.is_triple:
                self.history_latents = observed_latents
                if self.use_text_cache and "text_caches" in generated:
                    self.text_caches = generated["text_caches"]
        except Exception:
            self.reset()
            self.needs_reset = True
            raise
        self.history_pixels, self.camera_shapes = history_pixels, shapes
        self.prompt, self.control_mode, self.observation_step = prompt, mode, step
        self.revision += 1
        response = dict(
            action=split,
            available_control_modes=self.adapter.available_control_modes,
            execute_steps=self.size,
            observation_step=step,
            revision=self.revision,
            server_timing=timing_extra,
        )
        if self.revision == 1:
            response["structured_prompt"] = json.loads(sample["prompt"])
        return response


CausalPolicy = CausalInferenceSession
