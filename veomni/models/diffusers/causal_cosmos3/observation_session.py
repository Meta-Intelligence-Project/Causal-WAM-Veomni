# SPDX-License-Identifier: OpenMDW-1.1
"""Observed-input prediction and explicit real-feedback session ownership."""

import math

import torch

from veomni.data.causal_cosmos3.transforms.layout import VIEW_DESCRIPTION, format_prompt

from .cosmos3_core.generation import generate_action_block, generate_block, generate_video_block
from .cosmos3_core.inference_inputs import InferenceContext


def history_pixel_limit(pipe):
    geometry = pipe.geometry
    size = geometry.actions_per_block // geometry.actions_per_latent
    return 1 + math.ceil(pipe.history / size) * geometry.actions_per_block // geometry.video_stride


@torch.no_grad()
def predict_observation(pipe, images, state, *, caption, fps, state_mask, action_mask, task, index):
    """Predict from composed uint8 history and a physical, canonical current state.

    Caller owns feedback provenance. This function never updates observations.
    Images are already sampled at fps/video_stride, resized and camera-composed.
    State/masks are already mapped to the action template's channel order.
    """
    if task not in ("policy", "inverse_dynamics"):
        raise ValueError("Observed-input prediction supports policy or inverse_dynamics")
    geometry = pipe.geometry
    pixel_block = geometry.actions_per_block // geometry.video_stride
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[0] != 3 or images.shape[1] < 1:
        raise ValueError("Observations must be uint8 [3,T,H,W] composed images")
    if (images.shape[1] - 1) % pixel_block:
        raise ValueError("Observation history must contain the first frame and complete feedback blocks")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Control FPS must be finite and positive")
    state_mask = pipe.processor.template.validate_valid_mask(state_mask.cpu())
    action_mask = pipe.processor.template.validate_valid_mask(action_mask.cpu())
    if pipe.processor.statistics is not None and (action_mask & ~pipe.processor.statistics["action"][2]).any():
        raise ValueError("action statistics do not cover active source channels")
    if (
        not state_mask.any()
        or not action_mask.any()
        or (pipe.processor.template.required_anchor_mask(action_mask) & ~state_mask).any()
    ):
        raise ValueError("Relative actions require observed anchor state channels")
    if state.shape != state_mask.shape:
        raise ValueError("State must be a vector in canonical template channel order")
    anchor = pipe.processor.template.sanitize(state.cpu(), state_mask)
    if not torch.isfinite(anchor).all():
        raise ValueError("Observed state must be finite")
    normalized_state, clipped = pipe.processor.normalize(anchor.unsqueeze(0), state_mask, "state")
    viewpoint = pipe.source.get("viewpoint", "concat_view")
    prompt = format_prompt(
        dict(
            caption=caption,
            mode=task,
            image_size=images.shape[-2:],
            viewpoint=viewpoint,
            additional_view_description=VIEW_DESCRIPTION[viewpoint],
            conditioning_fps=fps / geometry.video_stride,
        ),
        structured=pipe.source.get("structured_prompt", True),
    )
    # Only actual supplied history enters the VAE; no future black frames or
    # fabricated action/state rows are passed through training preprocessing.
    history = pipe.vae.encode_pack([dict(video=images)])[0]["video_latents"].to(pipe.device)
    expected_frames = 1 + (images.shape[1] - 1) // geometry.temporal_compression
    if history.shape[1] != expected_frames:
        raise ValueError("Encoded history does not match observation geometry")
    context = InferenceContext(
        text_ids=torch.as_tensor(pipe.tokenize(prompt), dtype=torch.long),
        history_video=history,
        state=normalized_state[0].to(pipe.device),
        state_mask=state_mask.to(pipe.device),
        action_mask=action_mask.to(pipe.device),
        block_size=geometry.actions_per_block // geometry.actions_per_latent,
        actions_per_block=geometry.actions_per_block,
        history_frames=pipe.history,
        video_fps=fps / geometry.video_stride,
        action_fps=fps,
        caption=caption,
    )
    options = dict(pipe.sampling_options, seed_block=index)
    if options.get("kv_cache", False):
        # The two IDM stages/CFG branches share only this exact real visual window.
        # This dictionary dies with predict(); the next observation re-encodes RGB.
        options["observation_caches"] = {}
    if task == "inverse_dynamics":
        video = generate_video_block(pipe.model, context, **options)
        result = generate_action_block(pipe.model, context, video["video_latents"], **options)
        result["seeds"] = dict(video=video["seeds"], action=result["seeds"])
    else:
        result = generate_block(pipe.model, context, "policy", **options)
    actions = pipe.processor.decode(result["action"].cpu(), anchor.expand(geometry.actions_per_block, -1), action_mask)
    if not torch.isfinite(actions).all():
        raise ValueError("Nonfinite decoded actions")
    return dict(
        actions=actions,
        video_latents=torch.cat((history, result["video_latents"]), 1),
        seeds=result["seeds"],
        anchor=anchor,
        state_clip_fraction=clipped,
        context_blocks=context.block,
        prompt=prompt,
    )


class ObservationSession:
    """One real-feedback session, with no implicit action-as-state transition.

    observe receives the initial frame, then exactly one completed action block
    of *new* images (no repeated boundary frame) and its measured end state.
    timestamp is the final image/state time in seconds at a uniform control FPS.
    Only observe writes history. predict can be called once per observation.
    """

    def __init__(self, pipe, *, caption, fps, state_mask, action_mask, task="inverse_dynamics"):
        if task not in ("policy", "inverse_dynamics") or not caption.strip() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("A valid task, nonempty caption and positive FPS are required")
        self.pipe, self.caption, self.fps, self.task = pipe, caption, fps, task
        self.state_mask, self.action_mask = state_mask.cpu().clone(), action_mask.cpu().clone()
        self.reset()

    def reset(self):
        self._images = self._state = self._timestamp = None
        self._awaiting_feedback = False
        self._index = 0

    def observe(self, images, state, *, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("Observation timestamp must be finite")
        geometry = self.pipe.geometry
        first = self._images is None
        expected = 1 if first else geometry.actions_per_block // geometry.video_stride
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[:2] != (3, expected):
            raise ValueError(f"Expected {expected} new uint8 RGB frames")
        if min(images.shape[-2:]) < 1:
            raise ValueError("Observation spatial dimensions must be positive")
        if any(size % geometry.spatial_compression for size in images.shape[-2:]):
            raise ValueError("Observation spatial dimensions must be VAE aligned")
        if state.shape != self.state_mask.shape or not torch.isfinite(state).all():
            raise ValueError("Measured state must be finite and match the canonical state mask")
        if not first:
            if not self._awaiting_feedback:
                raise ValueError("Predict before submitting the next completed-block feedback")
            if images.shape[-2:] != self._images.shape[-2:]:
                raise ValueError("Observation image geometry cannot change within a session")
            elapsed = geometry.actions_per_block / self.fps
            if not math.isclose(timestamp - self._timestamp, elapsed, rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError("Feedback timestamp must advance by one complete action block")
        history = images.cpu().clone() if first else torch.cat((self._images, images.cpu()), 1)
        self._images = history[:, -history_pixel_limit(self.pipe) :].clone()
        self._state, self._timestamp = state.cpu().clone(), float(timestamp)
        self._awaiting_feedback = False

    def predict(self):
        if self._images is None or self._awaiting_feedback:
            raise ValueError("A new real observation is required before prediction")
        result = predict_observation(
            self.pipe,
            self._images,
            self._state,
            caption=self.caption,
            fps=self.fps,
            state_mask=self.state_mask,
            action_mask=self.action_mask,
            task=self.task,
            index=self._index,
        )
        result["observation_timestamp"] = self._timestamp
        result["action_timestamps"] = (
            self._timestamp + torch.arange(1, self.pipe.geometry.actions_per_block + 1, dtype=torch.float64) / self.fps
        )
        self._awaiting_feedback = True
        self._index += 1
        return result
