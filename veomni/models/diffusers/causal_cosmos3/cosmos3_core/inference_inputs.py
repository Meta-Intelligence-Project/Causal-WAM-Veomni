# SPDX-License-Identifier: OpenMDW-1.1
"""Build one inference block directly from observations and current targets.

Training packing is intentionally independent. No future reference payload,
duplicate history, historical action/state, or training loss mask is built.
"""

import math
from dataclasses import dataclass

import torch

from . import mrope
from .attention import CausalLayout
from .patching import patchify_video


@dataclass(frozen=True)
class InferenceContext:
    """Normalized observed latents and current state in model channel order.

    history_video contains the independent first latent followed by complete
    blocks. The supplied prefix is retained so deep historical representations
    have the same dependencies as training. FPS values are video/control FPS.
    """

    text_ids: torch.Tensor
    history_video: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    block_size: int
    actions_per_block: int
    history_frames: int
    video_fps: float
    action_fps: float
    caption: str = ""

    def __post_init__(self):
        video = self.history_video
        if video.ndim != 4 or min(video.shape) < 1 or not torch.isfinite(video).all():
            raise ValueError("history_video must be finite nonempty [C,T,H,W] latents")
        if any(type(n) is not int or n < 1 for n in (self.block_size, self.actions_per_block, self.history_frames)):
            raise ValueError("Block sizes and history_frames must be positive integers")
        if (video.shape[1] - 1) % self.block_size or self.actions_per_block % self.block_size:
            raise ValueError("Observed history and action counts must align with latent blocks")
        if self.text_ids.ndim != 1 or not self.text_ids.numel():
            raise ValueError("text_ids must be a nonempty vector")
        if self.state.ndim != 1 or self.state_mask.shape != self.state.shape:
            raise ValueError("state and state_mask must be matching vectors")
        if self.action_mask.ndim != 1 or any(m.dtype != torch.bool for m in (self.state_mask, self.action_mask)):
            raise ValueError("State/action channel masks must be boolean vectors")
        if not torch.isfinite(self.state).all() or not self.state_mask.any() or not self.action_mask.any():
            raise ValueError("Current state must be finite with active state/action channels")
        if any(not math.isfinite(fps) or fps <= 0 for fps in (self.video_fps, self.action_fps)):
            raise ValueError("Observation/control FPS must be finite and positive")

    @property
    def block(self):
        return (self.history_video.shape[1] - 1) // self.block_size

    @property
    def video_shape(self):
        c, _, h, w = self.history_video.shape
        return c, self.block_size, h, w

    @property
    def action_shape(self):
        return self.actions_per_block, self.action_mask.numel()


class InferenceInputs:
    """One stage's fixed layout/conditions; only target values vary with sigma.

    video: T, historical O, target video (ID video branch, no A/S).
    action: T, historical O + completed current O, target A, current S.
    policy: T, historical O, target video/A, current S.
    forward_dynamics: T, historical O, target video, conditioned A, current S.
    """

    def __init__(
        self,
        context,
        stage,
        *,
        patch_size,
        conditioned_video=None,
        conditioned_action=None,
        base_fps=24.0,
        temporal_compression=4,
        temporal_modality_margin=15000,
    ):
        if stage not in ("video", "action", "policy", "forward_dynamics"):
            raise ValueError("Unknown inference stage")
        if type(patch_size) is not int or patch_size < 1:
            raise ValueError("patch_size must be positive")
        if any(not math.isfinite(x) or x <= 0 for x in (base_fps, temporal_compression)):
            raise ValueError("Position FPS/compression must be positive")
        if not math.isfinite(temporal_modality_margin):
            raise ValueError("Position margin must be finite")
        for name, value, required, shape in (
            ("conditioned_video", conditioned_video, stage == "action", context.video_shape),
            ("conditioned_action", conditioned_action, stage == "forward_dynamics", context.action_shape),
        ):
            if required != (value is not None):
                raise ValueError(f"{name} must be supplied exactly for its conditioning stage")
            if value is not None and (tuple(value.shape) != shape or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite with shape {shape}")
        self.context, self.stage, self.patch_size = context, stage, patch_size
        self.video_active = stage != "action"
        self.action_active = stage in ("action", "policy")
        self.mode = "inverse_dynamics" if stage in ("video", "action") else stage
        device = context.history_video.device
        observed = context.history_video.float()
        fixed_video = (
            observed if conditioned_video is None else torch.cat((observed, conditioned_video.to(observed)), 1)
        )
        self.fixed_video_tokens = patchify_video(fixed_video, patch_size)
        self.conditioned_action = None if conditioned_action is None else conditioned_action.to(device).float()
        history = observed.shape[1]
        frames = history + context.block_size
        gh, gw = (observed.shape[2] + patch_size - 1) // patch_size, (observed.shape[3] + patch_size - 1) // patch_size
        self.history_rows = history * gh * gw
        self.video_target_rows = slice(self.history_rows, frames * gh * gw)
        # Coordinates retain training's temporal origins even though absent
        # modalities/duplicate video copies occupy no rows in this layout.
        text_pos, text_end = mrope.get_3d_mrope_ids_text_tokens(len(context.text_ids), 0, use_float_positions=True)
        video_pos, _ = mrope.get_3d_mrope_ids_vae_tokens(
            grid_t=frames,
            grid_h=gh,
            grid_w=gw,
            temporal_offset=text_end + temporal_modality_margin,
            reset_spatial_indices=True,
            fps=context.video_fps,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression,
            actual_temporal_compression_factor=temporal_compression,
        )
        roles = [-1] * len(context.text_ids)
        blocks = [-1] * len(roles)
        modalities = [0] * len(roles)
        frame_ids = [-1] * len(roles)
        for frame in range(frames):
            roles.extend([0 if frame < history or not self.video_active else 1] * (gh * gw))
            blocks.extend([(frame - 1) // context.block_size] * (gh * gw))
            modalities.extend([1] * (gh * gw))
            frame_ids.extend([frame] * (gh * gw))
        positions = [text_pos, video_pos]
        if stage != "video":
            count = context.actions_per_block
            roles.extend([2 if stage == "forward_dynamics" else 1] * count + [3])
            blocks.extend([context.block] * (count + 1))
            modalities.extend([2] * count + [3])
            frame_ids.extend([-1] * (count + 1))
            action_pos, _ = mrope.get_3d_mrope_ids_vae_tokens(
                grid_t=count,
                grid_h=1,
                grid_w=1,
                temporal_offset=text_end + temporal_modality_margin,
                reset_spatial_indices=True,
                fps=context.action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=temporal_compression,
                start_frame_offset=1 + context.block * count,
            )
            state_pos = torch.zeros(3, 1, dtype=torch.float32)
            state_pos[0] = video_pos[0, (history - 1) * gh * gw]
            positions.extend((action_pos, state_pos))
        self.layout = CausalLayout(
            sample_ids=(0,) * len(roles),
            block_ids=tuple(blocks),
            roles=tuple(roles),
            modalities=tuple(modalities),
            frame_ids=tuple(frame_ids),
            history_blocks=(context.history_frames,),
            block_sizes=(context.block_size,),
            modes=(self.mode,),
        )
        self.positions = torch.cat(positions, 1).to(device)

    def model_inputs(self, video, action, timestep, dtype):
        context, device = self.context, self.context.history_video.device
        tokens = self.fixed_video_tokens
        times = torch.zeros(len(tokens), dtype=torch.float32, device=device)
        if self.video_active:
            current = patchify_video(video, self.patch_size)
            tokens = torch.cat((tokens, current))
            times = torch.cat((times, timestep.float().expand(len(current))))
        result = dict(
            input_ids=context.text_ids.to(device),
            video_tokens=tokens.to(dtype),
            video_timesteps=times,
            layout=self.layout,
            position_ids=self.positions,
        )
        if self.stage != "video":
            mask = context.action_mask.to(device).expand(context.action_shape)
            action = action if self.action_active else self.conditioned_action
            result.update(
                action_tokens=action.masked_fill(~mask, 0).to(dtype),
                action_mask=mask,
                action_timesteps=(timestep.float() if self.action_active else timestep.new_zeros(())).expand(
                    context.actions_per_block
                ),
                state_tokens=context.state.to(device=device, dtype=dtype).unsqueeze(0),
                state_mask=context.state_mask.to(device).unsqueeze(0),
            )
        return result
