# SPDX-License-Identifier: OpenMDW-1.1
"""Training-free RTC action-prefix encoding and input-only flow guidance.

Equations 2--5 of https://arxiv.org/html/2506.07339v1 use flow time tau
increasing from noise to data. Cosmos uses sigma=1-tau and velocity with the
opposite sign: clean=x-sigma*v, so guided Cosmos velocity is v-gain*VJP.
Only the old action prefix supplies an error target. Its VJP may still alter
current video latents because video and action are generated jointly.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class RTCPrefix:
    """Remaining absolute targets, starting at the current observation time.

    The execution adapter removes already executed actions before constructing
    this object. Invalid template channels are cleared in ``prepare_prefix``;
    finite values are required in every valid channel. ``delay_steps`` is the
    number of committed actions while inference runs, not the prefix length.
    """

    previous_actions: torch.Tensor
    delay_steps: int
    beta: float = 5.0

    def __post_init__(self):
        actions = self.previous_actions
        if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
            raise ValueError("previous_actions must be a floating-point [P,D] tensor")
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("previous_actions must have shape [P,D] with D > 0")
        delay = _integer(self.delay_steps, "delay_steps")
        if delay > len(actions):
            raise ValueError("delay_steps cannot exceed the available action prefix")
        _nonnegative_real(self.beta, "beta")


@dataclass(frozen=True)
class PreparedRTCPrefix:
    action_indexes: torch.Tensor
    normalized_targets: torch.Tensor
    weights: torch.Tensor
    absolute_prefix: torch.Tensor
    delay_steps: int
    beta: float


def prefix_weights(delay_steps, overlap, horizon, *, device=None):
    """RTC Eq. 5 with overlap=P=H-s; the first d actions have weight one."""
    delay_steps = _integer(delay_steps, "delay_steps")
    overlap = _integer(overlap, "overlap")
    horizon = _integer(horizon, "horizon", minimum=1)
    if not delay_steps <= overlap <= horizon:
        raise ValueError("RTC prefix requires 0 <= delay_steps <= overlap <= horizon")
    indexes = torch.arange(horizon, device=device)
    weights = torch.zeros(horizon, device=device, dtype=torch.float32)
    weights[indexes < delay_steps] = 1.0
    middle = (indexes >= delay_steps) & (indexes < overlap)
    c = (overlap - indexes[middle]).float() / (overlap - delay_steps + 1)
    weights[middle] = c * torch.expm1(c) / math.expm1(1.0)
    return weights


@torch.no_grad()
def prepare_prefix(prefix, metadata, current_block, *, device):
    """Rebase absolute targets at the new anchor and normalize without clipping.

    A constant quantile channel can only represent its single physical delta.
    Matching constant channels use normalized zero and zero guidance weight;
    an incompatible prefix is rejected rather than silently changed.
    """
    if not isinstance(prefix, RTCPrefix):
        raise ValueError("prefix must be an RTCPrefix")
    current_block = _integer(current_block, "current_block")
    template = getattr(metadata, "template", None)
    statistics = getattr(metadata, "statistics", None)
    if template is None or statistics is None or getattr(statistics, "action", None) is None:
        raise ValueError("RTC needs an action template, block anchors and action statistics")
    width = _integer(template.width, "template width", minimum=1)
    if prefix.previous_actions.shape[1] != width:
        raise ValueError("RTC action width differs from the template")
    block_size = _integer(metadata.block_size, "block_size", minimum=1)
    frame_ids = metadata.action_frame_ids
    if (
        not isinstance(frame_ids, torch.Tensor)
        or frame_ids.ndim != 1
        or frame_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        or not len(frame_ids)
        or (frame_ids < 1).any()
        or (frame_ids[1:] < frame_ids[:-1]).any()
    ):
        raise ValueError("action_frame_ids must be a nonempty ordered positive integer vector")
    decision_window = getattr(metadata, "layout_mode", "block") == "decision_window"
    action_indexes = (
        metadata.action_indexes_for_group(current_block).to(device=device)
        if decision_window
        else torch.where((frame_ids.to(device=device, dtype=torch.long) - 1) // block_size == current_block)[0]
    )
    horizon = len(action_indexes)
    if not horizon or (action_indexes[1:] - action_indexes[:-1] != 1).any():
        raise ValueError("RTC requires a contiguous, nonempty current action block")
    overlap = len(prefix.previous_actions)
    if not prefix.delay_steps <= overlap <= horizon:
        raise ValueError("RTC prefix requires 0 <= delay_steps <= prefix length <= current horizon")

    masks = metadata.action_mask
    if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool or masks.shape != (len(frame_ids), width):
        raise ValueError("action_mask must be boolean [number of actions, template width]")
    current_masks = masks.to(device=device).index_select(0, action_indexes)
    mask = template.validate_valid_mask(current_masks[0]).to(device=device)
    if not mask.any() or not torch.equal(current_masks, mask.expand_as(current_masks)):
        raise ValueError("RTC current actions must share a nonempty dataset-level valid mask")
    anchors, state_masks = metadata.anchors, metadata.state_mask
    state_index = metadata.current_state_index if decision_window else current_block
    if (
        not isinstance(anchors, torch.Tensor)
        or not anchors.is_floating_point()
        or anchors.ndim != 2
        or anchors.shape[1] != width
        or state_index >= len(anchors)
        or not isinstance(state_masks, torch.Tensor)
        or state_masks.dtype != torch.bool
        or state_masks.shape != anchors.shape
    ):
        raise ValueError("RTC needs floating-point [blocks,D] anchors and boolean state masks")
    state_mask = template.validate_valid_mask(state_masks[state_index]).to(device=device)
    if not state_mask.any() or (template.required_anchor_mask(mask).to(device=device) & ~state_mask).any():
        raise ValueError("RTC prefix lacks a valid measured anchor for relative action fields")
    anchor = template.sanitize(anchors[state_index].detach().to(device=device, dtype=torch.float32), state_mask)
    # Encoding canonicalizes rotations, but physical export must preserve the
    # previous plan's valid-channel bytes, including tiny quaternion roundoff.
    absolute_prefix = (
        prefix.previous_actions.detach().to(device=device, dtype=torch.float32).masked_fill(~mask, 0).clone()
    )
    absolute = template.sanitize(absolute_prefix, mask)
    encoded = template.encode_action_delta(absolute, anchor, mask)
    if encoded.shape != absolute.shape or not torch.isfinite(encoded).all():
        raise ValueError("RTC template returned invalid encoded action targets")

    quantiles = statistics.action
    low, high, valid = quantiles.low, quantiles.high, quantiles.valid
    if (
        not all(isinstance(value, torch.Tensor) and value.shape == (width,) for value in (low, high, valid))
        or not low.is_floating_point()
        or not high.is_floating_point()
        or valid.dtype != torch.bool
    ):
        raise ValueError("RTC action statistics must have floating [D] bounds and boolean [D] validity")
    low, high, valid = (value.detach().to(device=device) for value in (low, high, valid))
    low, high = low.float(), high.float()
    if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any():
        raise ValueError("RTC action statistics must be finite and ordered")
    if (mask & ~valid).any():
        raise ValueError("RTC action statistics do not cover valid action channels")
    span = high - low
    if not torch.isfinite(span).all():
        raise ValueError("RTC action statistics have a nonfinite span")
    constant = mask & (span == 0)
    if constant.any() and not torch.isclose(encoded[:, constant], low[constant], rtol=1e-5, atol=1e-6).all():
        raise ValueError("RTC prefix conflicts with a constant action-statistics channel")
    targets = torch.zeros((horizon, width), device=device, dtype=torch.float32)
    if overlap:
        active = mask & ~constant
        targets[:overlap, active] = 2 * (encoded[:, active] - low[active]) / span[active] - 1
    weights = prefix_weights(prefix.delay_steps, overlap, horizon, device=device)[:, None] * mask.float()[None, :]
    weights[:, constant] = 0
    if not torch.isfinite(targets).all():
        raise ValueError("RTC prefix normalization produced nonfinite targets")
    return PreparedRTCPrefix(
        action_indexes=action_indexes,
        normalized_targets=targets,
        weights=weights,
        absolute_prefix=absolute_prefix,
        delay_steps=int(prefix.delay_steps),
        beta=float(prefix.beta),
    )


def guided_velocity(sample_flat, velocity_fn_tensor_to_tensor, sigma, target_flat, weights_flat, beta):
    """Return detached Cosmos velocity with RTC's full joint input VJP.

    This function locally enables input gradients even inside ``no_grad``.
    ``inference_mode`` must be disabled: it cannot be overridden by enable_grad.
    ``autograd.grad`` requests only the input derivative, never parameter grads.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC input guidance cannot run in inference_mode; use no_grad instead")
    beta = _nonnegative_real(beta, "beta")
    sigma = _nonnegative_real(sigma, "sigma")
    if sigma > 1:
        raise ValueError("sigma must lie in [0,1]")
    for name, value in (("sample_flat", sample_flat), ("target_flat", target_flat), ("weights_flat", weights_flat)):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.ndim != 1
            or not value.numel()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be a nonempty finite floating-point vector")
    if target_flat.shape != sample_flat.shape or weights_flat.shape != sample_flat.shape:
        raise ValueError("RTC sample, target and weights must have identical flattened shapes")
    if ((weights_flat < 0) | (weights_flat > 1)).any():
        raise ValueError("RTC guidance weights must lie in [0,1]")
    if not callable(velocity_fn_tensor_to_tensor):
        raise ValueError("velocity_fn_tensor_to_tensor must be callable")

    def velocity(x):
        value = velocity_fn_tensor_to_tensor(x)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.shape != x.shape
            or value.device != x.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("RTC velocity function must return a finite vector matching the input")
        return value

    if sigma == 0 or beta == 0 or not weights_flat.any():
        with torch.no_grad():
            return velocity(sample_flat.detach()).detach()
    with torch.enable_grad():
        x = sample_flat.detach().clone().requires_grad_(True)
        v = velocity(x)
        clean = x - sigma * v
        target = target_flat.detach().to(x)
        weights = weights_flat.detach().to(x)
        error = (target - clean.detach()) * weights
        gradient = torch.autograd.grad(clean, x, grad_outputs=error, create_graph=False)[0]
        tau = 1 - sigma
        gain = beta if sigma == 1 else min(beta, (tau * tau + sigma * sigma) / (tau * sigma))
        guided = v - gain * gradient
        if not torch.isfinite(guided).all():
            raise ValueError("RTC input guidance produced a nonfinite velocity")
        return guided.detach()
