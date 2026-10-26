# SPDX-License-Identifier: OpenMDW-1.1
"""55D physical layout; derived from Cosmos action branch 6066bdc.

Independent of dataset indexing and normalization. No registry or template base class.
"""

from types import MappingProxyType

import torch


class ActionTemplate55:
    """55D layout for processed state/action data.

    No dataset masks or quantiles are stored on the template. A dataset owns
    one [55] mask, broadcast at runtime over any leading tensor dimensions.
    Invalid values are cleared by sanitize/encode/decode, including NaNs.
    """

    template_id = "unified55-xyzw-absolute-gripper-hand-v4"
    width = 55
    fields = MappingProxyType(
        {
            "left_arm_joint": slice(0, 7),
            "right_arm_joint": slice(7, 14),
            "left_eef": slice(14, 21),
            "right_eef": slice(21, 28),
            "left_gripper": slice(28, 29),
            "right_gripper": slice(29, 30),
            "left_hand": slice(30, 36),
            "right_hand": slice(36, 42),
            "waist": slice(42, 46),
            "head": slice(46, 48),
            "mobile": slice(48, 51),
            "reserved": slice(51, 55),
        }
    )
    rotation_groups = (tuple(range(17, 21)), tuple(range(24, 28)))
    absolute_fields = ("left_gripper", "right_gripper", "left_hand", "right_hand")

    def required_anchor_mask(self, action_mask):
        """Return channels that require a measured anchor; absolute hands/grippers do not."""
        mask = self.validate_valid_mask(action_mask).clone()
        for name in self.absolute_fields:
            mask[self.fields[name]] = False
        return mask

    def validate_valid_mask(self, valid_mask, *, template_id=None):
        """Validate width, binary values, reserved slots and complete quaternion groups."""
        if template_id is not None and template_id != self.template_id:
            raise ValueError(f"Expected template_id={self.template_id}, got {template_id}")
        # Preserve Python float precision before checking exact 0/1 values.
        mask = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.as_tensor(valid_mask, dtype=torch.float64)
        if mask.shape != (self.width,):
            raise ValueError("valid_mask must be a dataset-level [55] vector")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("valid_mask must contain only bool or exact 0/1 values")
        mask = mask.bool()
        if mask[self.fields["reserved"]].any():
            raise ValueError("Reserved dimensions 51:55 must be invalid")
        for group in self.rotation_groups:
            enabled = mask[list(group)]
            if enabled.any() and not enabled.all():
                raise ValueError("A quaternion rotation group must be entirely valid or invalid")
        return mask

    def sanitize(self, values, valid_mask):
        """Clear invalid channels and canonicalize active xyzw quaternions."""
        mask = self.validate_valid_mask(valid_mask)
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise ValueError("Template values must be floating-point tensors")
        if values.ndim < 1 or values.shape[-1] != self.width:
            raise ValueError("Template values must have last dimension 55")
        mask = mask.to(values.device)
        result = values.masked_fill(~mask, 0)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite value in a valid template dimension")
        for group in self.rotation_groups:
            if mask[list(group)].all():
                q = result[..., list(group)]
                if (torch.linalg.vector_norm(q, dim=-1) < 1e-8).any():
                    raise ValueError("Active quaternion must be nonzero")
                result[..., list(group)] = self._unit_quaternion(q)
        return result

    def validate(self, state, action, valid_mask):
        """Validate values and broadcast compatibility."""
        state, action = self.sanitize(state, valid_mask), self.sanitize(action, valid_mask)
        try:
            torch.broadcast_shapes(state.shape, action.shape)
        except RuntimeError as exc:
            raise ValueError("State and action shapes are not broadcastable") from exc

    @staticmethod
    def _unit_quaternion(q):
        """Normalize xyzw and select a deterministic hemisphere."""
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        # w >= 0; at exactly 180 degrees use the largest xyz component.
        pivot = torch.where(q[..., 3:4] != 0, q[..., 3:4], q.gather(-1, q.abs().argmax(-1, keepdim=True)))
        return torch.where(pivot < 0, -q, q)

    @staticmethod
    def _multiply(a, b):
        """Compose xyzw quaternions as a * b."""
        av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
        return torch.cat(
            (aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)), -1
        )

    def _convert(self, values, anchor, valid_mask, *, decode):
        """Convert relative scalars/rotations while preserving absolute targets."""
        values = self.sanitize(values, valid_mask)
        anchor = self.sanitize(anchor, valid_mask)
        values, anchor = torch.broadcast_tensors(values, anchor)
        result = values + anchor if decode else values - anchor
        for name in self.absolute_fields:
            result[..., self.fields[name]] = values[..., self.fields[name]]
        mask = self.validate_valid_mask(valid_mask).to(values.device)
        for group in self.rotation_groups:
            indexes = list(group)
            if mask[indexes].all():
                a, b = self._unit_quaternion(anchor[..., indexes]), self._unit_quaternion(values[..., indexes])
                if not decode:
                    a = torch.cat((-a[..., :3], a[..., 3:]), -1)
                result[..., indexes] = self._unit_quaternion(self._multiply(a, b))
        return result.masked_fill(~mask, 0)

    def encode_action_delta(self, absolute_action, anchor_state, valid_mask):
        """Encode target minus anchor and inverse(anchor) * target rotations."""
        return self._convert(absolute_action, anchor_state, valid_mask, decode=False)

    def decode_action_delta(self, action_delta, anchor_state, valid_mask):
        """Decode relative actions using the same measured block anchor."""
        return self._convert(action_delta, anchor_state, valid_mask, decode=True)
