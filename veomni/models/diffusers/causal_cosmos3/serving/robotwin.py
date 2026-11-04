# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Server-owned conversion between RoboTwin native state/actions and unified55."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class RobotwinAdapter:
    def __init__(
        self,
        template,
        *,
        state_mask,
        action_mask,
        left_arm_dof=6,
        right_arm_dof=6,
        eef_pose_frame=None,
        T_world_base=None,
    ):
        if template.template_id != "unified55-xyzw-absolute-gripper-hand-v4":
            raise ValueError("RoboTwin native serving requires ActionTemplate55")
        for dof in (left_arm_dof, right_arm_dof):
            if type(dof) is not int or not 1 <= dof <= 7:
                raise ValueError("RoboTwin arm DOF must be an integer in [1,7]")
        self.template = template
        self.state_mask = template.validate_valid_mask(state_mask).cpu().clone()
        self.action_mask = template.validate_valid_mask(action_mask).cpu().clone()
        if not self.state_mask.any() or not self.action_mask.any():
            raise ValueError("RoboTwin state/action masks must be nonempty")
        if (template.required_anchor_mask(self.action_mask) & ~self.state_mask).any():
            raise ValueError("Relative action channels require valid state anchors")
        self.dofs = {"left": left_arm_dof, "right": right_arm_dof}
        self.joint_slots = {}
        for side, dof in self.dofs.items():
            slots = template.fields[f"{side}_arm_joint"]
            self.joint_slots[side] = np.arange(slots.start, slots.start + dof)
        self.gripper_slots = np.array([template.fields[f"{side}_gripper"].start for side in self.dofs])
        self.eef_slots = np.concatenate([np.arange(*template.fields[f"{side}_eef"].indices(55)) for side in self.dofs])
        joint_slots = np.concatenate(list(self.joint_slots.values()))
        supported = np.zeros(55, dtype=bool)
        supported[np.r_[joint_slots, self.eef_slots, self.gripper_slots]] = True
        if ((self.state_mask.numpy() | self.action_mask.numpy()) & ~supported).any():
            raise ValueError("Masks enable slots outside this robot's joints, EEFs and grippers")
        am = self.action_mask.numpy()
        self.available_control_modes = [
            mode
            for mode, slots in (("joint", joint_slots), ("eef", self.eef_slots))
            if am[np.r_[slots, self.gripper_slots]].all()
        ]
        if not self.available_control_modes:
            raise ValueError("Action mask must cover both arms and grippers for at least one execution mode")
        self.uses_eef = (self.state_mask | self.action_mask)[self.eef_slots].any().item()
        if self.uses_eef and eef_pose_frame is None:
            raise ValueError("Active EEF slots require eef_pose_frame ('base' or 'world'); pass --robot-config")
        self.eef_pose_frame = "world" if eef_pose_frame is None else eef_pose_frame
        if self.eef_pose_frame not in ("base", "world"):
            raise ValueError("eef_pose_frame must be 'base' or 'world'")
        self.T_world_base = None
        if self.eef_pose_frame == "base":
            if not isinstance(T_world_base, Mapping):
                raise ValueError("Base-frame EEF requires T_world_base with position and quaternion_wxyz")
            position = np.asarray(T_world_base.get("position"), dtype=np.float64)
            quaternion = np.asarray(T_world_base.get("quaternion_wxyz"), dtype=np.float64)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise ValueError("T_world_base needs position[3] and quaternion_wxyz[4]")
            if (
                not np.isfinite(position).all()
                or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) < 1e-8
            ):
                raise ValueError("T_world_base must be finite with a nonzero quaternion")
            quaternion = quaternion / np.linalg.norm(quaternion)
            self.T_world_base = dict(position=position.tolist(), quaternion_wxyz=quaternion.tolist())
            self._base_position, self._base_quaternion = position, quaternion
            for mask in (self.state_mask, self.action_mask):
                for side in self.dofs:
                    start = template.fields[f"{side}_eef"].start
                    xyz = mask[start : start + 3]
                    if xyz.any() and not xyz.all():
                        raise ValueError("Base-frame EEF position must enable all xyz slots or none")
        elif T_world_base is not None:
            raise ValueError("T_world_base only applies to eef_pose_frame='base'")

    @staticmethod
    def eef_config_from_dataset(root):
        """Read the training frame explicitly recorded by the EEF enrichment."""
        path = Path(root).expanduser() / "meta/info.json"
        if not path.is_file():
            raise ValueError(f"Cannot read EEF frame metadata at {path}; pass --robot-config")
        enrichment = json.loads(path.read_text()).get("custom", {}).get("eef_enrichment", {})
        frame = enrichment.get("unified_pose_frame")
        if enrichment.get("unified_quaternion") != "xyzw":
            raise ValueError(f"Expected explicit unified_quaternion='xyzw' in {path}; pass --robot-config")
        if frame in ("base", "shared ALOHA dual-arm URDF root/base frame"):
            return dict(eef_pose_frame="base", T_world_base=enrichment.get("T_world_base"))
        if frame == "world":
            return dict(eef_pose_frame="world")
        raise ValueError(f"Unknown unified EEF frame in {path}; pass --robot-config")

    @classmethod
    def from_statistics(cls, template, statistics, config_path=None, *, dataset_root=None):
        """Use exact source masks, never the union of group-statistics valid slots."""
        config = json.loads(Path(config_path).read_text()) if config_path else {}
        sources = statistics.provenance.get("sources", [])
        if ("state_mask" in config) != ("action_mask" in config):
            raise ValueError("--robot-config must provide both state_mask and action_mask when overriding masks")
        if "state_mask" not in config and "action_mask" not in config:
            pairs = {(tuple(source["state_mask"]), tuple(source["action_mask"])) for source in sources}
            if len(pairs) != 1:
                raise ValueError("Statistics must contain identical source masks; otherwise pass --robot-config")
            sm, am = pairs.pop()
            config.update(state_mask=list(sm), action_mask=list(am))
        sm = template.validate_valid_mask(config["state_mask"])
        am = template.validate_valid_mask(config["action_mask"])
        if (sm | am)[14:28].any() and "eef_pose_frame" not in config:
            if "T_world_base" in config:
                raise ValueError("T_world_base requires explicit eef_pose_frame='base'")
            if dataset_root is not None:
                frames = [cls.eef_config_from_dataset(dataset_root)]
            else:
                frames = []
                for source in sources:
                    if "eef_pose_frame" in source:
                        frames.append(
                            {key: source[key] for key in ("eef_pose_frame", "T_world_base") if key in source}
                        )
                    elif source.get("source"):
                        frames.append(cls.eef_config_from_dataset(source["source"]))
                    else:
                        raise ValueError("Statistics lack EEF frame metadata; pass --robot-config")
            if not frames or len({json.dumps(frame, sort_keys=True) for frame in frames}) != 1:
                raise ValueError("Statistics must have one EEF frame/transform; pass --robot-config")
            config.update(frames[0])
        adapter = cls(template, **config)
        statistics.validate(template.width, adapter.state_mask, adapter.action_mask)
        return adapter

    @property
    def metadata(self):
        return dict(
            state_format="robotwin-native",
            action_format="robotwin-native",
            template_id=self.template.template_id,
            quaternion_order="wxyz",
            native_eef_pose_frame="world",
            unified_eef_pose_frame=self.eef_pose_frame if self.uses_eef else None,
            T_world_base=self.T_world_base,
            left_arm_dof=self.dofs["left"],
            right_arm_dof=self.dofs["right"],
            available_control_modes=self.available_control_modes,
        )

    def _transform_pose(self, pose, *, to_world):
        """Transform xyz+wxyz poses; model normalization/delta coding stays outside."""
        pose = np.asarray(pose, dtype=np.float64)
        if self.eef_pose_frame == "world":
            return pose.astype(np.float32)
        rotation = self._base_quaternion.copy()
        xyz = pose[..., :3]
        if not to_world:
            rotation[1:] *= -1
            xyz = xyz - self._base_position
        uv = np.cross(rotation[1:], xyz)
        xyz = xyz + 2 * (rotation[0] * uv + np.cross(rotation[1:], uv))
        if to_world:
            xyz = xyz + self._base_position
        q = pose[..., 3:7]
        w = rotation[:1] * q[..., :1] - (rotation[1:] * q[..., 1:]).sum(axis=-1, keepdims=True)
        v = rotation[:1] * q[..., 1:] + q[..., :1] * rotation[1:] + np.cross(rotation[1:], q[..., 1:])
        return np.concatenate([xyz, w, v], axis=-1).astype(np.float32)

    def assemble_state(self, state):
        """Read get_obs()['joint_action'/'endpose']; apply server mask before normalization."""
        if not isinstance(state, Mapping):
            raise ValueError("state must contain RoboTwin native joint_action and endpose dictionaries")
        joint, eef = state.get("joint_action", {}), state.get("endpose", {})
        if not isinstance(joint, Mapping) or not isinstance(eef, Mapping):
            raise ValueError("joint_action and endpose must be dictionaries")
        result = np.zeros(55, dtype=np.float32)
        mask = self.state_mask.numpy()

        def read(values, key, width):
            if key not in values:
                raise ValueError(f"Missing state field: {key}")
            array = np.asarray(values[key], dtype=np.float32)
            expected = () if width == 1 else (width,)
            if array.shape not in (expected, (width,)):
                raise ValueError(f"{key} must contain {width} values, got shape {array.shape}")
            return array.reshape(width)

        for side, dof in self.dofs.items():
            slots = self.joint_slots[side]
            if mask[slots].any():
                result[slots] = read(joint, f"{side}_arm", dof)
            slots = self.template.fields[f"{side}_eef"]
            if mask[slots].any():
                pose = read(eef, f"{side}_endpose", 7)
                pose = self._transform_pose(pose, to_world=False)
                result[slots] = pose[[0, 1, 2, 4, 5, 6, 3]]  # xyz+wxyz -> xyz+xyzw
            slot = self.template.fields[f"{side}_gripper"].start
            if mask[slot]:
                key = f"{side}_gripper"
                result[slot] = read(joint if key in joint else eef, key, 1)[0]
        result[~mask] = 0
        # The template performs finite checks and quaternion sanitization in policy.infer.
        return result

    def split_actions(self, action):
        """Return ready-to-execute qpos/ee rows, including grippers; None means untrained."""
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 55 or not np.isfinite(action).all():
            raise ValueError("Generated unified action must be finite [T,55]")
        result = dict(joint=None, eef=None)
        for mode in self.available_control_modes:
            parts = []
            for side in self.dofs:
                if mode == "joint":
                    arm = action[:, self.joint_slots[side]]
                else:
                    arm = action[:, self.template.fields[f"{side}_eef"]]
                    arm = arm[:, [0, 1, 2, 6, 3, 4, 5]]  # xyz+xyzw -> xyz+wxyz
                    arm = self._transform_pose(arm, to_world=True)
                parts.extend([arm, action[:, self.template.fields[f"{side}_gripper"]]])
            result[mode] = np.concatenate(parts, axis=1)
        return result

    def state_for_replay(self, unified):
        """Translate an existing training observation into a native request for offline replay."""
        unified = np.asarray(unified, dtype=np.float32)
        if unified.shape != (55,):
            raise ValueError("Replay state must have shape [55]")
        joint, eef = {}, {}
        for side in self.dofs:
            joint[f"{side}_arm"] = unified[self.joint_slots[side]]
            joint[f"{side}_gripper"] = unified[self.template.fields[f"{side}_gripper"]][0]
            if self.state_mask[self.template.fields[f"{side}_eef"]].any():
                pose = unified[self.template.fields[f"{side}_eef"]]
                eef[f"{side}_endpose"] = self._transform_pose(pose[[0, 1, 2, 6, 3, 4, 5]], to_world=True)
        return dict(joint_action=joint, endpose=eef)
