# SPDX-License-Identifier: OpenMDW-1.1
"""Action resolution buckets, camera layouts and structured prompts."""

import json
import math
from collections.abc import Mapping

from ..readers.lerobot import Camera


# (width, height), preserving action branch 6066bdc's values and tie order.
RESOLUTION_BUCKETS = {
    "256": ((256, 256), (320, 256), (256, 320), (320, 256), (256, 320), (448, 256), (256, 448)),
    "480": ((512, 512), (640, 480), (480, 640), (640, 352), (352, 640)),
}
VIEW_ROLES = {"ego_view": ("head",), "concat_view": ("head", "left", "right")}
VIEW_TEXT = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}
VIEW_DESCRIPTION = {
    "ego_view": "The video shows the head-mounted camera view.",
    "concat_view": (
        "The top row shows the head-mounted camera view. "
        "The bottom row shows the left wrist camera on the left and the right wrist camera on the right."
    ),
}


def camera_layout(info, cameras, resolution="256", viewpoint="concat_view"):
    """Resolve source roles into output rectangles using the head's aspect ratio."""
    resolution = str(resolution)
    if resolution not in RESOLUTION_BUCKETS:
        raise ValueError("Action resolution must be 256 or 480")
    if viewpoint not in VIEW_ROLES:
        raise ValueError(f"Unsupported action viewpoint: {viewpoint}")
    feature = info["features"][cameras["head"]]
    names = feature["names"]
    height, width = (feature["shape"][names.index(dim)] for dim in ("height", "width"))
    width, height = min(RESOLUTION_BUCKETS[resolution], key=lambda wh: abs(height / width - wh[1] / wh[0]))
    rectangles = (
        (0, 0, height, width),
        (height, 0, height // 2, width // 2),
        (height, width // 2, height // 2, width - width // 2),
    )
    return tuple(Camera(cameras[role], *rect) for role, rect in zip(VIEW_ROLES[viewpoint], rectangles))


def _camera_name(key):
    """Drop the LeRobot image prefix while keeping bare test fixtures usable."""
    return key.removeprefix("observation.images.")


def select_cameras(cameras):
    """Select one head/left/right triple by camera name.

    Mirrors the role-selection rules used by OpenWAM's multiview preparation.
    Fisheye cameras are excluded from the head fallback so a standard
    high/front view wins over surround cameras. Returns the selected keys in
    ``[head, left, right]`` order, or an empty list when the names do not form
    one anatomical triple.
    """
    cameras = sorted(set(cameras))
    names = {camera: _camera_name(camera) for camera in cameras}
    left = [
        camera
        for camera in cameras
        if "left" in names[camera] and ("hand" in names[camera] or "wrist" in names[camera])
    ]
    right = [
        camera
        for camera in cameras
        if "right" in names[camera] and ("hand" in names[camera] or "wrist" in names[camera])
    ]
    # A stereo head keeps the left representative; the right view stays available
    # as an ordinary single view.
    heads = [camera for camera in cameras if "head" in names[camera]]
    heads.sort(
        key=lambda camera: (
            ("left" in names[camera] or "right" in names[camera]),
            "right" in names[camera],
            names[camera],
        )
    )
    fallback = [
        camera
        for camera in cameras
        if names[camera] in ("camera_front", "camera_top")
        or (("front" in names[camera] or "high" in names[camera]) and "fisheye" not in names[camera])
    ]
    head = heads or fallback
    if len(left) == 1 and len(right) == 1 and head:
        return [head[0], left[0], right[0]], "head/front above left/right wrist"
    return [], "requires camera-role review"


def _select_ego_head(cameras):
    """Pick the single head view for ``ego_view`` datasets."""
    cameras = sorted(set(cameras))
    names = {camera: _camera_name(camera) for camera in cameras}
    ego = [camera for camera in cameras if names[camera] == "ego_view"]
    if ego:
        return ego[0]
    heads = [camera for camera in cameras if "head" in names[camera]]
    heads.sort(
        key=lambda camera: (
            ("left" in names[camera] or "right" in names[camera]),
            "right" in names[camera],
            names[camera],
        )
    )
    if heads:
        return heads[0]
    fallback = [
        camera
        for camera in cameras
        if names[camera] in ("camera_front", "camera_top")
        or (("front" in names[camera] or "high" in names[camera]) and "fisheye" not in names[camera])
    ]
    if fallback:
        return fallback[0]
    return cameras[0] if len(cameras) == 1 else None


def _validate_camera_roles(features, roles, viewpoint, root=None):
    if viewpoint not in VIEW_ROLES:
        raise ValueError(f"Unsupported viewpoint: {viewpoint}")
    where = f"{root}: " if root is not None else ""
    selected = []
    for role in VIEW_ROLES[viewpoint]:
        key = roles.get(role)
        if key is None:
            raise ValueError(f"{where}missing camera role {role!r}")
        if not isinstance(key, str):
            raise ValueError(f"{where}camera {role!r} must map to a feature key")
        feature = features.get(key)
        if feature is None:
            raise ValueError(f"{where}camera {role!r} maps to unknown feature {key!r}")
        if feature.get("dtype") != "video":
            raise ValueError(f"{where}camera {role!r} feature {key!r} is not a video")
        selected.append(key)
    if len(set(selected)) != len(selected):
        raise ValueError(f"{where}camera roles must use distinct features: {roles}")


def resolve_camera_roles(features, cameras, viewpoint, *, root=None):
    """Resolve a role -> feature-key mapping for one dataset.

    ``features`` is ``info["features"]``. ``cameras`` may be an explicit
    mapping; when omitted, roles are selected from the dataset's video feature
    names using :func:`select_cameras`.
    """
    video = sorted(key for key, feature in features.items() if feature.get("dtype") == "video")
    if not video:
        where = f"{root}: " if root is not None else ""
        raise ValueError(f"{where}no video features")
    if cameras is not None:
        if not isinstance(cameras, Mapping):
            raise TypeError("cameras must be a mapping or omitted for automatic selection")
        roles = dict(cameras)
    elif viewpoint == "ego_view":
        head = _select_ego_head(video)
        if head is None:
            where = f"{root}: " if root is not None else ""
            raise ValueError(f"{where}cannot select an ego head camera from {video}; set cameras explicitly")
        roles = {"head": head}
    elif viewpoint == "concat_view":
        selected, reason = select_cameras(video)
        names = [_camera_name(key) for key in selected]
        if len(selected) != 3 or "left" not in names[1] or "right" not in names[2]:
            where = f"{root}: " if root is not None else ""
            raise ValueError(
                f"{where}cannot select head/left/right cameras ({reason}); "
                f"available video features: {video}; set cameras explicitly"
            )
        roles = dict(zip(("head", "left", "right"), selected, strict=True))
    else:
        raise ValueError(f"Unsupported viewpoint: {viewpoint}")
    _validate_camera_roles(features, roles, viewpoint, root=root)
    return roles


def format_prompt(sample):
    """Build the causal JSON text without changing the sample or its caption."""
    caption = (sample["caption"] or "").strip()
    if not caption:
        raise ValueError("A nonempty task caption is required")
    if not caption.endswith((".", "!", "?")):
        caption += "."

    height, width = map(int, sample["image_size"])
    fps = float(sample["conditioning_fps"])
    framing = VIEW_TEXT[sample["viewpoint"]]
    if description := sample.get("additional_view_description"):
        framing += " " + description.rstrip()
    # 在线执行时无法预知任务总时长，因此不将窗口时长或结束时间写进 prompt。
    action = {"description": caption}
    # 没有静止帧标注时不补造；inverse dynamics 不把该信息放进文本条件。
    if sample.get("mode") != "inverse_dynamics" and sample.get("idle_frames") is not None:
        idle = int(sample["idle_frames"])
        total = sample.get("idle_frames_total")
        if total is None and "action" in sample:
            total = len(sample["action"])
        action["idle_frame"] = f"{idle} out of {int(total)}." if total is not None else f"{idle}."
    divisor = math.gcd(width, height)
    return json.dumps(
        {
            "cinematography": {"framing": framing},
            "actions": [action],
            "fps": fps,
            "resolution": {"H": height, "W": width},
            "aspect_ratio": f"{width // divisor},{height // divisor}",
        }
    )
