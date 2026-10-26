# SPDX-License-Identifier: OpenMDW-1.1
"""Action resolution buckets, camera layouts and structured prompts."""

import json
import math

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


def format_prompt(sample):
    """Build the causal JSON text without changing the sample or its caption."""
    caption = sample["caption"].strip()
    if not caption:
        return ""
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
