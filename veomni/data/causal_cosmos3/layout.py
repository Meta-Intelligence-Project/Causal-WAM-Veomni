# SPDX-License-Identifier: OpenMDW-1.1
"""Action resolution buckets and camera layouts, independent of the source framework."""

from .lerobot import Camera


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
    """Append the actual viewpoint/layout and HxW once, before tokenization."""
    caption = sample["caption"].rstrip()
    if not caption:
        return caption
    viewpoint = sample["viewpoint"]
    height, width = sample["image_size"]
    separator = " " if caption.endswith(".") else ". "
    return (
        caption
        + separator
        + VIEW_TEXT[viewpoint]
        + " "
        + sample["additional_view_description"]
        + f" This video is {height}x{width}."
    )
