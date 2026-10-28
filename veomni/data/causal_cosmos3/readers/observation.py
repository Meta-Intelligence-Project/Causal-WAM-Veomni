"""Explicit inference windows and initial-observation-only input adapters."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from ..transforms.layout import VIEW_DESCRIPTION, VIEW_ROLES, camera_layout, format_prompt
from .lerobot import ActionColumns, plan_window, read_metadata, read_rows, read_window
from .video import decode_into


@dataclass
class InitialObservation:
    image: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    caption: str
    fps: float
    horizon: int
    timestamp: float
    identity: dict


def episode_window(root, episode, start, horizon, source, geometry):
    """Plan only from metadata, including an optional unpadded observation length."""
    viewpoint = source.get("viewpoint", "concat_view")
    info, episodes = read_metadata(root, [source["cameras"][role] for role in VIEW_ROLES[viewpoint]])
    rows = np.flatnonzero(episodes["episode_index"] == episode)
    if len(rows) != 1:
        raise ValueError(f"Unknown episode {episode} in {root}")
    ep = {key: value[int(rows[0])].item() for key, value in episodes.items()}
    # Match ActionDataset's source-FPS override without consuming future observations.
    annotations = Path(root) / "meta/episodes.jsonl"
    if annotations.exists():
        with annotations.open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if record["episode_index"] == episode and "source_fps" in record:
                        ep["source_fps"] = float(record["source_fps"])
    for path in sorted((Path(root) / "meta/episodes").glob("*/*.parquet")):
        with pq.ParquetFile(path) as table:
            if "original_length" in table.schema_arrow.names:
                for row in table.read(columns=["episode_index", "original_length"]).to_pylist():
                    if row["episode_index"] == episode:
                        length = int(row["original_length"])
                        if not 2 <= length <= ep["dataset_to_index"] - ep["dataset_from_index"]:
                            raise ValueError("Invalid original episode length")
                        ep["dataset_to_index"] = ep["dataset_from_index"] + length
    columns = ActionColumns(**source.get("columns", {}))
    cameras = camera_layout(info, source["cameras"], source.get("resolution", "256"), viewpoint)
    window = plan_window(
        root,
        episode,
        start,
        horizon,
        columns=columns,
        cameras=cameras,
        info=info,
        ep=ep,
        block_size=geometry.actions_per_block,
        video_stride=geometry.video_stride,
    )
    fps = ep["source_fps"]
    if math.isnan(fps):
        fps = window.fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source FPS must be positive and finite")
    preprocessing = dict(
        camera_shapes={camera.key: info["features"][camera.key]["shape"] for camera in cameras},
        camera_rectangles=[dict(key=c.key, top=c.top, left=c.left, height=c.height, width=c.width) for c in cameras],
        canvas=[window.height, window.width],
        video_stride=geometry.video_stride,
        spatial_compression=geometry.spatial_compression,
        temporal_compression=geometry.temporal_compression,
    )
    return window, float(fps), preprocessing


def task_caption(root, task_index):
    path = Path(root) / "meta/tasks.parquet"
    if path.exists():
        with pq.ParquetFile(path) as table:
            for row in table.read().to_pylist():
                if int(row["task_index"]) == task_index:
                    return str(row.get("task", row.get("__index_level_0__", "")))
    return ""


def make_sample(raw, *, source, processor, geometry, tokenize, caption, fps, task, history):
    """Use the training transform/prompt/token geometry with explicit inference tasks."""
    if task not in ("policy", "forward_dynamics", "inverse_dynamics") or history < 1:
        raise ValueError("An explicit task and positive history are required")
    if not caption.strip():
        raise ValueError("A nonempty fixed task caption is required")
    sample = dict(raw, **processor(raw, geometry))
    sample.pop("state")
    viewpoint = source.get("viewpoint", "concat_view")
    sample.update(
        mode=task,
        history_blocks=history,
        block_size=geometry.actions_per_block // geometry.actions_per_latent,
        caption=caption,
        source=source.get("name", "inference"),
        viewpoint=viewpoint,
        conditioning_fps=fps / geometry.video_stride,
        conditioning_fps_action=fps,
        image_size=tuple(raw["video"].shape[-2:]),
        resolution=str(source.get("resolution", "256")),
        template_id=processor.template.template_id,
        additional_view_description=VIEW_DESCRIPTION[viewpoint],
    )
    sample["prompt"] = format_prompt(sample)
    sample["text_token_ids"] = torch.as_tensor(tokenize(sample["prompt"]), dtype=torch.long)
    sample["token_length"] = sum(
        geometry.token_layout(len(sample["action"]), *sample["image_size"], len(sample["text_token_ids"])).values()
    )
    latents = 1 + len(sample["action"]) // geometry.actions_per_latent
    sample["condition_frame_indexes_vision"] = list(range(latents)) if task == "inverse_dynamics" else [0]
    sample["condition_frame_indexes_action"] = list(range(len(sample["action"]))) if task == "forward_dynamics" else []
    return sample


def read_teacher_window(
    root, episode, start, horizon, *, source, processor, geometry, tokenize, task, history, prompt=None
):
    window, fps, preprocessing = episode_window(root, episode, start, horizon, source, geometry)
    raw = read_window(window)
    caption = prompt if prompt is not None else task_caption(root, raw["task_index"])
    sample = make_sample(
        raw,
        source=source,
        processor=processor,
        geometry=geometry,
        tokenize=tokenize,
        caption=caption,
        fps=fps,
        task=task,
        history=history,
    )
    sample["identity"] = dict(
        dataset=str(Path(root).resolve()), episode=episode, start=start, preprocessing=preprocessing
    )
    physical, mask = processor._map(raw["action"], raw["action_mask"], processor.action_mapping)
    sample["ground_truth_actions"] = processor.template.sanitize(physical, mask)[: window.real_actions]
    return sample


def validate_feedback(source, processor, geometry, state_mask, action_mask):
    """Require declared next-state targets and complete, matching feedback channels."""
    columns = ActionColumns(**source.get("columns", {}))
    if columns.target != columns.state or columns.target_mask != columns.state_mask or columns.target_offset != 1:
        raise ValueError("Action-as-state rollout requires identical state/target columns and target_offset=1")
    identity = list(range(processor.template.width))
    state_mapping = identity if processor.state_mapping is None else list(processor.state_mapping)
    action_mapping = identity if processor.action_mapping is None else list(processor.action_mapping)
    if state_mapping != action_mapping:
        raise ValueError("Action-as-state rollout requires matching physical state/action channel mappings")
    provenance = processor.statistics_geometry
    for key, expected in (
        ("actions_per_block", geometry.actions_per_block),
        ("video_stride", geometry.video_stride),
        ("temporal_compression_factor", geometry.temporal_compression),
    ):
        if key in provenance and provenance[key] != expected:
            raise ValueError(f"Statistics geometry differs for {key}")
    if not torch.equal(state_mask, action_mask) or not state_mask.any():
        raise ValueError("Action-as-state rollout requires complete matching state/action masks")


def read_initial_observation(root, episode, start, horizon, *, source, processor, geometry, prompt=None):
    """Request exactly one state row and one timestamp per camera; no future GT."""
    window, fps, preprocessing = episode_window(root, episode, start, horizon, source, geometry)
    c = window.columns
    values = read_rows(
        window.data_path,
        window.row_start,
        window.row_start + 1,
        [c.state, c.state_mask, c.target_mask, "timestamp", "task_index"],
    )
    state, sm = processor._map(
        torch.from_numpy(values[c.state].copy()),
        torch.from_numpy(values[c.state_mask][0].copy()).bool(),
        processor.state_mapping,
    )
    _, am = processor._map(
        torch.from_numpy(values[c.state].copy()),
        torch.from_numpy(values[c.target_mask][0].copy()).bool(),
        processor.action_mapping,
    )
    sm, am = processor.template.validate_valid_mask(sm), processor.template.validate_valid_mask(am)
    validate_feedback(source, processor, geometry, sm, am)
    image = np.zeros((3, 1, window.height, window.width), dtype=np.uint8)
    timestamp = float(values["timestamp"][0])
    for camera, path, offset in zip(window.cameras, window.video_paths, window.video_offsets, strict=True):
        crop = image[:, :, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width]
        decode_into(path, [timestamp + offset], crop)
    caption = prompt if prompt is not None else task_caption(root, int(values["task_index"][0]))
    if not caption.strip():
        raise ValueError("A nonempty fixed task caption is required")
    return InitialObservation(
        torch.from_numpy(image),
        processor.template.sanitize(state[0], sm),
        sm,
        am,
        caption,
        fps,
        window.real_actions,
        timestamp,
        dict(
            dataset=str(Path(root).resolve()),
            episode=episode,
            start=start,
            image_size=[window.height, window.width],
            preprocessing=preprocessing,
        ),
    )


def read_npz_observation(path, *, source, processor, geometry, fps, horizon, prompt=None):
    """Load physical state and RGB/HWC images; use native bucket/layout/bicubic resize."""
    import av
    from av.video.reformatter import Interpolation

    if not math.isfinite(fps) or fps <= 0 or horizon < 1:
        raise ValueError("NPZ requires explicit positive fps and horizon")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        if metadata.get("color_order", "RGB") != "RGB" or metadata.get("image_layout", "HWC") != "HWC":
            raise ValueError("NPZ images must be RGB/HWC")
        cameras = dict(source.get("npz_cameras", source["cameras"]))
        images, features = {}, {}
        for key in cameras.values():
            image = data[key]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must be RGB uint8 [H,W,3]")
            images[key] = image.copy()
            features[key] = dict(shape=image.shape, names=["height", "width", "channels"])
        layout = camera_layout(
            dict(features=features), cameras, source.get("resolution", "256"), source.get("viewpoint", "concat_view")
        )
        canvas = np.zeros(
            (3, 1, max(c.top + c.height for c in layout), max(c.left + c.width for c in layout)), dtype=np.uint8
        )
        for camera in layout:
            frame = av.VideoFrame.from_ndarray(images[camera.key], format="rgb24")
            image = frame.reformat(
                width=camera.width, height=camera.height, format="rgb24", interpolation=Interpolation.BICUBIC
            ).to_ndarray()
            canvas[:, 0, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width] = (
                image.transpose(2, 0, 1)
            )
        # The documented POC schema has fixed Piper validity; other inputs provide masks explicitly.
        if "state_mask" in data:
            sm = torch.from_numpy(data["state_mask"].copy()).bool()
        elif metadata.get("robot_type", source.get("npz_robot")) == "piper":
            sm = torch.zeros(55, dtype=torch.bool)
            sm[:6], sm[7:13], sm[14:30] = True, True, True
        else:
            raise ValueError("NPZ requires an explicit state_mask for this robot")
        am = torch.from_numpy(data["action_mask"].copy()).bool() if "action_mask" in data else sm.clone()
        sm, am = processor.template.validate_valid_mask(sm), processor.template.validate_valid_mask(am)
        validate_feedback(source, processor, geometry, sm, am)
        state = torch.from_numpy(data["state"].copy()).float()
        if state.shape != (processor.template.width,) or not torch.isfinite(state).all():
            raise ValueError("NPZ state must be a finite physical template-width vector")
        caption = prompt if prompt is not None else str(data["prompt"].item()) if "prompt" in data else ""
        if not caption.strip():
            raise ValueError("NPZ requires a nonempty fixed task prompt")
    return InitialObservation(
        torch.from_numpy(canvas),
        processor.template.sanitize(state, sm),
        sm,
        am,
        caption,
        float(fps),
        horizon,
        0.0,
        dict(
            input_npz=str(Path(path).resolve()),
            image_size=list(canvas.shape[-2:]),
            metadata=metadata,
            preprocessing=dict(
                camera_shapes={key: value["shape"] for key, value in features.items()},
                camera_rectangles=[
                    dict(key=c.key, top=c.top, left=c.left, height=c.height, width=c.width) for c in layout
                ],
                canvas=list(canvas.shape[-2:]),
                video_stride=geometry.video_stride,
                spatial_compression=geometry.spatial_compression,
                temporal_compression=geometry.temporal_compression,
            ),
        ),
    )
