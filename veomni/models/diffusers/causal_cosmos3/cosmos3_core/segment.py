# SPDX-License-Identifier: OpenMDW-1.1
"""One differentiable episode segment with bounded, detached layer-input memory."""

from dataclasses import dataclass
from itertools import groupby

import torch

from .attention import AttentionPlan, TNDChunk


@dataclass(frozen=True)
class SegmentMemory:
    """Read-only clean history; normalization/K/V use the next segment's weights."""

    frame_ids: tuple[int, ...]
    position_ids: torch.Tensor
    layer_inputs: tuple[torch.Tensor, ...]
    input_ids: torch.Tensor
    text_position_ids: torch.Tensor
    context_id: str
    history_latents: int
    patches_per_frame: int
    model_id: int


@dataclass(frozen=True)
class SegmentLayerContext:
    memory: torch.Tensor
    memory_cos: torch.Tensor
    memory_sin: torch.Tensor
    cold_plan: AttentionPlan | None


def _frames(value, name):
    if not isinstance(value, tuple) or any(type(frame) is not int for frame in value):
        raise ValueError(f"{name} must be a tuple of integer latent frame IDs")
    if any(right != left + 1 for left, right in zip(value, value[1:])):
        raise ValueError(f"{name} must contain consecutive latent frames")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _conditions(value, size, name):
    if type(value) is bool:
        return (value,) * size
    if not isinstance(value, tuple) or len(value) != size or any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must be bool or one boolean per payload row")
    return value


def _position(value, rows, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (3, rows)
        or value.dtype != torch.float32
        or value.device != device
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{name} must be finite device-local FP32 [3,{rows}] positions")
    return value


def _plan(groups, key_rows, text_length, device, backend):
    """Assemble independent query groups, keeping their original token order."""
    if backend not in ("dense", "grouped", "npu_tnd"):
        raise ValueError("attention_backend must be dense, grouped, or npu_tnd")
    if backend == "npu_tnd" and torch.device(device).type != "npu":
        raise ValueError("npu_tnd requires an Ascend NPU")
    chunks, queries, keys, query_ends, key_ends = [], [], [], [], []
    count = 0

    def flush():
        if queries:
            chunks.append(TNDChunk.from_lists(queries, keys, query_ends, key_ends, device=device))
            queries.clear()
            keys.clear()
            query_ends.clear()
            key_ends.clear()

    for query_rows, visible in groups:
        if len(visible) > 1_000_000 or len(query_rows) > 1_000_000:
            raise ValueError("Segment attention exceeds the NPU TND per-group limit")
        if query_ends and (len(keys) + len(visible) > 131072 or len(query_ends) == 1000):
            flush()
        queries.extend(query_rows)
        keys.extend(visible)
        query_ends.append(len(queries))
        key_ends.append(len(keys))
        count += len(query_rows)
    flush()
    indexes = torch.arange(count, dtype=torch.long, device=device)
    text = torch.arange(text_length, dtype=torch.long, device=device)
    empty = text[:0]
    mask = None
    if backend == "dense":
        mask = torch.zeros((count, key_rows), dtype=torch.bool, device=device)
        for chunk in chunks:
            qstart = kstart = 0
            for qend, kend in zip(chunk.query_ends, chunk.kv_ends, strict=True):
                q, k = chunk.query_indexes[qstart:qend], chunk.kv_indexes[kstart:kend]
                mask[q[:, None], k[None, :]] = True
                qstart, kstart = qend, kend
    return AttentionPlan(
        modality_indexes=(text, empty, empty, empty),
        gen_indexes=indexes,
        text_samples=(text,),
        text_indexes=text,
        text_ends=(text_length,),
        chunks=tuple(chunks),
        mask=mask,
        backend=backend,
    )


def _prepare(model, segment, memory):
    if model.config.action_interface != "shared" or model.config.state_dim < 1:
        raise ValueError("Episode segments require the shared Action/state interface")
    if not isinstance(segment, dict):
        raise ValueError("stream_segment must be a prepared dictionary")
    context = segment["context_id"]
    if not isinstance(context, str) or not context:
        raise ValueError("Episode segments require a nonempty context_id")
    history = _positive(segment["history_latents"], "history_latents")
    patches = _positive(segment["patches_per_frame"], "patches_per_frame")
    clean_frames = _frames(segment["clean_frame_ids"], "clean_frame_ids")
    if len(clean_frames) > model.config.tbptt_history_latents:
        raise ValueError("A segment exceeds tbptt_keep_history_steps of newly observed time")
    device = segment["clean_video_tokens"].device
    input_ids = segment["input_ids"]
    if input_ids.ndim != 1 or not input_ids.numel() or input_ids.dtype != torch.long or input_ids.device != device:
        raise ValueError("A segment requires nonempty device-local long input_ids")
    text_positions = segment.get("text_position_ids")
    if text_positions is None:
        text_positions = torch.arange(len(input_ids), dtype=torch.float32, device=device).expand(3, -1)
    text_positions = _position(text_positions, len(input_ids), device, "text_position_ids")
    prefix_frames = _frames(segment["prefix_frame_ids"], "prefix_frame_ids")
    _position(segment["prefix_position_ids"], len(prefix_frames) * patches, device, "prefix_position_ids")
    _position(segment["clean_position_ids"], len(clean_frames) * patches, device, "clean_position_ids")
    for name, rows, width in (
        ("prefix_video_tokens", len(prefix_frames) * patches, model.config.patch_dim),
        ("clean_video_tokens", len(clean_frames) * patches, model.config.patch_dim),
        ("future_video_tokens", len(segment["future_video_tokens"]), model.config.patch_dim),
        ("action_tokens", len(segment["action_tokens"]), model.config.action_dim),
        ("state_tokens", len(segment["state_tokens"]), model.config.state_dim),
    ):
        value = segment[name]
        if value.shape != (rows, width) or value.device != device or not value.is_floating_point():
            raise ValueError(f"Invalid segment {name} payload shape/device/dtype")
    for kind in ("action", "state"):
        mask, payload = segment[f"{kind}_mask"], segment[f"{kind}_tokens"]
        if mask.shape != payload.shape or mask.dtype != torch.bool or mask.device != device:
            raise ValueError(f"Invalid segment {kind}_mask")
    _position(segment["future_position_ids"], len(segment["future_video_tokens"]), device, "future_position_ids")
    for kind in ("action", "state"):
        _position(segment[f"{kind}_position_ids"], len(segment[f"{kind}_tokens"]), device, f"{kind}_position_ids")
    for kind in ("future_video", "action"):
        times = segment[f"{kind}_timesteps"]
        if (
            times.shape != (len(segment[f"{kind}_tokens"]),)
            or times.dtype != torch.float32
            or times.device != device
            or not torch.isfinite(times).all().item()
            or not ((times >= 0) & (times <= 1000)).all().item()
        ):
            raise ValueError(f"Invalid segment {kind}_timesteps")
    if memory is not None:
        if not isinstance(memory, SegmentMemory):
            raise ValueError("stream_memory must be SegmentMemory")
        if (
            memory.context_id != context
            or memory.model_id != id(model)
            or memory.history_latents != history
            or memory.patches_per_frame != patches
            or not isinstance(memory.input_ids, torch.Tensor)
            or memory.input_ids.device.type != "cpu"
            or memory.input_ids.dtype != torch.long
            or not isinstance(memory.text_position_ids, torch.Tensor)
            or memory.text_position_ids.device.type != "cpu"
            or memory.text_position_ids.dtype != torch.float32
            or not torch.equal(memory.input_ids, input_ids.detach().cpu())
            or not torch.equal(memory.text_position_ids, text_positions.detach().cpu())
        ):
            raise ValueError("Episode memory context/text/model/geometry mismatch; reset explicitly")
        if prefix_frames or segment["prefix_video_tokens"].numel():
            raise ValueError("A continuing segment cannot supply a cold prefix")
        old_frames = _frames(memory.frame_ids, "memory.frame_ids")
        if len(old_frames) != history or len(memory.layer_inputs) != len(model.layers):
            raise ValueError("Episode memory must contain the complete bounded history and every decoder layer")
        _position(memory.position_ids, len(old_frames) * patches, device, "memory.position_ids")
        if memory.position_ids.requires_grad or memory.position_ids.grad_fn is not None:
            raise ValueError("Episode memory positions must be detached")
        for value in memory.layer_inputs:
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (len(old_frames) * patches, model.config.hidden_size)
                or value.device != device
                or not value.is_floating_point()
                or value.requires_grad
                or value.grad_fn is not None
            ):
                raise ValueError("Episode layer memory must be detached and match hidden shape/device/dtype")
        old_positions = memory.position_ids
    else:
        old_frames, old_positions = prefix_frames, segment["prefix_position_ids"]
        if len(old_frames) > history:
            raise ValueError("A cold prefix exceeds the bounded history")
    if clean_frames and old_frames and clean_frames[0] != old_frames[-1] + 1:
        raise ValueError("New observations must continue immediately after cached/prefix history")
    frames = old_frames + clean_frames
    if not frames:
        raise ValueError("A segment requires observed history")
    decisions = segment["decisions"]
    if not isinstance(decisions, tuple) or not decisions:
        raise ValueError("A segment requires a tuple of decisions")
    video_conditions, action_conditions, anchors = [], [], []
    vstop = astop = 0
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict) or any(
            type(decision.get(key)) is not int
            for key in ("anchor_frame", "video_start", "video_stop", "action_start", "action_stop", "state_row")
        ):
            raise ValueError("Decision slices, state rows and anchors must be integer metadata")
        anchor = decision["anchor_frame"]
        if type(anchor) is not int or anchor not in frames or (anchors and anchor <= anchors[-1]):
            raise ValueError("Decision anchors must be strictly increasing observed frame IDs")
        if anchor - history + 1 < frames[0]:
            raise ValueError("A decision is missing part of its complete observed history")
        if (
            decision["video_start"] != vstop
            or decision["action_start"] != astop
            or not vstop < decision["video_stop"] <= len(segment["future_video_tokens"])
            or (decision["video_stop"] - vstop) % patches
            or not astop < decision["action_stop"] <= len(segment["action_tokens"])
            or decision["state_row"] != index
        ):
            raise ValueError("Decisions must partition future/action rows and own one state each")
        video_conditions.extend(
            _conditions(decision["video_condition"], decision["video_stop"] - vstop, "video_condition")
        )
        action_conditions.extend(
            _conditions(decision["action_condition"], decision["action_stop"] - astop, "action_condition")
        )
        vstop, astop = decision["video_stop"], decision["action_stop"]
        anchors.append(anchor)
    if (
        vstop != len(segment["future_video_tokens"])
        or astop != len(segment["action_tokens"])
        or len(decisions) != len(segment["state_tokens"])
        or anchors[-1] != frames[-1]
    ):
        raise ValueError("Decision payload coverage/last observed anchor is inconsistent")
    return dict(
        device=device,
        context=context,
        history=history,
        patches=patches,
        input_ids=input_ids,
        text_positions=text_positions,
        old_frames=old_frames,
        old_positions=old_positions,
        clean_frames=clean_frames,
        frames=frames,
        decisions=decisions,
        video_conditions=tuple(video_conditions),
        action_conditions=tuple(action_conditions),
    )


def _attention_plans(meta, backend):
    """Independent clean causal stream and isolated, anchor-local decisions."""
    text = len(meta["input_ids"])
    patches, history = meta["patches"], meta["history"]
    old = len(meta["old_frames"]) * patches
    clean = len(meta["clean_frames"]) * patches
    future = len(meta["video_conditions"])
    actions = len(meta["action_conditions"])
    states = len(meta["decisions"])
    total = text + old + clean + future + actions + states
    clean_keys = {
        frame: tuple(range(text + index * patches, text + (index + 1) * patches))
        for index, frame in enumerate(meta["frames"])
    }

    def observed(anchor):
        return [key for frame in meta["frames"] if anchor - history < frame <= anchor for key in clean_keys[frame]]

    future_start, action_start, state_start = clean, clean + future, clean + future + actions
    labels = [("clean", frame, False) for frame in meta["clean_frames"] for _ in range(patches)]
    future_labels, action_labels, state_labels = [], [], []
    decision_keys = {}
    for index, decision in enumerate(meta["decisions"]):
        future_rows = range(future_start + decision["video_start"], future_start + decision["video_stop"])
        action_rows = range(action_start + decision["action_start"], action_start + decision["action_stop"])
        conditional, current = [], []
        for row, condition in zip(
            future_rows, meta["video_conditions"][decision["video_start"] : decision["video_stop"]], strict=True
        ):
            future_labels.append(("current", index, condition))
            (conditional if condition else current).append(text + old + row)
        for row, condition in zip(
            action_rows, meta["action_conditions"][decision["action_start"] : decision["action_stop"]], strict=True
        ):
            action_labels.append(("current", index, condition))
            (conditional if condition else current).append(text + old + row)
        state_key = text + old + state_start + index
        state_labels.append(("state", index, True))
        base = list(range(text)) + observed(decision["anchor_frame"])
        decision_keys[index, False] = sorted(base + conditional + current + [state_key])
        decision_keys[index, True] = sorted(base + conditional + [state_key])
        decision_keys[index, "state"] = list(range(text)) + [state_key]
    labels.extend(future_labels + action_labels + state_labels)
    groups = []
    for label, entries in groupby(enumerate(labels), key=lambda item: item[1]):
        rows = [row for row, _ in entries]
        kind, index, condition = label
        keys = (
            list(range(text)) + observed(index)
            if kind == "clean"
            else decision_keys[index, "state" if kind == "state" else condition]
        )
        groups.append((rows, keys))
    active_plan = _plan(groups, total, text, meta["device"], backend)
    prefix_groups = [
        (list(range(index * patches, (index + 1) * patches)), list(range(text)) + observed(frame))
        for index, frame in enumerate(meta["old_frames"])
    ]
    prefix_plan = _plan(prefix_groups, text + old, text, meta["device"], backend) if old else None
    return active_plan, prefix_plan


def forward_segment(model, segment, memory=None):
    """Run one shared-history segment; return output fields and a detached snapshot."""
    from .layers import rotary_tables

    meta = _prepare(model, segment, memory)
    backend = segment.get("attention_backend", "dense")
    active_plan, prefix_plan = _attention_plans(meta, backend)
    device, patches = meta["device"], meta["patches"]
    vc = torch.tensor(meta["video_conditions"], dtype=torch.bool, device=device)
    ac = torch.tensor(meta["action_conditions"], dtype=torch.bool, device=device)
    future = model.proj_in(segment["future_video_tokens"])
    future = future + model.time_embedder(
        segment["future_video_timesteps"].masked_fill(vc, 0) * model.config.timestep_scale
    )
    if len(segment["clean_video_tokens"]):
        clean = model.proj_in(segment["clean_video_tokens"])
        clean = clean + model.time_embedder(torch.zeros(len(clean), dtype=torch.float32, device=device))
    else:
        clean = future[:0]
    action = model.action_proj_in(segment["action_tokens"].masked_fill(~segment["action_mask"], 0))
    action = (
        action
        + model.action_modality_embed
        + model.time_embedder(segment["action_timesteps"].masked_fill(ac, 0) * model.config.timestep_scale)
    )
    state = model.state_proj_in(segment["state_tokens"].masked_fill(~segment["state_mask"], 0))
    state = state + model.state_modality_embed
    state = state + model.time_embedder(torch.zeros(len(state), dtype=torch.float32, device=device))
    gen, und = torch.cat((clean, future, action, state)), model.embed_tokens(meta["input_ids"])
    positions = torch.cat(
        (
            meta["text_positions"],
            meta["old_positions"],
            segment["clean_position_ids"],
            segment["future_position_ids"],
            segment["action_position_ids"],
            segment["state_position_ids"],
        ),
        dim=1,
    )
    cos, sin = rotary_tables(positions, model.config, gen.dtype)
    text, old_rows, clean_rows = len(meta["input_ids"]), len(meta["old_frames"]) * patches, len(clean)
    tables = ((cos[:text], sin[:text]), (cos[text + old_rows :], sin[text + old_rows :]))
    memory_tables = (cos[text : text + old_rows], sin[text : text + old_rows])
    if memory is None:
        with torch.no_grad():
            if old_rows:
                cold = model.proj_in(segment["prefix_video_tokens"])
                cold = cold + model.time_embedder(torch.zeros(len(cold), dtype=torch.float32, device=device))
            else:
                cold = gen[:0].detach()
    else:
        cold = None
        for value in memory.layer_inputs:
            if (
                value.shape != (old_rows, model.config.hidden_size)
                or value.dtype != gen.dtype
                or value.device != device
                or value.requires_grad
                or value.grad_fn is not None
            ):
                raise ValueError("Episode layer memory must be detached and match hidden shape/device/dtype")
    retained = min(meta["history"], len(meta["frames"]))
    keep_rows = retained * patches
    unchanged_memory = memory is not None and not meta["clean_frames"]
    snapshots = []
    for index, layer in enumerate(model.layers):
        past = cold if memory is None else memory.layer_inputs[index]
        # The snapshot is a separate, detached tensor and never changed inside
        # checkpoint recomputation; caller commits it only after a healthy step.
        if not unchanged_memory:
            snapshots.append(torch.cat((past.detach(), gen[:clean_rows].detach()))[-keep_rows:].clone())
        context = SegmentLayerContext(past, *memory_tables, prefix_plan if memory is None else None)
        if model.gradient_checkpointing and model.training and torch.is_grad_enabled():
            und, gen, cold = model._gradient_checkpointing_func(
                layer, und, gen, tables, active_plan, None, None, context
            )
        else:
            und, gen, cold = layer(und, gen, tables, active_plan, segment_context=context)
    gen = model.norm_moe_gen(gen)
    video_rows = slice(clean_rows, clean_rows + len(future))
    action_rows = slice(video_rows.stop, video_rows.stop + len(action))
    next_memory = memory
    if not unchanged_memory:
        next_memory = SegmentMemory(
            frame_ids=meta["frames"][-retained:],
            position_ids=torch.cat((meta["old_positions"], segment["clean_position_ids"]), dim=1)[:, -keep_rows:]
            .detach()
            .clone(),
            layer_inputs=tuple(snapshots),
            input_ids=meta["input_ids"].detach().cpu().clone(),
            text_position_ids=meta["text_positions"].detach().cpu().clone(),
            context_id=meta["context"],
            history_latents=meta["history"],
            patches_per_frame=patches,
            model_id=id(model),
        )
    return dict(
        video=model.proj_out(gen[video_rows]),
        action=model.action_proj_out(gen[action_rows]).masked_fill(~segment["action_mask"], 0),
        segment_memory=next_memory,
        segment_stats=dict(
            decisions=len(meta["decisions"]),
            new_clean_latents=len(meta["clean_frames"]),
            cold_prefix_latents=len(meta["old_frames"]) if memory is None else 0,
            reused_history_latents=len(meta["old_frames"]) if memory is not None else 0,
            retained_history_latents=retained,
        ),
    )
