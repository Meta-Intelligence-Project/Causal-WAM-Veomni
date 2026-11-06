# SPDX-License-Identifier: OpenMDW-1.1
"""Inference-owned recurrent clean memory with fresh future-token autograd."""

import torch

from ..cosmos3_core.segment import SegmentMemory
from ..cosmos3_core.segment_inputs import compact_stream_payload
from ..cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Output


def _frame_ids(value, name):
    if not isinstance(value, tuple) or any(type(frame) is not int for frame in value):
        raise ValueError(f"{name} must be a tuple of integer latent frame IDs")
    if any(right != left + 1 for left, right in zip(value, value[1:])):
        raise ValueError(f"{name} must contain consecutive latent frames")
    return value


class EpisodePrefixCache:
    """One request/CFG branch over a policy-owned bounded ``SegmentMemory``.

    Only observed clean hidden states survive calls. Every future-token forward,
    including the first RTC velocity, builds a fresh graph when gradients are
    enabled. A delayed request can cold-rebuild its measured prefix without
    replacing a newer policy-owned memory snapshot.
    """

    def __init__(
        self,
        model,
        sample,
        inputs,
        span,
        backend,
        *,
        stream_context,
        stream_memories,
        branch="cond",
        cache_mode="full",
        position_options=None,
    ):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Episode prefix caching requires frozen eval-mode model weights")
        if not isinstance(stream_context, dict) or not isinstance(stream_memories, dict):
            raise ValueError("episode_stream inference requires stream_context and persistent stream_memories")
        context = stream_context.get("context_id")
        if not isinstance(context, str) or not context or branch not in ("cond", "uncond"):
            raise ValueError("Episode inference requires a nonempty context_id and a CFG branch")
        self.ids = _frame_ids(stream_context.get("prefix_global_ids"), "prefix_global_ids")
        self.new_ids = _frame_ids(stream_context.get("new_frame_ids", ()), "new_frame_ids")
        observed = sample["observed_latents"]
        if len(self.ids) != observed:
            raise ValueError("prefix_global_ids must identify the complete measured history")
        self.new_latents = stream_context.get("new_latents")
        shape = sample["video_latents"].shape
        if (
            not isinstance(self.new_latents, torch.Tensor)
            or self.new_latents.ndim != 4
            or self.new_latents.shape != (shape[0], len(self.new_ids), *shape[2:])
            or not self.new_latents.is_floating_point()
        ):
            raise ValueError("new_latents must match the supplied observation frame IDs and geometry")
        self.model, self.sample, self.span = model, sample, span
        self.backend, self.branch = backend, branch
        self.memories = stream_memories
        self.context_id = f"{context}/{branch}"
        self.position_options = dict(stream_context.get("position_options") or {})
        self.position_options.update(position_options or {})
        self.position_options["patch_size"] = model.config.patch_size
        self.memory = stream_memories.get(branch)
        if self.memory is not None and (
            not isinstance(self.memory, SegmentMemory)
            or self.memory.context_id != self.context_id
            or self.memory.model_id != id(model)
            or self.memory.history_latents != observed
            or len(self.memory.frame_ids) != observed
            or not isinstance(self.memory.input_ids, torch.Tensor)
            or self.memory.input_ids.device.type != "cpu"
            or not torch.equal(self.memory.input_ids, inputs["input_ids"].detach().cpu())
        ):
            raise ValueError("Episode memory context/text changed; reset the persistent stream memories")
        if self.memory is not None:
            _frame_ids(self.memory.frame_ids, "memory.frame_ids")
        delayed = self.memory is not None and self.memory.frame_ids[-1] > self.ids[-1]
        self.persist = not delayed
        if delayed:
            self.memory = None
        patches = (span.current_video_rows.stop - span.current_video_rows.start) // shape[1]
        self.future_rows = (shape[1] - observed) * patches
        self.stats = dict(
            prefill_calls=0,
            replay_calls=0,
            model_calls=0,
            active_tokens=self.future_rows + len(sample["action"]) + 1,
            prefilled_history_latents=0,
            reused_history_latents=0,
            retained_history_latents=observed,
            delayed_cold_rebuild=int(delayed),
            maintenance_calls=0,
            cache_mode=cache_mode,
        )

    def __call__(self, inputs):
        history = self.sample["video_latents"][:, : self.sample["observed_latents"]].detach()
        if self.memory is None:
            prefix, prefix_ids, clean, clean_ids = history, self.ids, history[:, :0], ()
            reused = 0
        else:
            prefix, prefix_ids = history[:, :0], ()
            clean_ids = tuple(range(self.memory.frame_ids[-1] + 1, self.ids[-1] + 1))
            if len(clean_ids) > self.model.config.tbptt_keep_history_latents:
                raise ValueError("New episode observations exceed the segment length; reset the stream")
            lookup = {frame: index for index, frame in enumerate(self.new_ids)}
            if any(frame not in lookup for frame in clean_ids):
                raise ValueError("Episode observation updates must continue cached history without missing frames")
            selected = torch.tensor([lookup[frame] for frame in clean_ids], device=self.new_latents.device)
            clean = self.new_latents.index_select(1, selected.to(dtype=torch.long)).detach()
            reused = len(self.memory.frame_ids)
        payload = compact_stream_payload(
            [self.sample],
            inputs,
            [self.span],
            prefix_latents=prefix,
            prefix_frame_ids=prefix_ids,
            clean_latents=clean,
            clean_frame_ids=clean_ids,
            anchor_frame_ids=(self.ids[-1],),
            context_id=self.context_id,
            history_latents=len(self.ids),
            targets=None,
            position_options=self.position_options,
        )
        output = self.model(stream_segment=payload, stream_memory=self.memory, attention_backend=self.backend)
        candidate = output.segment_memory
        if not isinstance(candidate, SegmentMemory):
            raise ValueError("Episode model did not return a clean memory snapshot")
        self.memory = candidate
        if self.persist:
            self.memories[self.branch] = candidate
        self.stats["model_calls"] += 1
        if prefix_ids or clean_ids:
            self.stats["prefill_calls"] += 1
            self.stats["prefilled_history_latents"] += len(prefix_ids) + len(clean_ids)
        else:
            self.stats["replay_calls"] += 1
        self.stats["reused_history_latents"] = max(self.stats["reused_history_latents"], reused)
        video = output.video.new_zeros(len(inputs["video_tokens"]), output.video.shape[-1])
        start = self.span.current_video_rows.stop - self.future_rows
        video = video.index_copy(0, torch.arange(start, start + self.future_rows, device=video.device), output.video)
        return Cosmos3Output(
            video=video,
            action=output.action,
            segment_memory=candidate,
            segment_stats=output.segment_stats,
        )
