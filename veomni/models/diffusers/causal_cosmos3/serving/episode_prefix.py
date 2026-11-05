# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded episode-owned snapshots of an unchanged causal visual prefix."""

import threading
from contextlib import contextmanager

import torch


def model_signature(model):
    """Include mutable weights/modules and the numerical execution context."""
    device = next(model.parameters()).device.type
    return (
        id(model),
        tuple(
            (name, id(value), value.data_ptr(), value._version, str(value.dtype), str(value.device))
            for name, value in (*model.named_parameters(), *model.named_buffers())
        ),
        tuple((name, id(module), module.training) for name, module in model.named_modules()),
        repr(getattr(model, "config", None)),
        torch.is_autocast_enabled(device),
        str(torch.get_autocast_dtype(device)),
        torch.is_inference_mode_enabled(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


class EpisodePrefixCache:
    """Retain one immutable text/history snapshot per CFG branch, until clear().

    Each lookup verifies weights, text, layout, pixels' encoded tokens and RoPE.
    Reuse stops before the first changed *whole latent*: spatial patches within
    a latent attend bidirectionally. Request-owned buffers never alias snapshots.
    The serialized serving worker owns this object; it is not a model global.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.clear()

    def clear(self):
        self.entries = {}

    @contextmanager
    def request(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or reentrant episode prefix requests are unsupported")
        try:
            yield self
        except BaseException:
            self.clear()
            raise
        finally:
            self._lock.release()

    def match(self, entry, branch):
        if branch not in ("cond", "uncond"):
            raise ValueError("Episode prefix entries must belong to cond or uncond CFG branches")
        source = self.entries.get(branch)
        signature = model_signature(entry.model)
        if source is None:
            return None, 0, signature
        inputs = entry.inputs
        if (
            source["signature"] != signature
            or source["layout"] != inputs["layout"]
            or not torch.equal(source["input_ids"], inputs["input_ids"])
        ):
            del self.entries[branch]
            return None, 0, signature
        count = len(source["video_tokens"])
        frames = inputs["layout"].decision_windows[0]
        text = len(inputs["input_ids"])
        same_tokens = (source["video_tokens"] == inputs["video_tokens"][:count]).all(dim=-1)
        same_positions = (source["position_ids"] == inputs["position_ids"][:, : text + count]).all(dim=0)
        if not same_positions[:text].all().item():
            return None, 0, signature
        same = (same_tokens & same_positions[text:]).reshape(frames, -1).all(dim=1)
        shared = int(same.to(torch.int64).cumprod(dim=0).sum().item())
        return source, shared * (count // frames), signature

    def remember(self, entry, branch, signature):
        if branch in self.entries:
            return
        inputs, layout = entry.inputs, entry.inputs["layout"]
        video_count = sum(role == 0 for role in layout.roles)
        end = len(inputs["input_ids"]) + video_count
        self.entries[branch] = dict(
            signature=signature,
            layout=layout,
            input_ids=inputs["input_ids"].detach().clone(),
            video_tokens=inputs["video_tokens"][:video_count].detach().clone(),
            position_ids=inputs["position_ids"][:, :end].detach().clone(),
            video_output=entry.reference.video[:video_count].detach().clone(),
            layers={
                index: {name: value[:end].detach().clone() for name, value in layer.items()}
                for index, layer in entry.layers.items()
            },
        )
