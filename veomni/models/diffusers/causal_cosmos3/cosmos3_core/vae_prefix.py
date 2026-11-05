# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable frozen-VAE prefixes with caller-owned convolution forks."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace

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
        repr(getattr(getattr(model, "vae", None), "config", None)),
        torch.is_autocast_enabled(device),
        str(torch.get_autocast_dtype(device)),
        torch.is_inference_mode_enabled(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        (
            torch.npu.conv.allow_hf32,
            torch.npu.matmul.allow_hf32,
            repr(getattr(torch.npu.matmul, "cube_math_type", None)),
        )
        if device == "npu"
        else (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        if device == "cuda"
        else None,
    )


def _fork(state, latents, *, group=4):
    return replace(
        state,
        closed=False,
        frame_count=1 + (len(latents) - 1) * group if latents else state.frame_count,
        feat_cache=[None if value is None else value.detach().clone() for value in state.feat_cache],
        latent_chunks=list(latents),
    )


def _same_features(left, right):
    if len(left) != len(right):
        return False
    checks = []
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            if a is not b:
                return False
        elif a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return False
        else:
            checks.append((a == b).all())
    return bool(checks) and torch.stack(checks).all().item()


class RepeatedVAEPrefix:
    """Retain finite repeat-prefix checkpoints; never share writable conv state.

    Wan's transition uses only the feature cache and the next four pixels. If
    two successive caches are bitwise identical for identical repeat pixels,
    that transition has reached a fixed point. Later identical groups then use
    the same state/output exactly, without retaining duplicate checkpoints.
    No receptive-field length or approximate-convergence threshold is assumed.
    """

    @torch.no_grad()
    def __init__(self, tokenizer, first, latent_count):
        if tokenizer.vae.training or any(parameter.requires_grad for parameter in tokenizer.parameters()):
            raise ValueError("Repeated VAE prefixes require frozen eval-mode weights")
        self.tokenizer, self.signature = tokenizer, model_signature(tokenizer)
        self.states, self.latents = [], []
        state = None
        try:
            state = tokenizer.stream_encode_start(first)
            self.latents.append(state.latent_chunks[-1])
            self.states.append(_fork(state, []))
            group = first.expand(-1, -1, tokenizer.temporal_compression_factor, -1, -1).contiguous()
            for _ in range(1, latent_count):
                tokenizer.stream_encode_step(state, group)
                self.latents.append(state.latent_chunks[-1])
                if _same_features(state.feat_cache, self.states[-1].feat_cache):
                    self.latents.extend([self.latents[-1]] * (latent_count - len(self.latents)))
                    break
                self.states.append(_fork(state, []))
        except BaseException:
            self.clear()
            raise
        finally:
            if state is not None:
                tokenizer.stream_encode_abort(state)

    def valid(self):
        return bool(self.states) and self.signature == model_signature(self.tokenizer)

    def fork(self, count):
        if not 1 <= count <= len(self.latents):
            raise ValueError("Repeated VAE prefix must contain complete available latent groups")
        state = self.states[min(count, len(self.states)) - 1]
        self.tokenizer._validate_stream_state(state)
        return _fork(state, self.latents[:count], group=self.tokenizer.temporal_compression_factor)

    def clear(self):
        for state in self.states:
            self.tokenizer.stream_encode_abort(state)
        self.states.clear()
        self.latents.clear()


@dataclass
class _TrainingPrefix:
    first: torch.Tensor
    states: dict = field(default_factory=dict)
    latents: list = field(default_factory=list)

    @property
    def nbytes(self):
        tensors = [self.first, *self.latents]
        tensors.extend(value for state in self.states.values() for value in state.feat_cache if value is not None)
        return sum(
            {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in tensors}.values()
        )


def _snapshot(state, previous):
    """Deduplicate exactly equal immutable features, cloning all changed ones."""
    if previous is None:
        return _fork(state, [])
    checks, indexes = [], []
    for index, (value, old) in enumerate(zip(state.feat_cache, previous.feat_cache, strict=True)):
        if value is not None and old is not None and value.shape == old.shape and value.dtype == old.dtype:
            checks.append((value == old).all())
            indexes.append(index)
    same = dict(zip(indexes, torch.stack(checks).cpu().tolist(), strict=True)) if checks else {}
    return replace(
        state,
        feat_cache=[
            None if value is None else previous.feat_cache[index] if same.get(index) else value.detach().clone()
            for index, value in enumerate(state.feat_cache)
        ],
        latent_chunks=[],
    )


class DecisionVAEPrefixCache:
    """Bounded rank-local LRU, learned from normal first-frame/four-frame calls.

    Only complete groups of declared reset padding are reusable. Snapshots are
    sparse (one every eight latents plus the longest prefix), and every working
    convolution state is cloned. Cold samples follow the original encode path;
    no separate full-prefix prefill or fixed-point assumption is required.
    """

    def __init__(self, tokenizer, *, max_entries, max_bytes, checkpoint_stride=8):
        if any(type(value) is not int or value < 1 for value in (max_entries, max_bytes, checkpoint_stride)):
            raise ValueError("VAE prefix entry/byte limits and checkpoint stride must be positive integers")
        self.tokenizer = tokenizer
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.checkpoint_stride = checkpoint_stride
        self.entries = OrderedDict()
        self.recent = OrderedDict()
        self.signature = None

    @property
    def nbytes(self):
        return sum(entry.nbytes for entry in self.entries.values())

    def clear(self):
        for entry in self.entries.values():
            for state in entry.states.values():
                self.tokenizer.stream_encode_abort(state)
        self.entries.clear()
        self.recent.clear()
        self.signature = None

    def _observe(self, entry, state, count):
        current = len(state.latent_chunks)
        if current > count:
            return
        if current > len(entry.latents):
            entry.latents.extend(state.latent_chunks[len(entry.latents) :])
        regular = (current - 1) % self.checkpoint_stride == 0
        if current not in entry.states and (regular or current == count == len(entry.latents)):
            previous = entry.states[max(entry.states)] if entry.states else None
            snapshot = _snapshot(state, previous)
            # Keep a single non-grid endpoint, not one for every sampled anchor.
            for old in list(entry.states):
                if (old - 1) % self.checkpoint_stride and old < current:
                    self.tokenizer.stream_encode_abort(entry.states.pop(old))
            entry.states[current] = snapshot

    @torch.no_grad()
    def encode(self, sample):
        try:
            return self._encode(sample)
        except BaseException:
            self.clear()
            raise

    def _encode(self, sample):
        vae, video = self.tokenizer, sample["video"]
        window, ticks = sample.get("window"), sample.get("video_frame_ticks")
        if window is None or not ticks or not sample.get("subset"):
            return vae.encode_decision_window(video)
        leading = 0
        for tick in ticks:
            if tick != 0:
                break
            leading += 1
        group = vae.temporal_compression_factor
        count = 1 + (leading - 1) // group if leading else 0
        if count < 2:
            return vae.encode_decision_window(video)
        key = (sample["subset"], window.episode, tuple(video.shape[2:]))
        repeated = key in self.recent
        self.recent.pop(key, None)
        self.recent[key] = None
        while len(self.recent) > self.max_entries:
            self.recent.popitem(last=False)
        # Cold episodes never read snapshots, so avoid even a model-signature
        # scan until the episode has a reuse opportunity.
        if key not in self.entries and not repeated:
            return vae.encode_decision_window(video)
        state = None
        try:
            vae.vae.eval()
            if any(parameter.requires_grad for parameter in vae.parameters()):
                raise ValueError("Decision VAE prefix reuse requires frozen weights")
            signature = model_signature(vae)
            if signature != self.signature:
                self.clear()
                self.signature = signature
                self.recent[key] = None
            pixels = vae._decision_pixels(video)
            if len(ticks) != video.shape[1] or count > sample["observed_latents"]:
                raise ValueError("Reset prefix metadata must describe the observed video")
            end = 1 + (count - 1) * group
            first = video[:, :1]
            if not torch.equal(video[:, :end], first.expand(-1, end, -1, -1)):
                raise ValueError("Declared reset padding must repeat the actual reset pixels")
            entry = self.entries.pop(key, None)
            if entry is not None and not torch.equal(entry.first, first):
                for snapshot in entry.states.values():
                    vae.stream_encode_abort(snapshot)
                entry = None
            if entry is None:
                entry = _TrainingPrefix(first.detach().clone())
            self.entries[key] = entry
            if entry.states:
                shared = max(index for index in entry.states if index <= count)
                state = _fork(entry.states[shared], entry.latents[:shared], group=group)
            else:
                state = vae.stream_encode_start(pixels[:, :, :1])
                self._observe(entry, state, count)
            for start in range(state.frame_count, video.shape[1], group):
                vae.stream_encode_step(state, pixels[:, :, start : start + group])
                self._observe(entry, state, count)
            result = vae.stream_encode_finish(state).squeeze(0)
            while len(self.entries) > self.max_entries or self.nbytes > self.max_bytes:
                _, removed = self.entries.popitem(last=False)
                for snapshot in removed.states.values():
                    vae.stream_encode_abort(snapshot)
            return result
        finally:
            if state is not None:
                vae.stream_encode_abort(state)
