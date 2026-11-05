# SPDX-License-Identifier: OpenMDW-1.1
"""Exact, episode-owned Wan convolution checkpoints for repeated reset pixels."""

from dataclasses import replace

import torch

from .episode_prefix import model_signature


def _fork(state, latents):
    return replace(
        state,
        closed=False,
        frame_count=1 + (len(latents) - 1) * 4 if latents else state.frame_count,
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
        return _fork(state, self.latents[:count])

    def clear(self):
        for state in self.states:
            self.tokenizer.stream_encode_abort(state)
        self.states.clear()
        self.latents.clear()
