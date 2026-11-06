# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded frozen-VAE episode state shared by training and online observation."""

import torch


class BoundedEpisodeVAE:
    """Keep convolution history and a bounded normalized-latent ring.

    Complete-prefix retention in the underlying tokenizer is disabled. A caller
    may consume the returned cold/new latents, while this owner retains only
    ``capacity`` latest latent frames and scalar stream counters.
    """

    def __init__(self, tokenizer, *, max_latents):
        self.tokenizer = tokenizer
        self.group = int(tokenizer.temporal_compression_factor)
        if self.group < 1:
            raise ValueError("VAE temporal compression must be positive")
        for name in ("stream_encode_start", "stream_encode_step", "stream_encode_abort"):
            if not callable(getattr(tokenizer, name, None)):
                raise ValueError("Episode encoding requires a streaming VAE")
        self.state = None
        self._latents = None
        self._latent_count = 0
        self.capacity = self._positive(max_latents)

    @staticmethod
    def _positive(value):
        if type(value) is not int or value < 1:
            raise ValueError("max_latents must be a positive integer")
        return value

    @property
    def active(self):
        return self.state is not None and not self.state.closed

    @property
    def frame_count(self):
        return self.state.frame_count if self.active else 0

    @property
    def latent_count(self):
        return self._latent_count

    @property
    def latents(self):
        if not self.active or self._latents is None:
            raise ValueError("The episode VAE stream has not been initialized")
        return self._latents

    @property
    def latent_ids(self):
        count = 0 if self._latents is None else self._latents.shape[2]
        return tuple(range(self._latent_count - count, self._latent_count))

    def set_capacity(self, max_latents):
        self.capacity = self._positive(max_latents)
        if self._latents is not None and self._latents.shape[2] > self.capacity:
            # A narrow view would keep the larger allocation alive.
            self._latents = self._latents[:, :, -self.capacity :].detach().clone()

    def clear(self):
        try:
            if self.state is not None:
                self.tokenizer.stream_encode_abort(self.state)
        finally:
            self.state = self._latents = None
            self._latent_count = 0

    def _validate_frames(self, frames, *, first):
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 5
            or frames.shape[1] != 3
            or min(frames.shape) < 1
            or not frames.is_floating_point()
        ):
            raise ValueError("Episode VAE expects normalized floating RGB [B,3,T,H,W]")
        count = frames.shape[2] - int(first)
        if count < 0 or count % self.group:
            raise ValueError("Episode VAE requires one first frame followed by complete temporal groups")

    def _remember(self, new):
        if new.ndim != 5 or new.shape[2] < 1:
            raise ValueError("Streaming VAE returned invalid latent frames")
        new = new.detach()
        self._latent_count += new.shape[2]
        combined = new if self._latents is None else torch.cat((self._latents, new), dim=2)
        self._latents = combined[:, :, -self.capacity :].detach().clone()

    @torch.no_grad()
    def reset(self, frames, *, keep_latents=None):
        """Encode a cold prefix and return all of it; retain only the bounded tail."""
        self.clear()
        if keep_latents is not None:
            self.set_capacity(keep_latents)
        self._validate_frames(frames, first=True)
        chunks = []
        try:
            self.state = self.tokenizer.stream_encode_start(frames[:, :, :1], retain_latents=False)
            first = self.state.last_latent
            if first is None or first.ndim != 5 or first.shape[2] != 1:
                raise ValueError("Streaming VAE did not expose its first latent")
            chunks.append(first)
            self._remember(first)
            if frames.shape[2] > 1:
                chunks.append(self.append(frames[:, :, 1:]))
            return torch.cat(chunks, dim=2).detach()
        except BaseException:
            self.clear()
            raise

    @torch.no_grad()
    def append(self, frames):
        """Encode complete new groups; return only the new normalized latents."""
        if not self.active:
            raise ValueError("Initialize the episode VAE stream before appending")
        try:
            self._validate_frames(frames, first=False)
            new = self.tokenizer.stream_encode_step(self.state, frames).detach()
            if new.ndim != 5 or new.shape[2] != frames.shape[2] // self.group:
                raise ValueError("Streaming VAE returned an unexpected temporal latent count")
            self._remember(new)
            return new
        except BaseException:
            self.clear()
            raise
