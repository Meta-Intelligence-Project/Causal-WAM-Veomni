# SPDX-License-Identifier: OpenMDW-1.1
"""Measured observation windows; all methods run on the single model worker."""

import torch

from veomni.models.diffusers.causal_cosmos3.serving.history_window import build_history_window


class ObservationWindowEncoder:
    """Keep real RGB separately from a disposable, window-scoped VAE state."""

    def __init__(
        self,
        vae,
        *,
        history_blocks=None,
        history_steps=None,
        action_horizon=32,
        history_padding,
        actions_per_block=32,
        video_stride=4,
    ):
        self.history_blocks, self.history_padding = history_blocks, history_padding
        self.history_steps = history_steps if history_steps is not None else history_blocks * actions_per_block
        self.action_horizon = action_horizon
        self.explicit_steps = history_steps is not None
        self.actions_per_block, self.video_stride = actions_per_block, video_stride
        self.tokenizer = vae
        if not all(
            callable(getattr(self.tokenizer, name, None))
            for name in ("stream_encode_start", "stream_encode_step", "stream_encode_finish", "stream_encode_abort")
        ):
            raise ValueError("streaming_vae requires a streaming vision tokenizer")
        self.group = int(self.tokenizer.temporal_compression_factor)
        self.clear()

    def clear(self):
        self._discard_window()
        self.frames = {}
        self.first_image = None
        self.latest_tick = -1
        self.reanchors = 0

    def _discard_window(self):
        state = getattr(self, "state", None)
        self.state = self.window = self.anchor = None
        self.count = 0
        if state is not None:
            self.tokenizer.stream_encode_abort(state)

    def _image(self, image):
        image = torch.as_tensor(image).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError("Observed images must be CPU uint8 [3,H,W]")
        if self.first_image is not None and image.shape != self.first_image.shape:
            raise ValueError("Observed image geometry changed; reset is required")
        return image.clone()

    def add(self, tick, image, *, prune=True):
        if type(tick) is not int or tick < 0:
            raise ValueError("Observation tick must be a nonnegative integer")
        image = self._image(image)
        if tick in self.frames and not torch.equal(self.frames[tick], image):
            raise ValueError("A measured image cannot be overwritten at an existing tick")
        if tick == 0:
            if self.first_image is not None and not torch.equal(self.first_image, image):
                raise ValueError("The episode's first image is immutable")
            self.first_image = image
        self.frames[tick] = image
        self.latest_tick = max(self.latest_tick, tick)
        if prune:
            self.prune()
        self._feed_available()

    def prune(self):
        # A bounded extra action horizon permits a delayed RTC submission's
        # frozen anchor to be reconstructed after newer observations arrive.
        capacity = self.history_steps + self.action_horizon
        self.frames = {t: frame for t, frame in self.frames.items() if t >= self.latest_tick - capacity}

    def _normalized(self, images):
        mean = getattr(self.tokenizer, "latent_mean", None)
        device = mean.device if mean is not None else torch.device("cpu")
        if device.type != "cpu":
            getattr(torch, device.type).set_device(device)
        return torch.stack(images, dim=1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0

    def prepare(self, anchor):
        if type(anchor) is not int or anchor < 0:
            raise ValueError("Window anchor must be a nonnegative integer")
        if anchor == self.anchor:
            self._feed_available()
            return
        if self.state is not None:
            self.reanchors += 1
        self._discard_window()
        self.anchor = anchor
        self.window = build_history_window(
            anchor,
            **(
                dict(history_steps=self.history_steps)
                if self.explicit_steps
                else dict(history_blocks=self.history_blocks)
            ),
            history_padding=self.history_padding,
            actions_per_block=self.actions_per_block,
            video_stride=self.video_stride,
        )
        self._feed_available()

    def _frame(self, tick):
        return self.first_image if tick == 0 else self.frames.get(tick)

    @torch.no_grad()
    def _feed_available(self):
        if self.window is None:
            return
        try:
            ticks = self.window.observation_ticks
            while self.count < len(ticks):
                size = 1 if self.count == 0 else self.group
                selected = ticks[self.count : self.count + size]
                if len(selected) != size:
                    break
                frames = [self._frame(t) for t in selected]
                if any(frame is None for frame in frames):
                    break
                chunk = self._normalized(frames)
                if self.state is None:
                    self.state = self.tokenizer.stream_encode_start(chunk)
                else:
                    self.tokenizer.stream_encode_step(self.state, chunk)
                self.count += size
        except Exception:
            self._discard_window()
            raise

    @torch.no_grad()
    def finish(self, anchor):
        matched = self.anchor == anchor
        prepared = self.count if matched else 0
        try:
            self.prepare(anchor)
            if self.count != len(self.window.observation_ticks):
                missing = [t for t in self.window.required_observation_ticks if self._frame(t) is None]
                raise ValueError(f"missing real historical video observations at ticks {missing}")
            latent = self.tokenizer.stream_encode_finish(self.state)
        finally:
            self._discard_window()
        return latent.squeeze(0), dict(
            sampled_frames_encoded_before_predict=prepared,
            streaming_window_rebuilt=not matched,
            streaming_reanchors=self.reanchors,
        )
