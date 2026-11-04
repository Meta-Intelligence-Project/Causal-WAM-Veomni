# SPDX-License-Identifier: OpenMDW-1.1
"""Window timing, true-first padding, anchor changes and isolated VAE lifetime."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.diffusers.causal_cosmos3.serving.window_streaming import ObservationWindowEncoder


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Tokenizer:
    temporal_compression_factor = 4

    def __init__(self):
        self.events = []

    def stream_encode_start(self, video):
        self.events.append(("start", video.clone()))
        return SimpleNamespace(parts=[video[:, :1].mean(dim=(-1, -2), keepdim=True)], closed=False)

    def stream_encode_step(self, state, video):
        assert not state.closed and video.shape[2] == 4
        self.events.append(("step", video.clone()))
        state.parts.append(video[:, :1].mean(dim=(-1, -2), keepdim=True).mean(dim=2, keepdim=True))

    def stream_encode_finish(self, state):
        result = torch.cat(state.parts, dim=2)
        self.stream_encode_abort(state)
        return result

    def stream_encode_abort(self, state):
        state.closed = True
        state.parts.clear()


def image(tick):
    return torch.full((3, 16, 16), tick % 255, dtype=torch.uint8)


def encoder():
    model = SimpleNamespace(
        tokenizer_vision_gen=Tokenizer(), tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32}
    )
    return ObservationWindowEncoder(model.tokenizer_vision_gen, history_blocks=1, history_padding="repeat_first")


def expected(anchor):
    values = torch.tensor([max(0, t) for t in range(anchor - 32, anchor + 1, 4)]) / 127.5 - 1
    return torch.cat([values[:1], values[1:].reshape(-1, 4).mean(dim=1)])


def test_encoding_starts_before_trigger_and_group_tail_waits_for_real_frames():
    session = encoder()
    session.add(0, image(0))
    session.prepare(16)
    assert session.count == 5
    assert [event[0] for event in session.tokenizer.events] == ["start", "step"]
    for tick in range(1, 16):
        session.add(tick, image(tick))
    assert session.count == 5  # sampled 4/8/12 waits for observed 16, never pads the future.
    session.add(16, image(16))
    latent, timing = session.finish(16)
    torch.testing.assert_close(latent.flatten(), expected(16))
    assert timing["streaming_window_rebuilt"] is False
    assert session.state is None


def test_explicit_48_step_history_is_not_rounded_to_the_action_horizon():
    model = SimpleNamespace(
        tokenizer_vision_gen=Tokenizer(), tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32}
    )
    session = ObservationWindowEncoder(
        model.tokenizer_vision_gen, history_steps=48, action_horizon=32, history_padding="repeat_first"
    )
    session.add(0, image(0))
    session.prepare(16)
    assert session.window.window_start_tick == -32 and session.count == 9
    for tick in range(1, 17):
        session.add(tick, image(tick))
    latent, _ = session.finish(16)
    assert latent.shape == (1, 4, 1, 1)


def test_actual_trigger_reanchors_and_reencodes_overlap_at_its_own_phase():
    session = encoder()
    for tick in range(18):
        session.add(tick, image(tick))
    session.prepare(16)
    old = session.state
    latent, timing = session.finish(17)
    assert old.closed and timing["streaming_window_rebuilt"]
    assert timing["streaming_reanchors"] == 1
    torch.testing.assert_close(latent.flatten(), expected(17))


def test_episode_first_image_is_pinned_and_positive_missing_frames_are_errors():
    session = encoder()
    session.add(0, image(0))
    session.add(16, image(16))
    with pytest.raises(ValueError, match="missing real"):
        session.finish(16)
    with pytest.raises(ValueError, match="overwritten"):
        session.add(0, image(1))
    session.clear()
    assert session.first_image is None and session.frames == {}


def test_buffer_is_bounded_and_clear_closes_an_in_progress_session():
    session = encoder()
    for tick in range(101):
        session.add(tick, image(tick))
    assert len(session.frames) <= 65 and session.first_image[0, 0, 0].item() == 0
    session.prepare(100)
    state = session.state
    session.clear()
    assert state.closed


def test_a_delayed_request_can_reconstruct_its_frozen_snapshot_after_buffer_eviction():
    session = encoder()
    for tick in range(101):
        session.add(tick, image(tick))
    assert 4 not in session.frames
    for tick in range(0, 17, 4):
        session.add(tick, image(tick), prune=False)
    latent, _ = session.finish(16)
    torch.testing.assert_close(latent.flatten(), expected(16))
    session.prune()
    assert len(session.frames) <= 65


def test_missing_frames_discard_partial_vae_but_preserve_rgb_for_a_later_retry():
    session = encoder()
    session.add(0, image(0))
    session.prepare(16)
    old = session.state
    with pytest.raises(ValueError, match="missing real"):
        session.finish(16)
    assert old.closed and session.state is session.anchor is session.window is None
    for tick in range(1, 17):
        session.add(tick, image(tick))
    latent, _ = session.finish(16)
    torch.testing.assert_close(latent.flatten(), expected(16))


def test_encoder_failure_does_not_leave_a_closed_state_attached_to_the_window(monkeypatch):
    session = encoder()
    session.add(0, image(0))
    original = session.tokenizer.stream_encode_step
    failed = []

    def fail(state, video):
        failed.append(state)
        session.tokenizer.stream_encode_abort(state)
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(session.tokenizer, "stream_encode_step", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        session.prepare(16)
    assert failed[0].closed and session.state is session.anchor is session.window is None
    assert session.first_image is not None and 0 in session.frames
    monkeypatch.setattr(session.tokenizer, "stream_encode_step", original)
    session.prepare(16)
    assert session.count == 5 and not session.state.closed
