"""Bounded episode VAE state and shared online/training temporal geometry."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers import test_causal_cosmos3_vae
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.episode_vae import BoundedEpisodeVAE
from veomni.models.diffusers.causal_cosmos3.serving.episode_streaming import EpisodeObservationEncoder


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
encoder = test_causal_cosmos3_vae.encoder


class Tokenizer:
    """Causal recurrence makes accidental per-anchor resets observable."""

    temporal_compression_factor = 4

    def __init__(self):
        self.events = []

    def stream_encode_start(self, frames, *, retain_latents=True):
        first = frames[:, :1].mean(dim=(-1, -2), keepdim=True)
        self.events.append(("start", frames.shape[2]))
        return SimpleNamespace(
            closed=False,
            frame_count=1,
            feature=first.clone(),
            last_latent=first,
            latent_chunks=[first] if retain_latents else [],
            retain_latents=retain_latents,
        )

    def stream_encode_step(self, state, frames):
        if state.closed:
            raise ValueError("closed")
        self.events.append(("step", frames.shape[2]))
        values = []
        for start in range(0, frames.shape[2], 4):
            value = frames[:, :1, start : start + 4].mean(dim=(-1, -2), keepdim=True).mean(dim=2, keepdim=True)
            state.feature = state.feature * 0.75 + value
            state.last_latent = state.feature.clone()
            state.frame_count += 4
            if state.retain_latents:
                state.latent_chunks.append(state.last_latent)
            values.append(state.last_latent)
        return torch.cat(values, dim=2)

    def stream_encode_abort(self, state):
        state.closed = True
        state.feature = state.last_latent = None
        state.latent_chunks.clear()


def reference(frames):
    values = frames[:, :1].mean(dim=(-1, -2), keepdim=True)
    hidden = values[:, :, :1]
    outputs = [hidden]
    for start in range(1, values.shape[2], 4):
        hidden = hidden * 0.75 + values[:, :, start : start + 4].mean(dim=2, keepdim=True)
        outputs.append(hidden)
    return torch.cat(outputs, dim=2)


def image(tick):
    return torch.full((3, 16, 16), tick % 255, dtype=torch.uint8)


def pixels(ticks):
    return torch.stack([image(max(0, tick)) for tick in ticks], dim=1).unsqueeze(0).float() / 127.5 - 1


def test_bounded_episode_matches_full_prefix_without_accumulating_latent_chunks():
    tokenizer = Tokenizer()
    stream = BoundedEpisodeVAE(tokenizer, max_latents=3)
    frames = torch.linspace(-1, 1, 25).reshape(1, 1, 25, 1, 1).expand(-1, 3, -1, 16, 16)
    cold = stream.reset(frames[:, :, :9])
    parts = [cold]
    for start in range(9, 25, 4):
        parts.append(stream.append(frames[:, :, start : start + 4]))
        assert stream.latents.shape[2] <= 3
        assert stream.state.latent_chunks == []
        assert not stream.latents.requires_grad and stream.latents.grad_fn is None
        assert stream.latents.untyped_storage().nbytes() == stream.latents.numel() * stream.latents.element_size()
    expected = reference(frames)
    torch.testing.assert_close(torch.cat(parts, dim=2), expected)
    torch.testing.assert_close(stream.latents, expected[:, :, -3:])
    assert stream.frame_count == 25 and stream.latent_count == 7
    assert stream.latent_ids == (4, 5, 6)
    assert tokenizer.events.count(("start", 1)) == 1


def test_bounded_episode_capacity_clear_reset_and_interleaved_owners():
    tokenizer = Tokenizer()
    left, right = BoundedEpisodeVAE(tokenizer, max_latents=5), BoundedEpisodeVAE(tokenizer, max_latents=2)
    a, b = pixels(range(17)), pixels(range(20, 29))
    left.reset(a)
    right.reset(b[:, :, :5])
    right.append(b[:, :, 5:])
    torch.testing.assert_close(left.latents, reference(a))
    torch.testing.assert_close(right.latents, reference(b)[:, :, -2:])
    left.set_capacity(2)
    assert left.latent_ids == (3, 4) and left.latent_count == 5
    assert left.latents.untyped_storage().nbytes() == left.latents.numel() * left.latents.element_size()
    old = left.state
    left.clear()
    assert old.closed and old.feature is None and not left.active
    assert left.frame_count == left.latent_count == 0 and left.latent_ids == ()
    with pytest.raises(ValueError, match="initialized"):
        _ = left.latents
    torch.testing.assert_close(left.reset(a[:, :, :5], keep_latents=1), reference(a[:, :, :5]))
    assert left.capacity == 1 and left.latent_ids == (1,)
    assert right.active


@pytest.mark.parametrize("capacity", [0, -1, True, None, 1.5])
def test_bounded_episode_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        BoundedEpisodeVAE(Tokenizer(), max_latents=capacity)


def test_bounded_episode_invalid_append_releases_stream_and_other_owner_survives():
    tokenizer = Tokenizer()
    stream, other = BoundedEpisodeVAE(tokenizer, max_latents=3), BoundedEpisodeVAE(tokenizer, max_latents=3)
    stream.reset(pixels(range(5)))
    other.reset(pixels(range(5)))
    state = stream.state
    with pytest.raises(ValueError, match="complete temporal groups"):
        stream.append(pixels(range(3)))
    assert state.closed and stream.state is None and stream.latent_count == 0
    assert other.active
    with pytest.raises(ValueError, match="Initialize"):
        stream.append(pixels(range(4)))
    with pytest.raises(ValueError, match="complete temporal groups"):
        stream.reset(pixels(range(4)))
    assert not stream.active


def test_bounded_episode_rejects_inconsistent_tokenizer_temporal_counts(monkeypatch):
    tokenizer = Tokenizer()
    stream = BoundedEpisodeVAE(tokenizer, max_latents=3)
    stream.reset(pixels(range(5)))
    state = stream.state
    original = tokenizer.stream_encode_step

    def wrong_count(owner, frames):
        return original(owner, frames).repeat_interleave(2, dim=2)

    monkeypatch.setattr(tokenizer, "stream_encode_step", wrong_count)
    with pytest.raises(ValueError, match="temporal latent count"):
        stream.append(pixels(range(4)))
    assert state.closed and not stream.active and stream.latent_count == 0


def test_online_episode_encodes_each_group_once_and_matches_training_time_axis():
    tokenizer = Tokenizer()
    online = EpisodeObservationEncoder(tokenizer, history_steps=48, action_horizon=32)
    online.add(0, image(0))
    previous = 0
    for anchor in range(0, 129, 16):
        online.prepare(anchor)
        for tick in range(previous + 1, anchor + 1):
            online.add(tick, image(tick))
        history, _ = online.finish(anchor)
        expected = reference(pixels(range(-48, anchor + 1, 4)))
        torch.testing.assert_close(history, expected.squeeze(0)[:, -4:])
        target = (anchor + 48) // 16
        assert online.prefix_global_ids == tuple(range(target - 3, target + 1))
        assert online.new_frame_ids == ((0, 1, 2, 3) if anchor == 0 else (target,))
        assert online.last_new_latents.shape[1] == (4 if anchor == 0 else 1)
        assert online.count == 13 and online.episode.latents.shape[2] <= 6
        assert online.state.latent_chunks == []
        assert len(online.frames) <= 81
        previous = anchor
    assert online.root_origin == online.stream_origin_tick == -48
    assert online.latent_count == 12
    assert sum(kind == "start" for kind, _ in tokenizer.events) == 1
    assert sum(size for _, size in tokenizer.events) == 45
    training = BoundedEpisodeVAE(Tokenizer(), max_latents=6)
    cold = training.reset(pixels(range(-48, 33, 4)))
    torch.testing.assert_close(cold, reference(pixels(range(-48, 33, 4))))
    for end in (64, 96, 128):
        training.append(pixels(range(end - 28, end + 1, 4)))
    torch.testing.assert_close(training.latents.squeeze(0)[:, -4:], history)
    assert training.latent_count == online.latent_count


def test_online_background_encoding_waits_for_group_tail_and_never_reads_future():
    tokenizer = Tokenizer()
    online = EpisodeObservationEncoder(tokenizer, history_steps=48)
    online.add(0, image(0))
    online.prepare(16)
    assert online.count == 9
    for tick in range(1, 16):
        online.add(tick, image(tick))
    assert online.count == 9 and online.encoded_tick == 0
    with pytest.raises(ValueError, match="ticks \\[16\\]"):
        online.finish(16)
    online.add(16, image(16))
    first, _ = online.finish(16)
    frame_count = online.episode.frame_count
    for tick in range(17, 33):
        online.add(tick, image(tick))
    assert online.episode.frame_count == frame_count
    repeated, _ = online.finish(16)
    torch.testing.assert_close(first, repeated)
    assert online.new_frame_ids == () and online.last_new_latents.shape[1] == 0
    online.prepare(32)
    next_history, _ = online.finish(32)
    torch.testing.assert_close(next_history, reference(pixels(range(-32, 33, 4))).squeeze(0)[:, -4:])
    assert online.new_frame_ids == (4,)


def test_online_phase_errors_are_explicit_and_reset_starts_a_new_time_axis():
    online = EpisodeObservationEncoder(Tokenizer(), history_steps=48)
    for tick in range(18):
        online.add(tick, image(tick))
    online.finish(16)
    old = online.state
    with pytest.raises(ValueError, match="phase.*reset"):
        online.finish(17)
    assert online.state is old and not old.closed
    online.clear()
    assert old.closed and online.first_image is online.state is online.anchor is None
    assert online.new_frame_ids == online.prefix_global_ids == ()
    for tick in range(18):
        online.add(tick, image(tick))
    result, _ = online.finish(17)
    torch.testing.assert_close(result, reference(pixels(range(-31, 18, 4))).squeeze(0))
    assert online.root_origin == -31 and online.prefix_global_ids == (0, 1, 2, 3)


def test_online_delayed_anchor_uses_retained_latents_without_rewinding_stream():
    online = EpisodeObservationEncoder(Tokenizer(), history_steps=48)
    online.add(0, image(0))
    online.finish(0)
    for anchor in (16, 32, 48):
        for tick in range(anchor - 15, anchor + 1):
            online.add(tick, image(tick))
        online.finish(anchor)
    expected = reference(pixels(range(-48, 17, 4))).squeeze(0)
    delayed, _ = online.finish(16)
    torch.testing.assert_close(delayed, expected[:, -4:])
    assert online.new_frame_ids == () and online.last_finished_id == 6
    for tick in range(49, 65):
        online.add(tick, image(tick))
    online.finish(64)
    assert online.new_frame_ids == (7,)
    with pytest.raises(ValueError, match="outside.*retained"):
        online.finish(0)


def test_online_observation_and_missing_history_validation():
    online = EpisodeObservationEncoder(Tokenizer(), history_steps=48)
    online.add(0, image(0))
    online.add(0, image(0))
    with pytest.raises(ValueError, match="overwritten"):
        online.add(0, image(1))
    online.add(16, image(16))
    with pytest.raises(ValueError, match="ticks \\[4, 8, 12\\]"):
        online.finish(16)
    with pytest.raises(ValueError, match="nondecreasing"):
        online.add(4, image(4))
    with pytest.raises(ValueError, match="geometry"):
        online.add(17, torch.zeros(3, 32, 16, dtype=torch.uint8))


def test_real_vae_nonretaining_state_and_bounded_episode_match_full_encode(encoder):
    torch.manual_seed(31)
    frames = torch.rand(1, 3, 17, 16, 16) * 2 - 1
    stream = BoundedEpisodeVAE(encoder, max_latents=3)
    result = [stream.reset(frames[:, :, :9]), stream.append(frames[:, :, 9:13]), stream.append(frames[:, :, 13:])]
    actual = torch.cat(result, dim=2)
    with torch.no_grad():
        mean = encoder.vae.encode(frames).latent_dist.mode()
        expected = (mean - encoder.latent_mean) * encoder.latent_inv_std
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(stream.latents, expected[:, :, -3:], atol=2e-5, rtol=2e-4)
    assert stream.state.latent_chunks == [] and stream.state.last_latent.shape[2] == 1
    assert stream.frame_count == 17 and stream.latent_ids == (2, 3, 4)
    with pytest.raises(ValueError, match="does not retain"):
        _ = stream.state.latents
    state = stream.state
    stream.clear()
    assert state.closed and not state.feat_cache and state.last_latent is None
