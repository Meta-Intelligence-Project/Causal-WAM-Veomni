"""VAE pixel/latent contracts, causal prefixes and independent variable-size samples."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import AutoencoderKLWan

from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video, unpatchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae import (
    ENCODE_CHUNK_FRAMES,
    Cosmos3VideoVAE,
    _encode_chunk_frames,
)
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    path = tmp_path_factory.mktemp("wan_vae")
    torch.manual_seed(7)
    vae = AutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=2,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0.2, -0.3],
        latents_std=[0.5, 1.5],
        in_channels=12,
        out_channels=12,
        patch_size=2,
        is_residual=True,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
    )
    vae.save_pretrained(path)
    return Cosmos3VideoVAE(path, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("frames", [1, 5, 9, 37])
def test_vae_normalization_shapes_and_patches(encoder, frames):
    video = torch.randint(0, 256, (3, frames, 32, 48), dtype=torch.uint8)
    calls = []
    hook = encoder.vae.encoder.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[2]))
    try:
        latent = encoder(video)
    finally:
        hook.remove()
    chunk = _encode_chunk_frames(*video.shape[-2:])
    assert calls == [1] + [min(chunk, frames - start) for start in range(1, frames, chunk)]
    with torch.no_grad():
        mean = encoder.vae.encode(video.float().unsqueeze(0) / 127.5 - 1).latent_dist.mode()
    expected = ((mean - encoder.latent_mean) * encoder.latent_inv_std).squeeze(0)
    torch.testing.assert_close(latent, expected)
    assert latent.shape == (2, 1 + (frames - 1) // 4, 2, 3)
    assert latent.dtype == torch.float32 and not latent.requires_grad
    assert not encoder.vae.training and not any(p.requires_grad for p in encoder.parameters())
    assert all(x is None for x in encoder.vae._enc_feat_map)
    patches = patchify_video(latent, 2)
    assert patches.shape == (latent.shape[1] * 2, 8)
    torch.testing.assert_close(unpatchify_video(patches, latent.shape, 2), latent)


def test_vae_pack_and_future_frames_do_not_contaminate_prefix(encoder):
    video = torch.randint(0, 256, (3, 9, 32, 32), dtype=torch.uint8)
    changed = video.clone()
    changed[:, 5:] = 255 - changed[:, 5:]
    samples = [dict(video=video, caption="a"), dict(video=changed[:, :5], caption="b")]
    encoded = encoder.encode_pack(samples)
    assert [s["caption"] for s in encoded] == ["a", "b"]
    assert all("video" not in s for s in encoded) and samples[0]["video"] is video
    torch.testing.assert_close(encoded[0]["video_latents"][:, :2], encoded[1]["video_latents"])
    torch.testing.assert_close(encoder(changed)[:, :2], encoded[0]["video_latents"][:, :2])
    torch.testing.assert_close(encoder(video), encoded[0]["video_latents"])


@pytest.mark.parametrize(
    "shape,dtype", [((3, 4, 32, 32), torch.uint8), ((3, 5, 31, 32), torch.uint8), ((3, 5, 32, 32), torch.float32)]
)
def test_vae_rejects_implicit_truncation_or_resize(encoder, shape, dtype):
    with pytest.raises(ValueError):
        encoder(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize(
    "height,width,chunk",
    [
        (256, 320, 32),
        (384, 320, 24),
        (720, 640, 24),
        (656, 960, 8),
        (960, 1280, 8),
        (768, 1360, 8),
        (1184, 880, 8),
        (992, 1024, 8),
        (2048, 2048, 8),
    ],
)
def test_chunk_size_uses_composed_canvas(height, width, chunk):
    assert _encode_chunk_frames(height, width) == chunk
    assert _encode_chunk_frames(width, height) == chunk


@pytest.mark.parametrize("chunk", [8, 24, 32, 68])
def test_chunk_boundaries_tail_and_causality(encoder, monkeypatch, chunk):
    # 小画布跑真实网络，覆盖每个块大小的完整块和不足一块的尾部。
    monkeypatch.setitem(ENCODE_CHUNK_FRAMES, "256", chunk)

    @torch.no_grad()
    def encode_reserved(video):
        try:
            return encoder._encode_pixels(video.float().unsqueeze(0) / 127.5 - 1).squeeze(0)
        finally:
            encoder.vae.clear_cache()

    video = torch.randint(0, 256, (3, chunk + 9, 16, 16), dtype=torch.uint8)
    calls = []
    hook = encoder.vae.encoder.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[2]))
    try:
        latent = encode_reserved(video)
    finally:
        hook.remove()
    assert calls == [1, chunk, 8]
    with torch.no_grad():
        native = encoder.vae.encode(video.float().unsqueeze(0) / 127.5 - 1).latent_dist.mode()
    expected = ((native - encoder.latent_mean) * encoder.latent_inv_std).squeeze(0)
    torch.testing.assert_close(latent, expected, atol=2e-5, rtol=2e-4)
    changed = video.clone()
    changed[:, chunk + 1 :] = 255 - changed[:, chunk + 1 :]
    torch.testing.assert_close(encode_reserved(changed)[:, : 1 + chunk // 4], latent[:, : 1 + chunk // 4])
    assert all(x is None for x in encoder.vae._enc_feat_map)


def test_vae_clears_cache_after_chunk_failure(encoder, monkeypatch):
    original = encoder.vae.encoder.forward

    def fail_after_prime(x, **kwargs):
        if x.shape[2] != 1:
            raise RuntimeError("test chunk failure")
        return original(x, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(encoder.vae.encoder, "forward", fail_after_prime)
        with pytest.raises(RuntimeError, match="test chunk failure"):
            encoder(torch.zeros(3, 9, 16, 16, dtype=torch.uint8))
    assert all(x is None for x in encoder.vae._enc_feat_map)
    assert torch.isfinite(encoder(torch.zeros(3, 5, 16, 16, dtype=torch.uint8))).all()


def decision_sample(dense, anchor, *, episode=7, root="dataset", history=128):
    ticks = (anchor + torch.arange(-history, 33, 4)).clamp_min(0)
    return dict(
        video=dense[:, ticks],
        layout_mode="decision_window",
        video_frame_ticks=ticks.tolist(),
        window=SimpleNamespace(episode=episode),
        subset=root,
        observed_latents=1 + history // 16,
    )


def test_training_reset_cache_is_bitwise_causal_sparse_and_sample_isolated(encoder):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae_prefix import DecisionVAEPrefixCache

    dense = torch.randint(0, 256, (3, 193, 16, 16), dtype=torch.uint8)
    cache = DecisionVAEPrefixCache(encoder, max_entries=3, max_bytes=2**28, checkpoint_stride=4)
    # Include incomplete reset groups, reverse anchor order and full real history.
    for anchor in [64, 0, 16, 1, 100, 128, 160, 0]:
        sample = decision_sample(dense, anchor)
        expected = encoder.encode_decision_window(sample["video"])
        actual = cache.encode(sample)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    entry = next(iter(cache.entries.values()))
    assert set(entry.states) == {1, 5, 9}
    pointers = [value.data_ptr() for state in entry.states.values() for value in state.feat_cache if value is not None]
    assert len(set(pointers)) < len(pointers)  # Equal frozen features share storage.
    saved = [value.clone() for state in entry.states.values() for value in state.feat_cache if value is not None]
    sample = decision_sample(dense, 16)
    sample["video"][:, 33:] = 255 - sample["video"][:, 33:]
    changed = cache.encode(sample)
    original = cache.encode(decision_sample(dense, 16))
    torch.testing.assert_close(changed[:, :9], original[:, :9], atol=0, rtol=0)
    after = [value for state in entry.states.values() for value in state.feat_cache if value is not None]
    assert all(torch.equal(a, b) for a, b in zip(saved, after, strict=True))
    cache.clear()


def test_training_reset_cache_bounds_keys_pixels_weights_and_failure(encoder, monkeypatch):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae_prefix import DecisionVAEPrefixCache

    dense = torch.randint(0, 256, (3, 65, 16, 16), dtype=torch.uint8)
    cache = DecisionVAEPrefixCache(encoder, max_entries=1, max_bytes=2**28)
    first = decision_sample(dense, 0, history=32)
    cache.encode(first)
    assert not cache.entries
    cache.encode(first)
    old = next(iter(cache.entries.values()))
    cache.encode(decision_sample(dense, 0, history=32, root="another dataset"))
    cache.encode(decision_sample(dense, 0, history=32, root="another dataset"))
    assert all(state.closed for state in old.states.values())
    old = next(iter(cache.entries.values()))
    changed = decision_sample(255 - dense, 0, history=32, root="another dataset")
    actual = cache.encode(changed)
    torch.testing.assert_close(actual, encoder.encode_decision_window(changed["video"]), atol=0, rtol=0)
    assert all(state.closed for state in old.states.values())
    old = next(iter(cache.entries.values()))
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(not previous_determinism)
        cache.encode(changed)
        cache.encode(changed)
        assert all(state.closed for state in old.states.values())
    finally:
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn)
    old = next(iter(cache.entries.values()))
    saved_mean = encoder.latent_mean.clone()
    with torch.no_grad():
        encoder.latent_mean.add_(0.125)
    try:
        actual = cache.encode(changed)
        actual = cache.encode(changed)
        torch.testing.assert_close(actual, encoder.encode_decision_window(changed["video"]), atol=0, rtol=0)
        assert all(state.closed for state in old.states.values())
        bad = {**changed, "video": changed["video"].clone()}
        bad["video"][:, 1] = 255 - bad["video"][:, 1]
        with pytest.raises(ValueError, match="reset padding"):
            cache.encode(bad)
        assert not cache.entries
        cache.max_bytes = 1
        cache.encode(changed)
        cache.encode(changed)
        assert not cache.entries and cache.nbytes == 0
        cache.max_bytes = 2**28
        cache.encode(changed)
        cache.encode(changed)
        with monkeypatch.context() as patch:
            patch.setattr(
                encoder, "stream_encode_step", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("test failure"))
            )
            with pytest.raises(RuntimeError, match="test failure"):
                cache.encode(changed)
        assert not cache.entries
    finally:
        with torch.no_grad():
            encoder.latent_mean.copy_(saved_mean)
        cache.clear()


def test_training_reset_cache_dtype_move_clears_and_pack_retains_metadata(encoder):
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae_prefix import DecisionVAEPrefixCache

    encoder.reset_prefix_cache = DecisionVAEPrefixCache(encoder, max_entries=2, max_bytes=2**28)
    try:
        dense = torch.randint(0, 256, (3, 65, 16, 16), dtype=torch.uint8)
        sample = decision_sample(dense, 0, history=32)
        encoded = encoder.encode_pack([sample])
        encoder.encode_pack([sample])
        assert encoded[0]["window"] is sample["window"]
        assert "video" not in encoded[0] and "video" in sample
        assert encoder.reset_prefix_cache.entries
        encoder.to(dtype=torch.float32)
        assert not encoder.reset_prefix_cache.entries
    finally:
        encoder.reset_prefix_cache.clear()
        encoder.reset_prefix_cache = None


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Ascend frozen VAE reset-prefix parity")
def test_training_reset_cache_npu_bitwise(encoder, tmp_path):
    path = tmp_path / "tiny_vae"
    encoder.vae.save_pretrained(path)
    native = Cosmos3VideoVAE(path, device="npu", dtype=torch.bfloat16, reset_prefix_cache_mb=16)
    dense = torch.randint(0, 256, (3, 193, 16, 16), dtype=torch.uint8)
    for anchor in [64, 0, 16, 1, 100, 128, 160]:
        sample = decision_sample(dense, anchor)
        sample["video"] = sample["video"].to("npu")
        expected = native.encode_decision_window(sample["video"])
        actual = native.encode_pack([sample])[0]["video_latents"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    native.reset_prefix_cache.clear()
