"""VAE pixel/latent contracts, causal prefixes and independent variable-size samples."""

import pytest
import torch
from diffusers import AutoencoderKLWan

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_inference import feedback_fixture
from veomni.data.causal_cosmos3.readers.observation import make_sample
from veomni.data.causal_cosmos3.transforms.processing import Geometry
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.patching import patchify_video, unpatchify_video
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae import (
    ENCODE_CHUNK_FRAMES,
    Cosmos3VideoVAE,
    _encode_chunk_frames,
)
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline, teacher_forcing


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_measured_replay_with_real_vae_and_transformer_resets_between_samples(encoder):
    initial, processor, source = feedback_fixture(horizon=35)
    states = torch.zeros(65, 55)
    states[:, 0] = 2 + torch.arange(65)
    raw = dict(
        video=torch.randint(0, 256, (3, 17, 16, 16), dtype=torch.uint8),
        state=states,
        action=states[1:].clone(),
        state_mask=initial.state_mask,
        action_mask=initial.action_mask,
        real_action_steps=35,
        timestamps=torch.arange(65) / 30,
    )
    sample = make_sample(
        raw,
        source=source,
        processor=processor,
        geometry=Geometry(),
        tokenize=lambda prompt: [1, 2],
        caption="move",
        fps=30,
        task="policy",
        history=1,
    )
    sample["ground_truth_actions"] = raw["action"][:35].clone()
    with ops_config_scope(eager_ops_config()):
        model = Cosmos3Model(tiny_config(action_dim=55, state_dim=55)).eval()
    pipe = CausalCosmos3Pipeline(
        model,
        encoder,
        processor,
        lambda prompt: [1, 2],
        dict(source=source, sampling=dict(kv_cache=True, history=1, steps=2)),
    )
    calls = []
    handle = model.embed_tokens.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    try:
        first = teacher_forcing(pipe, sample)
        second = teacher_forcing(pipe, sample)
    finally:
        handle.remove()
    assert calls == [2, 2]  # One text prefill per complete replay, not per block.
    assert first.actions.shape == (35, 55) and torch.isfinite(first.actions).all()
    assert first.video.shape == first.reconstruction.shape == (3, 9, 16, 16)
    torch.testing.assert_close(first.actions, second.actions, atol=0, rtol=0)
    torch.testing.assert_close(first.video, second.video, atol=0, rtol=0)
    assert [record["input_pixel_frames"] for record in first.blocks] == [1, 8]
    assert first.blocks[-1]["valid_steps"] == 3


def test_streaming_measured_encoder_matches_full_episode_and_survives_decode(encoder, monkeypatch):
    video = torch.randint(0, 256, (3, 17, 16, 16), dtype=torch.uint8)
    # from_pretrained() has not initialized the per-call encode cache on cold start.
    monkeypatch.delattr(encoder.vae, "_enc_conv_num", raising=False)
    stream = encoder.stream_encoder()
    first = stream.encode(video[:, :1])
    expected = encoder(video)
    second = stream.encode(video[:, 1:9])
    # A prediction-video decode and an unrelated video encode must not clear
    # this session's observation convolution history.
    encoder.decode(torch.cat((first, second), dim=1))
    encoder(255 - video[:, :5])
    third = stream.encode(video[:, 9:])
    torch.testing.assert_close(torch.cat((first, second, third), dim=1), expected, atol=2e-5, rtol=2e-4)
    assert any(value is not None for value in stream.features)
    stream.reset()
    assert stream.shape is None and all(value is None for value in stream.features)
    torch.testing.assert_close(stream.encode(video[:, :1]), first)


def test_streaming_encoder_rejects_partial_chunks_and_clears_failed_session(encoder, monkeypatch):
    stream = encoder.stream_encoder()
    video = torch.zeros(3, 9, 16, 16, dtype=torch.uint8)
    with pytest.raises(ValueError, match="initial RGB"):
        stream.encode(video)
    stream.encode(video[:, :1])
    with pytest.raises(ValueError, match="complete temporal"):
        stream.encode(video[:, :3])
    original = encoder.vae.encoder.forward

    def fail(*args, **kwargs):
        raise RuntimeError("stream failed")

    monkeypatch.setattr(encoder.vae.encoder, "forward", fail)
    with pytest.raises(RuntimeError, match="stream failed"):
        stream.encode(video[:, 1:])
    assert stream.shape is None and all(value is None for value in stream.features)
    monkeypatch.setattr(encoder.vae.encoder, "forward", original)
    assert stream.encode(video[:, :1]).shape[1] == 1


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
