"""Triple-Expert online inference server, DCP loading, TextCache reuse, and wire transport."""

import copy
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml
from diffusers import AutoencoderKLWan

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.client_server.client.runtime import InferenceClient, run_episode
from tasks.client_server.server import server as server_module
from tasks.client_server.server.adapter import Unified55Adapter
from tasks.client_server.server.session import CAMERA_KEYS, CausalInferenceSession, build_adapter
from tasks.client_server.wire import packb, unpackb
from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.data.causal_cosmos3.transforms.processing import ActionProcessor
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.vae import Cosmos3VideoVAE
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import Cosmos3Config
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline, load_trained_model


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


ROBOTWIN_MASK = [
    1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1,
    *([0] * 25),
]


def _make_stats():
    return {
        "provenance": {
            "sources": [{"state_mask": ROBOTWIN_MASK, "action_mask": ROBOTWIN_MASK}],
            "geometry": {"actions_per_block": 32, "video_stride": 4, "temporal_compression_factor": 4},
        },
        "state": {"low": [-5.0] * 55, "high": [5.0] * 55, "valid": ROBOTWIN_MASK},
        "action": {"low": [-5.0] * 55, "high": [5.0] * 55, "valid": ROBOTWIN_MASK},
    }


def _tiny_triple_config(**overrides):
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mrope_section=[2, 1, 1],
        latent_channels=2,
        patch_size=2,
        action_interface="shared",
        action_dim=55,
        state_dim=55,
        expert_layout="triple",
        action_hidden_size=16,
        action_intermediate_size=32,
        action_num_hidden_layers=1,
        action_layer_map=[2],
    )
    defaults.update(overrides)
    return Cosmos3Config(**defaults)


@pytest.fixture(scope="module")
def tiny_wan_vae(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_server_wan_vae")
    torch.manual_seed(11)
    vae = AutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=2,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0.1, -0.2],
        latents_std=[0.5, 1.2],
        in_channels=12,
        out_channels=12,
        patch_size=2,
        is_residual=True,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
    )
    vae.save_pretrained(path)
    return Cosmos3VideoVAE(path, device="cpu", dtype=torch.float32)


class FastMockVAE:
    """Fast CPU VAE stub preserving exact [2, T_lat, 24, 20] geometry for 384x320 concat_view."""

    def __init__(self):
        self.stream_calls = []
        self.pack_calls = []
        self.decode_calls = []

    def encode_pack(self, samples):
        results = []
        for sample in samples:
            video = sample["video"]
            self.pack_calls.append(video.clone())
            latents = video[:2, ::4, ::16, ::16].float() / 127.5 - 1.0
            results.append({**{k: v for k, v in sample.items() if k != "video"}, "video_latents": latents})
        return results

    def stream_encoder(self):
        owner = self

        class Stream:
            def __init__(self):
                self.initial = True

            def reset(self):
                self.initial = True

            def encode(self, video):
                owner.stream_calls.append(video.clone())
                if self.initial:
                    assert video.shape[1] == 1
                    self.initial = False
                    return video[:2, :1, ::16, ::16].float() / 127.5 - 1.0
                assert video.shape[1] % 4 == 0
                return video[:2, 3::4, ::16, ::16].float() / 127.5 - 1.0

        return Stream()

    def decode(self, latents):
        self.decode_calls.append(latents.clone())
        frames = 1 + (latents.shape[1] - 1) * 4
        h, w = latents.shape[2] * 16, latents.shape[3] * 16
        base = ((latents[0].clamp(-1, 1) + 1) * 127.5).round().byte()
        frame0 = base[:1].repeat_interleave(16, 1).repeat_interleave(16, 2)
        if latents.shape[1] == 1:
            return frame0.unsqueeze(0).expand(3, -1, -1, -1).contiguous()
        rest = base[1:].repeat_interleave(4, 0).repeat_interleave(16, 1).repeat_interleave(16, 2)
        out = torch.cat((frame0, rest), dim=0).unsqueeze(0).expand(3, frames, h, w).contiguous()
        return out


def _build_triple_pipe(vae=None, *, idm_video_mode="partial_denoised", observation_mode="streaming", text_cache=True, guidance=1.0):
    with ops_config_scope(eager_ops_config()):
        torch.manual_seed(19)
        model = Cosmos3Model(_tiny_triple_config(idm_video_mode=idm_video_mode)).eval().requires_grad_(False)
    processor = ActionProcessor(_make_stats())
    config = dict(
        expert_layout="triple",
        task="inverse_dynamics",
        statistics=_make_stats(),
        source=dict(name="robotwin", resolution="256", viewpoint="concat_view"),
        sampling=dict(
            observation_mode=observation_mode,
            text_cache=text_cache,
            idm_video_mode=idm_video_mode,
            history=4,
            partial_video_steps=2,
            full_video_steps=2,
            action_steps=2,
            video_completion_steps=2,
            video_stop_sigma=0.9,
            guidance=guidance,
            negative_prompt="",
            seed=42,
            attention_backend="grouped",
        ),
        attention_backend="grouped",
    )

    def tokenize(text):
        return [1 + (ord(ch) % 60) for ch in (text or "empty")[:12]]

    return CausalCosmos3Pipeline(model, vae or FastMockVAE(), processor, tokenize, config, device="cpu")


def _sample_protocol_state(offset=0.0):
    return {
        "joint_action": {
            "left_arm": np.linspace(0.1, 0.6, 6, dtype=np.float32) + offset,
            "left_gripper": np.float32(0.8),
            "right_arm": np.linspace(-0.6, -0.1, 6, dtype=np.float32) - offset,
            "right_gripper": np.float32(0.2),
        },
        "endpose": {
            "left_endpose": np.array([0.2 + offset, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "right_endpose": np.array([0.2 - offset, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
    }


def _sample_cameras(seed_val, count=1, h=48, w=64):
    rng = np.random.default_rng(seed_val)
    frames = [
        {key: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for key in CAMERA_KEYS}
        for _ in range(count)
    ]
    return frames[0] if count == 1 else frames


@contextmanager
def _running_server(session):
    pytest.importorskip("websockets")
    server = server_module.create_server(session, host="127.0.0.1", port=0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{port}", server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_wire_roundtrip_preserves_numpy_arrays_and_scalars():
    payload = {
        "arr_f32": np.arange(6, dtype=np.float32).reshape(2, 3),
        "arr_u8": np.array([[0, 255]], dtype=np.uint8),
        "scalar": np.float32(1.25),
        "flag": np.bool_(True),
        "text": "pick bottle",
    }
    restored = unpackb(packb(payload))
    np.testing.assert_array_equal(restored["arr_f32"], payload["arr_f32"])
    assert restored["arr_f32"].dtype == np.float32
    np.testing.assert_array_equal(restored["arr_u8"], payload["arr_u8"])
    assert restored["arr_u8"].dtype == np.uint8
    assert restored["scalar"] == np.float32(1.25)
    assert restored["flag"] is np.bool_(True)


@pytest.mark.parametrize("fps", [None, 0, -30, float("nan"), float("inf"), True, "30"])
def test_server_rejects_missing_or_invalid_fps_before_loading_weights(tmp_path, monkeypatch, fps):
    config = dict(device="cpu", expert_layout="triple", task="inverse_dynamics")
    if fps is not None:
        config["fps"] = fps
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(CausalCosmos3Pipeline, "from_pretrained", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["server", "--config", str(path), "--checkpoint", "unused"])
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match=r"fps.*--fps"):
        server_module.main()
    assert calls == []


def test_dcp_checkpoint_loads_from_step_dir_and_model_ckpt_dir(tmp_path):
    run_dir = tmp_path / "run"
    assets_dir = run_dir / "model_assets"
    ckpt_dir = run_dir / "checkpoints" / "global_step_20" / "model" / "ckpt"
    assets_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    config = _tiny_triple_config()
    config.save_pretrained(assets_dir)
    with ops_config_scope(eager_ops_config()):
        torch.manual_seed(99)
        source_model = Cosmos3Model(config).eval()
    dcp.save(
        state_dict={"model": source_model.state_dict()},
        storage_writer=dcp.FileSystemWriter(str(ckpt_dir)),
        no_dist=True,
    )
    for target in (run_dir / "checkpoints" / "global_step_20", ckpt_dir):
        loaded = load_trained_model(target, dtype=torch.float32)
        assert loaded.weight_loading_report["mode"] == "complete"
        assert loaded.weight_loading_report["format"] == "dcp"
        for key, value in source_model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("idm_video_mode", ["partial_denoised", "full_denoised"])
@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_text_cache_reuse_across_rounds_is_bit_identical_and_skips_text_layers(idm_video_mode, guidance):
    pipe_cached = _build_triple_pipe(idm_video_mode=idm_video_mode, text_cache=True, guidance=guidance)
    pipe_uncached = copy.deepcopy(pipe_cached)
    pipe_uncached.config["sampling"]["text_cache"] = False

    text_layer_executions = []

    def count_text_ops(_module, args):
        if len(args) >= 5 and isinstance(args[4], dict) and args[4].get("operation") == "text":
            text_layer_executions.append(1)

    hook = pipe_cached.model.layers[0].register_forward_pre_hook(count_text_ops)
    try:
        sess_cached = CausalInferenceSession(pipe_cached, fps=30.0)
        sess_uncached = CausalInferenceSession(pipe_uncached, fps=30.0)
        sess_cached.infer({"reset": True, "prompt": "Pick up the bottle", "control_mode": "joint", "record_video": True})
        sess_uncached.infer(
            {"reset": True, "prompt": "Pick up the bottle", "control_mode": "joint", "record_video": True}
        )

        branches = 2 if guidance != 1.0 else 1
        for round_idx in range(3):
            step = 32 * round_idx
            req = {
                "obs": _sample_cameras(100 + round_idx, count=1 if round_idx == 0 else 32),
                "state": _sample_protocol_state(0.02 * round_idx),
                "observation_step": step,
            }
            if round_idx > 0:
                req["executed_steps"] = 32
            before_count = len(text_layer_executions)
            out_cached = sess_cached.infer(copy.deepcopy(req))
            out_uncached = sess_uncached.infer(copy.deepcopy(req))
            delta_text_runs = len(text_layer_executions) - before_count
            if round_idx == 0:
                assert delta_text_runs == branches
                assert out_cached["server_timing"]["expert_calls"]["text"] == branches
            else:
                assert delta_text_runs == 0
                assert out_cached["server_timing"]["expert_calls"]["text"] == 0
            for mode in ("joint", "eef"):
                np.testing.assert_allclose(out_cached["action"][mode], out_uncached["action"][mode], atol=0, rtol=0)
        for lat_c, lat_u in zip(sess_cached.video_latents, sess_uncached.video_latents, strict=True):
            torch.testing.assert_close(lat_c, lat_u, atol=0, rtol=0)
    finally:
        hook.remove()


def test_streaming_and_window_observation_modes_match_with_real_wan_vae(tiny_wan_vae):
    pipe_stream = _build_triple_pipe(vae=tiny_wan_vae, observation_mode="streaming", text_cache=True)
    pipe_window = copy.deepcopy(pipe_stream)
    pipe_window.vae = tiny_wan_vae
    pipe_window.config["sampling"]["observation_mode"] = "window"

    sess_stream = CausalInferenceSession(pipe_stream, fps=30.0)
    sess_window = CausalInferenceSession(pipe_window, fps=30.0)
    sess_stream.infer({"reset": True, "prompt": "Adjust bottle", "control_mode": "joint", "record_video": True})
    sess_window.infer({"reset": True, "prompt": "Adjust bottle", "control_mode": "joint", "record_video": True})

    for round_idx in range(2):
        req = {
            "obs": _sample_cameras(200 + round_idx, count=1 if round_idx == 0 else 32),
            "state": _sample_protocol_state(0.05 * round_idx),
            "observation_step": 32 * round_idx,
        }
        if round_idx > 0:
            req["executed_steps"] = 32
        r_stream = sess_stream.infer(copy.deepcopy(req))
        r_window = sess_window.infer(copy.deepcopy(req))
        np.testing.assert_allclose(r_stream["action"]["joint"], r_window["action"]["joint"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(r_stream["action"]["eef"], r_window["action"]["eef"], atol=1e-4, rtol=1e-4)


def test_end_to_end_websocket_client_episode_records_prediction_mp4_and_uses_real_feedback(tmp_path, monkeypatch):
    import shutil
    from tasks.client_server.client import runtime as client_runtime

    if shutil.which("ffmpeg") is None:
        monkeypatch.setattr(client_runtime.VideoRecorder, "append", lambda self, images: setattr(self, "frames", self.frames + 1))
    else:
        try:
            import cv2  # noqa: F401
        except ImportError:
            import types
            from av.video.reformatter import Interpolation

            fake_cv2 = types.ModuleType("cv2")
            fake_cv2.INTER_LINEAR = 1

            def _resize(img, dsize, interpolation=1):
                w, h = dsize
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                return frame.reformat(width=w, height=h, format="rgb24", interpolation=Interpolation.BILINEAR).to_ndarray()

            fake_cv2.resize = _resize
            monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    vae = FastMockVAE()
    pipe = _build_triple_pipe(vae=vae, idm_video_mode="partial_denoised", observation_mode="streaming", text_cache=True)
    session = CausalInferenceSession(pipe, fps=30.0)

    class FakeRobotDevice:
        def __init__(self, max_steps=64):
            self.step = 0
            self.max_steps = max_steps
            self.executed_actions = []

        def observe(self):
            return _sample_cameras(300 + self.step, count=1), _sample_protocol_state(0.001 * self.step)

        def execute(self, action):
            self.executed_actions.append(action.copy())
            self.step += 1

        def is_done(self):
            return self.step >= self.max_steps

        def is_success(self):
            return self.step >= self.max_steps

    with _running_server(session) as (uri, _):
        with InferenceClient(uri, control_mode="joint") as client:
            assert client.metadata["protocol"] == "cosmos-causal-action-v2"
            assert client.metadata["video_decode_mode"] == "whole_sequence"
            assert client.metadata["mode"] == "inverse_dynamics"
            assert client.metadata["idm_video_mode"] == "partial_denoised"

            # Second simultaneous client must be rejected with 1013.
            from websockets.exceptions import ConnectionClosedError
            from websockets.sync.client import connect

            with connect(uri) as extra_conn:
                with pytest.raises(ConnectionClosedError) as exc_info:
                    extra_conn.recv()
                assert exc_info.value.rcvd.code == 1013

            out_dir = tmp_path / "episode0"
            device = FakeRobotDevice(max_steps=64)
            metrics = run_episode(device, client, prompt="Stack two blocks", output=out_dir, video_fps=30)

    assert metrics["success"] is True
    assert metrics["executed_steps"] == 64
    assert len(device.executed_actions) == 64
    # Two chunks of 32 steps -> prediction.mp4 has 1 initial + 2 * 8 = 17 frames at 7.5 fps.
    pred_mp4 = Path(metrics["prediction_video_path"])
    assert pred_mp4.is_file() and pred_mp4.stat().st_size > 0
    assert metrics["prediction_video_frames"] == 17
    assert metrics["prediction_video_fps"] == pytest.approx(7.5)
    with av.open(str(pred_mp4)) as container:
        decoded_frames = list(container.decode(video=0))
        assert len(decoded_frames) == 17
    # Streaming VAE encoded initial frame (1) + chunk 0 real feedback (8 frames), and decoded whole sequence once.
    assert [call.shape[1] for call in vae.stream_calls] == [1, 8]
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0].shape[1] == 5  # 1 initial latent + 2 blocks * 2 latents
    requests_log = json.loads((out_dir / "requests.json").read_text())
    assert len(requests_log) == 2
    assert requests_log[0]["action_condition_video_sigma"] == pytest.approx(0.9)
    assert "video_completion" in requests_log[0]["stage_seconds"]
    assert requests_log[0]["expert_calls"]["text"] == 1
    assert requests_log[1]["expert_calls"]["text"] == 0


def test_session_rejects_out_of_order_requests_and_recovers_after_failure(monkeypatch):
    pipe = _build_triple_pipe()
    session = CausalInferenceSession(pipe, fps=30.0)
    with _running_server(session) as (uri, _):
        with InferenceClient(uri, control_mode="eef") as client:
            client.reset(prompt="Beat block with hammer", record_video=True)
            first = client.infer({"obs": _sample_cameras(1, 1), "state": _sample_protocol_state(), "observation_step": 0})
            assert first["action"]["eef"].shape == (32, 16)

            # Out-of-order observation_step is rejected preflight without corrupting valid session.
            with pytest.raises(RuntimeError, match="Expected observation_step=32"):
                client.infer(
                    {
                        "obs": _sample_cameras(2, 32),
                        "state": _sample_protocol_state(),
                        "executed_steps": 32,
                        "observation_step": 64,
                    }
                )
            # Valid step=32 still succeeds because preflight validation did not poison the session.
            second = client.infer(
                {
                    "obs": _sample_cameras(2, 32),
                    "state": _sample_protocol_state(0.01),
                    "executed_steps": 32,
                    "observation_step": 32,
                }
            )
            assert second["revision"] == 2

            # Simulate an internal model failure -> session sets needs_reset=True until reset() is called.
            def boom(*args, **kwargs):
                raise RuntimeError("simulated NPU kernel failure")

            monkeypatch.setattr("tasks.client_server.server.session.sample_block", boom)
            with pytest.raises(RuntimeError, match="simulated NPU kernel failure"):
                client.infer(
                    {
                        "obs": _sample_cameras(3, 32),
                        "state": _sample_protocol_state(0.02),
                        "executed_steps": 32,
                        "observation_step": 64,
                    }
                )
            monkeypatch.undo()
            with pytest.raises(RuntimeError, match="send reset before continuing"):
                client.infer(
                    {
                        "obs": _sample_cameras(3, 32),
                        "state": _sample_protocol_state(0.02),
                        "executed_steps": 32,
                        "observation_step": 0,
                    }
                )
            # Reset recovers cleanly.
            client.reset(prompt="Beat block with hammer", record_video=False)
            recovered = client.infer(
                {"obs": _sample_cameras(4, 1), "state": _sample_protocol_state(), "observation_step": 0}
            )
            assert recovered["revision"] == 1
