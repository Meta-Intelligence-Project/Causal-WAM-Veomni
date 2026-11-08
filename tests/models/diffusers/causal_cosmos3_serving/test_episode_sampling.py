"""Recurrent serving history, CFG branch ownership and first-velocity RTC VJP."""

import copy

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.causal_cosmos3_serving.test_native import window
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.packing import pack_model_inputs
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.rtc import RTCPrefix
from veomni.models.diffusers.causal_cosmos3.serving.sampling import sample_decision
from veomni.models.diffusers.causal_cosmos3.serving.segment_cache import EpisodePrefixCache


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def eager_ops():
    with ops_config_scope(eager_ops_config()):
        yield


def model():
    torch.manual_seed(137)
    return (
        Cosmos3Model(
            tiny_config(
                action_dim=55,
                state_dim=55,
                enable_tbptt=True,
                tbptt_keep_history_steps=64,
                history_encoding="episode_stream",
            )
        )
        .eval()
        .requires_grad_(False)
    )


def observed(tick=0, *, new=True):
    _, sample = window(history_steps=32, horizon=32, tick=tick)
    generator = torch.Generator().manual_seed(83)
    timeline = torch.randn(2, 32, 1, 1, generator=generator)
    anchor = 2 + tick // 16
    ids = tuple(range(anchor - 2, anchor + 1))
    sample["video_latents"] = sample["video_latents"].clone()
    sample["video_latents"][:, :3] = timeline[:, list(ids)]
    new_ids = ids if tick == 0 else ((anchor,) if new else ())
    context = dict(
        context_id="episode0",
        prefix_global_ids=ids,
        new_frame_ids=new_ids,
        new_latents=timeline[:, list(new_ids)],
    )
    return sample, context


def assert_memory(memory, end):
    assert memory.frame_ids == tuple(range(end - 2, end + 1))
    assert len(memory.layer_inputs) == 2
    assert all(not value.requires_grad and value.grad_fn is None for value in memory.layer_inputs)
    assert not memory.position_ids.requires_grad and memory.position_ids.grad_fn is None


@pytest.mark.parametrize("cache_mode", ["off", "plans", "full"])
@pytest.mark.parametrize("guidance", [1.0, 2.0])
def test_episode_sampler_modes_share_the_same_recurrent_trajectory(cache_mode, guidance):
    net = model()
    negative = torch.tensor([3, 4, 5]) if guidance > 1 else None

    def rollout(mode):
        memories, results = {}, []
        previous = None
        for tick in (0, 16):
            sample, context = observed(tick)
            results.append(
                sample_decision(
                    net,
                    sample,
                    steps=2,
                    cache_mode=mode,
                    guidance=guidance,
                    negative_text_ids=negative,
                    stream_context=context,
                    stream_memories=memories,
                )
            )
            torch.testing.assert_close(results[-1]["video_latents"][:, :3], sample["video_latents"][:, :3])
            assert_memory(memories["cond"], 2 + tick // 16)
            if guidance > 1:
                assert_memory(memories["uncond"], 2 + tick // 16)
                assert memories["uncond"].context_id != memories["cond"].context_id
                assert not torch.equal(memories["uncond"].input_ids, memories["cond"].input_ids)
            stats = results[-1]["inference_cache"]["cond"]
            assert stats["prefilled_history_latents"] == (3 if tick == 0 else 1)
            assert stats["prefill_calls"] == stats["replay_calls"] == 1
            if previous is not None:
                assert previous.frame_ids == (0, 1, 2)
            previous = memories["cond"]
        return results

    expected, actual = rollout("full"), rollout(cache_mode)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left["video_latents"], right["video_latents"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(left["action"], right["action"], atol=2e-6, rtol=2e-5)
    assert all(parameter.grad is None for parameter in net.parameters())


def test_episode_first_velocity_keeps_joint_input_gradients_and_detaches_memory():
    net = model()
    sample, context = observed()
    frames = sample["video_latents"].shape[1]
    sample["token_length"] = 2 + frames + 32 + 1
    inputs, spans, _ = pack_model_inputs(
        [sample], [sample["video_latents"]], [sample["action"]], timesteps=[torch.full((frames,), 500.0)]
    )
    inputs["video_tokens"] = inputs["video_tokens"].detach().clone().requires_grad_()
    inputs["action_tokens"] = inputs["action_tokens"].detach().clone().requires_grad_()
    memories = {}
    cache = EpisodePrefixCache(
        net, sample, inputs, spans[0], "grouped", stream_context=context, stream_memories=memories
    )
    output = cache(inputs)
    # At the first origin, a cold recurrent prefix has exactly the native
    # independent-window semantics. Compare values and full active-input VJP
    # against that separate path, keeping measured history constant there too.
    ordinary = copy.deepcopy(net)
    ordinary.config.history_encoding = "window"
    ordinary_video = inputs["video_tokens"].detach().clone().requires_grad_()
    ordinary_action = inputs["action_tokens"].detach().clone().requires_grad_()
    measured = torch.arange(len(ordinary_video)) < 3
    ordinary_inputs = dict(
        inputs,
        video_tokens=torch.where(measured[:, None], ordinary_video.detach(), ordinary_video),
        action_tokens=ordinary_action,
    )
    expected = ordinary(**ordinary_inputs, attention_backend="grouped")
    torch.testing.assert_close(output.video[3:], expected.video[3:], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output.action, expected.action, atol=2e-6, rtol=2e-5)
    vg, ag = torch.autograd.grad(output.action.square().sum(), (inputs["video_tokens"], inputs["action_tokens"]))
    evg, eag = torch.autograd.grad(expected.action.square().sum(), (ordinary_video, ordinary_action))
    torch.testing.assert_close(vg, evg, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(ag, eag, atol=2e-6, rtol=2e-5)
    assert not vg[:3].any() and vg[3:].abs().sum() > 0 and ag.abs().sum() > 0
    assert cache.stats["model_calls"] == 1
    assert_memory(memories["cond"], 2)
    assert all(parameter.grad is None for parameter in net.parameters())


def test_episode_rtc_guides_its_first_solver_step_without_parameter_gradients():
    net = model()
    sample, context = observed()
    previous = torch.zeros(16, 55)
    previous[:, 0] = torch.linspace(0.05, 0.2, 16)
    ordinary = sample_decision(net, sample, steps=1, stream_context=context, stream_memories={})
    guided = sample_decision(
        net,
        sample,
        steps=1,
        rtc=RTCPrefix(previous, 8),
        stream_context=context,
        stream_memories={},
    )
    assert guided["rtc_vjp_calls"] == 1
    assert not torch.equal(guided["video_latents"][:, 3:], ordinary["video_latents"][:, 3:])
    torch.testing.assert_close(guided["action"][:8], guided["prepared_prefix"].normalized_targets[:8], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in net.parameters())


def test_cfg_excluded_branch_still_advances_clean_memory_without_noisy_contamination():
    net = model()
    negative = torch.tensor([3, 4, 5])
    full, excluded = {}, {}
    sample, context = observed()
    options = dict(steps=2, guidance=2.0, negative_text_ids=negative, stream_context=context)
    sample_decision(net, sample, stream_memories=full, **options)
    result = sample_decision(net, sample, guidance_interval=(2000, 3000), stream_memories=excluded, **options)
    assert result["inference_cache"]["uncond"]["maintenance_calls"] == 1
    for branch in ("cond", "uncond"):
        assert_memory(excluded[branch], 2)
        for actual, expected in zip(excluded[branch].layer_inputs, full[branch].layer_inputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sample, context = observed(16)
    options["stream_context"] = context
    expected = sample_decision(net, sample, stream_memories=full, **options)
    actual = sample_decision(net, sample, stream_memories=excluded, **options)
    torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
    torch.testing.assert_close(actual["video_latents"], expected["video_latents"], rtol=0, atol=0)


@pytest.mark.parametrize("cache_mode", ["off", "plans", "full"])
def test_episode_model_never_silently_falls_back_to_window_history(cache_mode):
    sample, context = observed()
    net = model()
    with pytest.raises(ValueError, match="stream_context.*stream_memories"):
        sample_decision(net, sample, steps=1, cache_mode=cache_mode)
    with pytest.raises(ValueError, match="stream_context.*stream_memories"):
        sample_decision(net, sample, steps=1, cache_mode=cache_mode, stream_context=context)
    with pytest.raises(ValueError, match="graphs.*episode_stream"):
        sample_decision(net, sample, steps=1, npu_replay_graph=True, stream_context=context, stream_memories={})


def test_delayed_anchor_cold_rebuild_is_causal_and_does_not_rewind_newer_memory():
    net, memories = model(), {}
    for tick in (0, 16, 32):
        sample, context = observed(tick)
        sample_decision(net, sample, steps=1, stream_context=context, stream_memories=memories)
    latest = memories["cond"]
    sample, context = observed(16, new=False)
    delayed = sample_decision(net, sample, steps=1, stream_context=context, stream_memories=memories)
    cold = sample_decision(net, sample, steps=1, stream_context=context, stream_memories={})
    torch.testing.assert_close(delayed["action"], cold["action"], rtol=0, atol=0)
    assert delayed["inference_cache"]["cond"]["delayed_cold_rebuild"] == 1
    assert memories["cond"] is latest and latest.frame_ids[-1] == 4
    sample, context = observed(48)
    sample_decision(net, sample, steps=1, stream_context=context, stream_memories=memories)
    assert_memory(memories["cond"], 5)


def test_episode_missing_increment_fails_and_clears_persistent_memory():
    net, memories = model(), {}
    sample, context = observed()
    sample_decision(net, sample, steps=1, stream_context=context, stream_memories=memories)
    snapshot = copy.deepcopy(memories["cond"].layer_inputs)
    sample, context = observed(16, new=False)
    with pytest.raises(ValueError, match="without missing frames"):
        sample_decision(net, sample, steps=1, stream_context=context, stream_memories=memories)
    assert memories == {} and all(not value.requires_grad for value in snapshot)


def test_repeated_anchor_changes_noisy_inputs_without_changing_clean_memory():
    net, memories = model(), {}
    sample, context = observed()
    sample_decision(net, sample, steps=1, seed=1, stream_context=context, stream_memories=memories)
    snapshot = memories["cond"]
    expected = [value.clone() for value in snapshot.layer_inputs]
    sample, context = observed(new=False)
    sample_decision(net, sample, steps=2, seed=2, stream_context=context, stream_memories=memories)
    for actual, original in zip(memories["cond"].layer_inputs, expected, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_legacy_window_model_still_samples_without_recurrent_context():
    net = model()
    net.config.history_encoding = "window"
    sample, _ = observed()
    actual = sample_decision(net, sample, steps=1)
    assert torch.isfinite(actual["action"]).all()
