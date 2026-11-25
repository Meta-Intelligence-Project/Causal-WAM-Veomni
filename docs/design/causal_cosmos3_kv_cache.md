# Causal Cosmos3 window KV cache

`sampling.kv_cache: true` reuses immutable text and real observed-video K/V
within one encoded window. The default, including the server example, is false;
configurations without this field also use the full recomputation path. Both
paths use the same RGB window, VAE, noise seeds, local positions and trained attention rules.
The server accepts `--kv-cache` and `--no-kv-cache`; `observation_mode` stays `window`.

## Ownership and lifetime

The implementation adapts text/clean-block prefill from
[PR #21](https://github.com/Meta-Intelligence-Project/Causal-WAM-Veomni/pull/21).
It does not adopt that PR's state prefill or Policy video prior for IDM.

| Component | Responsibility |
| --- | --- |
| [ObservationKVCache](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L26) | Own one CFG branch's text/history K/V for exactly one encoded observed prefix. Construction requires eval, no-grad, single-process execution and shared action projections. |
| [_prefill_text](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L103) | Evaluate causal text once, preserving UND K normalization and the separate GEN text K normalization. |
| [_append_clean](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L114) | Evaluate the first latent and complete historical blocks in order, retaining at most the configured history-frame budget per layer. Old context already embedded in retained representations is preserved. |
| [bind](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L136) | Validate model, backend, dtype/device, text, observed latent values, geometry/history budget and prefix positions before binding a stage. Reject stale or incompatible prefixes. |
| [_CachedBlock.predict](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L163) | Evaluate all current-block rows each denoising step. Read cached prefix K/V without modifying it; return current video/action velocities. State output is discarded. |

These are inference-owned helpers, not callbacks that a robot device must implement.
Callers must keep model weights and the bound stage inputs unchanged while sampling.
KV cache does not run with attention probing, which requires the full model forward.

The production call flow is:

1. [predict_observation](../../veomni/models/diffusers/causal_cosmos3/observation_session.py#L21)
   encodes only supplied real RGB history and creates a local cache dictionary.
2. [generate_block](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/generation.py#L22)
   constructs native stage inputs and prefills separate conditional/unconditional caches.
   IDM's video and action stages share that dictionary for their identical text/history.
3. Every solver step evaluates current queries. Only text/history projection and MLP
   work is skipped. Action/state K/V are never held fixed across steps.
4. Returning from prediction releases the caches. New observations, a cropped window,
   reset or reconnect start from a fresh encoded prefix and local positions.

No cache is attached to model weights or retained in a long-lived server session.
Prediction latents are never appended as measured history. RGB history and VAE
encoding remain unchanged; this feature does not implement streaming VAE encoding
or persistent cross-window Transformer KV.

## Current-block visibility

| Stage | Video queries read | Action/state queries read |
| --- | --- | --- |
| IDM video | Text, real historical video, current noisy video | Absent |
| IDM action | Text, real historical video, fixed current video | Text, real historical video, fixed current video, noisy actions, state |
| Policy | Text, real historical video, current noisy video/actions, state | The same current stream |
| Forward dynamics | Text, real historical video, current noisy video, measured actions/state | Text, real historical video, measured actions/state |

State is a clean physical input at timestep zero. Its hidden representations can
depend on current targets, so all state rows are recomputed together with actions.
In this first implementation even fixed current video is recomputed; only text
and real historical video are cached. The attention mask, training loss, state
projections, parameter names and checkpoint format are unchanged.

## Verification

[Inference tests](../../tests/models/diffusers/test_causal_cosmos3_inference.py)
compare full/cached generation across stages, model variants, CFG branches with
different prompt lengths, small history budgets and multiple block positions.
They check unchanged global RNG/seeds, immutable cached history, repeated dynamic
state projections, stale-prefix rejection, and NPU FP32/BF16 TND equivalence.
[Serving tests](../../tests/models/diffusers/test_causal_cosmos3_robotwin.py)
compare cached server actions and prediction latents against uncached native
sessions over measured feedback and a cropped window.

Floating-point reduction shapes can change with cached queries. Compare with
dtype-appropriate numerical tolerances; deterministic reset/replay within the
same execution path should still reproduce its results. Hardware performance and
real-checkpoint precision must be measured separately from tiny-model tests.

The 2026-11-25 acceptance used the step-4000 IDM checkpoint and five recorded
feedback boundaries from `RoboTwin__adjust_bottle/clean`, episode 0. The real
WebSocket client sent identical RGB/state inputs for both paths, with 20 video
and 20 action steps on one NPU. Text was evaluated once per cached prediction
instead of 40 times; state was still evaluated 20 times. Both paths reproduced
their first action block exactly after reset.

With the production BF16/NPU operations, steady request time fell from about
5.69 s to 3.16 s (1.80x in this replay, including diagnostic capture). The largest
joint difference was 0.02198 rad. Video-latent differences failed the predeclared
`atol=0.02, rtol=0.02` elementwise check; the maximum was 0.33957. This is not a
passing BF16 equivalence result. Layer comparison found bitwise-identical text
KV, but historical video projections differed from the first layer when batch
shapes changed. In FP32, identical-input predictions across all four stages
matched within 4.3e-5 absolute error. These checks support the visibility and
cache-lifetime implementation. A complete 20+20-step FP32 IDM sample on the same
real history had maximum differences of 1.01e-4 in video latents and 6.56e-7 in
normalized actions. These checks do not establish BF16 task quality or a
successful physical RoboTwin rollout. Keep cache opt-in until the intended
checkpoint and simulation are accepted.
