# Causal Cosmos3 observation processing and KV reuse

Observation processing and Transformer KV reuse are separate choices. Fix the
observation strategy when comparing cache on/off; changing it changes model
inputs, not just compute cost. Policy and IDM support both strategies in offline
teacher forcing, model-only rollout and live serving.

## Configuration and compatibility

```yaml
sampling:
  observation_mode: streaming  # streaming (default) or window
  kv_cache: true
```

Both CLI entries accept `--observation-mode streaming|window` and
`--kv-cache` / `--no-kv-cache`. CLI values override YAML independently. There is
no implicit strategy switch when KV is disabled. FD still requires KV disabled.

| Observation mode | KV on | KV off |
| --- | --- | --- |
| `streaming` | Incremental VAE encoding; text and clean historical KV survive observation revisions | The same incremental VAE encoding; retain the encoded prefix and recompute full-pack Transformer outputs |
| `window` | Encode the same full/bounded RGB window as the reference; rebuild text/history KV within that window, reuse it during denoising | Encode that window; run full-pack Transformer computation every step |

Window KV never borrows history from an earlier window: shifted local positions
and changed VAE context invalidate that assumption. This simple reference-aligned
implementation also rebuilds window caches per teacher-forcing block; it does not
promise cross-block prefill savings in window mode. Streaming is the production
path for persistent observation-history reuse.

Previously, `kv_cache=false` also selected bounded-window RGB reconstruction.
To request that observation strategy explicitly, use
`--observation-mode window --no-kv-cache`. Default no-KV inference now uses the
same streaming observations as default cached inference. Pipeline sampling now
uses target-block noise; outputs from earlier revisions can differ under the
same seed. The low-level sampler retains its legacy noise mode for existing
explicit callers, while pipeline entries opt into aligned block noise.

## Names and ownership

| Component | Responsibility |
| --- | --- |
| [`InferenceServer`](../../tasks/client_server/server/server.py#L19) | Transport, single-client reservation, serialized requests and connection cleanup |
| [`CausalInferenceSession`](../../tasks/client_server/server/session.py#L60) in `tasks/client_server/server/session.py` | Task, observation validation, real feedback, action conversion, reset and recording |
| [`InferenceClient`](../../tasks/client_server/client/runtime.py#L166) | Task-independent client requests and reset messages |
| [`ObservationSequence`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L763) | Streaming VAE encoder, optional episode KV branches, revision and reference/decode latent prefix |
| [`ObservationKVCache`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L23) | One CFG branch's immutable text and rolling clean-video K/V |
| [`IDMBlockKVCache`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L330) | One action-sampling pass's condition-video and state K/V, borrowing observation history |

`PolicyServer`, `CausalPolicy`, `CausalPolicyClient`, `OnlinePolicyKVCache`,
`create_policy_caches`, and `compact_policy_sample` remain compatibility aliases.
The old server and client `policy.py` modules are small import shims; implementations
live in `server/session.py` and `client/runtime.py`. The device interface is named
`RobotDevice`, with `PolicyDevice` retained as an alias for existing integrations.
The wire protocol identifier `cosmos-causal-action-v2` and the actual `task=policy`
value are unchanged.

No cache is attached globally to the model. Conditional and negative-text CFG
branches own distinct caches because text-dependent video states and positions
may differ. Weights remain shared and immutable during inference. Cache use
requires eval/no-grad, shared action projections and single-process execution.

## Shared observation and sampling helpers

These helpers are framework-provided implementation details, not extension
methods that other models must implement.

- [`create_observation_caches`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L236) creates CFG branches from one initial latent, text
  and geometry. `ObservationKVCache.check_inference` enforces the execution
  contract; `attend` is the shared backend primitive used by observation and IDM
  condition caches. Neither cache reaches through private attention methods.
- [`encode_observation_block`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L692) encodes newly arrived pixels, optionally advances
  historical KV, and constructs a compact target sample. `ObservationSequence`
  calls it for both KV settings. It rejects out-of-order revisions.
- [`expand_observation_sample`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L730) reconstructs the full-pack reference using the
  exact streaming latent prefix and absolute positions. Earlier action/state
  rows are causally invisible to the current prediction; clean-history rows do
  not depend on them. Keeping only a cropped latent prefix would change deeper
  clean representations, so the no-KV streaming oracle retains the full prefix.
- [`sample_window_block`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L788) samples a prepared window. With KV enabled it initializes
  the window prefix, predicts a compact current block, and restores the result
  to the full-window layout for the caller. With KV disabled it calls the same
  sampler using full-pack forward.
- [`sample_rollout_block`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L834) implements a Policy prior followed, for IDM, by
  fixed-video action sampling. Both online and offline rollout use it.
- [`sample_block`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L306) owns noise, CFG, condition masks and scheduler updates. It
  creates per-pass IDM condition caches and calls their action-only `predict`.

The observation mode selects preprocessing; `kv_cache` selects Transformer
execution. Live session reset discards both, while ordinary requests preserve
the selected strategy. The client does not manipulate cache objects.

## What is reusable?

`LayerKV` stores text K/V and clean historical video K/V per layer. K has already
passed its relevant normalization and RoPE; V is projected. Initial text is
prefilled once per streaming CFG branch. `update(observed, revision=...)` jointly
prefills one complete incoming latent block and atomically commits its layers,
then evicts video rows to the configured latent-frame budget. Absolute episode
positions are independent of the compact target container's local block zero.

Policy predicts current video, action and state rows each denoising step. Current
state is not yet separately cached on the Policy path. The reference uses the
same Scheme B visibility and the same model weights.

IDM condition-video Q reads text, earlier clean video, same-block condition video
and state. State Q reads text and itself. Neither reads noisy target actions.
`IDMBlockKVCache` therefore prefills their K/V once at zero time; subsequent
`predict(action, timestep)` calls compute only changing action rows. Action Q
reads `[text | clean history | condition video | noisy action | state]` in packed
order. Advancing history invalidates that pass by its revision guard.

Condition-video KV depends on state and must never be appended as clean-history
KV. For IDM teacher forcing, the current real video latent can be retained until
the next block, but its clean-history K/V must be computed separately. For live
IDM, the next request contains actual executed observations; the model's previous
predicted video is not committed as real feedback.

## Streaming VAE and feedback

A streaming encoder retains causal convolution features, not Transformer K/V.
It receives the independent first RGB frame and then complete temporal chunks,
encoding only newly arrived pixels with the prior convolution context. Its state
is isolated from other VAE encoding and decoding calls. Unknown future black
placeholders must not be encoded as observations.

Live serving receives real pixels from the client. Offline teacher forcing reads
real dataset pixels; IDM additionally reads this block's known video condition.
Model-only rollout explicitly uses decoded generated RGB as feedback and the
last valid action as its next state. Those feedback semantics are independent
of the KV switch.

Streaming live serving with KV keeps no full latent archive unless prediction
recording is enabled separately. Streaming without KV retains the full encoded
prefix for reference computation, so CPU storage and recomputation cost grow
with episode length. Offline streaming rollout retains latents for whole-prefix
video decoding under both KV settings; decode cost still grows with horizon.
Window inference bounds rollout/serving RGB context but re-encodes that context.

## Randomness and numerical comparisons

Pipeline sampling generates video/action noise for the target block, then places
it in either compact or full tensors. Inactive reference slots do not consume
those noise streams. Scheduler updates operate on the same active target vector;
this also aligns fresh fixed-step SDE noise. IDM action-stage noise is independent
of the preceding Policy prior's action noise. CFG uses separate text branches.

Thus fixing observation mode, weights, source, masks, seed and schedule aligns
inputs and random draws across the KV switch. FP32 tiny-model tests compare
both observation strategies, Policy/IDM, teacher forcing/rollout, online feedback,
window shifts, CFG, partial tails and UniPC/fixed-step sampling. Matrix shapes and
kernel grouping still differ: bitwise equality and BF16/NPU tolerance are not
claimed from CPU tests. Closed-loop roundoff can compound over a long rollout.
Changing observation mode intentionally changes the input contract and is not a
pure KV benchmark.

## Lifecycle and historical implementation

New offline calls own new sessions. Online connect, explicit reset, disconnect
and shutdown clear the episode state. A sampling failure requires reset; invalid
preflight input does not advance the valid session. The server serializes requests;
callers must not reset while another thread is computing. Dropping references
releases tensors when no other references remain; allocator-reserved memory need
not immediately decrease. Model weights are not reloaded.

The old `opt/inverse-dynamics-action-with-kv-cache-reuse` / `dffb3338` implementation
verified full-window fixed rows and history prefixes through layer hooks. This
branch instead explicitly advances observations and computes compact targets.
Its Scheme B visibility differs, so the old implementation was not copied
wholesale. IDM condition prefill is checked against the current full-pack oracle.

See the [inference guide](../usage/causal_cosmos3_inference.md) and
[client/server guide](../usage/causal_cosmos3_robotwin.md). Regression cases live in
`tests/models/diffusers/test_causal_cosmos3_inference.py` and
`tests/models/diffusers/test_causal_cosmos3_robotwin.py`, both included by the
existing `tests/models/` test suite.
