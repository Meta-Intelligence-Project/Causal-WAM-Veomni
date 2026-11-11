# Causal Cosmos3 teacher-forcing Policy KV cache

## Ownership and defaults

The cache is enabled by default for single-process teacher-forcing Policy
inference. `sampling.kv_cache: false` or `--no-kv-cache` disables it;
`--kv-cache` explicitly enables it. FD, ID and rollout require disabling the
cache. Training and gradient-enabled cache use are rejected explicitly.
The model weights and checkpoint layout do not change.

The inference pipeline owns a fresh branch dictionary per teacher-forcing
sample. Each `PolicyKVCache` binds one eval model, immutable encoded sample and
text branch. Conditional and negative-text branches must not share video KV:
their deep video states and text-dependent position offsets can differ.
Branch reuse validates the model/sample identity, text IDs, position options and
attention backend, rejecting incompatible reuse. Calling teacher forcing again creates new caches; there is no global cache or
attention dispatcher replacement.

## Framework-provided interface

These are inference implementation helpers, not methods that another model
must implement. The sampling loop can disable caching; existing full-pack forward is
the reference and continues to serve training and uncached inference.

| Function | Contract and call site |
| --- | --- |
| [`PolicyKVCache(...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L32) | Called lazily by [`sample_block.predict`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L294) for each text branch. Builds the original pack positions once and prefills text through UND, storing each layer's GEN-facing text K and projected V. |
| [`advance(block)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L171) | Called by cached prediction. Appends measured clean video up to the requested prediction boundary; allows repeated calls for the same block and forward jumps, rejects backward access. Block/frame numbering starts at zero, with independent latent frame 0. |
| [`predict(video, action, timestep, block)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L181) | Called by `sample_block.predict`. Accepts the existing normalized FP32 sampler state and model time in 0..1000. Returns only this block's video/action velocity and action row indexes. After `advance`, denoising reads history without modifying it. |

`sample_block` scatters these velocities into its existing full-size FP32
sampling state. Existing masks, independent noise streams, CFG intervals,
velocity normalization, UniPC/fixed-step updates and condition restoration
remain shared with uncached inference. The teacher-forcing loop retains its
original VAE encoding, absolute physical action decoding and video export.

## Prefill and visibility

- Text self-attention remains causal. Cache GEN-facing normalized/rotated text
  K separately from UND self-attention K when `k_norm_und_for_gen` is enabled.
- Prefill the independent first latent alone. Then prefill each complete clean
  training block jointly: clean queries read text, their old history window,
  and all clean frames in their own block. A two-latent block must not be
  appended one latent at a time.
- Save video K after normalization/RoPE and V after projection for every layer.
  Finish every layer before committing the new block. An exception leaves
  previous layers, frame IDs and the append cursor unchanged.
- Evict only after computing the new clean block, retaining the latest
  `history` latent frames. This budget may cut a block and can be smaller than
  the block size. It uses VeOmni's existing frame budget, not a block count.
- Current target queries read text, historical clean KV and same-block
  video/action/state KV. State queries read text and their own state only.
  State, noisy target and generated video KV never enter measured history.

All RoPE coordinates are sliced from the original pack, including after history
eviction; compact execution must not renumber positions. This preserves the
original video/action FPS mapping and state anchor coordinates. Clean input
timesteps remain zero. Only current target video/action receive the sampling
timestep.

Historical token projections, attention queries and FFNs do not execute during
denoising. Current state still executes per step; it is not a long-term cache.
Each branch/layer retains text KV and at most `history * patches_per_frame`
video KV rows. Persistent cache memory therefore grows with layers, KV heads,
frame patch count and history budget; attention still reads that history.

## RoboTwin online session

The [RoboTwin policy](../../veomni/models/diffusers/causal_cosmos3/robotwin_policy.py#L99)
owns one framework-provided
[`OnlinePolicyKVCache`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L231)
per text branch, and one isolated
[`StreamingVideoEncoder`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L123).
These helpers are internal to the native Policy backend; other models do not
implement or override them. Serving enables them through `sampling.kv_cache: true`
or `--kv-cache`. See the [launch and protocol guide](../usage/causal_cosmos3_robotwin.md).

| Framework-provided call | Owner and effect |
| --- | --- |
| [`stream_encoder()`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L101) | Policy reset creates a new encoder with empty convolution features. Features belong to this session and survive unrelated VAE encoding and prediction-video decoding. |
| [`encode(video)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L140) | The policy supplies resized uint8 `[3,T,H,W]` real observations: exactly one initial frame, then positive multiples of the temporal factor with unchanged spatial geometry. Returns normalized FP32 latent frames. Validation rejects incomplete input; an encoder execution failure clears its feature state. |
| [`OnlinePolicyKVCache(...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L239) | The first request creates each branch from a compact sample containing an actual initial latent plus one unknown prediction block. Prefills text and the initial latent once. |
| [`update(sample, measured, revision=...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L261) | Each subsequent request supplies the new measured latent block and latest normalized state. Requires the next consecutive revision, fixed text/masks/geometry and a complete measured block. Joint prefill commits all layers, then evicts to the latent-frame budget. Rebinds compact targets without re-running text or old video. |

The request flow is:

1. The server validates the observation counter and full-block feedback.
2. The policy selects real RGB endpoints and calls the isolated streaming encoder.
3. The policy creates or updates conditional/negative-text caches independently.
4. `sample_block` calls cached `predict` on compact block zero for every denoise
   step; measured history remains read-only. Absolute episode frame/action indexes
   reproduce the full-pack FP32 position calculation, including after eviction.
5. The policy decodes absolute actions against the latest physical state. No state,
   predicted action, future placeholder or generated video is appended to observed KV.

Reset, disconnect and model execution errors drop all branches and VAE features;
the loaded weights remain available. A partially updated multi-branch request is
never retried with its old cache: the server requires reset after execution failure.
`--no-kv-cache` uses the older bounded-RGB reconstruction path. Its truncated VAE
prefix and rebased positions differ on long episodes, so numerical parity tests
use an untruncated measured episode with identical target noise instead.

## Usage and validation

```bash
OUTPUT_DIR=outputs/policy_reference \
  bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh --no-kv-cache
OUTPUT_DIR=outputs/policy_cache \
  bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh --kv-cache
```

Keep checkpoint, source window, statistics, preprocessing, dtype, operators,
noise and sampler schedule identical. The cache preserves mathematical
visibility, but changes matrix shapes and kernel grouping. FP32 and BF16 need
separate error measurements; bitwise equality is not promised. Compare
per-step velocities and final normalized/physical actions and video latents,
then assess exported pixels and timing separately.

The existing
[`inference tests`](../../tests/models/diffusers/test_causal_cosmos3_inference.py)
cover sampling parity across Edge/Nano/Super, CFG, partial-frame history
eviction, exact layer visibility against full-pack KV, read-only prediction,
compact projection sizes, atomic failure and invalid-mode guards. They belong
to the already collected `tests/models/` accelerator unit selection.
Online cases additionally compare five consecutive requests with a full measured
episode across Edge/Nano/Super, CFG branches with different text lengths and
history budgets of 1, 3 and 32 latent frames. Real small Wan VAE tests verify
streaming/full-episode encoding and independence from decode. RoboTwin tests
exercise cache reuse, reset/error cleanup and local WebSocket reconnects.
