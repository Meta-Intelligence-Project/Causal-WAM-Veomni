# Causal Cosmos3 online Policy KV cache and measured replay

## Ownership and defaults

The cache is enabled by default for single-process teacher-forcing Policy
inference. `sampling.kv_cache: false` or `--no-kv-cache` disables it;
`--kv-cache` explicitly enables it. FD, ID and rollout require disabling the
cache. Training and gradient-enabled cache use are rejected explicitly.
The model weights and checkpoint layout do not change.

Live serving and cached offline teacher forcing both use `OnlinePolicyKVCache`.
The session owns a fresh branch dictionary and an isolated streaming VAE encoder;
offline inference replays its complete sample as incrementally arriving RGB.
Each cache owns one eval model, immutable text/geometry and rolling measured KV.
It never owns a prediction sample, state or target placeholders. Conditional and
negative-text branches must not share video KV:
their deep video states and text-dependent position offsets can differ.
Branch reuse validates the model identity, text IDs, geometry, revision, position
options and attention backend. Prediction masks are validated per call; they are
not part of observation history. Calling teacher forcing again creates new caches; there is no global cache or
attention dispatcher replacement.

## Framework-provided interface

These are inference implementation helpers, not methods that another model
must implement. The sampling loop can disable caching; existing full-pack forward is
the reference and continues to serve training and uncached inference.

| Function | Contract and call site |
| --- | --- |
| [`create_policy_caches(...)`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L231) | Called by offline teacher forcing and the online policy with one real initial latent, positive-text IDs and both FPS values. Uses pipeline geometry/options to create independent conditional/negative-text caches before prediction samples are built. |
| [`OnlinePolicyKVCache(model, initial_latent, ...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L31) | Session-owned cache. Accepts exactly one real `[C,1,H,W]` latent on the model device, nonempty long text IDs and fixed block/action/history/FPS/position configuration. Prefills text and the initial latent once. Does not accept states, actions or unknown target slots. |
| [`update(measured, revision=...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L268) | The caller supplies one complete measured `[C,B,H,W]` latent block with unchanged spatial geometry/device and the next consecutive revision. Appends observed KV independently of prediction metadata or states. |
| [`predict(video, action, state, timestep, ...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L215) | Called by [`sample_block.predict`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L366). Receives only current noisy video `[C,B,H,W]`, actions `[A,action_dim]`, one measured state `[1,state_dim]`, boolean feature masks and the current revision. Returns video/action velocities without changing history or retaining inputs. |
| [`validate_branch(...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L108) | The sampler validates its prediction sample's geometry, Policy conditions, local indexes and `cache_revision`, plus branch text/model/options. Sample identity is not bound; equivalent prediction samples are allowed. |

`sample_block` scatters these velocities into its compact FP32
sampling state. Existing masks, independent noise streams, CFG intervals,
velocity normalization, UniPC/fixed-step updates and condition restoration
remain shared with uncached inference. Cached calls require local block zero
and preinitialized branch caches; the sampler never constructs or updates a
cache. The old full-sample cache and implicit `advance(block)` API are removed.

The sampler still uses a compact sample with a layout prefix and target slots
to preserve its shared conditioning/noise/scheduler interface. That sample stays
outside the cache. The cache neither packs it nor projects its prefix during
prediction; the sampler passes only the target slice and explicit state/masks.

For example, with two latent frames and 32 actions per block:

```python
with torch.no_grad():
    cache = OnlinePolicyKVCache(
        model.eval(),
        initial_latent,  # [C, 1, H, W], only the real first observation
        text_ids=text_ids,
        block_size=2,
        actions_per_block=32,
        history=32,
        conditioning_fps=7.5,
        conditioning_fps_action=30.0,
    )
    velocity_video, velocity_action = cache.predict(
        noisy_video,  # [C, 2, H, W], no prefix slot
        noisy_action,  # [32, action_dim]
        state,  # [1, state_dim]
        timestep,
        action_mask=action_mask,
        state_mask=state_mask,
        revision=0,
    )
    cache.update(measured_latents, revision=1)  # [C, 2, H, W]
```

The constructor fixes `block_size`, `actions_per_block`, `history` (latent frames),
video/action FPS, spatial dimensions and position options. `predict` receives
normalized sampler values and time in model units 0..1000. Masks are one-dimensional
boolean valid-feature masks. Revision identifies the absolute prediction block;
the cache constructs RoPE tables directly from fixed geometry once per revision,
without full-sample packing or retained state/action payloads.

## Offline measured replay

The [teacher-forcing loop](../../veomni/models/diffusers/causal_cosmos3/inference.py#L469)
owns the full sample and evaluation targets. Each prediction follows this order:

1. Encode only the initial RGB frame on the first iteration. On iteration `k > 0`,
   encode only the preceding complete block of newly arrived real RGB frames.
   The isolated encoder keeps its causal convolution state across iterations.
2. Initialize the caches from the first real latent and text, or update each
   branch with the new measured block. These operations do not receive a state
   or prediction sample.
3. Build a [compact sample](../../veomni/models/diffusers/causal_cosmos3/inference.py#L254)
   containing one true normalized block state, unknown video/action placeholders,
   local frame indexes and fixed text/geometry. Full RGB, future GT and physical
   anchors stay outside the cache. Only iteration zero populates the sampler's
   prefix slot; later compact prefixes are placeholders, not history.
4. Predict local block zero using `seed_block=k`. The sampler passes the noisy
   target slice, measured state/masks and `cache_revision=k` explicitly to
   `predict`; absolute positions are maintained by the cache.
5. Decode actions against the dataset's measured block anchor. Decode predicted
   video together with the actual measured latent prefix, never the compact zero
   prefix or previously generated video. Retain only valid output steps/frames.

A short final block is padded for prediction but does not produce another
observation update. Padded/generated pixels never enter measured KV or the
observation encoder. Full-window GT VAE encoding happens after all predictions,
solely for reconstruction export. Replaying a dataset still uses measured
states and observations; it does not model how predicted actions change an
environment. FD/ID and `--no-kv-cache` keep full-window inference.

## Prefill and visibility

- Text self-attention remains causal. Cache GEN-facing normalized/rotated text
  K separately from UND self-attention K when `k_norm_und_for_gen` is enabled.
- Prefill the independent first latent alone. Then prefill each complete clean
  training block jointly: clean queries read text, their old history window,
  and all clean frames in their own block. A two-latent block must not be
  appended one latent at a time.
- Save video K after normalization/RoPE and V after projection for every layer.
  Finish every layer before committing the new block. An exception leaves
  previous layers, prediction RoPE tables, revision, frame IDs and the append
  cursor unchanged.
- Evict only after computing the new clean block, retaining the latest
  `history` latent frames. This budget may cut a block and can be smaller than
  the block size. It uses VeOmni's existing frame budget, not a block count.
- Current target queries read text, historical clean KV and same-block
  video/action/state KV. State queries read text and their own state only.
  State, noisy target and generated video KV never enter measured history.

RoPE coordinates are calculated at absolute frame/action indexes with the same
FP32 arithmetic as a full pack, including after history eviction. This preserves the
original video/action FPS mapping and state anchor coordinates. Clean input
timesteps remain zero. Only current target video/action receive the sampling
timestep.

Historical token projections, attention queries and FFNs do not execute during
denoising. Current state still executes per step; it is not a long-term cache.
Each branch/layer retains text KV and at most `history * patches_per_frame`
video KV rows. Persistent cache memory therefore grows with layers, KV heads,
frame patch count and history budget; attention still reads that history.

## Unified55 online session

The [online policy](../../tasks/client_server/server/policy.py#L103)
owns one framework-provided
[`OnlinePolicyKVCache`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L23)
per text branch, and one isolated
[`StreamingVideoEncoder`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L123).
These helpers are internal to the native Policy backend; other models do not
implement or override them. Serving enables them through `sampling.kv_cache: true`
or `--kv-cache`. See the [launch and protocol guide](../usage/causal_cosmos3_robotwin.md).

| Framework-provided call | Owner and effect |
| --- | --- |
| [`stream_encoder()`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L101) | Policy reset creates a new encoder with empty convolution features. Features belong to this session and survive unrelated VAE encoding and prediction-video decoding. |
| [`encode(video)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L140) | The policy supplies resized uint8 `[3,T,H,W]` real observations: exactly one initial frame, then positive multiples of the temporal factor with unchanged spatial geometry. Returns normalized FP32 latent frames. Validation rejects incomplete input; an encoder execution failure clears its feature state. |
| [`OnlinePolicyKVCache(...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L31) | The first request creates each branch from the real initial latent through the same `create_policy_caches` helper as offline replay. Prefills text and the initial latent once. |
| [`update(measured, revision=...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L268) | Each subsequent request supplies only the new complete measured latent block and next consecutive revision. Joint prefill commits all layers, then evicts to the latent-frame budget. Current state/masks belong to prediction calls. |

The request flow is:

1. The server validates the observation counter and full-block feedback.
2. The policy selects real RGB endpoints and calls the isolated streaming encoder.
3. The policy creates or updates conditional/negative-text caches independently.
4. `sample_block` passes the noisy target block and latest state/masks/revision to
   cached `predict` for every denoise step; measured history remains read-only. Absolute episode frame/action indexes
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
noise and sampler schedule identical. A seed alone does not align full-window
and compact sampling: different tensor shapes change initial noise and fixed-step
SDE noise placement. Strict parity tests supply corresponding initial-noise
slices with UniPC; they do not assume CLI outputs match under the same seed.
Streaming/full-window VAE encoding and exported pixels are checked separately.
The cache preserves mathematical
visibility, but changes matrix shapes and kernel grouping. FP32 and BF16 need
separate error measurements; bitwise equality is not promised. Compare
per-step velocities and final normalized/physical actions and video latents,
then assess exported pixels and timing separately.

The existing
[`inference tests`](../../tests/models/diffusers/test_causal_cosmos3_inference.py)
cover sampling parity across Edge/Nano/Super, CFG, partial-frame history
eviction, exact layer visibility against full-pack KV, read-only prediction,
compact projection sizes, initialization from a single real latent without targets,
explicit state changes with read-only KV, stale-revision guards, atomic failure
and invalid-mode guards. They belong
to the already collected `tests/models/` accelerator unit selection.
Replay cases compare five consecutive requests with a full measured
episode across Edge/Nano/Super, CFG branches with different text lengths and
history budgets of 1, 3 and 32 latent frames, with 16/32-action blocks. Offline
pipeline tests verify incrementally arriving RGB, measured states, branch reuse,
decode context, short tails and evaluation-only GT encoding. Real small Wan VAE tests verify
streaming/full-episode encoding and independence from decode. RoboTwin tests
exercise cache reuse, reset/error cleanup and local WebSocket reconnects.
