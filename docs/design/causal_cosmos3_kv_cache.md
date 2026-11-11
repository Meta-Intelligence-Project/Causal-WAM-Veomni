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
| [`PolicyKVCache(...)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L31) | Called lazily by [`sample_block.predict`](../../veomni/models/diffusers/causal_cosmos3/inference.py#L294) for each text branch. Builds the original pack positions once and prefills text through UND, storing each layer's GEN-facing text K and projected V. |
| [`advance(block)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L159) | Called by cached prediction. Appends measured clean video up to the requested prediction boundary; allows repeated calls for the same block and forward jumps, rejects backward access. Block/frame numbering starts at zero, with independent latent frame 0. |
| [`predict(video, action, timestep, block)`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py#L169) | Called by `sample_block.predict`. Accepts the existing normalized FP32 sampler state and model time in 0..1000. Returns only this block's video/action velocity and action row indexes. After `advance`, denoising reads history without modifying it. |

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
