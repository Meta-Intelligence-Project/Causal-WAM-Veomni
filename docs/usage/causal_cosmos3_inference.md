# Causal Cosmos3 independent inference

The [CLI](../../tasks/infer/infer_causal_cosmos3.py) runs a complete trained model
on one NPU, reusing the native network, action transforms, tokenizer, packing
and Wan VAE. It creates no Trainer, optimizer, training FSDP wrapper or validation
callback. Existing training and validation loss are unchanged.

## Two conditioning modes

- `teacher_forcing`: every block uses measured clean video history and its
  measured state. `forward_dynamics` conditions on measured actions;
  `inverse_dynamics` generates video and action in isolated streams: actions
  condition on the measured video copy while video targets use text and visual
  history. `policy` predicts both jointly.
  `--task all` runs all three separately. Predictions are saved, while the next
  block again uses ground truth. This is a preview under measured conditions.
- `rollout`: Policy starts with exactly one image/state observation. Generated
  RGB uint8 pixels are decoded and re-encoded through the VAE for history.
  The last valid absolute action becomes the next block's state. The reader
  requests one state row and one timestamp per camera, without future targets
  or future subtask text. This is model-only open-loop feedback, assuming the
  predicted targets are executed; it does not measure environment success.
  `--task inverse_dynamics` also starts from one observation. Each block first
  generates video, then uses the completed generated latents as the observed
  video copy to infer action. Both passes retain IDM attention. Generated RGB
  pixels and the final valid absolute action feed the next block. Neither pass
  reads measured future video or actions.

## Optional KV reuse across denoising steps and blocks

`sampling.use_kv_cache` is a YAML boolean, defaulting to `false`. The CLI
`--use-kv-cache` enables it and `--no-use-kv-cache` overrides YAML back to the
uncached path. The effective setting is included in `run.json`.

```yaml
sampling:
  use_kv_cache: false
```

Each inference request owns a private cache session for the
[block sampler](../../veomni/models/diffusers/causal_cosmos3/inference.py),
with separate conditional/unconditional text branches. It passes
that state to the network only when enabled. On a cache miss, the first forward computes the
whole network and records each layer's text K/V (including the GEN-specific
text key normalization), clean/observed-video K/V, state K/V, and fixed hidden
outputs. Later forwards compute only the remaining GEN query rows, refresh
their K/V slots, and reuse fixed rows. Training and validation loss do not use
this path. The network rejects cache use with training mode, enabled autograd,
or loss targets.

Cache ownership and lifecycle are implemented in
[`inference_cache.py`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/inference_cache.py),
with consumption in the model forward and
[`layers.py`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/layers.py).
This is private sampler state, not a general `past_key_values` API: text,
observed latents, states, weights and position coordinates must remain fixed
while full layer caches are reused. Teacher forcing owns an immutable encoded
window and reuses its fixed rows across all blocks. Rollout creates a new
block workspace on each sampling call and retains text plus verified clean
history snapshots across stages and blocks. Text reuse checks token IDs, position coordinates, sample grouping,
model identity, dtype, device and backend when binding a new window. Weights
must remain immutable during a request. The fixed-window contract relies on
sample identity and caller ownership, not a tensor-content hash: do not mutate
an encoded teacher-forcing sample in place. Nothing is retained on the model
or shared between requests. A standalone `sample_block()` call without a
private session still has call-local lifetime.

| Execution | Reuse boundary |
| --- | --- |
| Teacher forcing, any task | Reuse text and all fixed observed-video/state rows across blocks of the same encoded window. |
| Policy rollout | Reuse text and an exactly matching clean history prefix; recompute new/changed video and all states. |
| IDM rollout, video pass | Reuse verified history from the preceding block; compute current observations/targets and states. |
| IDM rollout, action pass | Verify/reuse historical clean rows from the video pass; recompute the current clean observation after generated video replaces its placeholder. |
| CFG | Positive and negative prompts have separate caches, including all text-dependent video/state features. |

Rollout history reuse is enabled by the same `use_kv_cache` switch. It preserves
pixel feedback and VAE re-encoding. Once per sampling prefill, it compares the
clean video tokens at the model input precision, timesteps and RoPE coordinates
against owned snapshots. Only historical role-0 rows before the current block
are stored; noisy target KV, current/future placeholders and states are never
promoted into cross-block video history. A changed frame invalidates its entire
causal block (clean frames within a block attend each other) and every later
block. An unchanged earlier prefix can still be reused. A moved local window,
changed history limit, task, geometry or text branch invalidates incompatible
history. This conservative policy does not reuse stale visual features after
VAE re-encoding. It can miss most/all historical rows, so it is not a guarantee
of additional speedup.

Each layer injects the matching history K/V into the new block's full workspace,
computes only fresh prefill query rows, and restores cached history hidden
outputs. Later denoising steps use the existing block-local cache. History
snapshots are published only after all decoder layers finish, and own their
storage independently of mutable denoising buffers. Weights must remain fixed
for the request. Snapshots also retain input tokens for validation and increase
memory use; validation includes a device-to-host comparison transfer once per
prefill, not per layer or denoising step.

Rollout `blocks.jsonl` records include `kv_cache_totals` when caching is enabled:
`prefill_calls`, `candidate_tokens`, and `reused_tokens` are cumulative across
sampling stages and CFG branches, counted once per prefill (not multiplied by
decoder depth). Subtract successive block records for that block's counts.
`reused_tokens > 0` confirms actual history rows skipped decoder work; it does
not measure latency or include text/block-local cache hits. With CFG disabled,
policy normally has one prefill per block and IDM has two (video then action).

Only text, role-0 observed video and role-3 states are cached. Role-2 condition
rows are conservatively recomputed: in IDM their attention can depend on
target rows. Dynamic actions/video are never reused from a previous timestep.
The current implementation also recomputes packing, input projections, rotary
tables and output projections; it caches the attention plan and fixed
Transformer work, not the whole inference pipeline. It keeps full K/V
workspaces and fixed hidden outputs, increasing memory use. Cache mode is
mathematically equivalent but backend/GEMM rounding can differ; compare with
appropriate dtype tolerances rather than requiring bit-identical samples.

```bash
# The existing launcher forwards these CLI overrides unchanged.
MODES=teacher_forcing EPISODES=0 OUTPUT_DIR=outputs/idm_cached \
  bash scripts/causal_cosmos3/infer_idm_robotwin.sh --use-kv-cache
MODES=teacher_forcing EPISODES=0 OUTPUT_DIR=outputs/idm_uncached \
  bash scripts/causal_cosmos3/infer_idm_robotwin.sh --no-use-kv-cache
```

Use the same checkpoint, inputs, seed, schedule and backend for comparisons;
choose separate output directories. Measure synchronized device latency and
peak memory before drawing speed conclusions. No acceleration factor is
guaranteed by this switch.

## Load trained weights

[`CausalCosmos3Pipeline`](../../veomni/models/diffusers/causal_cosmos3/inference.py)
exposes `teacher_forcing(dataset, episode, start, horizon, task=...)` and
`rollout(dataset=... | input_npz=..., horizon=..., task=..., ...)`. It requires the final
trained config, including the shared 55D action/state interfaces. Missing trained
parameters fail. Base-model initialization skip rules are disabled.
The pipeline, block sampling, generation loops and result export share the
single `inference.py` entrypoint, following MiniMax H3's organization. The UniPC
scheduler lives in `cosmos3_core/unipc.py`.

Native VeOmni DCP step directories (`model/ckpt/.metadata`) can be loaded
directly, including a checkpoint saved by multiple ranks. The loader reads
only model weights in bounded shards without initializing a process group or
opening optimizer/trainer state. It uses the run's `model_assets/config.json`;
the optional `model_config` inference setting overrides that path. A raw DCP
weights directory is also accepted. Missing trained tensors and shape/name
mismatches fail as they do for HF weights. The existing
[HF export](checkpoint_conversion.md) remains available.

For legacy Cosmos-framework DCP, run the standalone
[conversion tool](../../scripts/causal_cosmos3/convert_inference_checkpoint.py)
before starting inference:

```bash
python scripts/causal_cosmos3/convert_inference_checkpoint.py \
  --checkpoint /path/to/cosmos/checkpoints/iter_000001000 \
  --model-config configs/model_configs/causal_cosmos3/causal_cosmos3_edge.json \
  --output outputs/poc_iter_1000_hf
```

The converter reads the run's `config.yaml` with a primitive YAML loader;
`--training-config` overrides its location. It combines the architecture with
trained action/state dimensions, checks every name and shape, and exports model
tensors in bounded safetensors shards. It never opens optimizer/trainer state.
`conversion_report.json` records source identity, mappings, shapes, dtypes and
dtype conversion errors. The destination must be new; source files are retained.
All legacy metadata compatibility and parameter mapping live in this tool.
The inference package loads the resulting complete native/HF checkpoint and
does not import or run the converter.

Set `vae` to the matching Diffusers VAE directory containing `config.json` and
its model weights, using the existing `AutoencoderKLWan.from_pretrained` loader.
Encoding, normalization, camera layout and RGB resize follow the native training
conventions.

## Run inference

Edit [the configuration](../../configs/infer/causal_cosmos3.yaml) for the
tokenizer, VAE, measured statistics, cameras and device. Its resource paths
refer to the authorized 1030 POC example. Keep the exported model's `config.json`
with its weights.

For the RoboTwin IDM run, use
[`causal_cosmos3_idm_robotwin.yaml`](../../configs/infer/causal_cosmos3_idm_robotwin.yaml).
It preserves the training statistics, cameras, geometry, 64-latent history and
`state_unified` / `action_unified` columns with `target_offset=0`.
`source.rollout_state: action_as_state` explicitly declares that absolute
predicted commands can act as the next physical state. This declaration still
requires matching channel mappings and masks. It models execution of the
command without a simulator. Other sources retain the stricter default
requiring identical state/target columns and `target_offset=1`.

In an activated Torch/CANN environment, run three episodes in both modes:

```bash
EPISODES="0 1 2" MAX_ACTIONS=64 STEPS=20 \
  bash scripts/causal_cosmos3/infer_idm_robotwin.sh
```

`CHECKPOINT` defaults to
`outputs/causal_cosmos3_idm_robotwin/checkpoints/global_step_4000`.
Override `DATASET`, `OUTPUT_DIR`, `START`, `MODES`, or `PYTHON_BIN` as needed;
additional CLI arguments are forwarded. A single mode/episode can also use the
CLI directly:

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_idm_robotwin.yaml \
  --checkpoint outputs/causal_cosmos3_idm_robotwin/checkpoints/global_step_4000 \
  --dataset /mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__adjust_bottle/clean \
  --episode 0 --max-actions 64 --mode rollout --task inverse_dynamics \
  --output outputs/idm_robotwin_episode0_rollout
```

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode teacher_forcing --task all \
  --dataset /path/to/lerobot_v3 --episode 2 --start 0 --max-actions 64 \
  --output outputs/teacher_episode2

python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode rollout --task policy \
  --dataset /path/to/lerobot_v3 --episode 2 --start 0 --max-actions 366 \
  --output outputs/rollout_episode2

python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode rollout --task policy \
  --input-npz /path/to/initial_observation.npz --fps 30 --max-actions 64 \
  --output outputs/rollout_npz
```

`--steps`, `--history`, `--seed`, `--guidance`, `--prompt`, `--device`, `--dtype`,
`--attention-backend`, `--guidance-interval LO HI`, `--normalize-cfg` /
`--no-normalize-cfg` and `--fixed-step-sigmas SIGMA ...` override configuration.
The CLI initializes NPU only for an NPU `device`, such as `npu:0`. For CPU
checks, select `device: cpu`, `attention_backend: grouped` and eager operations.
Defaults in the POC configuration are BF16, 20 UniPC steps, guidance 1 and 32
history latents; the RoboTwin recipe uses 64 history latents. Training evaluation
cadence is separate.

`ops_implementation` must match the training recipe. The NPU example uses
`torch_npu` RoPE and RMSNorm, as in `configs/dit/causal_cosmos3_edge_action.yaml`.
Switching these operations to eager changes BF16 rounding and model predictions.
The low-level `load_trained_model` API accepts the same mapping; without it,
that API retains eager operations for standalone CPU checks.

Sampling extensions use the same block sampler for all three teacher-forcing
tasks, policy rollout and both passes of IDM rollout:

| `sampling` option | Behavior |
| --- | --- |
| `guidance_interval: [200, 800]` | Apply CFG only when `200 < timestep < 800`, using model time (`1000 * sigma`). Bounds are strict. Outside this interval use the conditional prediction and skip the unconditional forward. `null` applies CFG at every step when guidance differs from 1. |
| `normalize_cfg: true` | Scale the guided velocity by `min(1, norm(conditional) / (norm(guided) + 1e-8))`. Compute one norm across the active video/action vector; this only reduces magnitude. Default: `false`. |
| `fixed_step_sampler_config` | An explicit `t_list` and `sample_type: sde` select fixed-step distilled sampling. A null/missing value inherits this field from the checkpoint config; otherwise use UniPC. |

For a matching DMD2-distilled checkpoint, the schedule can be stored in its
`config.json` or supplied explicitly:

```yaml
sampling:
  fixed_step_sampler_config:
    t_list: [0.999, 0.75, 0.5, 0.25]
    sample_type: sde
```

The [fixed-step sampler](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/fixed_step.py)
appends a final zero, predicts `x0 = x_sigma - sigma * velocity`, then draws fresh
noise to construct the next sigma state. `steps` and `shift` do not change this
schedule. The pipeline follows the source distilled entry by resolving guidance
to 1 and exporting that effective setting in `run.json`. The standalone legacy
checkpoint converter preserves the distilled schedule metadata.

Initial video/action noise and fixed-step re-noising use separate block-local
Generators. Fixed-step block records include a `sampler` seed as well as the
existing `video` and `action` seeds. Conditions and invalid channels are restored
after every update. Fixed-step quality requires matching distilled weights;
running its code with an ordinary flow-matching checkpoint only checks runtime
compatibility.

## Observation, sampling and feedback contract

For the POC, state/target both use `action_unified`, masks use `mask_action`, and
`target_offset=1`. `state_unified` is a zero placeholder. Targets must represent
next-step state with complete feedback channels. Initial-observation readers reject incompatible
column offsets and state/action masks. Use measured
training statistics. POC XYZ is in millimeters, joints in radians, grippers in
meters and quaternions in xyzw order; display conversion must not change inputs.
Unit labels are optional metadata. Inference does not validate a unit whitelist
or convert physical units; input values and normalization statistics must use
the same conventions as training. Agibot grippers can retain their native scale.

Episode `source_fps` overrides in `meta/episodes.jsonl` are read just as during
training. They control prompt FPS and mRoPE spacing; storage FPS still determines
which video frames are decoded. Initial NPZ inputs use the explicit FPS instead.

An initial NPZ contains physical floating-point `state[55]`, RGB/HWC uint8 camera
arrays and a nonempty `prompt` (or `--prompt`). `npz_cameras` maps camera names.
Supply `state_mask[55]` and `action_mask[55]` for a custom robot; the documented
Piper schema also supports its fixed valid channel layout. `metadata_json` may
describe RGB/HWC layout and robot. NPZ requires explicit FPS and horizon.

Only current-block target regions are sampled. The first, observed video copy,
conditioned actions and invalid channels remain fixed. An explicit
`target_frame_indexes_vision` independently selects the second video copy's
targets, including for IDM. Video/action use the same block-time
mapping; condition times are zero. UniPC updates FP32 state with a deterministic
steps/shift schedule. CFG changes text while retaining physical conditions.
The current native training recipe has no explicit caption dropout, so guidance
above 1 requires checking that the checkpoint learned unconditional text inputs.
Video/action seeds are independent per block. `sample_block(initial_noise=...)`
accepts actual noise tensors for algorithm comparisons across implementations.

History is cropped at complete block boundaries, retaining an independent first
frame. Partial final blocks use full geometry and export only valid outputs.
Feedback selects the final valid action before padding. Video is sampled every
`video_stride` actions: 366 actions with stride 4 produce 92 frames, ending at
observation 364, and retain all 366 action outputs. Autonomous generation does
not read future ground truth.

## Artifacts and verification

Each new output directory contains `run.json`, `blocks.jsonl`, `generated.mp4`,
`actions.npz` and `actions.svg`. Arrays contain absolute physical actions, valid
channel masks and action/video timestamps. SVG blue curves are predictions;
orange curves are measured targets when available. Teacher forcing also saves
ground-truth and VAE reconstruction videos and target action arrays. Run metadata
records effective configuration, checkpoint and input semantics. Block records
include anchors, feedback state, seeds, history, valid lengths, actual latent
shapes and temporal padding.
Action timestamps use the resulting-state convention: action zero is at the
initial observation time plus `1 / fps`. In RoboTwin this is the resulting
state's time, one step after the corresponding command row's timestamp.

Native VeOmni buckets, resize and padding are preserved. Different reference
preprocessing can change latent shape, noise and final pixels despite equal
seeds. Compare algorithms using common prepared tensors/noise, and record each
native pipeline's geometry for end-to-end comparisons. The POC checkpoint used
no held-out split, so these examples are training-data replay.

Regression coverage is in
[`test_causal_cosmos3_inference.py`](../../tests/models/diffusers/test_causal_cosmos3_inference.py)
and the inference cases in
[`test_causal_cosmos3_pipeline.py`](../../tests/data/test_causal_cosmos3_pipeline.py).
CPU and tiny-model checks establish contracts. NPU regression coverage also
compares training and inference BF16 predictions with their configured operations.
Full-checkpoint hardware acceptance records are kept in
`dev-docs/causal_cosmos3_train_infer_audit_20261028/` in the surrounding workspace.
Graph compilation, EMA, multi-device inference
and environment control are outside this implementation.
