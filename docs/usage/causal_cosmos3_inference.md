# Causal Cosmos3 independent inference

The [CLI](../../tasks/infer/infer_causal_cosmos3.py) runs a complete trained model
on one NPU, reusing the native network, action transforms, tokenizer, packing
and Wan VAE. It creates no Trainer, optimizer, training FSDP wrapper or validation
callback. Existing training and validation loss are unchanged.

## Two conditioning modes

- `teacher_forcing`: every block uses measured clean video history and its
  measured state. `forward_dynamics` conditions on measured actions;
  `inverse_dynamics` conditions on measured video; `policy` predicts both.
  `--task all` runs all three separately. Predictions are saved, while the next
  block again uses ground truth. This is a preview under measured conditions.
- `rollout`: Policy starts with exactly one image/state observation. Generated
  RGB uint8 pixels are decoded and re-encoded through the VAE for history.
  The last valid absolute action becomes the next block's state. The reader
  requests one state row and one timestamp per camera, without future targets
  or future subtask text. This is model-only open-loop feedback, assuming the
  predicted targets are executed; it does not measure environment success.

## Load trained weights

[`CausalCosmos3Pipeline`](../../veomni/models/diffusers/causal_cosmos3/inference.py)
exposes `teacher_forcing(dataset, episode, start, horizon, task=...)` and
`rollout(dataset=... | input_npz=..., horizon=..., ...)`. It requires the final
trained config, including the shared 55D action/state interfaces. Missing trained
parameters fail. Base-model initialization skip rules are disabled.
The pipeline, block sampling, generation loops and result export share the
single `inference.py` entrypoint, following MiniMax H3's organization. The UniPC
scheduler lives in `cosmos3_core/unipc.py`.

For VeOmni DCP, use the existing [HF export](checkpoint_conversion.md), retaining
its matching model assets. For legacy Cosmos-framework DCP, run the standalone
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
The CLI always initializes NPU
and selects the configured `device`, such as `npu:0`. Defaults are BF16, 20 UniPC
steps, guidance 1 and 32 history blocks. Training evaluation cadence is separate.

`ops_implementation` must match the training recipe. The NPU example uses
`torch_npu` RoPE and RMSNorm, as in `configs/dit/causal_cosmos3_edge.yaml`.
Switching these operations to eager changes BF16 rounding and model predictions.
The low-level `load_trained_model` API accepts the same mapping; without it,
that API retains eager operations for standalone CPU checks.

Sampling extensions use the same block sampler for all three teacher-forcing
tasks and policy rollout:

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

Only current-block unknown regions are sampled. Observed video, conditioned
actions and invalid channels remain fixed. Video/action use the same block-time
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
KV caching, graph compilation, EMA, multi-device inference
and environment control are outside this implementation.

## Warm rollout profiling on Ascend 910B

From the repository root, with the NPU Python environment activated:

```bash
bash scripts/inference/infer_causal_cosmos3.sh
# Select another physical card or checkpoint:
ASCEND_RT_VISIBLE_DEVICES=6 CHECKPOINT=ckpts/episode_iter_000020000_hf \
  OUTPUT=outputs/my_profile bash scripts/inference/infer_causal_cosmos3.sh
# Optional operator/memory trace (run separately from the latency baseline):
bash scripts/inference/infer_causal_cosmos3.sh --profile-npu-trace
```

The [script](../../scripts/inference/infer_causal_cosmos3.sh) defaults to physical
card 6 / logical `npu:0`, BF16, 20 sampling steps, Agibot task 352 episode 0,
and `--max-actions 640` (20 consecutive 32-action chunks). It uses the tracked
[Agibot configuration](../../configs/infer/causal_cosmos3_agibot.yaml), matching
the IDE setup. `PYTHON`, `CONFIG`, `CHECKPOINT`, `DATASET`, `EPISODE`, `START`,
`OUTPUT`, and `WARMUP_RUNS` are environment overrides. Extra arguments are passed
to the CLI last. Default output directories are unique; use a new directory for
each run. Unless `PYTHON` is set, the script prefers `.venv/bin/python`, then
the existing `/mnt/sfs_turbo/public/apps/miniforge3/envs/infer-wam-veomni/bin/python`,
then `python` on PATH. Local asset paths must exist on the machine running the script.

`--profile` enables synchronized timing for rollout only. By default, one full
rollout of the same horizon warms all growing-history shapes, including the
saturated history window. Its outputs are discarded. The measured rollout
restarts from the original observation with the same seeds in the same process,
retaining loaded weights, compiled kernels and allocator caches. No cache flush
occurs between passes. `--profile-warmup-runs N` requires N >= 1; increase N to
check stability. This measures a warm process with a fresh episode, not a
continuation after 20 already-executed chunks. All 20 measured chunks, including
the short initial context, contribute to the arithmetic mean. For another
geometry/horizon the actual chunk count is recorded; a partial tail counts as
one chunk.

[Rollout instrumentation](../../veomni/models/diffusers/causal_cosmos3/inference.py)
records these disjoint stages in seconds:

| Stage | Scope |
| --- | --- |
| `prepare_sample` | Feedback-state normalization, history/future sample construction and tokenization |
| `vae_encode` | Encode the current pixel history and padded future |
| `to_device` | Move encoded tensors/masks/state/action to the selected device |
| `sampling` | Noise, packing, all denoiser forwards, CFG and scheduler updates |
| `action_decode` | Device-to-host action transfer, physical decoding, validation and feedback state |
| `feedback_decode` | VAE decode to CPU RGB uint8, used as simulated client feedback latency |
| `history_update` | Validate, append generated pixels and crop history |

The feedback proxy decodes the entire current latent window as required by the
existing rollout. It is not measured network, camera or robot execution latency.
`mean_chunk_seconds` is total synchronized chunk wall time / chunk count,
including bookkeeping and timing synchronization. Do not add `feedback_decode`
again: it is already included. Stage boundary synchronization can reduce overlap;
these are instrumented serial-path wall times, not pure kernel durations.
Initial input reading, model loading, final result assembly and MP4/NPZ/SVG export
are excluded from the chunk mean. `model_load_seconds` and
`rollout_wall_seconds` are reported separately; the latter includes input reading,
result assembly, progress logging and any in-loop trace export.

`profile.json` contains per-chunk data, stage means, min/max latency, warmup
summaries, device/software information and memory peaks. `blocks.jsonl` also
contains the measured stages and peaks alongside seeds/context shapes. Memory
is in bytes: `peak_allocated_bytes` is the maximum live tensor allocation and
`peak_reserved_bytes` is allocator-reserved memory, including warmup cache.
Peak counters reset at each measured chunk; overall peaks are their maxima,
not averages. These include resident model/VAE weights but exclude driver and
other-process allocations, so they are not total board HBM usage from `npu-smi`.

`--profile-npu-trace` uses
[Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0033.html)
to collect CPU/NPU operators and allocation events under `npu_trace/`, with
`rollout/<stage>` annotations. By default chunk index 16 prepares the collector,
indices 17 and 18 are captured (zero-based). Configure with
`--profile-trace-start` (>= 1) and `--profile-trace-chunks`; the window must fit
the measured rollout. Shape/memory recording and trace export add overhead;
`trace_enabled=true` labels these reports. Use a separate default run for latency
comparison. Generated Ascend trace artifacts can be analyzed with MindStudio;
see the vendor documentation linked above for supported viewers and formats.

Within `sample_block`, profiling also records three synchronized sub-stages:
`noise_initialization` (input/mask preparation, seed/noise creation and initial
latent assembly), `denoising` (scheduler setup and all denoising steps together),
and `result_assembly` (reshape video/action outputs and mask invalid channels).
Per-chunk values are in `sampling_stages_seconds`; their per-chunk averages are
in `profile.json` under `mean_sampling_stages_seconds`. These are nested inside
`sampling`, so do not add them to the top-level stage totals. No synchronization
is added per denoising step. With NPU tracing, their annotations are named
`rollout/sampling/<sub-stage>`. A block with no active prediction elements skips
denoising; missing sub-stage times count as zero in averages across chunks.

### Reusing denoising metadata

`sample_block()` reuses a request-local packing and attention plan within each
block and CFG branch. Token roles, positions, attention indexes and RoPE tables
stay fixed; video/action values and diffusion times are updated each step.
Plans are released at the end of the block and never shared across requests.
Training forward is unchanged. External autocast, inference-mode contexts and
models left in training mode use the reference path.

For numerical comparisons, call `sample_block(..., reuse_denoise_packing=False)`
to rebuild the original inputs each step. Prepared model plans reject mismatched
models, layouts, position tensors/versions, devices, dtypes and attention
backends; they require evaluation mode with gradients disabled.

Prepared sampling skips complete Query groups that cannot affect the current
block. It retains all earlier clean frames (including indirect history
dependencies), current-block targets/conditions and the current state. Retained
queries never read discarded GEN rows. Since the remaining projection, norm and
FFN operations are token-local, this dependency closure holds through every
Transformer layer. Fixed queries additionally cannot read noisy target rows.

With `sample_block(..., prune_attention=True)` (the default), sampling removes
unused groups on NPU too, then repacks surviving groups across the original chunk boundaries. Each group's
ordered KV sequence is unchanged. The original plan's KV budget (default 131072),
1000000-query and 1000-sequence limits still apply; a single indivisible group
may exceed the KV budget, as in the original planner. Projection and FFN tensor
shapes remain unchanged. This is query pruning, not physical token compaction.

Repacking is mathematically equivalent for retained outputs but **not bitwise
identical** on NPU BF16. Real checkpoint inputs reproduce shape-dependent
`npu_fusion_attention` differences; multilayer and multistep sampling can amplify
them. `prune_attention=False` disables both pruning/repacking and fixed-attention
output caching. There is no separate repacking switch or whole-chunk fallback.

Within each prepared block, every layer caches its first-step text and fixed GEN
attention outputs. Later steps select only current noisy target groups and repack
them again, copying their compact results directly into the cached output layout.
Thus short histories can benefit even if the original attention used one chunk.

Caches remain local to the block and CFG branch. No noisy KV is appended, and no
projection/FFN matrix is shortened. `cache_fixed_attention=False` disables output
caching; `prune_attention=False` disables pruning and output caching;
`reuse_denoise_packing=False` restores the complete reference path. Never reuse
an internal packing object after changing fixed inputs, model weights, block,
text, positions or history.
