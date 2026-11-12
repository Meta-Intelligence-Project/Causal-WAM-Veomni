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
and selects the configured `device`, such as `npu:0`. Defaults are BF16, 10 UniPC
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
Graph compilation, EMA, multi-device inference
and environment control are outside this implementation.

## Opt-in fused inference

The shell entry supports two independent fusions, both disabled by default:

| Selection | Computation | Scope and tradeoff |
| --- | --- | --- |
| `--add-rms-norm-implementation torch_npu` | Post-attention residual add + RMSNorm via `torch_npu.npu_add_rms_norm` | UND and GEN, including compact cold/warm/history paths. Requires `rms_norm_implementation: torch_npu`. Training and grad-enabled calls retain the original operations. |
| `--fuse-qkv` / `--no-fuse-qkv` | Three projection GEMMs become one packed `F.linear`, independently for UND and GEN | No-grad eval with explicit precision, outside external autocast/inference-mode contexts. Preserves GQA widths, biases, contiguous outputs, and original checkpoint keys. |

Equivalent YAML:

```yaml
ops_implementation:
  attn_implementation: eager
  rms_norm_implementation: torch_npu
  rotary_pos_emb_implementation: torch_npu
  add_rms_norm_implementation: torch_npu
inference_optimizations:
  fuse_qkv: true
```

The fusion returns both normalized values and the residual sum; the latter
continues through the MLP skip connection. It uses the model's epsilon. BF16
normalization of the internally added values can round differently from
normalization of a separately rounded sum, so this is not bitwise equivalence.
The [vendor API](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md)
returns `(y, rstd, sum)`; the registered adapter exposes `(y, sum)`.

QKV packs are created lazily on first use and reused for subsequent shapes and
blocks. They are rebuilt after tracked parameter changes and device/dtype
transfers. `Cosmos3Model.set_fused_qkv(False)` clears them. Plain, unwrapped
`nn.Linear` projections without forward hooks are required; wrapped projections
raise on this explicit path. Training and grad-enabled calls still execute the
original three modules. The pack does not add trainable parameters or change
the exported checkpoint. Packed weights consume extra HBM: all UND/GEN packs
for stock BF16 Edge total 896 MiB, Nano 3.375 GiB and Super 12.5 GiB, excluding
biases. `profile.json.packed_qkv_bytes` reports actual resident pack storage;
this is already included in allocator peaks, not additional to them.

The default entry already selects fused TND attention, RMSNorm and RoPE.
Edge uses **ReLU-squared**, not GELU/SwiGLU, so FastGELU and fused SwiGLU/quant
are not replacements for its MLP. LayerNorm/AddLayerNorm is not a replacement
for RMSNorm. No quantized `npu_ffn`, activation approximation, timestep precision
change, VAE alteration or causal-mask change is enabled by these options.

For a four-way ablation, set the same local `CONFIG`, `CHECKPOINT`, `DATASET`,
`PYTHON` and physical card variables for every run. Use a new `OUTPUT` each time:

```bash
OUTPUT=outputs/fusion_reference bash scripts/inference/infer_causal_cosmos3.sh \
  --no-fuse-qkv --add-rms-norm-implementation eager
OUTPUT=outputs/fusion_add_norm bash scripts/inference/infer_causal_cosmos3.sh \
  --no-fuse-qkv --add-rms-norm-implementation torch_npu
OUTPUT=outputs/fusion_qkv bash scripts/inference/infer_causal_cosmos3.sh \
  --fuse-qkv --add-rms-norm-implementation eager
OUTPUT=outputs/fusion_combined bash scripts/inference/infer_causal_cosmos3.sh \
  --fuse-qkv --add-rms-norm-implementation torch_npu
```

First compare with `sampling.residual_reuse_full_steps: null`, identical noise
seed, operators, geometry and history settings. Then repeat with the actual
deployment residual-reuse budget. Use the non-trace warm timings for fusion
latency, and paired teacher-forcing errors against physical ground truth for
quality. Report action groups and units separately, excluding invalid channels
and padding. Baseline similarity and CPU tests alone do not demonstrate task
success or NPU speedup. Native operator and small BF16 sampling regressions are
provided, but full-checkpoint hardware acceptance is still required.

## Denoising row assembly

The default prepared denoising path assembles video, action and state embeddings
in GEN order with a single concatenation for an ordinary single-sample layout.
It removes the full zero allocation and the three successive out-of-place
`index_copy` operations. This path is also used on residual-reuse steps.
Unprepared calls and training retain their original implementation.

The fixed-attention cache stores dynamic KV in a contiguous tail. Warm layer
evaluations update that tail with two slice `copy_` calls. Compact attention
plans remap storage positions while retaining each query's original ordered
keys, including merged groups. Cold KV, full-layout hidden outputs and clean
history snapshots are assembled from precomputed contiguous spans, avoiding
their scatter writes. Hidden outputs and history snapshots always own storage;
later steps cannot mutate earlier results. Uncomputed hidden rows remain zero,
and history keys remain pre-RoPE.

Row plans are prepared once per sampling branch from host metadata. Up to eight
source/zero spans use concatenation; more fragmented layouts use compact gathers
and an inverse permutation, bounding per-layer Python work. No row-index scalar
reads or host transfers are added inside decoder layers. The optimization is
automatic when `reuse_denoise_packing` is enabled (the default); fixed KV updates
also require the existing `cache_fixed_attention` option. No new weights or
configuration fields are introduced.

For a latency baseline without trace collection, run:

```bash
OUTPUT=outputs/row_assembly_wall bash scripts/inference/infer_causal_cosmos3.sh \
  --denoise-only --denoise-block 17 --denoise-runs 10 \
  --no-fuse-qkv --add-rms-norm-implementation eager
```

Use `denoise_profile.json.without_profiling` for latency, and the profiling
wrapper below to inspect operators separately. To compare the original scatter
implementation, run the same command from parent commit
`f60a9bca7d3edbc744400e76f2d923eda6c946c8` in a separate worktree, with identical
checkpoint, data, seed, block, operators and residual-reuse budget. Disabling
fixed-attention caching changes the amount of model computation and is not a
controlled row-copy ablation. CPU correctness and operator-dispatch checks do
not establish NPU speedup; measure the same workload on the target server.
Historical revisions such as `f60a9bca` require a compatible shared config:
omit the newer `compact_denoise` key and explicitly use `sampling.steps: 20`
with `sampling.residual_reuse_full_steps: 8` on both sides. Do not compare their
old 20-step defaults with the current 10-step shipped configuration.

## Compact warm hidden and persistent attention KV

`--compact-denoise` opts into a second cache layout optimization, independent
of QKV and residual-norm fusion. It defaults to disabled and can also be set
with `inference_optimizations.compact_denoise: true`. It requires the prepared,
query-pruned fixed-attention path to have an effect. Training and unprepared
calls continue to use their existing computation.

On the cold full-backbone evaluation of each block/CFG branch, decoder layers
compute the dependency closure and save clean history as before. In addition,
they pack exactly each warm attention chunk's ordered keys into persistent
K/V buffers, retaining the existing TND sequence/token limits. The warm union
buffers are replaced by these working sets; a second union is not retained.
Fixed keys may still be duplicated between attention groups, so resident KV
memory can increase relative to the union cache. Measure allocator peaks too.

On subsequent full-backbone evaluations, every decoder layer receives and
returns only `[dynamic_tokens, hidden_size]`: there is no full-GEN hidden
assembly or hidden-row gather between layers. With the sampler's default
`compact_boundary=True`, selected embeddings feed that stack directly,
residual update/apply and output heads stay in dynamic-row order, and only the
prediction payloads are expanded back to their public shapes. With
`compact_boundary=False`, the model instead gathers dynamic hidden at the
Transformer entrance and restores full hidden at the exit. The first prefill
evaluation still expands selected embeddings and uses full-layout layer
caches. Neither path removes every network copy.

Warm attention updates only dynamic locations in each persistent working set,
then directly consumes it. Query/chunk accesses are views; no Q/K/V row gathers
are performed inside the ordinary warm layer. Native NPU execution still uses
the same `npu_fusion_attention` TND call, mask semantics and scale. The CPU
reference attends each packed group independently. No attention approximation
or per-group softmax combination is introduced. For more than eight disjoint
dynamic update spans in a chunk, one indexed update bounds launch/Python
overhead; fragmented layouts therefore may still use `index_select/index_copy`.
All plans are resolved outside decoder layers, not by device scalar reads.

Only mutable attention working sets are updated in place. Full-layout cold
hidden and clean history snapshots remain independent and immutable. Residual
reuse steps continue to bypass the stack, and the next full refresh updates
dynamic KV before consuming it.

Compare both modes on the same current commit, checkpoint, data and sampling
settings, using normal rollout and all denoising blocks:

```bash
OUTPUT=outputs/compact_off bash scripts/inference/infer_causal_cosmos3.sh \
  --max-actions 640 --no-compact-denoise \
  --no-fuse-qkv --add-rms-norm-implementation eager
OUTPUT=outputs/compact_on bash scripts/inference/infer_causal_cosmos3.sh \
  --max-actions 640 --compact-denoise \
  --no-fuse-qkv --add-rms-norm-implementation eager
```

Use `profile.json.total_denoise_seconds` for all-block denoising latency and
`peak_allocated_bytes`/`peak_reserved_bytes` for the storage tradeoff. Trace
separately with `--profile-npu-trace` to inspect kernel/host behavior. CPU
regressions check output/history parity and reject row-index operations inside
ordinary warm layers; native BF16 regressions cover the same switch when an NPU
is available. Full-checkpoint performance and trajectory quality still require
hardware measurements.

### Check checkpoint accuracy before timing

The same loaded model can run compact off/on on identical prepared tensors and
actual initial noise. Each side creates its own history/cache, and the model's
original compact setting is restored afterwards. Check both a cold block and
a block with history reuse, using a new output directory for each:

```bash
OUTPUT=outputs/accuracy_cold bash scripts/inference/infer_causal_cosmos3.sh \
  --check-compact-accuracy --denoise-block 0 \
  --accuracy-atol 0.001 --accuracy-rtol 0.001
OUTPUT=outputs/accuracy_history bash scripts/inference/infer_causal_cosmos3.sh \
  --check-compact-accuracy --denoise-block 17 \
  --accuracy-atol 0.001 --accuracy-rtol 0.001
```

This mode uses the configured dtype/operators, CFG and residual budget. It
performs one complete target denoising per side, without timing warmups, trace
collection, full rollout or artifact export. It requires prepared query-pruned
fixed-attention caching and at least two full-backbone evaluations to exercise
warm compact layers. Optional preceding-block history prefill is independent on
each side. The history input is measured data, not generated rollout feedback.

`compact_accuracy.json` reports current-block video-latent and valid normalized
action errors: max/mean absolute error, RMSE, finite status, element count and
count outside `atol + rtol * abs(reference)`. Invalid action channels and padded
action steps/video frames are excluded. Nonfinite values fail, and JSON error
statistics are null when they cannot be computed. A mismatch writes the report
and exits with status 1; passing exits with status 0. Default tolerances are
`atol=rtol=0.001`, numerical checks in normalized/latent units, not physical
trajectory or task-success thresholds. Set acceptance tolerances for the actual
checkpoint and validate rollout quality separately.

## FIA v2 GEN attention ablation

`--gen-attention-implementation torch_npu_fia_v2` replaces only no-grad GEN
attention with `torch_npu.npu_fused_infer_attention_score_v2`. It is stored in
`ops_implementation.gen_attention_implementation`. The default `eager` value
retains the selected backend; in particular, `attention_backend: npu_tnd`
continues to use the original `npu_fusion_attention` FAS implementation.
UND causal attention and all training/grad-enabled execution remain unchanged.

The registered `attention/tnd_inference` adapter takes contiguous FP16/BF16
`[T,N,D]` Q/K/V with D=128, native GQA and strictly increasing host cumulative
Q/KV group ends. It keeps each group's exact ordered keys, `sparse_mode=0` and
scale `1/sqrt(D)`, without causal masking, quantization or LSE output. Both the
prefill gather path and compact warm persistent-KV path use the bound instance
handle. No Q/K/V transpose, padding or contiguous-copy conversion is inserted.
Missing FIA v2 APIs, incompatible torch_npu/CANN pairs, unsupported inputs and
vendor errors fail explicitly, without fallback. The Python API must be present
in the installed environment; this repository does not upgrade CANN or torch_npu.

First check real-weight numerical parity on identical measured inputs/noise:

```bash
OUTPUT=outputs/fia_v2_accuracy bash scripts/inference/infer_causal_cosmos3.sh \
  --check-fia-v2-accuracy --denoise-block 17 --compact-denoise \
  --fuse-qkv --add-rms-norm-implementation torch_npu \
  --accuracy-atol 0.001 --accuracy-rtol 0.001
```

This accuracy mode independently builds each side's history/cache, keeps the
compact/fusion/sampler settings fixed, and toggles only the GEN kernel. Every
candidate FIA call is additionally compared with FAS on exactly the same QKV
and group ends; then final current-block normalized action and video-latent
outputs are compared. `fia_v2_accuracy.json.attention` aggregates raw-attention
errors and call counts, while `metrics` contains final-output errors.
`target_fia_calls > 0` is required, so a skipped candidate cannot pass. Any
tolerance violation or nonfinite result fails with exit status 1. Nonfinite
raw attention aborts denoising and records a partial failure with
`completed_denoising: false`. Original model handles/flags are restored even
on errors. Use `--denoise-block 0` for an additional no-history case.

Default tolerances are `atol=rtol=0.001` in attention/normalized/latent units;
acceptance tolerances must follow the actual checkpoint requirements. This is
a numerical check, not a physical trajectory/task-quality guarantee. Accuracy
mode deliberately performs additional kernels and CPU transfers: its duration
must not be used as an inference performance measurement.

After acceptance, compare full rollout denoising totals on the same commit:

```bash
OUTPUT=outputs/fas_wall bash scripts/inference/infer_causal_cosmos3.sh \
  --gen-attention-implementation eager --compact-denoise \
  --fuse-qkv --add-rms-norm-implementation torch_npu
OUTPUT=outputs/fia_v2_wall bash scripts/inference/infer_causal_cosmos3.sh \
  --gen-attention-implementation torch_npu_fia_v2 --compact-denoise \
  --fuse-qkv --add-rms-norm-implementation torch_npu
```

Read `profile.json.total_denoise_seconds` and compare identical solver schedules,
horizon, data, card and operator switches. To inspect Cube/MTE counters and
stacks separately, use `profile_causal_cosmos3.sh` with the same FIA selector.
UND still uses FAS, so FAS events can legitimately remain in that trace. There
is no assumed speedup: confirm actual API/kernel dispatch and wall time on the
installed hardware. CPU reference/mock tests and NPU-gated raw/model cases are
provided; native NPU validation is outstanding on the CPU development host.

## Denoising-only benchmark

Use this entry when measuring the denoiser rather than whole-rollout latency:

```bash
# Unfused operator path; same CONFIG/CHECKPOINT/DATASET/PYTHON/card variables.
bash scripts/inference/profile_causal_cosmos3_denoise.sh \
  --no-fuse-qkv --add-rms-norm-implementation eager \
  --denoise-block 17 --denoise-runs 10
# Fused candidate with identical workload and sampling settings.
bash scripts/inference/profile_causal_cosmos3_denoise.sh \
  --fuse-qkv --add-rms-norm-implementation torch_npu \
  --denoise-block 17 --denoise-runs 10
```

The wrapper forwards `--denoise-only --profile-npu-trace --compare-profile-wall`
to the normal inference entry and defaults to three warmups (`WARMUP_RUNS`
overrides this). The same mode is available from the earlier profiling wrapper
by adding `--denoise-only`. Ordinary inference and rollout profiling keep their
previous behavior.

It reads and VAE-encodes one **measured-history policy** window, transfers it
to the selected device and draws initial video/action noise once. Every repeat
uses those same latent tensors and actual noise values, with a fresh block-local
cache. No `pipe.rollout()`, VAE decode, physical-action decode or video/result
export is performed. This is a fixed-input microbenchmark, not a generated
pixel-feedback rollout and not an environment quality evaluation.

`--denoise-block` is a zero-based global target block relative to `START`;
default `0` measures the cold first block. A later block such as `17` exercises
history. Preparation crops measured history to the same complete-block limit
as rollout, so the local block index can be smaller than the global index.
The noise seed retains the global index. The report records the source window,
local/global indices, latent/action shapes, text length, dtype and operators.
Choose a block available in the selected episode. `--max-actions` and the
rollout `--profile-trace-start/chunks` settings do not select this workload.

For a later block with native caches enabled, the benchmark evaluates the
preceding block once outside measurement to prepare clean/text snapshots.
Clean rows cannot read noisy targets, so a single evaluation suffices for this
repeatable history-hit workload; its generated output is discarded. Every
repeat forks the same immutable snapshots rather than consuming the previous
repeat's cache. This does not reproduce the approximate values accumulated in
a generated pixel-feedback history. `--no-denoise-history-cache` measures a
cross-block cache miss instead. Disabling native fixed-attention/pruning also
disables history prefill. The initial noise and encoded inputs are unchanged.

Only the existing `sample_block` **denoising** region is synchronized and timed:
scheduler setup and all solver iterations, including per-step model input
updates, first-use block packing/cache binding, Transformer work, output heads,
CFG, solver updates, masks and the existing finite checks. Noise/mask
initialization and final sample reshaping are outside this region. No new
per-step timers, device synchronizations or peak resets are added. The NPU
profiler starts/stops at the same denoising boundaries, records Python stacks,
and captures the whole invocation without a rollout-chunk schedule. VAE and
noise initialization are not in this trace.

`denoise_profile.json` contains:

| Field | Meaning |
| --- | --- |
| `without_profiling` | Warm denoising mean/median/min/max/p95 and samples in **milliseconds**, over `--denoise-runs` repeats (default 10) |
| `with_profiling` | One additional traced denoising invocation, in milliseconds |
| `profiler_context_ms` | That trace including profiler setup, teardown and stack/trace export; separate from denoising wall time |
| `profiling_overhead_ratio` | Traced denoising wall / unprofiled median - 1; the one traced sample is not a statistical estimate |
| `denoising` | Actual solver steps, full-backbone refresh indices/evaluations and residual-reuse evaluations |
| `profiled_output_max_abs_difference` | Normalized action/latent consistency diagnostic against the last unprofiled repeat, outside timing |

CPU/NPU operator traces and folded CPU stacks are under `npu_trace/`, including
`npu_trace/stacks_cpu.txt`. Full model/data/VAE preparation still runs once and
must fit memory, but is excluded from denoising metrics. The shipped 10-step
schedule refreshes the backbone at `[0, 1, 6, 8]`. Setting
`residual_reuse_full_steps: null` changes four full evaluations to ten while
retaining all ten solver updates.
Keep this and the selected block/cache settings identical for fusion comparisons.

## Warm rollout profiling on Ascend 910B

### Total denoising across the normal rollout

To diagnose a gap between `stages_seconds.sampling` and summed kernel time,
first run the normal `--profile` command without a vendor trace. Logs now
print separate sampling and denoising wall/event pairs. Each chunk's
`sampling_timing` contains entries for `sampling`, `noise_initialization`,
`denoising` and `result_assembly`, in seconds. Each entry reports
`wall_seconds`, `event_seconds`, `host_call_seconds` and `final_wait_seconds`.
The last two add up to wall time; host call time includes internal blocking
operations and is not pure CPU compute time. Sampling and denoising overlap
by nesting; do not add their durations together. Summary fields include
`total_sampling_seconds` and `total_sampling_event_seconds`.

Then add `--profile-sampling-details` to that same command (also supported
with `--denoise-only`) for synchronized diagnostic attribution. Per-chunk
`sampling_details` aggregates counts and seconds by region, while
`sampling_detail_records` preserves each invocation with zero-based solver
`step` and conditional `branch` metadata (`true` for the conditional branch).
Regions include scheduler setup, step input, packing/cache preparation,
packing updates, model forward, prediction assembly, CFG decision/combine,
scheduler update and finite checks. The finite-check region includes the
Python boolean conversion of `torch.isfinite(...).all()`. Cold packing is
reported separately from warm input updates. The denoising-only report
stores summaries/records per measured repeat under `sampling_details`.

Detailed mode synchronizes before and after each region, changes scheduling,
and increases measured latency. Use it to locate expensive regions, then
rerun without it for the performance comparison. A region's final wait
reports only its explicit end synchronization, not internal waits. These
timers do not report kernel active time; obtain that from a matching trace
window, accounting for overlapping kernels and streams. Existing latency
fields and sampling behavior are preserved.

Rollout profiling prints denoising `wall` and current-stream `event` seconds
for each chunk, in addition to the complete chunk latency. `profile.json`
retains `total_denoise_seconds` (wall), adds `total_denoise_event_seconds`,
and records `denoise_event_seconds` per chunk. The denoising-only benchmark
retains `without_profiling` for wall statistics and adds
`without_profiling_event` (and `with_profiling_event` when tracing), in the
same millisecond statistics format. CPU event values are null.

Both timers cover scheduler setup and the complete denoising loop, including
lazy packing/cache preparation, all model evaluations and scheduler updates.
No extra per-step synchronization is added. Wall time uses `perf_counter`
and waits for device completion at the boundary. Events bracket the loop on
the selected device's current stream; event elapsed time can include stream
idle gaps caused by host dispatch. It is not the sum of kernel durations,
and wall minus event is not a CPU-time measurement. Independent streams need
explicit dependencies to be represented in that interval. Run latency
comparisons without `--profile-npu-trace` to avoid trace overhead.

Run the full pipeline when comparing the total denoising cost of all generated
blocks, with real VAE/feedback/history updates between blocks:

```bash
OUTPUT=outputs/after bash scripts/inference/infer_causal_cosmos3.sh \
  --max-actions 640 --no-fuse-qkv --add-rms-norm-implementation eager
```

Do not add `--denoise-only`, `--denoise-block`, `--denoise-runs`,
`--profile-npu-trace` or `--compare-profile-wall` for this measurement. The shell
already enables synchronized stage wall timing with `--profile`, without a
vendor operator trace. The default geometry generates 20 blocks for 640 actions.
One separate full-rollout warmup is discarded; only the following measured
rollout contributes to `profile.json.total_denoise_seconds`. This is the sum of
every `chunks_detail[*].sampling_stages_seconds.denoising`, including the first
block and any partial tail. `denoise_block_count` counts timed denoising regions;
`mean_denoise_block_seconds` is their arithmetic mean, or null if none ran.

All pipeline stages still execute, but the denoising total excludes VAE,
observation processing, transfers outside the sampler, noise initialization,
sampler result assembly and artifact export. It includes every solver update,
model/CFG evaluation and lazy packing/cache setup inside the denoising region.
Stage boundary device synchronization remains enabled; this is instrumented
wall time, not a sum of kernel durations or wholly uninstrumented latency.

For the scatter-versus-assembly comparison, use commit
`f60a9bca7d3edbc744400e76f2d923eda6c946c8` in a separate checkout for the before
run, with `OUTPUT=outputs/before`. Use absolute `CONFIG`, `CHECKPOINT`, `DATASET`
and `PYTHON` paths when sharing inputs and the environment between checkouts.
Keep horizon, seed, sampling, residual budget, history, operator switches and
device fixed. Earlier reports lack the new total fields but contain the same
per-block timing records. Compare both formats with:

```bash
python - /absolute/before/profile.json /absolute/after/profile.json <<'PY'
import json
import sys

reports = [json.load(open(path)) for path in sys.argv[1:]]
for key in ("requested_actions", "actions_per_chunk", "chunks"):
    assert reports[0][key] == reports[1][key], f"Mismatched workload: {key}"
assert not any(report["trace_enabled"] for report in reports), "Use trace-free runs"
rows = [[row["sampling_stages_seconds"]["denoising"]
         for row in report["chunks_detail"]
         if "denoising" in row["sampling_stages_seconds"]] for report in reports]
assert len(rows[0]) == len(rows[1]) > 0, "Denoise block counts must match"
before, after = map(sum, rows)
assert before > 0 and after > 0
print(f"Blocks: {len(rows[0])}")
print(f"Before: {before:.6f} s; after: {after:.6f} s")
print(f"Speedup: {before / after:.3f}x; time reduction: {(1 - after / before) * 100:.2f}%")
PY
```

Repeat full runs into fresh output directories to assess variability. The
comparison above is one complete rollout per version; it does not repeat a
single fixed block ten times.

### Run options and stage timing

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
card 6 / logical `npu:0`, BF16, 10 sampling steps, Agibot task 352 episode 0,
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

### Code stacks and unprofiled wall comparison

Every `--profile-npu-trace` run now sets `with_stack=True`, `record_shapes=True`
and `profile_memory=True` for `torch_npu.profiler`, retaining CPU/NPU activities
and the `rollout/<stage>` annotations. Python stacks are collected for host
operator attribution; device kernel events themselves do not have Python stacks.
The trace callback also explicitly exports folded CPU call stacks to
`npu_trace/stacks_cpu.txt` using `export_stacks(..., metric="self_cpu_time_total")`.

### AI Core and Cube counters

Both rollout and denoise-only traces explicitly request
`_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
aic_metrics=AiCMetrics.PipeUtilization, data_simplification=False,
export_type=ExportType.Text)`. CPU/NPU activities, shapes, memory and Python
stacks remain enabled. Earlier commits without this configuration did not
explicitly enable AI Core PMU counters. The requested settings are recorded in
`npu_trace/capture_config.json`; that file is not proof that hardware counters
were successfully collected.

Run normal rollout on each side, keeping other operators and inputs fixed:

```bash
OUTPUT=outputs/trace_compact_off bash scripts/inference/profile_causal_cosmos3.sh \
  --no-compact-denoise --max-actions 640 \
  --profile-trace-start 17 --profile-trace-chunks 2
OUTPUT=outputs/trace_compact_on bash scripts/inference/profile_causal_cosmos3.sh \
  --compact-denoise --max-actions 640 \
  --profile-trace-start 17 --profile-trace-chunks 2
```

The entire 20-block rollout executes, but only zero-based blocks 17 and 18
are traced. The wrapper runs full warmups and a separate uninstrumented rollout
before the traced pass. For a longer trace excluding the schedule warmup at
block 0, use `--profile-trace-start 1 --profile-trace-chunks 19`. To trace just
one fixed denoising workload, use `profile_causal_cosmos3_denoise.sh` instead.

Inspect the exported kernel details/Kernel View and retained CANN
`op_summary*.csv` under `npu_trace/`. See the
[CANN 9.0 operator-field definitions](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0067.html).
Depending on device and tool versions, the pipe prefixes are `aic_`, `aiv_`, or
absent. Key fields include:

| Field family | Meaning |
| --- | --- |
| `*_mac_ratio`, `*_mac_time(us)` | Cube MAC instruction active-cycle ratio/time |
| `cube_utilization(%)` | Separate Cube operator utilization metric, when supported |
| `*_vec_ratio`, `*_scalar_ratio` | Vector and Scalar instruction active-cycle ratios |
| `*_mte1_ratio`, `*_mte2_ratio`, `*_mte3_ratio` | L1-to-L0, external-memory-to-core, core-to-external-memory pipe ratios |
| `aic_fixpipe_ratio`, `memory_bound` | Fixpipe activity and the tool's memory-bound indicator |

Field presence and non-N/A values depend on actual hardware, CANN and
torch_npu. Missing/N/A counters do not mean zero utilization. Pipe ratios are
per-kernel core-cycle ratios, can overlap, and must not be summed or treated as
the whole denoising wall-time utilization or achieved fraction of peak FLOPs.
Arithmetic/Memory/L2 metric groups require separate captures; PipeUtilization
does not collect every PMU metric at once. Keep the ordinary trace-free rollout
for `total_denoise_seconds` performance comparisons, as counter/stack collection
adds overhead. Native hardware counter availability remains unverified on the
CPU-only development host.

The dedicated wrapper also measures the cost of profiling:

```bash
# Same environment/path overrides as infer_causal_cosmos3.sh; default 640 actions.
bash scripts/inference/profile_causal_cosmos3.sh
# Include the opt-in fusions in both sides of the comparison:
bash scripts/inference/profile_causal_cosmos3.sh \
  --fuse-qkv --add-rms-norm-implementation torch_npu
# Short trace smoke test: three chunks, recording chunk index 1.
bash scripts/inference/profile_causal_cosmos3.sh \
  --max-actions 96 --profile-trace-start 1 --profile-trace-chunks 1
```

This is equivalent to passing `--profile-npu-trace --compare-profile-wall`
to the ordinary inference shell entry. After the configured full-horizon
warmups, it runs one **uninstrumented** rollout (`profiler=None`) and one
profiled rollout in the same process with the same loaded model, input,
sampling options and seeds. Each call owns a fresh episode history cache.
No allocator/cache flush occurs. The baseline has only boundary device
synchronizations, without stage timers, peak resets or profiler annotations.
The profiled pass includes the usual stage synchronization overhead and the
configured trace window; it is not a trace of every chunk by default.

`wall_comparison.json` (also embedded in `profile.json.wall_comparison`) records:

| Field | Scope |
| --- | --- |
| `without_profiling_wall_seconds` | Whole uninstrumented rollout, device-synchronized at its boundaries |
| `with_profiling_rollout_wall_seconds` | Same rollout with stage measurement and NPU tracing; includes any in-loop trace export |
| `with_profiling_context_wall_seconds` | Includes profiler construction, context entry/exit and trace export |
| `rollout_overhead_seconds`, `rollout_overhead_ratio` | Profiled rollout minus baseline, and `(profiled / baseline) - 1` |
| `context_overhead_seconds`, `context_overhead_ratio` | Corresponding totals including the profiler context |
| `action_max_abs_difference`, `action_mean_abs_difference`, `video_identical` | Output consistency diagnostics between the two passes |

Both whole-rollout wall measurements include initial input reading and result
assembly, and exclude model loading and final MP4/NPZ/SVG export. They therefore
have a different scope from `mean_chunk_seconds`. This is one ordered pair,
not a statistical speedup estimate; repeat with new output directories to
assess variability. Do not use the trace-inflated timing as fusion throughput.
The reusable `timed_rollout()` helper and CPU tests verify boundary timing,
the profiler-free baseline, stack recording and CLI/shell forwarding.

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

### Denoising and history caches

`sample_block()` prepares packing metadata once per block and CFG branch. With
`cache_fixed_attention=True` (the default) and no cross-block snapshot,
the first denoise step computes only the dependency closure of the current noisy
targets. This includes clean video, conditions and state that the targets read,
plus the earlier clean rows needed by those retained queries across decoder
layers. Merely keeping direct target-visible KV would omit indirect dependencies
at a truncated history-window boundary.

The first step allocates temporary K/V only for the union read by these retained
queries. Prefill query rows, KV read/write indices and RoPE slices are prepared once
in `BlockCache`; each layer saves only the smaller target-visible KV needed by
later steps. The prefill union can be larger than this persistent cache. Full hidden
layout shapes remain compatible with packing: retained outputs are scattered back
and unused rows are zero. Predictions or hidden states for discarded rows are not
complete-layout outputs; the sampler restores fixed input values and uses only
current targets.

`sample_block(compact_boundary=True)` projects only the prefill dependency closure
on the first evaluation, then current target rows. Final GEN normalization,
output projections and reused residuals also retain only targets; other prediction
rows are zero placeholders restored by the sampler. Decoder history caches keep
their existing full hidden layouts. This requires prepared packing and fixed
attention caching; training and diagnostic outputs keep the original path.
Smaller projections can change BF16 rounding. Set `compact_boundary=False` for
full boundary computation; direct `DenoisePacking` calls default to this reference.
The boundary row plans use segment assemblies for embedding construction and
prediction restoration. Enabling `--compact-denoise` additionally skips full
hidden expansion/recovery on warm refreshes; the residual remains compact in
both the full-refresh and reuse steps.

Later denoise steps update the persistent buffer's dynamic slots in place and
restore retained fixed hidden rows. Adjacent query groups with the exact same
ordered KV are merged on this block-local path, including across chunk boundaries
when the TND limits permit it. Every query retains its original ordered keys;
queries within the current target block still attend each other bidirectionally.
Projection/MLP row counts and attention group shapes change, so BF16 results can
differ from the full-layout path through operator rounding. Dense attention can
also differ slightly when masked-out KV columns are physically removed.

Training and uncached forward behavior are unchanged. Cross-block history caching
uses the same compact prefill KV and merged attention groups, including its initial
cache miss. Subsequent hidden outputs retain their own storage rather than
mutating the cached hidden tensor in place.

Rollout and teacher forcing automatically create a fresh `HistoryCache` per call;
cross-block history reuse is always enabled on the supported optimized path.
The `sampling.cache_history` YAML switch has been removed. Legacy values are ignored
and stripped from the pipeline's effective/exported configuration, including
`false`; they no longer disable history reuse. The caller's configuration object
is not mutated.

History caching adds per-layer clean hidden/KV storage; each old layer
snapshot is released after its replacement is saved on the first full evaluation,
rather than retained throughout denoising. It also retains history through VAE feedback.
Account for this persistent storage when sizing the history window. It reuses
per-layer text and already-observed clean-video hidden/KV
across blocks. Prefill planning starts from the current targets and the observed
clean rows needed for the next snapshot. It expands fresh dependencies, stopping
at cached clean rows. Only fresh retained rows run decoder projection, attention
and MLP; obsolete current/condition, action/state and future clean rows outside
this dependency set are not evaluated. With boundary compaction enabled, input
embeddings cover only retained rows; scattered hidden layouts keep their full shapes.

Matched history keys/values are gathered directly into compact KV slots. Keys
are stored before RoPE and rotated at current positions. New history is evaluated
through the clean path; noisy target KV is never promoted to clean KV. Snapshot
writes cover fresh and reused observed clean rows in their stable frame/patch
order, without constructing a full-layout raw-KV intermediate. Snapshots own their
storage and remain unchanged during later denoise steps. Each rollout/teacher-
forcing call owns a fresh cache, with separate CFG branches. Consuming a previous
snapshot does not modify its tensors or the previous snapshot dictionary. A cache missing any
decoder layer (for example, after an interrupted forward) is discarded on the
next binding instead of mixing partially populated snapshots.

Without a previous snapshot, history caching computes the compact dependency
closure and initializes clean snapshots. With hits, it computes only fresh
dependencies and reuses old clean results. Later denoise steps in either mode
already use the block-local cache; history reuse does not save that computation
again. Warm attention groups with identical ordered KV are merged in both modes.
Compared with full-layout history caching, projection and attention shapes change;
BF16 outputs can therefore differ through operator rounding even on identical
immutable history. Measure latency and memory with the actual task and history
length.

History reuse assumes immutable historical latents and model weights. The current
pixel-feedback rollout re-encodes history and eventually shifts its window;
reuse under these conditions is explicitly approximate. For a numerical reference,
direct `sample_block` calls with `history_cache=None` recompute clean history from
the current encoded window on each block; outputs can therefore differ from the
approximate cached rollout. `prune_attention` and `cache_fixed_attention` remain
enabled by default. Direct sequential `sample_block` calls can pass one
`HistoryCache` via `history_cache`; start a new object for each episode and pass
continuous global `seed_block` values when local window indexes shift.

For complete-layout comparisons, use `prune_attention=False` to disable query
selection and both caches while retaining metadata reuse, or
`reuse_denoise_packing=False` to rebuild inputs each step. Setting
`cache_fixed_attention=False` retains query-only pruning but disables both layer
caches. Training, external autocast and inference-mode contexts retain the
reference fallback without history reuse; the optimized path uses evaluation mode
with no gradients. Creating the episode cache does not override these execution
guards or the low-level reference/debug options.

### Fixed residual reuse (experimental)

Both shipped Cosmos3 inference configs default to
`sampling.steps: 10` and `sampling.residual_reuse_full_steps: [0, 1, 6, 8]`,
refreshing at zero-based steps `0, 1, 6, 8`. All ten UniPC updates still run.
Set the reuse field to `null` to disable
residual reuse and restore the full-backbone inference path. Configs that omit
the field retain the original behavior.

Choose `2`, `4`, `6`, or `8` (CLI: `--residual-reuse-full-steps`) to retain
20 UniPC updates while evaluating the complete Transformer layer stack only
at the uniformly spaced interval starts `floor(i * 20 / F)`. The optional
value `20` is a full-refresh equivalence check. These legacy schedules require
`sampling.steps: 20`; the additional 10-step integer-count schedule accepts
budget 4. Explicit index lists can select other valid refresh counts.
Both schedules require guidance 1 and no distilled fixed-step sampler. Disable
residual reuse explicitly before using any other step count, CFG, or a distilled
checkpoint with the shipped configs.

For schedule ablations, use the same field with a list:
`sampling.residual_reuse_full_steps: [0, 2, 5, 8]`
(CLI: `--residual-reuse-full-steps 0 2 5 8`). A positive CLI value selects a
refresh count; values starting with zero select explicit indices (`0` alone
means `[0]`). Explicit indices must be strictly increasing, unique integers in
`[0, steps)`, starting at zero to initialize the cache. Their count is inferred
from the list. To disable reuse, set `sampling.residual_reuse_full_steps: null`.
An explicit schedule still requires guidance 1 and the standard UniPC sampler.
Explicitly choosing the integer value `4` with 10 steps selects `[0, 3, 6, 9]`;
the shipped configs instead use the explicit list `[0, 1, 6, 8]`.

Each refresh stores `GEN_after_layers.float() - GEN_before_layers.float()`
before the final normalization. Other steps skip the layer stack and add that
FP32 residual to the **current** GEN embedding, then cast back to model dtype.
Current GEN latent/time embeddings, final GEN normalization, output heads, masks
and all solver updates still execute. Reuse steps skip text embedding, and final
text normalization runs only when text logits or hidden states are requested
(both are prohibited in residual-reuse mode). This approximates hidden features, not the last
velocity or the final denoised sample. It changes predictions and needs quality
validation for each checkpoint.

The residual belongs to one `sample_block` invocation. The existing fixed
hidden/KV cache remains active during full evaluations; reuse steps do not
read or update its dynamic KV slots. A later refresh recomputes those slots
from the current inputs. Residuals never persist across blocks or episodes.
The model requires no-grad eval mode and rejects text logits, hidden-state
outputs and training targets with residual reuse. `blocks.jsonl` records the
actual solver count, refresh indices, full evaluations and reuse evaluations
under `denoising`.

Compare against the disabled-reuse 20-step path with the same checkpoint, operators,
input, noise seed and history settings. Use warmed rollout profiler results
for latency, and paired teacher-forcing errors against measured ground truth
for quality. Report physical action groups separately and exclude invalid
channels and padding; similarity to baseline alone does not demonstrate task
success. The schedule is a fixed-budget adaptation of
[RealtimeWAM section 3.4](https://arxiv.org/html/2610.10079v1#S3.SS4), not its
motion-dependent budget selector or its 10-update Euler sampler.
