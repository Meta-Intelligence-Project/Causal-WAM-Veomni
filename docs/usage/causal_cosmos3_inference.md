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

### Code stacks and unprofiled wall comparison

Every `--profile-npu-trace` run now sets `with_stack=True`, `record_shapes=True`
and `profile_memory=True` for `torch_npu.profiler`, retaining CPU/NPU activities
and the `rollout/<stage>` annotations. Python stacks are collected for host
operator attribution; device kernel events themselves do not have Python stacks.
The trace callback also explicitly exports folded CPU call stacks to
`npu_trace/stacks_cpu.txt` using `export_stacks(..., metric="self_cpu_time_total")`.

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
queries. Cold query rows, KV read/write indices and RoPE slices are prepared once
in `BlockCache`; each layer saves only the smaller target-visible KV needed by
later steps. The cold union can be larger than this persistent cache. Full hidden
layout shapes remain compatible with packing: retained outputs are scattered back
and unused rows are zero. Predictions or hidden states for discarded rows are not
complete-layout outputs; the sampler restores fixed input values and uses only
current targets.

Later denoise steps update the persistent buffer's dynamic slots in place and
restore retained fixed hidden rows. Adjacent query groups with the exact same
ordered KV are merged on this block-local path, including across chunk boundaries
when the TND limits permit it. Every query retains its original ordered keys;
queries within the current target block still attend each other bidirectionally.
Projection/MLP row counts and attention group shapes change, so BF16 results can
differ from the full-layout path through operator rounding. Dense attention can
also differ slightly when masked-out KV columns are physically removed.

Training and uncached forward behavior are unchanged. Cross-block history caching
uses the same compact cold KV and merged attention groups, including its initial
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
across blocks. Cold planning starts from the current targets and the observed
clean rows needed for the next snapshot. It expands fresh dependencies, stopping
at cached clean rows. Only fresh retained rows run decoder projection, attention
and MLP; obsolete current/condition, action/state and future clean rows outside
this dependency set are not evaluated. The input embeddings and scattered hidden
layout still retain their full shapes.

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
`sampling.residual_reuse_full_steps: 8`. Set this field to `null` to disable
residual reuse and restore the full-backbone inference path. Configs that omit
the field retain the original behavior.

Choose `2`, `4`, `6`, or `8` (CLI: `--residual-reuse-full-steps`) to retain
20 UniPC updates while evaluating the complete Transformer layer stack only
at the uniformly spaced interval starts `floor(i * 20 / F)`. The optional
value `20` is a full-refresh equivalence check. This feature requires
`sampling.steps: 20`, guidance 1, and no distilled fixed-step sampler. Disable
residual reuse explicitly before using another step count, CFG, or a distilled
checkpoint with the shipped configs.

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
