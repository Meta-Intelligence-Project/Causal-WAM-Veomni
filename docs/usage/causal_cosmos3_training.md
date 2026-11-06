# Causal Cosmos3 training

This branch trains Action models with one decision window per sample. The
[Action recipe](../../configs/dit/causal_cosmos3_edge.yaml) and
[source configuration](../../configs/data/causal_cosmos3.yaml) use the same
history/current-state/future layout as the
[streaming service](causal_cosmos3_serving.md). There is one Action training
path and one recipe; the former `_decision` recipe names have been removed.

[`tasks/train_causal_cosmos3.py`](../../tasks/train_causal_cosmos3.py) uses the
native `DiTTrainer`, `DiTModelRuntime` and FSDP2. Network parameter shapes,
flow matching, the shared 55D template and FD/ID/policy objectives are unchanged.
The window contract borrows Long-WAM's finite observed history and joint future
prediction; it does not add its separate Wan/ActionDiT experts or IDM objective.

For video-only pretraining, use the distinct
[Video recipe](../../configs/dit/causal_cosmos3_video_edge.yaml). For offline
checkpoint inspection, see [independent inference](causal_cosmos3_inference.md).
Its explicit multi-block diagnostic and rollout helpers remain separate from
Action training; their block packing/math are also used by Video pretraining.

## Action decision windows

Configure dataset roots, cameras and measured statistics in the source YAML.
Training controls and model resources live in the DiT recipe.

```yaml
stage: action
sampling: sample_uniform
validation:
  split_ratio: 0.05
  max_episodes_per_source: 100
  seed: 73
geometry:
  actions_per_block: 32
  video_stride: 4
  temporal_compression: 4
  spatial_compression: 16
  patch_size: 2
candidates:
  start_stride: 1
  use_subtask: true
sample:
  mode: joint
  mode_weights: [1, 1, 1]  # forward_dynamics, inverse_dynamics, policy
  history_steps: 1024
  action_horizon: 32
```

P=`history_steps` and H=`action_horizon` count raw action-time intervals.
Defaults are P1024/H32. Both must be multiples of
`video_stride * temporal_compression` (16 here), with P nonnegative and H
positive. P0/H16 and P48/H32 are also supported. The source factory derives
`candidates.max_actions` from H; an explicit override must equal H. Action
candidates always use `frame_window` with zero overlap. `start_stride` controls
spacing between legal anchors, including anchors near the episode tail.
`geometry.actions_per_block` retains the measured-statistics provenance check;
H determines the current decision length.

`ActionDataset` has no training-layout selector or random-history option.
`full_history` and `history_range` belong to Video datasets. The output metadata
still records `layout_mode="decision_window"` for shared packing and serving.
See the [dataset contract](../data/causal_cosmos3/datasets.md).

At measured anchor t, an Action sample contains:

- RGB observations sampled from `[t-P, t+H]` at stride four. The causal VAE
  produces `M=1+P/16` clean history latents and `K=H/16` future latents.
- One current state. In the AgiBot recipe, the reader takes row t from
  `state_unified`, with `mask_state` selecting valid channels of the 55D
  robot-state template. State is supplied by the dataset, not estimated by
  the loader. The processor canonicalizes the physical anchor and normalizes
  a copy for the model condition.
- H physical targets encoded against that same current anchor using measured
  quantiles and channel masks. AgiBot uses `target=state_unified` and
  `target_offset=1`, so targets are states t+1 through t+H. Future states never
  become state conditions. The template preserves absolute hand/gripper targets
  and uses relative scalar/rotation targets where specified.
- One sampled future sigma. History and task conditions have zero input times
  and no loss. Future video and action share the sampled sigma; historical
  actions, additional state tokens and a duplicate clean-video sequence are absent.

For P1024/H32 this means 265 sampled RGB frame positions, 65 observed video
latents, two future video latents, 32 action targets and one state token.
Negative episode ticks repeat the actual reset frame. Future tail padding
repeats final readable values and is excluded from supervision. Missing real
history and non-contiguous timestamps are errors. Subtask text is selected at
anchor t, even when the future horizon crosses a subtask boundary.

`mode: joint` samples FD, ID and policy with normalized `mode_weights` in that
order; `[1, 1, 1]` gives probability 1/3 each. Fixed modes are also supported.

| Task | Additional clean condition | Supervised future target |
| --- | --- | --- |
| `forward_dynamics` | All future actions | Future video |
| `inverse_dynamics` | All future video | Future actions |
| `policy` | None | Future video and actions |

Every task also conditions on observed visual history and the current state.
Historical video uses one causal group per latent; a historical latent cannot
read later latents. All future video/action targets form one joint decision
and may read one another according to their task roles. Conditions cannot read
noisy targets. State reads only text and itself. Mixed packs remain sample-isolated.

The frozen VAE uses
[`encode_decision_window`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py):
the first-frame call followed by complete four-frame groups, matching online
streaming operation shapes. Future pixels cannot affect observed prefix latents.
Each training sample owns and releases its writable convolution state.
The NPU BF16 encoder delegates symmetric spatial zero padding to native Conv3D and
skips no-op padding copies. Temporal padding and feature caches remain causal;
the first-frame/four-frame calls, loaded parameters, normalization and precision
boundaries are preserved. The adapter is local to each frozen encoder and uses
the stock Diffusers forward for other devices, dtypes or padding settings.
Training reads original RGB data; derived pixels, latents, embeddings and feature
caches are not persisted as dataset files. Reuse is bounded volatile memory.
The Action recipe sets `model.condition_model_cfg.vae_reset_prefix_cache_mb: 4096`
to retain up to 4 GiB of immutable reset-prefix snapshots per rank, with at most
eight source/episode entries. `0` disables reuse. An episode must recur within
the last eight eligible episodes before admission; isolated cold episodes use
the original encoder without building snapshots. Snapshots are collected during
normal encoding every eight latents and at the longest observed reset prefix;
exactly equal immutable features share storage across these snapshots.
each window clones a preceding checkpoint and replays intervening four-frame
groups. Only declared reset padding with identical pixels is reusable. Real
history, state, future targets and task sampling follow the original path.
Changed reset pixels, model weights/dtype or numerical context invalidate reuse;
errors and device/dtype moves clear the cache. Retained snapshots are bounded
after each sample; encoding and writable forks additionally require transient
memory. Benefit depends on episode locality and reset-padding length; the cache
does not change sampling order to increase hits. Cache contents are not
checkpointed and resume starts cold.
Workers return CPU
pixels; the training rank encodes them. Training does not share writable VAE
state or DiT K/V across samples, or replay frozen inference graphs during backward.

Streaming VAE, asynchronous H/S/R execution, NPU graph replay and RTC are
inference capabilities; they do not introduce a trainable VAE, asynchronous
optimizer or RTC training loss. Native Cosmos flow-matching loss weights remain
uniform. Long-WAM's independently sampled video/action times, IDM partial video
denoising and detached imagined-video K/V are not implemented here.

## Optional history-gradient truncation (TBPTT)

Action and Video training default to full temporal gradients. The shared
`enable_tbptt` flag must be a boolean and defaults to `false`. Both stages use
one positive-integer length K, `tbptt_keep_history_latents`, defaulting to 16.
Only the stage's interpretation of K differs; the algorithms below retain
their original forward attention visibility and loss masks. To enable the
default K16 policy in either DiT recipe, only set:

```yaml
model:
  model_config:
    enable_tbptt: true
```

Set `model.model_config.tbptt_keep_history_latents` explicitly only to choose
a different K.

### Action: recent observed history

K counts the most recent **observed video latents**, including the current
observed latent, whose history-encoding paths retain gradients. It does not
change the source YAML's P/H, image sampling, task probabilities, packing or loss
masks. P1024 still supplies all 65 observed latents: K16 stops the encoding
gradient through the oldest 49 while all 65 remain available to attention.
The old hidden states' normalization and K/V readout projections, current state,
text conditions, recent history and future predictions retain their applicable
training gradients. FD and policy video predictions use the same history path
as action predictions; video loss is not bypassed or detached. K at least as
large as a sample's observed-latent count does not truncate that sample's
gradients. A pack with no truncated history
uses the original layer path.

Every independently sampled window rebuilds its old history under the current
weights. No DiT hidden state or K/V is reused between windows or optimizer
updates. The implementation runs the old-history Q, attention and MLP without
recording gradients, then preserves the original packed ordering and attention
visibility. Trainable work is scheduled before the independent old-history
output so non-reentrant checkpoint early stopping can skip that old branch on
recomputation. The default decoder-layer FSDP2 wrapping, checkpoint and
attention-SAC paths remain in use; frozen VAE encoding is unaffected.

### Video: temporal chunks

Independent `stage: video` pretraining supports a separate chunk-boundary
gradient policy using the same `enable_tbptt` and `tbptt_keep_history_latents`
settings. Here K counts the post-first video latents per temporal chunk. It
must be a multiple of each sample's `latent_block_size` L: a chunk must not
split a block whose clean tokens attend
bidirectionally within that block. With B=K/L blocks per chunk, block b>=0
belongs to chunk `b // B`; the independent first latent at block -1 belongs
to chunk 0. For K16/L2, chunk 0 contains the independent first latent and
blocks 0 through 7, then chunk 1 contains blocks 8 through 15.

For a query's temporal chunk, a clean hidden state from another chunk is
detached **before generation-stream normalization** at every layer. Its
normalization and K/V projection parameters still receive readout gradients.
Within-chunk history, text conditions and current predictions keep their
applicable gradients. The clean/current teacher-forcing layout, original
attention keys, forward values and loss masks are retained. Every originally
supervised current-video target in every block remains supervised; no early
blocks or targets are dropped.

This is temporal-chunk truncation, not a sliding guarantee of K recent history
latents for each prediction. A prediction near a chunk start has a shorter
within-chunk encoder history than a prediction near its end. The policy also
does not select only the final K clean latents of the whole sequence. If all
post-first latents fit in one chunk, there is no cross-chunk history to detach
and the original full-gradient path applies. K merely covering the configured
per-layer history visibility does not imply this fallback when a clip spans
multiple chunks.

All chunks are rebuilt within the current forward using the current weights.
No hidden state or K/V survives into another window or optimizer update.
Video chunk truncation does not promise faster training or lower peak memory:
all prediction blocks remain supervised and historical clean positions can
still train through their own chunk. Action recent-K timing results are not
Video performance evidence.

### Execution and validation boundaries

[`Cosmos3Config.validate_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/configuration_cosmos3_transformer.py#L110)
validates the model settings.
[`Cosmos3Model.forward`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/modeling_cosmos3_transformer.py#L363)
calls [`partition_history`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L143)
once per forward. Action builds
[`HistoryPartition`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L12)
for the decoder's
[`forward_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/layers.py#L221)
path. Video's
[`_partition_video_history`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L29)
builds [`VideoHistoryPartition`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L22)
for [`forward_video_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/layers.py#L205).
Partition metadata is forward-local, not checkpoint or episode state.

Default decoder-layer FSDP2 wrapping and checkpoint/SAC retain their native
paths. Independently FSDP-wrapping `Cosmos3Attention` (for example, through
`model.basic_modules: [Cosmos3Attention]`) is unsupported and raises an error
when either history partition is active; the altered attention dispatch must
not bypass the wrapped module's hooks.

The flag applies only to gradient-enabled training. The shared
`tbptt_keep_history_latents` setting follows the stage's policy described above.
Evaluation, no-grad forwards, online inference and RTC input gradients use
their ordinary paths even when the saved model config
contains the flag. The settings are retained in model configuration exports;
set the desired values explicitly when resuming a training experiment.
[`Cosmos3Config.from_dict`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/configuration_cosmos3_transformer.py#L120)
normalizes older exported model configs with separate stage lengths on load
to the length effective for their saved action interface; new exports
contain only the shared length. Training YAML must use the canonical field:
obsolete Video-only length settings raise an explicit rename error.

This deliberately changes the training gradient and is not a lossless
optimization. Split operator shapes can also change BF16 rounding. In a short
four-card Action Edge/AgiBot P1024/H32, 384 x 320 test with 48 windows per update, K16
reduced warm update time from 34.48 to 29.25 seconds and peak allocated tensor
memory from 30.68 to 28.92 GiB per rank. These measurements do not establish
convergence, use of long history or robot task quality. Compare held-out
performance at equal supervision and equal wall time before choosing K.

## Activation recomputation

The model's block loop calls HF's `_gradient_checkpointing_func`, which VeOmni
binds to its per-layer recomputation policy before FSDP wrapping. Standalone
models keep non-reentrant checkpointing by default, including callers that
toggle `gradient_checkpointing` directly. Evaluation and disabled checkpointing
use direct block calls.

The Edge Action recipe enables selective activation checkpointing (SAC) for all
28 layers: `model.accelerator.gradient_checkpointing.recompute_layers: -1` and
`selective_recompute_layers: 28`. SAC saves attention outputs and recomputes other
operators when needed, trading activation memory for less repeated computation.
The saved outputs belong to one forward/backward pass; they are not historical
K/V or a cache reused after an optimizer update.

Set `selective_recompute_layers: 0` with `recompute_layers: -1` to restore full
recomputation. Setting `recompute_layers: 24` and `selective_recompute_layers: 0`
keeps the first four Edge layers' activations and fully recomputes the last 24.
The selective count starts at the front of the recomputed range and is clamped
to its length. Layer selection changes memory requirements; check actual peak
memory for the resolution, packed token budget and FSDP layout in use. Network
weights, losses, causal visibility and the FP32 timestep MLP are unchanged.

## Data loading and packing

[`LeRobotDataset`](../../veomni/data/causal_cosmos3/datasets/dataset.py) owns
metadata, captions, prompts, task RNG state and sample error handling.
`ActionDataset` enumerates every legal decision anchor and calls
[`read_decision_window`](../../veomni/data/causal_cosmos3/readers/lerobot.py).
`sample.window` describes its future interval; `history_steps` and
`video_frame_ticks` describe the additional visual history. No full-video payload
is loaded during dataset construction.

The factory resolves source root globs once and requires measured Action
statistics, optionally expressed as `'{root}/meta/causal_action_stats.json'`.
Train/validation splitting pools `(root, episode)` identities within each source.
It selects ceil(5% of eligible episodes), caps validation at 100 episodes per
source and retains training data. A single-episode source stays in training;
100 is a validation cap, not a training-data cap. Split membership is deterministic
and recorded in `episode_split_manifest.json` under the checkpoint output directory.

`sampling: sample_uniform` concatenates candidate windows and uses the native
shuffled distributed sampler. Source/episode proportions therefore follow
candidate counts; source `weight` is unused in this mode. `source_weighted`
chooses sources using normalized weights, then uniformly draws windows within
the chosen source. Sampling does not imply temporally consecutive windows.

Each successful sample resolves its task after reading and before prompt
formatting/tokenization. Explicit validation mode/history leave task RNG untouched.
State dictionaries store task RNG only; no success/failure counters are maintained.

When no subtask description covers an anchor, the caption falls back to the
matching `task_index` description in `meta/tasks.parquet`. Every materialized
fallback emits a WARNING with source/root, episode, candidate index, start, task
index, reason and the complete `fallback_caption`. This includes missing/invalid
annotations, gaps and disabled subtask selection. `repr` escapes quotes/newlines;
matched subtasks produce no fallback warning. Missing nonempty task text or an
empty custom prompt raises before tokenization. `make_sample` propagates errors;
`__getitem__` logs ordinary failures and returns an empty-list skip marker.
Memory exhaustion propagates. Validation retains zero-weight empty packs for skips.

The canonical loader uses four workers per rank, `worker_num_threads: 1`,
`prefetch_factor: 2`, background prefetching and two FFmpeg threads per video
decoder (`sources[].read.decode_threads`). These are per-rank/per-decoder
limits: six ranks create 24 loader workers, and increasing workers also increases
prefetched RGB memory. Set the decoder limit explicitly to avoid automatic thread
pools competing across ranks. Worker and decoder threading change throughput;
timestamps, sampled pixels, masks and captions retain the same read rules.
The Action and Video recipes set `train.gc_steps: 0`, retaining Python's automatic garbage
collection in a fresh training process. Positive values disable automatic GC in
the trainer and schedule explicit collection at the configured update interval. Native loader
workers independently enable automatic GC at startup, including when no explicit
Torch thread limit is configured: forked workers must not inherit a trainer's
disabled-GC policy because they do not run its update callbacks. PyAV frame-format
objects contain cycles that otherwise accumulate during repeated decoding.
This changes object lifetime management, not sampled data or training arithmetic.
Setting zero does not undo GC disabled by earlier application code in the same
process; launch a fresh training process for the recipe.
Joint task selection uses worker-local RNG streams, so changing the worker count
changes the particular FD/ID/policy draws even with the same seed; their
configured probabilities and supervision rules are preserved.

CPU-side greedy packing uses a 54000-token budget and an eight-candidate minimum
buffer. `micro_batch_size=1` means one pack, which can hold several independent
windows. Oversized samples form standalone packs and are not truncated.
`grad_accum_iter=2` consumes two packs per rank per optimizer update; on eight DP
ranks that is 16 packs, not a fixed count of windows. The source YAML does not
set optimizer batch size or infer training steps from dataset length.
`train.max_steps=10000` counts updates per epoch; the loader cycles as needed.

## Launch and initialization

After replacing placeholder resource paths:

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge.yaml
```

The recipe retains backbone initialization. For fine-tuning a complete native
Action model, set `model.model_path` to that model and remove `action_*` from
`model.model_config.weight_loading.skip_patterns` to load trained action heads.
Inference requires complete trained weights; initialization checks do not
establish policy quality.

Keep P/H, camera layout, source FPS, state/action template, statistics and
preprocessing aligned with serving. Training and online serving default to
resolution `256` (a 384 x 320 concat canvas for 4:3 cameras); keep any explicit
overrides aligned. Match RoPE/RMSNorm implementations
as well: BF16 eager and `torch_npu` operations can produce different predictions.
Training with P48 does not validate deployment with P1024. Different context lengths
require evaluation. Window-aligned training still incurs VAE encoding and DiT prefill.

## Video pretraining

The [Video source YAML](../../configs/data/causal_cosmos3_video.yaml) selects
`stage: video`; its model uses `action_interface: none` and `state_dim: 0`.
No action/state columns, templates or statistics are read, even if present in a
source. The same entry point, trainer, native loader, split and validation paths
are used. Stage/condition/model-interface mismatches fail before loading data.

Video retains the generic block layout: `latent_block_size` defaults to two,
`full_history: false` samples latent-frame history from `history_range: [1,32]`,
and an explicit positive history or `full_history: true` overrides sampling.
It uses source-frame stride two above 16 FPS and stride one at or below 16 FPS.
Reader padding aligns intervals to `stride * temporal_compression * latent_block_size`.
Only the independent first latent is a video condition. Video supports generic
`frame_window` and `segmented` candidates, including `max_actions: -1` for full
ranges. These options are Video functionality, not a second Action training mode.

The Video recipe keeps `enable_tbptt: false` and
`tbptt_keep_history_latents: 16` explicitly. Setting only `enable_tbptt: true`
enables its default temporal-chunk policy, described in
[TBPTT](#optional-history-gradient-truncation-tbptt). The shared length has a
different interpretation in Action. `train.gc_steps: 0` retains automatic GC;
it does not change the video training targets or time-block layout.

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_video_edge.yaml
```

## Validation

`eval_steps=1000` evaluates held-out episodes without optimizer updates. Joint
Action validation enumerates every decision in each of FD, ID and policy,
including tasks with zero training sampling weight. A fixed Action mode evaluates
only that mode; Video evaluates its windows once. Validation does not resample
training tasks or consume training/global/condition RNG. No generated videos or
action success metrics are implied by this loss-only evaluation.

Identities include source/root/episode/start, future valid/padded counts and
Action P/H/layout. History and time/video/action noise seeds are stable across
rank, ordering, training step and repeated passes. DP ranks run equal rounds;
repeated padding samples have weight zero. Failed reads/preparation are coordinated
before model collectives. Per-mode losses average supervised windows; counts
separately report unsupervised, failed and rank-padding windows. A mode with no
supervised windows raises rather than reporting a misleading mean.

Model/condition modes and all RNG states are restored even after validation
failure. FSDP parameters are resharded on cleanup. Evaluation records and
`validation.jsonl` remain tied to the split/validation identity. See
[`datasets/validation.py`](../../veomni/data/causal_cosmos3/datasets/validation.py)
and [`CausalCosmos3Validator`](../../veomni/trainer/causal_cosmos3.py).

## Checkpoints

Training uses native VeOmni DCP. Model, optimizer, scheduler, training step and
training RNG state are restored. Data recovery is always **restart**: a resumed
epoch starts a fresh data stream; sample order and later losses need not match
an uninterrupted run. The Cosmos3 loader adapter returns no cursor state, so
background prefetch/checkpoint callbacks do not snapshot residual raw videos.
Old data cursors are ignored; `loader/rank_{R}.pt` retains a null placeholder
and checkpoint metadata records `data_resume_mode: restart`.

Full checkpoint recovery skips original backbone initialization. HF weight
export/conversion is separate from DCP recovery. Shared loader/checkpoint and
worker-lifetime behavior are unchanged; removing residual-video snapshots does
not promise elimination of every host-memory allocation.

## Verification

Decision contracts are covered by
[`test_causal_cosmos3_decision.py`](../../tests/data/test_causal_cosmos3_decision.py):
anchors, episode reset/tails, target offsets, task weights/RNG recovery, exact
conditions, attention isolation, loss masks, train/serve positions and VAE prefixes.
[`test_causal_cosmos3_pipeline.py`](../../tests/data/test_causal_cosmos3_pipeline.py)
covers native source sampling/packing, stage dispatch, episode splits,
caption warnings and deterministic/distributed validation. Video and explicit
offline diagnostic block contracts remain covered independently. NPU tests
exercise checkpointed backward across P0/P48/P1024 and all Action tasks.
[`test_causal_cosmos3_tbptt.py`](../../tests/models/diffusers/test_causal_cosmos3_tbptt.py)
checks the YAML/config round trip, disabled/full-K and single-chunk equivalence,
independent Action and per-query-chunk Video gradient oracles, conditions,
mixed geometries, early Video-target gradients, checkpoint/SAC and unchanged
evaluation/no-grad paths. Oracle agreement validates the split
implementation, not equivalence to untruncated training.
Full convergence and robot task success require a trained-policy evaluation.
