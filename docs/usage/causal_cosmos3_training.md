# Causal Cosmos3 training

This branch supervises Action decisions against one current measured state and
one future horizon. Full-gradient training uses independent windows; optional
episode TBPTT groups consecutive decisions into differentiable segments. The
[Action recipe](../../configs/dit/causal_cosmos3_edge_action.yaml) and
[source configuration](../../configs/data/action_datasets.yaml) use the same
history/current-state/future layout as the
[streaming service](causal_cosmos3_serving.md). The same Action recipe selects
these paths through `enable_tbptt`; the former `_decision` recipe names remain
removed.

[`tasks/train_causal_cosmos3.py`](../../tasks/train_causal_cosmos3.py) uses the
native `DiTTrainer`, `DiTModelRuntime` and FSDP2. Network parameter shapes,
flow matching, the shared 55D template and FD/ID/policy objectives are unchanged.
The window contract borrows Long-WAM's finite observed history and joint future
prediction; it does not add its separate Wan/ActionDiT experts or IDM objective.

For video-only pretraining, use the distinct
[Video recipe](../../configs/dit/causal_cosmos3_edge_video.yaml). For offline
checkpoint inspection, see [independent inference](causal_cosmos3_inference.md).
Its explicit multi-block diagnostic and rollout helpers remain separate from
Action training; their block packing/math are also used by Video pretraining.

## Action decision windows

Configure dataset roots, cameras and measured statistics in the source YAML.
Its only top-level sections are `sampling`, `validation` and `sources`.
The DiT recipe owns model/training controls and these sample-construction settings
under `data`:

```yaml
data:
  train_path: configs/data/action_datasets.yaml
  stage: action
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

Source YAMLs containing the former top-level `stage`, `geometry`, `candidates`
or `sample` sections are rejected. Move those sections into the recipe's `data`
mapping; they cannot silently override the experiment through the source file.
Other unknown source top-level keys are rejected as well.

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
Workers return CPU pixels; the training rank encodes them. Independent windows
do not share writable VAE state or DiT history. Episode TBPTT instead owns a
bounded streaming VAE and detached clean-history memory per ordered lane, as
described below. Neither training path replays frozen inference graphs during
backward.

Asynchronous H/S/R execution, NPU graph replay and RTC are inference
capabilities. The frozen streaming VAE is also used by episode TBPTT; it does
not introduce a trainable VAE, asynchronous optimizer or RTC training loss.
Native Cosmos flow-matching loss weights remain
uniform. Long-WAM's independently sampled video/action times, IDM partial video
denoising and detached imagined-video K/V are not implemented here.

## Optional history-gradient truncation (TBPTT)

Action and Video training default to full temporal gradients. The shared
`enable_tbptt` flag must be a boolean and defaults to `false`. Both stages use
one canonical temporal segment span, `tbptt_segment_steps`, defaulting to
**256 steps**. This is not the retained-history length. It must be a positive
integer and a **multiple of 16**. The internal read-only property
`tbptt_segment_latents` converts it using **16 steps = 1 latent time interval**,
so the default span is 16 latent time intervals. The stage determines the
training organization: Action steps are raw action/control ticks under the
supported geometry; Video uses the same
equivalent temporal-step scale, not MP4 source-frame indices.
Each decision retains its original task conditions and loss masks. To enable the
default 256-step policy in either DiT recipe, only set:

```yaml
model:
  model_config:
    enable_tbptt: true
```

The Action launcher also accepts an explicit command-line override without
editing the YAML. Run one of:

```bash
bash scripts/causal_cosmos3/training_scripts/train_action.sh --enable-tbptt true
bash scripts/causal_cosmos3/training_scripts/train_action.sh --enable-tbptt false
```

`--tbptt-segment-steps 256` optionally overrides the temporal segment span.
These task-level options also work when calling `tasks/train_causal_cosmos3.py`
directly, for either stage. Their canonical aliases are
`--model.model_config.enable_tbptt` and `--model.model_config.tbptt_segment_steps`.
Only explicitly supplied options override the YAML; the other model settings
remain intact. The boolean option accepts the parser's boolean values, including
`true` and `false`; the segment span is parsed as an integer and retains the
stage's existing alignment validation.

Set `model.model_config.tbptt_segment_steps` explicitly only to choose
a different step span. For example, `256` means at most eight H32 Action windows
per segment; text boundaries and episode tails can end it earlier. `16` passes
the shared multiple-of-16 rule but cannot form an H32
Action window; stage-specific horizon/block alignment remains required.

### Action: continuous episode segments

With `stage: action` and `enable_tbptt: true`, the trainer selects an
episode-owned FIFO stream automatically. Let S be `tbptt_segment_steps`
and H the Action horizon. S limits the raw action/control steps covered by one
segment, not a sliding history length. Enabled Action TBPTT requires
`video_stride * temporal_compression == 16`, matching the fixed steps-to-latents
conversion. In addition to `S % 16 == 0`, a segment must contain complete
prediction windows: `S % H == 0`. Each full segment contains `D = S / H`
decisions at stride H. Defaults S256/H32 give eight anchors
`s, s+32, ..., s+224` and a segment ending at
`s+256`. P1024 still sets each decision's retained history to 65 observed
latents; S does not increase that history budget.
This bounds directly readable memory, not the full forward receptive field:
retained hidden states can already summarize older observations.

Every decision keeps its own measured current state, H targets relative to
that state, independently sampled FD/ID/policy task and joint future sigma.
Each segment shares the clean-history encoder across its decisions and computes
all their video/action losses. A forecast can read only the observed history
available at its own anchor; the raw segment's later ground-truth frames are
targets or observations for later decisions, not early conditioning.

The segment boundary detaches clean layer-input hidden states before their
generation normalization. The next segment uses its current norm/K/V projection
weights to read that memory. Within-segment history and every active decision
retain their applicable gradients. One backward covers a segment's supervised
decisions; the ordinary optimizer step follows the configured accumulation of
segments. Loss scaling uses the actual number of decisions across every DP rank
and accumulated segment, so text-shortened segments, episode tails and
zero-weight validation padding do not acquire extra per-decision weight.

Detached layer inputs may survive an optimizer update. They were produced under
the preceding weights, so this is an additional approximation beyond simply
stopping gradients through a freshly recomputed prefix. The cold/reset prefix is
encoded under the current weights without retaining its old encoding graph.
This path is not advertised as full-gradient-equivalent or lossless.

#### Segment data and runtime contract

[`build_episode_dataset`](../../veomni/data/causal_cosmos3/datasets/pipeline.py#L170)
requires `sampling: sample_uniform`. `source_weighted` fails explicitly because
selecting whole episodes would change the existing per-window weight semantics.
Ranks own complete episodes before splitting their owned sets across loader
workers. Each worker is one ordered lane. An episode is consumed before its lane
chooses another one. Episode order is shuffled between passes; the legal anchor
phase rotates on each visit. With the default `start_stride: 1`, phases cover
all tick offsets modulo H instead of permanently training only anchors `0,H,...`.
The sampling unit changes to FIFO segment slots across rank/worker lanes.
Lanes receive equal microbatch slots and consume their complete owned episodes;
unequal episode/segment mass across lanes changes long-run candidate frequencies.
Short episodes also cycle a smaller set of legal phases and revisit their
anchors more often per episode visit than episodes spanning a full H phases.
Consequently, the accepted `sample_uniform` source setting does not preserve the
old global candidate-uniform measure. The trainer automatically exports
`training_sampling_unit="episode_lane_fifo"` to record this scientific change;
per-update actual-decision normalization does not remove the sampling difference.
Changing final prompt token IDs ends the current context and resets both VAE
and DiT history before the next decision.

[`EpisodeDataset`](../../veomni/data/causal_cosmos3/datasets/episode.py#L39) yields one
indivisible segment dictionary. Its `lane_id` selects state ownership;
`stream_id`, `segment_id`, `reset`, `final` and `start_tick`/`end_tick` define
ordered continuation. `decisions` contains the actual ordered decision records,
each retaining the ordinary per-decision action/state masks, anchor, task,
prompt and valid/padded supervision. Empty training ranks fail; finite validation
may yield no local segment and receives coordinated zero-weight padding.

The call sequence is:

1. [`read_episode_rows`](../../veomni/data/causal_cosmos3/readers/episode.py#L10)
   reads one union of state/target/timestamp/mask/task rows. A cheap next-decision
   row lookahead identifies text changes at a full segment boundary.
   [`read_episode_video`](../../veomni/data/causal_cosmos3/readers/episode.py#L57)
   decodes the actual shared RGB span once per camera. A cold span includes P
   history; a continuation includes the boundary frame and only new observations.
2. [`encode_segment`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_condition/episode_condition.py#L47)
   validates lane/context/order and removes the already-encoded boundary from a
   continuation. [`BoundedEpisodeVAE`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/episode_vae.py#L7)
   retains convolution state and a bounded latent ring, not the full episode.
3. [`prepare_segment`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_condition/episode_condition.py#L132)
   preserves the native per-decision flow-matching inputs/targets and compacts
   their shared clean observations into one model payload.
4. [`forward_segment`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/segment.py#L354)
   performs one layer traversal for the segment and returns a detached
   [`SegmentMemory`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/segment.py#L13)
   snapshot. [`episode_forward_backward`](../../veomni/trainer/causal_cosmos3_episode.py#L51)
   commits a healthy snapshot to its lane after backward; reset, final, errors,
   epoch boundaries and restart recovery clear the corresponding state.

By default (`train.micro_batch_size=1`), one microbatch is one entire segment.
Action episode training also supports a packed microbatch of B independent
streams. `grad_accum_iter` counts sequential microbatches per rank per optimizer
update; the update consumes `DP_size * B * grad_accum_iter` complete segments.
Each segment retains its actual decision count. Segments are not sorted or split
by greedy token packing; `data.max_seq_len` does not change S, D, B or the number of decisions.
It must not be interpreted as a request to truncate segment supervision.
Use the step span S, P/H, resolution and the actual peak allocation to size a segment.
VAE state, normalized latents and layer-input memory are bounded volatile
host/device state, never derived dataset disk caches or live autograd checkpoint
payloads. Restart recovery discards them and starts cold.

#### Multiple episode streams per forward

For Action episode TBPTT, use the existing batch parameters:

```yaml
train:
  micro_batch_size: 2
  grad_accum_iter: 1
```

On NPU this also requires `CLOSE_MATMUL_K_SHIFT=1` **before starting Python**.
The Action launcher exports it by default:

```bash
bash scripts/causal_cosmos3/training_scripts/train_action.sh \
  --enable-tbptt true --train.micro_batch_size 2 --train.grad_accum_iter 1
```

For a direct launch, export it explicitly; choose the visible devices and
process/shard counts for the available hardware:

```bash
export CLOSE_MATMUL_K_SHIFT=1
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tasks/train_causal_cosmos3.py configs/dit/causal_cosmos3_edge_action.yaml \
  --enable-tbptt true --train.micro_batch_size 2 --train.grad_accum_iter 1
```

The Trainer rejects NPU B>1 when this variable is absent or differs from `1`.
The launcher preserves an explicitly supplied value, so setting it to `0`
does not silently opt out of the B>1 check. Use the same environment for the
paired B1 baseline. This requirement does not alter the existing B1 or Video
batch-size rules. See the numerical evidence below.

Each rank needs at least B active worker lanes. Their count is
`min(num_workers or 1, owned_episode_count)` after complete-episode DP ownership;
all ranks validate this before worker iteration. Increase eligible episodes or
workers when this condition is not met. Independent-window and Video training
continue to require `micro_batch_size=1`. Validation retains the existing finite
single-stream evaluator.

The worker loader emits one raw segment at a time.
[`EpisodeBatchDataLoader`](../../veomni/data/causal_cosmos3/datasets/episode_batching.py#L29), a rank-local FIFO batcher,
selects one queue head from each of B different lanes, before background
prefetching. It never pairs dependent consecutive segments of one lane in the
same forward, skips a short segment, or pads its decision count to S/H. Lookahead
is bounded, and missing/non-producing lanes or an incomplete finite update raise
instead of retaining unbounded raw videos or silently discarding supervision.

The frozen VAE/condition is prepared per lane.
[`forward_segment_batch`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/segment_batch.py#L113) then concatenates the
streams' modality tokens and executes each decoder layer once: projections,
QKV and MLP use the combined token matrices. Packed TND plans isolate each
stream's text, history, current queries and state. RoPE keeps each stream's own
FP32 coordinates, with no pack-offset shift of time. Reset streams compute their
own cold prefix while continuing streams use their own detached layer memory.
Their outputs and bounded next-memory snapshots are split back by stream.

Loss retains the native decision mean: streams with eight and two actual
decisions contribute eight and two decisions, respectively, rather than equal
stream means. Trainer then normalizes across all accumulated microbatches and
DP ranks using the update's actual decision count. There is one model forward
and backward per packed microbatch. Every candidate memory is validated before
any batch memory is committed; preparation/validation errors invalidate the
affected lanes. Forward/backward collective failures abort the distributed job,
rather than pretending to skip a failed batch. Epoch/end/error cleanup stops
the background reader before shutting down the owned worker iterator.

Batching adds no history compression or extra gradient truncation: P/H/S, masks,
task draws and per-lane cache rules retain their meaning. B1's existing worker
batching can place dependent same-lane segments in one update; B2 deliberately
groups independent lanes and therefore can change update composition. A fair
performance/parity comparison uses the same independent segment pairs, weights,
noise and initial memories for `B1 * accum2` and `B2 * accum1`. Check losses,
complete parameter gradients and next caches before comparing throughput and
peak allocation. Floating-point matrix shapes can change rounding; neither a
lower iteration time nor equal nominal segment counts establishes equal
convergence quality. Grad accumulation alone still computes its segments
serially and does not enlarge a single GEMM.

##### NPU Matmul numerical setting

The local complete-Edge audit used one 910B3, FP32 master parameters and explicit
NPU BF16 forward autocast. It compared two independent real AgiBot cold segments
with eight decisions each, without optimizer updates. The prepared payload
digests were identical across the two environment experiments, weights stayed
unchanged, and a repeated B1 run had exactly zero full-gradient error.

| Local same-payload B2 versus B1 audit | Complete-gradient relative L2 | Gradient cosine |
| --- | --- | --- |
| `CLOSE_MATMUL_K_SHIFT` unset | 15.6124% | 0.9951914 |
| `CLOSE_MATMUL_K_SHIFT=1` | 0.274966% | 0.9999964 |

In this experiment, disabling Matmul K-shift restored identical video/action
forward outputs and all next-memory values (maximum absolute difference zero),
and reduced the full-gradient error. The model equations, targets, loss weights
and truncation policy were unchanged. Remaining gradients differ because the
BF16 batched backward and serial accumulation can round differently; identical
forward/cache values do not establish bitwise gradient equality.

This evidence is limited to the tested local hardware/software stack and fixed
payload pair. It does not certify every Ascend/CANN version or input shape,
multi-card equivalence, or convergence quality. Fresh environments still need
same-payload output, complete-gradient and cache checks. Throughput runs made
with this variable unset are historical diagnostics and cannot stand in for
performance of the recommended configuration; they must be rerun with `1`.

#### Checkpoint and serving behavior

The Action trainer records `history_encoding="episode_stream"` automatically
in model and condition configuration. This metadata selects continuous VAE
observations and stable stream RoPE coordinates in serving. Old window-trained
checkpoints retain `history_encoding="window"` and their existing behavior.
The encoding contract is a model property; changing an inference toggle is not
a substitute for using the checkpoint's trained history representation.
Episode-stream serving supports eager execution and RTC. NPU replay graphs are
currently unsupported for this representation and must fail explicitly rather
than replaying a window graph with different history semantics.
[`ServingModelPolicy`](../../veomni/models/diffusers/causal_cosmos3/serving/serving_model.py#L72)
selects the checkpoint representation; the
[`EpisodeObservationEncoder`](../../veomni/models/diffusers/causal_cosmos3/serving/episode_streaming.py#L12)
owns the bounded continuous observation stream. Decision anchors must preserve
the VAE phase: with default geometry, anchor advances are multiples of 16 action
ticks. Fixed RTC S16/R24 has aligned observation anchors; R is the handoff budget,
not the observation-trigger stride. Adaptive clients must quantize trigger
anchors to this phase or receive an explicit incompatibility error.
An error in an episode observation, preparation or prediction entry invalidates
the stream and requires a fresh reset. The older offline `CausalCosmos3Pipeline`
and `sample_block` paths support window checkpoints only and explicitly reject
episode-stream checkpoints; use `ServingModelPolicy` for this representation.

#### Standalone recent-K compatibility

The standalone ordinary Action `Cosmos3Model.forward` retains its earlier
recent-K gradient path for diagnostics and compatibility: it detaches the
encoding of observed rows older than K while retaining their trainable memory
readout. Here K is the internal latent count `tbptt_segment_steps / 16`.
This is not the Action trainer's `enable_tbptt: true` training path.
Historical recent-K timing measurements must not be presented as measurements
of continuous episode TBPTT.

### Video: temporal chunks

Independent `stage: video` pretraining supports a separate chunk-boundary
gradient policy using the same `enable_tbptt` and `tbptt_segment_steps`
settings. Here the length is the span of each gradient-truncation temporal segment
within a clip. Steps use the fixed equivalent temporal scale of 16 steps per latent;
they are not the MP4 source-frame count. Video's existing FPS-based downsampling
is independent of this parameter and is unchanged. Convert to the internal
latent span `K = tbptt_segment_latents = tbptt_segment_steps / 16`. K counts the
post-first video latents per temporal chunk. It must be a multiple of each
sample's `latent_block_size` L: a chunk must not
split a block whose clean tokens attend
bidirectionally within that block. With B=K/L blocks per chunk, block b>=0
belongs to chunk `b // B`; the independent first latent at block -1 belongs
to chunk 0. For 256 steps (K16)/L2, chunk 0 contains the independent first latent and
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

[`Cosmos3Config.validate_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/configuration_cosmos3_transformer.py#L128)
validates the model settings.
The ordinary standalone
[`Cosmos3Model.forward`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/modeling_cosmos3_transformer.py#L265)
calls [`partition_history`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L143)
once per forward. Its diagnostic Action path builds
[`HistoryPartition`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L12)
for the decoder's
[`forward_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/layers.py#L284)
path. Video's
[`_partition_video_history`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L29)
builds [`VideoHistoryPartition`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/tbptt.py#L22)
for [`forward_video_tbptt`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/layers.py#L268).
This ordinary partition metadata is forward-local, not episode memory. The
Action trainer's episode path instead passes `stream_segment` and
`stream_memory` to the shared-history segment implementation.

Default decoder-layer FSDP2 wrapping and checkpoint/SAC retain their native
paths. Independently FSDP-wrapping `Cosmos3Attention` (for example, through
`model.basic_modules: [Cosmos3Attention]`) is unsupported and raises an error
when either history partition is active; the altered attention dispatch must
not bypass the wrapped module's hooks.

The shared `tbptt_segment_steps` setting follows the stage's policy above.
The flag alone does not enable optimizer updates or gradient truncation in an
ordinary eval/no-grad forward. Episode validation and serving do retain the
checkpoint's recorded `history_encoding` representation. The settings are
retained in model configuration exports; set the desired values explicitly when
resuming a training experiment.
[`Cosmos3Config.from_dict`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/configuration_cosmos3_transformer.py#L151)
accepts only the canonical `tbptt_segment_steps` field, as do training YAML
and keyword/command-line overrides. Obsolete `tbptt_keep_history_steps`,
`tbptt_keep_history_latents` and separate Video-only latent length fields raise
an explicit rename error in all configuration entry points, including JSON;
no compatibility migration is
performed. Exports contain only the steps field. The derived latent count is
an internal read-only value rather than a second configuration field.

These policies deliberately change training gradients and are not lossless
optimizations. Segment/operator shapes can also change BF16 rounding. Earlier
standalone recent-K timings do not measure the new episode path. Evaluate
throughput, peak memory and held-out quality at matched supervision and wall
time; neither implementation correctness nor a short stability run establishes
convergence, long-history use or robot task success.

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

In the default independent-window path, `sampling: sample_uniform` concatenates candidate windows and uses the native
shuffled distributed sampler. Source/episode proportions therefore follow
candidate counts; source `weight` is unused in this mode. `source_weighted`
chooses sources using normalized weights, then uniformly draws windows within
the chosen source. Sampling does not imply temporally consecutive windows.
Action episode TBPTT uses the whole-episode-owned stream described above and
requires `sample_uniform`; it does not pass segments through this sampler.

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
`__getitem__` logs ordinary failures and returns an empty-list skip marker in
the independent-window path. Episode reads propagate failures so the trainer
can invalidate the lane instead of silently reusing history across a missing
segment.
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

Independent-window CPU greedy packing uses a 54000-token budget and an eight-candidate minimum
buffer. `micro_batch_size=1` means one pack, which can hold several independent
windows. Oversized samples form standalone packs and are not truncated.
`grad_accum_iter=2` consumes two packs per rank per optimizer update; on eight DP
ranks that is 16 packs, not a fixed count of windows. The source YAML does not
set optimizer batch size or infer training steps from dataset length.
`train.max_steps=10000` counts updates per epoch; the loader cycles as needed.
For Action episode TBPTT, one microbatch contains B FIFO segments from different
lanes (`micro_batch_size=B`, default1), and the accumulation count consumes that
many microbatches per rank. A complete segment is never split or truncated to
meet `max_seq_len`; this token budget does not select the number of streams.

## Launch and initialization

[parse_causal_cosmos3_args](../../tasks/train_causal_cosmos3.py#L25) reads the recipe, extracts `data.stage`,
`data.geometry`, `data.candidates` and `data.sample`, then reads the source YAML
once and returns `(args, dataset_config)`. Typed TBPTT command-line overrides
remain in `args.model.model_config`; the other model fields are preserved.
An explicitly incompatible Condition stage is rejected before it is replaced
with the resolved recipe stage.

[CausalCosmos3Trainer](../../veomni/trainer/causal_cosmos3.py#L93) receives
`args, dataset_config=dataset_config, tokenize=tokenize`, copies the assembled
dataset configuration, validates the effective stage/model
interface and uses it for the existing training and validation paths. It does
not reopen the source YAML. Global rank zero writes `dataset_config.yaml` under
`train.checkpoint.output_dir` after dataset construction. The snapshot includes
the extracted sample-construction settings and the source contents, unlike
`veomni_cli.yaml`; root globs remain configured patterns. This is configuration
metadata, not a persisted data cursor, episode split, latent or stream cache.

After replacing placeholder resource paths:

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge_action.yaml
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

The [Video recipe](../../configs/dit/causal_cosmos3_edge_video.yaml) selects
`data.stage: video`; its model uses `action_interface: none` and `state_dim: 0`.
The [Video source YAML](../../configs/data/video_datasets.yaml) supplies only
the source mixture and validation split.
No action/state columns, templates or statistics are read, even if present in a
source. The same entry point, DiT runtime and episode-split machinery are used.
Video retains its window loader and evaluator rather than the Action episode
stream. Stage/condition/model-interface mismatches fail before loading data.

Video retains the generic block layout: `latent_block_size` defaults to two,
`full_history: false` samples latent-frame history from `history_range: [1,32]`,
and an explicit positive history or `full_history: true` overrides sampling.
It uses source-frame stride two above 16 FPS and stride one at or below 16 FPS.
Reader padding aligns intervals to `stride * temporal_compression * latent_block_size`.
Only the independent first latent is a video condition. Video supports generic
`frame_window` and `segmented` candidates, including `max_actions: -1` for full
ranges. These options are Video functionality, not a second Action training mode.

The Video recipe keeps `enable_tbptt: false` and
`tbptt_segment_steps: 256` explicitly. Setting only `enable_tbptt: true`
enables its default temporal-chunk policy, described in
[TBPTT](#optional-history-gradient-truncation-tbptt). The shared length has a
different interpretation in Action. `train.gc_steps: 0` retains automatic GC;
it does not change the video training targets or time-block layout.

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge_video.yaml
```

## Validation

`eval_steps=1000` evaluates held-out episodes without optimizer updates. In the
independent-window baseline, joint Action validation enumerates every decision in each of FD, ID and policy,
including tasks with zero training sampling weight. A fixed Action mode evaluates
only that mode; Video evaluates its windows once. Validation does not resample
training tasks or consume training/global/condition RNG. No generated videos or
action success metrics are implied by this loss-only evaluation.

Action episode TBPTT uses finite continuous episode streams at a fixed visit
phase, with the same segment representation and per-decision task conditions.
Joint training evaluates each explicit mode separately. Ranks with fewer or no
local held-out episodes participate through coordinated zero-weight padding;
only real supervised decisions enter the metric count. This phase-fixed stream
grid is not the baseline's exhaustive all-anchor validation enumeration.
[`EpisodeValidator`](../../veomni/trainer/causal_cosmos3_episode.py#L272) seeds time
and noise from the validation seed, explicit task, context identity and segment
index. It uses its own model-memory map and temporarily replaces the condition's
lane-state map. Cleanup discards only validation streams, then restores the
original training stream map, RNG state and module flags. Validation must not
abort the VAE state needed by the next already-queued training continuation.

Baseline validation identities include source/root/episode/start, future valid/padded counts and
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
Episode TBPTT recovery also clears every per-lane VAE ring/convolution state and
detached DiT memory. These are volatile training state, not resumable tensor
payloads. The exported `history_encoding` field remains part of the model's
inference contract even though a resumed data stream starts cold.

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
Continuous data ordering, complete-episode ownership, phase coverage, shared
reads, per-anchor masks/tasks and cold/text/tail boundaries are specified by
[`test_causal_cosmos3_episode.py`](../../tests/data/test_causal_cosmos3_episode.py).
The differentiable shared-history segment and detached carry contract are
specified by
[`test_causal_cosmos3_segment.py`](../../tests/models/diffusers/test_causal_cosmos3_segment.py).
New episode measurements must be reported separately from historical
independent-window recent-K results; no speed, memory or convergence benefit
is inferred solely from enabling the flag.
Full convergence and robot task success require a trained-policy evaluation.
