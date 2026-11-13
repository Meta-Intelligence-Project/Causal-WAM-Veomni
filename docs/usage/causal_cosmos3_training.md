# Causal Cosmos3 training

For manual checkpoint generation, see [independent inference](causal_cosmos3_inference.md).

`tasks/train_causal_cosmos3.py` connects video/action datasets to the standard
`DiTTrainer` and `DiTModelRuntime`. Start from the
[Action recipe](../../configs/dit/causal_cosmos3_edge_action.yaml) or
[Video recipe](../../configs/dit/causal_cosmos3_edge_video.yaml).
Select dataset roots and cameras in the source YAML referenced by
`data.train_path`; Action also requires measured normalization statistics.

Cosmos3 action/video data recovery always uses **restart**, without a mode flag
or YAML switch. Model, optimizer, scheduler, training step and training RNG
state are restored; data starts from a fresh stream in the resumed epoch.
Sample order and losses after resume need not match uninterrupted training.
The Cosmos3-specific dataloader adapter returns no cursor state, so background
prefetch does not copy residual raw videos from the packing buffer. Its
checkpoint callback ignores old data cursors and records
`data_resume_mode: restart`; the existing `loader/rank_{R}.pt` layout retains a null placeholder.
Worker lifetime settings and the shared VeOmni iterator/checkpoint components
are unchanged. This removes snapshot copying, but does not guarantee that all
host-memory growth is eliminated.

The model package separates input preparation in `cosmos3_condition`, the
trainable network in `cosmos3_transformer`, and shared implementations in
`cosmos3_core`. The condition and network packages each own their config and
model registration. See the [source layout](causal_cosmos3_network.md#代码组织)
for responsibilities and import paths.

Data code lives in `veomni/data/causal_cosmos3/`: `readers/` handles LeRobot
and video I/O, `transforms/` handles action/state processing, camera layouts
and prompt tokenization, and `datasets/` builds datasets, source sampling and
dataloaders. Package-level imports remain available; direct module imports
use these subpackages.

`LeRobotDataset` in `datasets/dataset.py` owns metadata, camera layouts,
task/subtask captions, window enumeration and lookup, history selection, common
sample metadata and prompt tokenization, sampling RNG state, and `__getitem__`
error handling. Its initialization requires no action columns, processor or
statistics. `CandidateConfig.validate()` checks window settings independently;
passing an action geometry also checks action-block alignment.

Subclasses implement `make_sample()` and may restrict the exclusive frame bounds
through `_episode_bounds(length)`. The default bounds cover the whole episode.
`ActionDataset` restricts them for `target_offset`, plans action reads in
`plan_read_window()`, and keeps task sampling, action/state processing, conditions and token
budgeting in `make_sample()`. It uses `get_window_metadata()` for one shared index lookup,
`select_history()` for history in **latent frames**, and `prepare_sample()` to
attach common fields and tokenize the final prompt after task fields are set.
The existing output key `history_blocks` retains that latent-frame unit.
Both subclasses expose `num_video_latents(window)` so training and validation
count the same padded video latents. `select_history(..., rng=...)` accepts an
independent validation generator; omitted `rng` retains training behavior.
See the [dataset history contract](../data/causal_cosmos3/datasets.md).
Explicit history does not consume sampling RNG; Action preserves the original
read-success, task-selection, history-selection order. Direct `make_sample()`
calls propagate errors, while `__getitem__` logs and skips ordinary sample
failures and propagates resource exhaustion.

`VideoDataset` implements its own `plan_read_window()` and `make_sample()` on this base.
It reads only timestamps, task indices and videos, even when the source also
contains action/state columns. No action template, statistics or action/state
payload fields are needed. Its deterministic stride is `2` when `source_fps > 16`
and `1` otherwise, including exactly 16 FPS. All cameras use this stride and the
same timeline. The prompt's FPS is `source_fps / stride`; decoding positions
still use the stored timestamps and video offsets.

Video blocks have a fixed latent-frame size (`latent_block_size`, default 2).
The reader pads raw frame intervals to a multiple of
`stride * temporal_compression * latent_block_size`, so changing stride does not
change the latent block size. Video defaults to random history in `[1, 32]`;
`full_history=True` or an explicit `make_sample(index, history=...)` use the shared
history mechanism. The first latent is the sole video condition (`[0]`). Common
helpers calculate valid video latents and text/clean-video/target-video token
counts; Action adds its own action/state counts. Neither history nor block sizes
count raw video frames.

For dataset-level use:

```python
from veomni.data.causal_cosmos3 import CandidateConfig, VideoDataset

dataset = VideoDataset(
    root,
    source="video_source",
    tokenize=tokenize,
    cameras={"head": "observation.images.head"},
    viewpoint="ego_view",
    candidates=CandidateConfig(mode="segmented", max_actions=160),
    latent_block_size=2,
    history_range=(1, 32),
)
sample = dataset.make_sample(0, history=8)
```

`CandidateConfig.max_actions` retains its existing name but represents raw frame
intervals for Video; it need not be an action-block multiple. Both datasets use
the same `Window` and `read_window()`. Explicit `plan_window(..., columns=None)`
selects video/task-only reading; omitting `columns` retains the default Action
schema. `Window.real_intervals` and `Window.intervals` expose the common units
while retaining the existing `real_actions`/`actions` storage fields for Action
callers and validation records. Video windows have `columns`, `action_width` and
`state_width` set to `None`, without dummy action/state tensors.

The shared `build_dataset(config, tokenize, ...)` factory selects the Dataset
using the assembled configuration's `stage`: `action` or `video`. The training
entry requires an explicit `data.stage` in the DiT recipe; standalone factory
callers retain the existing Action default.
It replaces the former `build_action_dataset` entry point. Each invocation uses
one stage for all sources. Source/root resolution, episode selections, candidate
overrides, seed assignment and both sampling modes use the same implementation.
Only the Action branch constructs `Geometry`, `ActionColumns` and the cached
`ActionProcessor`; the Video branch never accesses source statistics or action
column settings. Video `geometry` supplies the `VideoDataset` arguments shown
above, and `sample` configures history. Action-only task/geometry settings should
remain in the Action configuration.

See `configs/data/video_datasets.yaml` for video source configuration. After
initializing the usual VeOmni parallel state, build it with:

```python
from veomni.data.causal_cosmos3 import build_dataset

dataset = build_dataset(config, tokenize)
```

When migrating existing callers, import `build_dataset` instead of
`build_action_dataset`. Remove `CandidateConfig.borrow_subtask_frames`; windows
now retain their real boundaries and pad only at the tail. Direct `CausalLayout`
construction also requires `frame_ids` (the video latent frame ID, or `-1` for
other modalities), per-sample `block_sizes`, and per-sample `modes` (`video`,
`forward_dynamics`, `inverse_dynamics`, or `policy`). Action packing requires an
explicit sample `mode`; task mixtures are resolved before packing. Prefer `build_packed_layout()` to
derive these fields from sample metadata.

Trainer and the existing `build_action_dataloader()` helper now call this shared
factory. The shared `split_dataset_config()` partitions episodes for either
stage before construction, and `iter_validation_batches()` enumerates either
stage. Condition, Packing, model loss and the validation executor support both
stages. `parse_causal_cosmos3_args()` in `tasks/train_causal_cosmos3.py` reads
the DiT recipe and CLI overrides, extracts `data.stage`, `data.geometry`,
`data.candidates` and `data.sample`, and loads the source YAML once. Configure
`sampling`, optional `validation`, and `sources` in that file; keep experiment
parameters in the DiT recipe. The function returns `(args, dataset_config)`
using the existing DiT argument classes and sets
`model.condition_model_cfg.stage` from the recipe's `data.stage`.

`CausalCosmos3Trainer(args, dataset_config=dataset_config, tokenize=tokenize)`
copies the assembled config and checks the effective model interface before
model construction. It never reopens the source YAML. Training and validation
use this same config, including its `validation` episode split. Global rank zero
saves `dataset_config.yaml` under `train.checkpoint.output_dir` after dataset
construction; unlike `veomni_cli.yaml`, it includes the extracted experiment
settings and expanded source YAML contents. Root globs remain configured patterns;
the snapshot is not a persisted episode split or a data-cursor checkpoint.

The shared base builds an explicit `windows` index during initialization for all
candidate modes. Each row stores the episode metadata row, actual start frame,
valid source-frame interval count and subtask text ID for one candidate. A count
of N intervals covers N+1 frames, so the shared variable is `frame_intervals`.
`get_window_metadata(index)` resolves the index and episode metadata;
`plan_read_window(index)` creates the file-access and padding plan without I/O.
Video and action data are loaded on
demand. Frame windows select their text by start frame, while segmented windows
retain their original subtask boundaries and text. The four-column
`int64` index uses 32 bytes per candidate, excluding temporary construction data.

All candidate modes retain the real window extent. The reader rounds the output
length up to an action block and pads only at the tail: state/action repeat their
respective final values, and video queries clamp to the last real frame. Padding
does not read adjacent subtask content to fill a block or change the window's
start or valid length. Action validity and video latent validity are computed
from the real extent; purely padded positions are excluded from supervision,
while token budgets include the full padded length. A latent containing both
real sampled frames and padding remains valid. Frame windows can still cross
subtask boundaries according to their configured size; this is window selection,
not padding. Long segmented ranges retain their existing right-aligned tail window.

Set `candidates.max_actions: -1` to disable length-based windowing in both
`frame_window` and `segmented`. With `use_subtask: true`, each complete subtask
becomes one sample with its subtask text; episodes without valid subtasks fall
back to one complete episode with task text. With `use_subtask: false`, each
complete episode becomes one sample with task text. These ranges cover their
valid action steps; `start_stride` and `overlap` do not
alter them, and the reader still pads the tail to a complete block. Positive
`max_actions` values retain the existing windowing behavior.

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge_action.yaml
```

For video pretraining, use `configs/dit/causal_cosmos3_edge_video.yaml` instead.
The Bash wrapper `scripts/causal_cosmos3/training_scripts/train_video.sh` reuses the root `train.sh`
launcher with `configs/data/video_datasets.yaml` sources. Set `DATA_CONFIG`
to use another source YAML.
Run it in an activated torch_npu/CANN environment. It defaults to NPUs 2,3,4,6;
setting `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` selects four and derives the process
and shard counts accordingly. `MODEL_PATH`, `DATA_CONFIG`, `OUTPUT_DIR`, and
`DP_SHARD_SIZE` can override its defaults; trailing CLI arguments override the
training settings, for example `--train.max_steps 10 --train.eval_steps 0`.
Its source YAML retains `validation.split_ratio`, and the recipe enables periodic
validation with `train.eval_steps`. CPU tests cover stage propagation, data
splitting, variable-pack gradient accumulation and an optimizer update for both
stages. The Video recipe still requires a real checkpoint/data and multi-device
training smoke test on the target hardware.

Both video and action wrappers use W&B offline mode. With
`train.wandb.enable=true`, run data is saved under
`$OUTPUT_DIR/wandb/offline-run-*` by default. The wrappers create the W&B base
directory before launching training. Set `WANDB_DIR` to use a different base
directory. To move checkpoints and W&B logs together, set `OUTPUT_DIR` on the
wrapper rather than overriding only `--train.checkpoint.output_dir`.

Use at least two devices; this migration does not include singleton FSDP2
materialization support. The recipe uses FSDP2, meta initialization, BF16
computation, FP32 optimizer parameters and gradient recomputation.
`cast_forward_inputs: false` preserves FP32 coordinates and diffusion
times; condition processing converts only modality payloads to BF16.
On CUDA and Ascend, time embedding keeps the sinusoid and both Linear layers
in FP32 under autocast, then casts the result to the model compute dtype. Its
frequency constants are computed in FP32 on the input device. Casting the
sinusoid or intermediate time MLP activations to BF16 changes conditioning and
gradients even when the scalar timesteps remain FP32.
SP/CP must remain one. Compilation and DDP are outside the validated recipe.
The model currently uses full, non-reentrant layer checkpointing; runtime
controls for selective recomputation, reentrant mode and custom checkpoint
contexts are not yet connected to its checkpoint calls.

## Assets and model selection

`model.model_path` is the complete pretrained root containing `transformer/`,
`text_tokenizer/` and `vae/`. The entry point resolves these paths and supplies
the real prompt tokenizer to the dataset. Structure and condition configuration
are separate: `model.config_path` and `model.condition_model_path`.

Before loading model weights, the trainer checks the effective structure config
(architecture JSON plus `model.model_config` overrides). Video requires
`action_interface: none` and `state_dim: 0`; Action requires
`action_interface: shared` and positive `action_dim`/`state_dim`.
The trainer reports incompatible settings instead of changing the model heads.
The common Condition JSON may retain its Action default: the experiment stage overrides
that default when Condition is constructed.

The Action Edge recipe selects shared action/state interfaces with 55 channels. Its
weight-loading policy skips audio and the original action interface. With
`initialize_missing_patterns` omitted, all missing parameters, including backbone
parameters, are initialized using the model's initialization rules. A rank-zero
warning lists the affected modules and every missing parameter's name and shape;
the loading report records every skipped and initialized name. Review this warning
to check that the checkpoint contains the expected pretrained backbone weights.
To restrict initialization to action/state parameters, set
`model.model_config.weight_loading.initialize_missing_patterns` to
`["action_*", "state_*"]`; other missing parameters then fail the audit.
Set it to `[]` to reject any missing parameters.
For Nano or Super, select the corresponding structure JSON and a complete
matching pretrained root; changing `variant` alone does not change the model.
These selections do not imply full-weight validation of those variants.

## Training parameters

`model.optimizer.eps` controls the AdamW numerical stability epsilon and defaults
to `1e-8` when omitted. The Action recipe explicitly uses `1e-8`; the Video recipe
uses `1e-6` to match the original Cosmos video optimizer. This changes Video's
previous implicit `1e-8` value. It is unrelated to diffusion noise or RMSNorm epsilon.

`model.model_config.train_generation_only` defaults to `false`. Both Edge
recipes enable it; the Action recipe matches the original action-training optimizer scope.
`Cosmos3Model.freeze_model()` is called by the official runtime before FSDP
wrapping, weight loading and optimizer construction. It selects GEN attention,
GEN MLP/norm parameters, the UND key norm read by GEN, time embedding,
video/action/state projections and action/state modality embeddings. Other
parameters are frozen. Optimizers consume the resulting `requires_grad` flags.

One microbatch is one variable-size packed list of samples, so
`train.micro_batch_size` must be one. Set `train.grad_accum_iter` in the DiT
training YAML to a positive integer to choose the number of packs each DP rank
processes before an optimizer update. Both Edge recipes set it to two. The
Cosmos3 entry point uses its local `configure_gradient_accumulation` function
to set the framework's internal
`global_batch_size = micro_batch_size * dp_size * grad_accum_iter`,
so changing DP size preserves the requested accumulation count. For example,
two accumulation steps mean global batch 16 with eight ranks and global batch
8 with four ranks.

Omit `train.global_batch_size` when specifying `train.grad_accum_iter`.
Existing configurations that only specify `train.global_batch_size` retain
the standard VeOmni derivation. If both are present, `train.grad_accum_iter`
takes precedence. CLI overrides such as `--train.grad_accum_iter 4` are
supported. The entry point extracts the local option before constructing the
original `VeOmniDiTArguments`; shared DiT argument classes remain unchanged.
The conversion reuses the standard batch configuration and training loop.
`train.max_steps` counts optimizer updates per epoch.

`data.max_seq_len` is the greedy packing budget. An oversized individual
sample may still form a singleton pack; this setting is not a hard memory
limit. `data.dyn_bsz_buffer_size` controls packing lookahead.
Cosmos3 source YAML has its own schema; the causal trainer disables the
generic text multisource tracker that expects a `names` field.
It owns only `sampling`, `validation` and `sources`. The DiT recipe owns
`data.stage`, `data.geometry`, `data.candidates`, `data.sample`, packing and
dataloader options, plus the optimizer-step, accumulation and seed settings in
`train`. Set `stage`, `geometry`, `candidates` and `sample` directly in the DiT
YAML; these experiment settings have no per-field CLI flags. The entry point
passes them unchanged to the dataset factory, which validates their fields and
values. Standard DiT options such as `--data.train_path` and `--train.max_steps`
still support CLI overrides.

`data.sample.mode` accepts one task name (`forward_dynamics`, `inverse_dynamics`,
or `policy`), `joint` for all three tasks, or an explicit task list:

```yaml
data:
  sample:
    mode: [forward_dynamics, inverse_dynamics]
    mode_weights: [1, 1]
```

List weights correspond to tasks in the listed order; omitted weights default
to equal probabilities. Lists must be nonempty and contain unique task names.
Weights must match the list length, be finite and nonnegative, and have a
positive sum. `joint` retains the FD, ID, Policy weight order. Single-task
strings continue to accept legacy three-entry weights, which do not affect
their fixed task selection.

The condition model performs VAE encoding, timestep/noise sampling and packing.
VAE encoding primes the causal cache with the first frame, then uses temporal
chunks selected from the composed canvas: 32 frames for the 256 tier, 24 for
the 480 tier, and 8 for higher tiers. The final partial chunk is encoded without
extra temporal padding; caches are cleared between samples and on failure.
Its `stage` setting defaults to `action`. `prepare()` processes the whole video
batch first, then the whole action batch inside one `has_action` branch,
preserving the video-before-action noise draw order and the all-FD special case.
With `CausalCosmos3ConditionConfig(stage="video")`, preparation skips action
fields, action noise and action targets, and passes `current_actions=None` and
`has_action=False` to Packing. Stage/sample mismatches and video-stage `epsilon_action` are rejected
before RNG draws. The final dtype conversion only touches payloads returned by
Packing.

ID trains video and action denoising in one packed forward with the existing
output heads and weighted video/action losses. `CausalLayout.modes` carries the
task for each sample into attention; ID uses the visibility below. State Query/KV
visibility follows the task's action stream. No model parameters or token streams
are added.

For block `b`, `C_b` is the first (observed) video copy, `V_b` the second
(current) video copy, `A_b` the action stream, and `S_b` the state. `H_b` is the
existing latent-frame window of earlier observed video. Queries in each row
can read the KV columns marked yes:

| Query | Text | H_b | C_b | V_b | A_b | S_b |
| --- | --- | --- | --- | --- | --- | --- |
| C_b | yes | yes | yes | no | no | no |
| V_b | yes | yes | no | yes | no | no |
| A_b | yes | yes | yes | no | yes | yes |
| S_b | yes | yes | yes | no | yes | yes |

Same-block attention is bidirectional, including non-target positions within
the current video stream; other blocks contribute only the bounded observed
video history. In ID, neither video copy reads state, preventing state from
reaching later video blocks through clean-history KV. Video denoising uses only
visual/text conditions; actions still read current state. State Query/KV follow
the action stream's visibility in each task (conditioned actions for FD, target
actions for ID/Policy), while state inputs remain clean at timestep zero without
a denoising loss. Policy conditions cannot read target-like state KV. Text retains
its separate causal attention path. ID action queries always read the first video
copy, including during inference when the second copy has no video targets.

The grouped/TND planner distinguishes ID query modality as well as sample,
role and block, and gathers the appropriate KV set for each sequence. Shared
text/history/state KV may appear in multiple sequences; projections are shared
and backward accumulates their gradients. Existing chunk limits and fused TND
execution are reused without a dense pair mask. The dense reference constructs
visibility independently. Regression tests cover mixed-task and reordered packs,
shared-KV gradients, multilayer isolation, dual losses, and action prediction
parity between training and inference preparation. NPU parity cases additionally
exercise ID video targets on the actual TND backend.

`ActionDataset.make_sample()` defines observation and training-target indexes
together: `condition_frame_indexes_vision` contains all frames for ID and only
the first latent for FD/Policy; `target_frame_indexes_vision=[1, ..., T-1]`
selects video targets for all three tasks. Condition preparation passes these
indexes through without adding or overriding the task definition.
Packing keeps the first video copy clean and unsupervised, while the second
copy contains noisy video targets after the independent first latent. Padding
remains in the payload but is excluded from supervision. Token budgets, spans,
position coordinates, action/state payloads and valid-channel masks are retained.
Dataset samples are not mutated. The separate inference sample builder does
not set training-target indexes; its ID Packing calls retain the existing
observation-only video behavior.

`prepare(sigmas=..., action_sigmas=...)` accepts separate FP32 `[T]` or `[1]`
sigma tensors per sample, where `T` is the video latent count. Action sigmas
map to action steps through `action_frame_ids`. By default, ID action times
are sampled independently by calling the existing blockwise sampler again.
`train_time_distribution` and `shift` configure video sampling;
`action_time_sampler` and `action_shift` optionally override ID action
sampling. The action distribution defaults to `null`, inheriting the video
distribution. Action shift defaults to `5.0`, independently of video resolution
and video shift.
Both shifts accept a scalar or a resolution-keyed mapping. The supported
distributions remain `uniform`, `logitnormal`, `waver`, and `ltx2`.
FD and Policy retain their shared video/action time behavior even when action
overrides are configured; Video stage does not sample action times.

For example, to use the existing video settings and independently sample ID
action times from a shifted uniform distribution:

```yaml
model:
  model_config:
    action_time_weight: bsmntw
  condition_model_cfg:
    train_time_distribution: waver
    shift: {"256": 3.0, "480": 5.0}
    action_time_sampler: uniform
    action_shift: 5.0
    cond_noise_prob: 0.5
```

Training and validation select action settings with
`sample_sigmas(..., for_action=True)`. Both reuse the existing block mapping.
Video and action use successive independent draws from the existing time RNG;
no new generator or checkpoint RNG field is added.
Explicitly supplying both sigma lists avoids target-timestep RNG draws.
Validation samples both lists with its local seeded generator and passes the
results to `prepare()`, without advancing training RNG.

`model.model_config.action_time_weight` defaults to `uniform`; the action
recipe enables `bsmntw` for all supervised action targets, including Policy.
The loss does not branch on task mode. FD actions remain excluded by the
existing condition mask, and video losses retain uniform time weights.
This setting changes neither the sampling distribution nor the overall
`action_loss_weight` multiplier.

The reusable `bsmntw_weight(sigmas, shift=...)` function in
`cosmos3_core/flow_matching.py` uses the continuous-time formula from Long-WAM:

```text
g(sigma) = exp(-2 * (sigma - 0.5)^2) - exp(-0.5)
weight(sigma) = g(sigma) / mean(g(full_shifted_uniform_grid))
```

The reference grid has 1000 points, generated from `linspace(1, 0, 1001)[:-1]`
and shifted with the action shift. Its normalization constant is computed on
CPU on each call, independent of the current batch. The same reference grid is
used even if the configured sampler is not uniform; weights then need not
have expectation one under that sampler. Continuous times are evaluated
directly, without snapping to the nearest grid point.

Condition preparation only records each sample's effective sampling shift in
`targets["action_shifts"]`, alongside the actual action times after
`action_frame_ids` mapping. ID uses its independent action shift; FD/Policy
retain the video shift corresponding to their shared schedule. This keeps
sampling configuration in one place, including for fixed validation inputs.
No weight tensors are computed or passed during input preparation.

The shared loss calls the existing `TrainTimeWeight` interface to compute
FP32 weights from the action times and shifts, using the reusable BSMNTW
function. It multiplies squared errors by these weights before masked reduction;
the denominator remains the number of valid elements, not the sum of weights.
Padding, conditioned actions and invalid action dimensions remain excluded.
Video can later reuse the same function and loss interface with its own times
and shift, without introducing another scheduler class.

VeOmni's existing `veomni/schedulers/flow_match.py` also implements BSMNTW
through a discrete scheduler table. The continuous function is checked against
that table at its grid points; it does not use the scalar nearest-timestep
lookup for the per-block continuous training times. The shared scheduler is
not modified.

`model.condition_model_cfg.cond_noise_prob` defaults to `0.0`; the action
recipe enables `0.5`. It gates condition-video noise augmentation per ID sample.
FD, Policy and pure-video samples retain their existing preparation. For each
selected ID sample, `prepare()` draws fresh condition sigmas with the existing
video sampler and shift: one independent sigma per block, shared within the
block. This follows OpenWAM/Long-WAM's reuse of the video sampling distribution,
without LingBot's additional restriction to the latter half of its timestep
table. The Gaussian noise is independently drawn per tensor element.

The first latent remains clean with timestep zero. Later condition latents use
`(1 - sigma_cond) * clean + sigma_cond * noise_cond` and carry their actual
`sigma_cond * 1000` timesteps into the network's existing time embedding.
Augmentation runs inside the video loop, after constructing each sample's noisy
video and target. Both modalities' noisy inputs and targets use the original
clean data; source samples are not mutated. Condition-noise draws precede action
noise draws, so enabling augmentation changes the subsequent random draws while
preserving their distributions and checkpoint replay.

Packing accepts an optional pair `condition_videos` / `condition_video_timesteps`
to replace the first video copy and its time labels. The historical name
`clean_video` continues to identify that condition role; it no longer guarantees
zero noise when augmentation is enabled. The second video's conditioned positions
still use the original clean payload. Roles, attention visibility and loss masks
are unchanged; the augmented condition copy has no direct denoising loss.

The Bernoulli gate and condition sigmas use the existing time RNG; condition
epsilon uses the existing noise RNG. Checkpoint restoration covers both without
additional state. Probability zero consumes no augmentation RNG. Fixed loss-only
validation passes `augment_video_condition=False`, retaining clean conditions
and leaving training RNG untouched. Inference retains its existing clean
observation/cache inputs.

Packing keeps three responsibilities: `build_packed_layout()` registers final
token spans, `build_position_ids()` constructs CPU FP32 mRoPE coordinates, and
`pack_model_inputs()` assembles payloads, timesteps and supervision masks.
All three accept keyword-only `has_action=True` for compatibility with Action
callers. Video uses the same text, clean-video and current-video path with
`has_action=False`; action/state spans have zero length, `action_rows` is `None`,
and no action/state payload or action supervision mask is returned. Only populated
modality buffers are concatenated. Temporal padding remains in the input sequence
and is excluded from video supervision; history visibility retains its latent-frame
budget. Packs contain one stage, and action tensors are rejected in Video mode.
Optional sample metadata `target_frame_indexes_vision` selects unique video
target frames independently of observation indexes; it excludes the first latent
and defaults to the complement of video conditions. Packing derives current-video
roles, input restoration, timesteps and supervision from that selection.
`pack_model_inputs(action_timesteps=...)` accepts separate FP32 `[A]` or `[1]`
raw 0..1000 times per sample (`A` is the action-step count). If omitted, the
existing video-to-action timestep mapping is used. Action condition times are
still zeroed. Pure-video preparation/Packing rejects action sigma/time inputs.

Pure-video `prepare()` → Packing → `Cosmos3Model.forward(targets=...)` → backward
is covered with real CPU FP32/BF16 tests using `action_interface="none"` and
`state_dim=0`. Loss always handles video and only processes action when action
targets exist. Video returns only `video_loss`; Action keeps both loss entries.
The trainer checks and propagates the assembled experiment stage to Condition before
construction, so the training and validation paths select the same preparation
and loss behavior. Video needs no separate Trainer, Condition or Packing class.

Its timestep stream uses `train.seed + dp_rank`, and its device-local noise
stream adds 32768 to that seed. DP ranks therefore draw independently while
single-rank behavior is unchanged. DCP stores and restores both stream states
per rank rather than reseeding at a resumed optimizer step.
`Cosmos3Model.forward(targets=...)` returns `outputs.loss`, with separate
weighted video/action losses. Loss normalizes each sample by active elements,
then averages all samples within a pack, including fully conditioned samples
whose loss is zero and remains connected to the gradient graph. This does not
implement averaging over supervised samples across a whole optimizer update.
The standard trainer applies gradient
accumulation and FSDP applies DP averaging. Padding and conditioned positions
are excluded using the registered masks.

## Checkpoints

Use the native DCP manager. `train.checkpoint.load_path` points to a
`checkpoints/global_step_N` directory. Full non-LoRA restoration skips HF
initialization and restores model, optimizer, scheduler, training state,
and the condition model's timestep/noise generators. The Cosmos3 dataloader
restarts its stream in the restored epoch instead of restoring a data cursor.
Frozen parameters remain in full DCP checkpoints; restoring does not initialize
action/state heads again. Preserve the same GEN selection and optimizer
configuration when resuming.

Restart recovery does not reproduce the uninterrupted sample sequence or later
losses. The Cosmos3 adapter deliberately omits loader snapshots, including raw
video buffers, while retaining training progress and RNG recovery. The shared
dynamic loader keeps its existing exact-replay limitations; validate the selected
worker configuration and topology independently.

## Automatic episode split

Configure the held-out episode split in the source YAML at
`data.train_path`, such as
[`action_datasets.yaml`](../../configs/data/action_datasets.yaml):

```yaml
validation:
  split_ratio: 0.05
  max_episodes_per_source: 100
  seed: 73
```

All eligible `(resolved root, episode ID)` pairs within each source form one
pool. For a source with N >= 2, the validation count is
`min(ceil(split_ratio * N), max_episodes_per_source, N-1)`. The split ratio must
be a finite number strictly between zero and one. The cap must be a positive
integer; omit it or set it to `null` to use only the ratio and training-data
requirement. The seed must be an integer in `[0, 2**63)`, and defaults to 42.
Unrelated validation configuration keys are preserved without a key whitelist.

The cap applies to each source independently, after pooling all its roots.
Three sources capped at 100 can provide up to 300 held-out episodes. Each
source with at least two eligible episodes retains training data and supplies
held-out data. Single-episode sources remain in training; startup fails if no
source can supply validation episodes. Individual roots can fall entirely on
either side. An eligible episode must contain at least two frames within its
readable range. For `stage: action` (the default), this range accounts for
`target_offset`; for `stage: video`, it is the full episode, independent of
action columns or statistics. Excluded episodes are listed in the in-memory
split summary. Splitting, Dataset window construction and Reader planning share
`episode_frame_bounds()` in `readers/lerobot.py`. This checks episode frame
bounds only; subtask ranges and candidate enumeration remain Dataset concerns.

Selection uses a stable SHA-256 order derived from the seed, resolved root and
episode ID. Source/root ordering, training seed, source sampling weights, DP
rank and world size do not change membership. Source names must be unique,
and the same physical root cannot occur twice across automatic-split sources.
Copies at different paths are not detected as duplicates.

The trainer uses the held-out episodes' exact complement for training. It
retains the held-out source configuration as `trainer.base.validation_config`
and the split summary as `trainer.base.episode_split_manifest`. When
`validation.split_ratio` is absent, all configured episodes remain available
for training and `validation_config` is `None`. With a held-out dataset, the
Cosmos3 evaluation callback computes the selected stage's validation loss at the
configured cadence. Validation and Condition stages must match; validation does
not change the Condition stage automatically.

For direct use, call
[`split_dataset_config`](../../veomni/data/causal_cosmos3/datasets/split.py),
which replaces both Action-specific split entry points:

```python
from veomni.data.causal_cosmos3.datasets.split import split_dataset_config

train_config, heldout_config, manifest = split_dataset_config(config)
```

Both outputs inherit camera, geometry, candidate, sample, processing and
normalization settings. Root-specific statistics keep their original
`{root}` resolution; statistics are neither refitted nor audited for original
episode membership. The episode cap limits episodes only, leaving candidate
window generation unchanged.

Source `episodes: [0, 3]` can restrict the input universe before splitting.
An `episodes` mapping can select IDs per resolved root and must cover every
root in the source. Lists under `read.episodes` are also supported. Unknown
episode IDs, duplicate roots and conflicting selections fail at startup.

Every startup, including checkpoint resume, recomputes membership from the
current source configuration and metadata. Splitting reads metadata only,
does not consume training RNG and does not copy payload files. No split file
or checkpoint sidecar is read or written; old `data_split.json` files are
ignored. The in-memory summary records per-source/root counts, episode
membership and a split ID. Dataset-construction errors are synchronized across
the model's DP group.

## Fixed validation data

When automatic splitting is enabled, `CausalCosmos3Trainer` also builds
`trainer.base.validation_dataset` through the existing `build_dataset()`
factory, using the held-out configuration with `sampling: sample_uniform`.
Validation enumeration reuses `ActionDataset` or `VideoDataset` instances,
enumerating every held-out source regardless of training weights. Geometry,
cameras, processing, normalization, sample and candidate settings are inherited
from the assembled dataset configuration. Without `validation.split_ratio`,
`validation_dataset` and `validation_config` are `None`.

Every candidate of each held-out episode is retained. Global and source-level
`candidates` settings are shared with training: `frame_window` uses the same
window size and start stride, and `segmented` uses the same subtask boundaries,
overlap, fallback and tail padding. Each subtask keeps its own start, valid
length and text even when padded output lengths are equal.
The episode cap does not impose a window cap.

FD, ID and Policy runs retain their own mode. An explicit task list expands
each window once per listed task in the configured order; `joint` expands it
for FD, ID and Policy. Both include listed modes with zero training sampling
weight. Video enumerates each window once with mode `video`. One iterator accepts
one stage; mixed Action/Video datasets are rejected. History is fixed per
candidate, shared across Action modes, and follows the training `full_history`
or `history_range` settings through the same Dataset methods. The independent
history generator neither initializes nor advances the training generator.

[`ActionDataset.make_sample()`](../../veomni/data/causal_cosmos3/datasets/dataset.py)
accepts explicit `mode` and `history` arguments. These values are
resolved before prompt formatting, tokenization, condition indexes and token
length calculation. Training `__getitem__()` delegates to this method with its
original default sampling behavior. `VideoDataset.make_sample()` accepts explicit
`history` without an Action task argument. Validation supplies these fixed inputs, so
training sampling RNG is not initialized or consumed. Subtask captions fall back
to task text. If neither provides nonempty text, prompt formatting raises a
`ValueError`; `__getitem__` logs the failure and returns the existing empty-list
skip marker. Empty or whitespace-only custom prompts are also rejected before
tokenization. Training skips these candidates, and validation retains an empty
pack with `weight=0`, as it does for failed reads. Sample errors are caught
and logged on every occurrence by `__getitem__`, which validation also calls with
fixed mode/history. `make_sample` propagates errors directly; memory exhaustion
also propagates through `__getitem__`. There are no success/failure counters;
dataset state stores only the RNG state and ignores legacy checkpoint counters. Successful items contain
a singleton pack:

```python
from veomni.data.causal_cosmos3.datasets.validation import iter_validation_batches

dataset = trainer.base.validation_dataset
parallel = trainer.base.model.parallel_state
seed = trainer.base.validation_config.get("validation", {}).get("seed", 42)
for item in iter_validation_batches(dataset, parallel.dp_rank, parallel.dp_size, seed=seed):
    batch, record, weight = item["batch"], item["record"], item["weight"]
    # batch is {"samples": [sample]}, or {"samples": []} for a failed read.
    # record and weight are evaluator metadata.
```

Each call creates a finite iterator starting from the beginning. It visits sources
and roots in sorted order, retaining each dataset's candidate order, and computes
fixed mode/history/seeds only for the current sample. It reads in the calling
process, without building or storing a separate full validation plan. Each DP rank
receives an equal number of rounds for each mode. Repeated legal samples pad the
final round with `weight=0`; successful non-padding samples have `weight=1`.
This also handles fewer windows than ranks. Metric consumers must use these
weights to exclude padding and failed reads. The validation executor must
synchronize pack availability across DP ranks and skip a round on every rank
if any pack is empty, before condition processing or model forward. The data
iterator retains these empty slots so per-rank round counts remain equal.
Creating and consuming the iterator does not draw from default training RNG.

SHA-256-derived seeds fix history and provide independent timestep, video-noise
and (for Action only) action-noise streams for the loss evaluator. Seeds are independent
of global step, rank, worker and reading order. Each item's `record` contains its
source/window identity, geometry, history, mode and seeds. Action records and
seed inputs retain their existing format. Video identity uses `real_intervals`
and `intervals`, and its geometry contains `latent_block_size`,
`temporal_compression`, `spatial_compression` and `patch_size`; it creates no
Action geometry. The serialized Reader window retains its legacy count fields.
Dataset construction
and iteration do not collect a full file manifest or write validation artifacts.
The Cosmos3 validation executor consumes either stage through VeOmni's evaluation
callback, as described below. Video uses one `video` pass; Action uses its selected
FD/ID/Policy modes. The executor checks each iterator record's mode before
preparing model inputs.

## Periodic validation loss

With `validation.split_ratio` enabled, `CausalCosmos3Trainer` creates a
[`CausalCosmos3Validator`](../../veomni/trainer/causal_cosmos3.py)
and replaces its base trainer's empty evaluation callback with
`CausalCosmos3EvaluateCallback`. The callback inherits VeOmni's existing cadence:

```yaml
train:
  eval_steps: 1000
  eval_epochs: 0
```

Each value set to zero disables that trigger. When a step and epoch trigger
coincide at the same global step, evaluation runs once. Without a held-out
dataset, the trainer keeps the existing callback and training behavior.

The call sequence is:

1. `EvaluateCallback.on_step_end()` or `on_epoch_end()` calls the Cosmos3 callback.
2. `CausalCosmos3Validator.evaluate()` iterates `trainer.base.validation_dataset`
   through `iter_validation_batches()` with the model's DP rank, size and group.
3. The existing condition model encodes raw samples. `fixed_validation_inputs()`
   generates independent seeded timestep and video-noise streams, plus an action
   stream only for Action. Video does not require an action seed or action fields.
   It then calls the existing `sample_sigmas()` and `prepare()` methods. Time
   distributions, shifts, block mapping, masks and model loss weights are shared
   with training. `sample_sigmas(..., generator=...)` accepts an independent
   generator. Validation draws ID action times separately and supplies both
   `sigmas` and `action_sigmas` to `prepare()`. Training uses its own generator.
4. The current model runtime executes under `torch.no_grad()` and its own
   parallel state and forward context. Validation does not call backward or
   the training postforward reduction that divides by gradient accumulation.
5. Each mode accumulates sample loss sums and counts, reduces them with SUM over
   the model's DP group, and divides once. Rank means are not averaged and no
   cross-mode overall loss is produced.

The per-window loss is the model's weighted `video_loss` for Video, or
`video_loss + action_loss` for Action. Missing required loss fields raise an error
rather than being replaced with zero. FD contributes zero action loss; ID
supplies both video and action supervision masks. Completely
unsupervised windows are excluded. Valid zero-weight padding samples still run
forward to preserve collective ordering. If any rank reads an empty pack, all
ranks skip that round before condition/model forward; this also excludes healthy
peer windows in that round. Condition preparation errors abort evaluation on
all ranks. A mode with no supervised windows raises an error instead of reporting
a zero mean.

Metrics use `eval/video` for Video, or `eval/fd`, `eval/id` and `eval/policy`
for the selected Action modes:
`loss`, `count`, `unsupervised_count`, `padding_count` and `skipped_count`.
`skipped_count` counts real windows excluded by synchronized empty-pack skips,
including healthy peer windows. Rank 0 writes the logger and
`<checkpoint.output_dir>/validation.jsonl`, with global step, epoch, duration,
configuration and a `validation_id`. The ID describes configuration, seed version
and episode split identity; it does not fingerprint all dataset files. If WandB
is enabled and a run is already initialized, the callback logs `eval/*` with
`eval/global_step` without changing the existing training history step.

Validation restores Python, NumPy, Torch CPU/device RNG, the condition model's
own generators and each existing submodule's training flag in a `finally` block.
FSDP modules are explicitly resharded because evaluation has no backward pass.
The evaluator reuses the current model and leaves optimizer, scheduler and
training dataloader state untouched. A lazily loaded VAE remains in eval mode.
