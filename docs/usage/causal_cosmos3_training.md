# Causal Cosmos3 training

For manual checkpoint generation, see [independent inference](causal_cosmos3_inference.md).

`tasks/train_causal_cosmos3.py` connects real action datasets to the standard
`DiTTrainer` and `DiTModelRuntime`. Start from
[`configs/dit/causal_cosmos3_edge.yaml`](../../configs/dit/causal_cosmos3_edge.yaml).
Select the dataset roots, measured normalization statistics and cameras in the
source YAML referenced by `data.train_path`.

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

Set `candidates.max_actions: -1` to disable length-based windowing in both
`frame_window` and `segmented`. With `use_subtask: true`, each complete subtask
becomes one sample with its subtask text; episodes without valid subtasks fall
back to one complete episode with task text. With `use_subtask: false`, each
complete episode becomes one sample with task text. These ranges cover their
valid action steps; `start_stride`, `overlap` and `borrow_subtask_frames` do not
alter them, and the reader still pads the tail to a complete block. Positive
`max_actions` values retain the existing windowing behavior.

```bash
torchrun --standalone --nproc_per_node=8 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge.yaml
```

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

The Edge recipe selects shared action/state interfaces with 55 channels. Its
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

`model.model_config.train_generation_only` defaults to `false`. The Edge
recipe enables it to match the original action-training optimizer scope.
`Cosmos3Model.freeze_model()` is called by the official runtime before FSDP
wrapping, weight loading and optimizer construction. It selects GEN attention,
GEN MLP/norm parameters, the UND key norm read by GEN, time embedding,
video/action/state projections and action/state modality embeddings. Other
parameters are frozen. Optimizers consume the resulting `requires_grad` flags.

One microbatch is one variable-size packed list of samples, so
`train.micro_batch_size` must be one. Set `train.grad_accum_iter` in the DiT
training YAML to a positive integer to choose the number of packs each DP rank
processes before an optimizer update. The Edge recipe sets it to two. The
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
Action-source YAML has its own schema; the causal trainer disables the
generic text multisource tracker that expects a `names` field.
It contains only sources, sampling and geometry. Optimizer steps, accumulation,
seed, packing and dataloader options belong in the DiT training YAML under
`train` and `data`, rather than being duplicated in the source YAML.

The condition model performs VAE encoding, timestep/noise sampling and packing.
Its timestep stream uses `train.seed + dp_rank`, and its device-local noise
stream adds 32768 to that seed. DP ranks therefore draw independently while
single-rank behavior is unchanged. DCP stores and restores both stream states
per rank rather than reseeding at a resumed optimizer step.
`Cosmos3Model.forward(targets=...)` returns `outputs.loss`, with separate
weighted video/action losses. Loss normalizes each sample by active elements,
then averages samples within a pack; the standard trainer applies gradient
accumulation and FSDP applies DP averaging. Padding and conditioned positions
are excluded using the registered masks.

## Checkpoints

Use the native DCP manager. `train.checkpoint.load_path` points to a
`checkpoints/global_step_N` directory. Full non-LoRA restoration skips HF
initialization and restores model, optimizer, scheduler, training state,
dataloader state and the condition model's timestep/noise generators.
Frozen parameters remain in full DCP checkpoints; restoring does not initialize
action/state heads again. Preserve the same GEN selection and optimizer
configuration when resuming.

Data order recovery remains subject to the existing dynamic-loader limitations.
Exact-replay regression cases are currently expected to fail with both zero
and two workers. Restoring a DCP checkpoint therefore does not establish an
identical subsequent sample sequence. Validate the selected worker configuration
and topology independently.

## Automatic episode split

Configure the held-out episode split in the complete source YAML at
`data.train_path`, such as
[`causal_cosmos3.yaml`](../../configs/data/causal_cosmos3.yaml):

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
either side. Episodes too short for one action target after applying
`target_offset` are excluded and listed in the in-memory split summary.

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
Cosmos3 evaluation callback computes validation loss at the configured cadence.

For direct use, call
[`split_action_config`](../../veomni/data/causal_cosmos3/datasets/split.py)
or its source-level implementation:

```python
from veomni.data.causal_cosmos3.datasets.split import split_action_config_by_source

train_config, heldout_config, manifest = split_action_config_by_source(config)
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
`trainer.base.validation_dataset` through the existing `build_action_dataset()`
factory, using the held-out configuration with `sampling: sample_uniform`.
Validation reuses `ActionDataset` instances and their standard concat container,
enumerating every held-out source regardless of training weights. Geometry,
cameras, processing, normalization, sample and candidate settings are inherited
from the complete source configuration. Without `validation.split_ratio`,
`validation_dataset` and `validation_config` are `None`.

Every candidate of each held-out episode is retained. Global and source-level
`candidates` settings are shared with training: `frame_window` uses the same
window size and start stride, and `segmented` uses the same subtask boundaries,
borrowing, overlap, fallback and tail padding. Candidates with different
subtask text remain separate even when borrowing gives them the same span.
The episode cap does not impose a window cap.

FD, ID and Policy runs retain their own mode. A `joint` run expands each window
once for each of FD, ID and Policy, including modes with zero training sampling
weight. History is fixed per candidate, shared across modes, and follows the
training `full_history` or `history_range` settings.

[`ActionDataset.make_sample()`](../../veomni/data/causal_cosmos3/datasets/dataset.py)
accepts explicit `mode` and `history` arguments. These values are
resolved before prompt formatting, tokenization, condition indexes and token
length calculation. Training `__getitem__()` delegates to this method with its
original default sampling behavior. Validation supplies fixed mode/history, so
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
and action-noise streams for the loss evaluator. Seeds are independent
of global step, rank, worker and reading order. Each item's `record` contains its
source/window identity, geometry, history, mode and seeds. Dataset construction
and iteration do not collect a full file manifest or write validation artifacts.
The Cosmos3 validation executor consumes this iterator through VeOmni's evaluation
callback, as described below.

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
   generates independent seeded timestep, video-noise and action-noise streams,
   then calls the existing `sample_sigmas()` and `prepare()` methods. Time
   distributions, shifts, block mapping, masks and model loss weights are shared
   with training. `sample_sigmas(..., generator=...)` accepts an independent
   generator; training retains its original default generator.
4. The current model runtime executes under `torch.no_grad()` and its own
   parallel state and forward context. Validation does not call backward or
   the training postforward reduction that divides by gradient accumulation.
5. Each mode accumulates sample loss sums and counts, reduces them with SUM over
   the model's DP group, and divides once. Rank means are not averaged and no
   cross-mode overall loss is produced.

The per-window loss is the model's weighted `video_loss + action_loss`. FD and
ID naturally contribute zero for their fully conditioned modality. Completely
unsupervised windows are excluded. Valid zero-weight padding samples still run
forward to preserve collective ordering. If any rank reads an empty pack, all
ranks skip that round before condition/model forward; this also excludes healthy
peer windows in that round. Condition preparation errors abort evaluation on
all ranks. A mode with no supervised windows raises an error instead of reporting
a zero mean.

Metrics use `eval/fd`, `eval/id` and `eval/policy` prefixes for the selected modes:
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
