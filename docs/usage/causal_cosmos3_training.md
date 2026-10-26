# Causal Cosmos3 training

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
