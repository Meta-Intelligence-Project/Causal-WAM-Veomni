# Causal Cosmos3 training

`tasks/train_causal_cosmos3.py` connects real action datasets to the standard
`DiTTrainer` and `DiTModelRuntime`. Start from
[`configs/dit/causal_cosmos3_edge.yaml`](../../configs/dit/causal_cosmos3_edge.yaml).
Select the dataset roots, measured normalization statistics and cameras in the
source YAML referenced by `data.train_path`.

```bash
torchrun --standalone --nproc_per_node=1 tasks/train_causal_cosmos3.py \
  configs/dit/causal_cosmos3_edge.yaml
```

Use `--nproc_per_node=8` for eight devices. The recipe uses FSDP2, meta
initialization, BF16 computation, FP32 optimizer parameters and gradient
recomputation. Singleton FSDP2 uses the same materialization and checkpoint
path. `cast_forward_inputs: false` preserves FP32 coordinates and diffusion
times; condition processing converts only modality payloads to BF16.
SP/CP must remain one. Singleton compilation is not supported by this recipe.

## Assets and model selection

`model.model_path` is the complete pretrained root containing `transformer/`,
`text_tokenizer/` and `vae/`. The entry point resolves these paths and supplies
the real prompt tokenizer to the dataset. Structure and condition configuration
are separate: `model.config_path` and `model.condition_model_path`.

The Edge recipe selects shared action/state interfaces with 55 channels. Its
weight-loading policy skips audio and the original action interface, then
initializes missing action/state parameters. Missing backbone parameters still
fail the audit. The loading report records every skipped and initialized name.
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
`train.micro_batch_size` must be one. `train.global_batch_size` counts packs
per optimizer update across DP ranks and must be divisible by DP size. For
example, global batch 16 with eight ranks accumulates two packs per rank.
`train.max_steps` counts optimizer updates per epoch.

`data.max_seq_len` is the greedy packing budget. An oversized individual
sample may still form a singleton pack; this setting is not a hard memory
limit. `data.dyn_bsz_buffer_size` controls packing lookahead.
Action-source YAML has its own schema; the causal trainer disables the
generic text multisource tracker that expects a `names` field.

The condition model performs VAE encoding, timestep/noise sampling and packing.
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

Data order recovery remains subject to the existing dynamic-loader and worker
limitations. Validate the selected worker configuration independently; a
successful zero-worker restore does not establish identical recovery with
asynchronous workers or a changed topology.
