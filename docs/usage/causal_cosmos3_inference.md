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

The loader reads the trained model configuration from `config.json` inside
the checkpoint directory for both DCP and native/HF weights. There is no
`model_config` setting in the inference YAML; existing non-null values are
rejected. Users must place the matching trained config beside the weights.
For VeOmni DCP, pass the `model/ckpt/` directory itself and copy the configuration
from that training run before inference:

```bash
cp outputs/robotwin_adjust_bottle_policy_8npu/model_assets/config.json \
  outputs/robotwin_adjust_bottle_policy_8npu/checkpoints/global_step_2300/model/ckpt/config.json
```

The DCP directory must contain:

```text
model/ckpt/
├── config.json
├── .metadata
└── *.distcp
```

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config /path/to/matching_robotwin_inference.yaml \
  --checkpoint outputs/robotwin_adjust_bottle_policy_8npu/checkpoints/global_step_2300/model/ckpt \
  --mode teacher_forcing --task policy \
  --dataset /path/to/lerobot_v3 --episode 5 --start 0 --max-actions 160 \
  --output outputs/teacher_step2300_episode5
```

The loader detects DCP by its `.metadata` file. A missing `config.json` fails
with the expected path and an instruction to copy the matching training config.
Only model weights are read, without optimizer, scheduler, trainer state or a
distributed process group. Parameter names and shapes must match the complete
native model. The step root and optimizer directory are not inference inputs.
When `.metadata` is absent, the existing safetensors / `.bin` loader is used,
including sharded HF weights. No format argument is needed. The existing
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

The [Policy launcher](../../scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh)
follows the training wrappers: run it in an activated torch_npu/CANN environment
from any working directory. It selects physical NPU 0 by default and invokes
the single-process CLI directly, without `torchrun`. Its
[configuration](../../configs/infer/causal_cosmos3_edge_action_policy.yaml) matches
the RoboTwin adjust_bottle run: measured state/action columns, training VAE and
statistics, resolution 256 and concatenated cameras.

```bash
bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh

ASCEND_RT_VISIBLE_DEVICES=0 EPISODE=8 OUTPUT_DIR=outputs/policy_step2300_ep8 \
  bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh --seed 123
```

Defaults are `teacher_forcing`, `policy`, the step 2300 `model/ckpt` directory
under `outputs/robotwin_adjust_bottle_policy_8npu`, episode 5, start 0, 160 action
steps and output `outputs/causal_cosmos3_action_policy_infer`. Set `PYTHON_BIN`,
`INFER_CONFIG`, `CHECKPOINT_PATH`, `DATASET_ROOT`, `OUTPUT_DIR`, `EPISODE`, `START`,
`MAX_ACTIONS`, `NUM_STEPS`, `HISTORY`, `GUIDANCE` or `SEED` to override these
settings; trailing CLI arguments take precedence. For a different training run,
place its trained `config.json` inside the selected checkpoint directory
and align the data resources. Both DCP and native/HF weight directories are
accepted through `CHECKPOINT_PATH`. Use a new output directory for another run.

The launcher enables the sample-local measured history cache by default.
Use `--no-kv-cache` to run the uncached baseline:

```bash
OUTPUT_DIR=outputs/policy_uncached \
  bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh --no-kv-cache
OUTPUT_DIR=outputs/policy_cached \
  bash scripts/causal_cosmos3/infer_scripts/infer_action_policy.sh --kv-cache
```

This first version supports only `--mode teacher_forcing --task policy`, on a
single process in eval/no-grad mode. `sampling.kv_cache` defaults to `true`.
FD, ID, `--task all` and rollout require `--no-kv-cache` or
`sampling.kv_cache: false`; they do not silently fall back to uncached inference.
It prefills the independent first latent, then complete measured clean blocks
jointly. Denoising reads per-layer text/history KV and executes only the current
video/action block and state. Predicted outputs never enter measured history.
`history` counts video latent frames, including when it cuts a historical block.
VAE encoding, random seeds, sampling schedules, physical action decoding and
video export retain the uncached behavior. Different matrix shapes may change
BF16 rounding; compare latent/action errors rather than assuming bitwise equality.
See the [cache contract](../design/causal_cosmos3_kv_cache.md) for ownership,
positions, failure handling and numerical validation.

Edit [the configuration](../../configs/infer/causal_cosmos3.yaml) for the
tokenizer, VAE, measured statistics, cameras and device. Its resource paths
refer to the authorized 1030 POC example. Keep the exported model's `config.json`
with its weights.

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode teacher_forcing --task all --no-kv-cache \
  --dataset /path/to/lerobot_v3 --episode 2 --start 0 --max-actions 64 \
  --output outputs/teacher_episode2

python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode rollout --task policy --no-kv-cache \
  --dataset /path/to/lerobot_v3 --episode 2 --start 0 --max-actions 366 \
  --output outputs/rollout_episode2

python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3.yaml \
  --checkpoint outputs/poc_iter_1000_hf \
  --mode rollout --task policy --no-kv-cache \
  --input-npz /path/to/initial_observation.npz --fps 30 --max-actions 64 \
  --output outputs/rollout_npz
```

`--steps`, `--history`, `--seed`, `--guidance`, `--prompt`, `--device`, `--dtype`,
`--attention-backend`, `--guidance-interval LO HI`, `--normalize-cfg` /
`--no-normalize-cfg` and `--fixed-step-sigmas SIGMA ...` override configuration.
The CLI always initializes NPU
and selects the configured `device`, such as `npu:0`. Defaults are BF16, 20 UniPC
steps, guidance 1 and 32 historical video latent frames. Training evaluation cadence is separate.

`ops_implementation` must match the training recipe. The NPU example uses
`torch_npu` RoPE and RMSNorm, as in `configs/dit/causal_cosmos3_edge_action.yaml`.
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
Graph compilation, EMA, multi-device inference and environment control are
outside this implementation. Rollout and FD/ID currently require the uncached path.
