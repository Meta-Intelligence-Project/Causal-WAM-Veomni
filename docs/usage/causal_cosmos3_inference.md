# Causal Cosmos3 independent inference

The [CLI](../../tasks/infer/infer_causal_cosmos3.py) runs a complete trained model
on one NPU, reusing the native network, action transforms, tokenizer, packing
and Wan VAE. It creates no Trainer, optimizer, training FSDP wrapper or validation
callback. Existing training and validation loss are unchanged.

## Triple-expert inference

The [triple configuration](../../configs/infer/causal_cosmos3_triple.yaml) requires
a complete checkpoint with `expert_layout: triple`. Teacher forcing supports
`video`, `forward_dynamics` and `inverse_dynamics`; `--task all` selects those
three tasks separately. Model-only rollout supports `video` and
`inverse_dynamics`. Policy and FDM rollout are disabled.
The legacy configuration below retains its FD/ID/policy and rollout behavior.
For `--task video`, a complete video-only checkpoint with `action_interface: none`
is also supported. FD/ID continue to require the complete trained shared 55D
action/state interface. A standalone FDM run trains the condition encoder and
video expert, not the action denoising head; use an IDM-trained checkpoint for
meaningful inverse-dynamics actions.

The [RoboTwin configuration](../../configs/infer/causal_cosmos3_triple_robotwin.yaml)
matches [action_robotwin.yaml](../../configs/data/action_robotwin.yaml): it selects
the training Cosmos3-Edge VAE, measured global RoboTwin statistics, matching
camera columns and separate `state_unified` / `action_unified` columns with
`target_offset: 0`. Source FPS is read from dataset metadata. The generic triple
configuration retains the original 1030 POC resource example.

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple.yaml \
  --checkpoint /path/to/trained_triple_hf \
  --mode teacher_forcing --task inverse_dynamics \
  --dataset /path/to/lerobot_v3 --episode 2 --max-actions 32 \
  --partial-video-steps 4 --video-stop-sigma 0.9 --action-steps 20 \
  --output outputs/triple_idm
```

The model exposes `expert_operation="full"`, `"video"` and `"action"`.
Training uses `full` for the current task; inference uses video/action forwards.
The first video forward builds missing text and FDM clean-action/state caches
internally and returns them for later forwards to reuse. Text and condition
prefill are not separate public operations. All calls enter through the full
model forward so they also work with model wrappers. Each block owns separate
text/cache branches for conditional and unconditional prompts when CFG is
requested.

| Task | Generation stages | Default steps |
| --- | --- | --- |
| Video | Text prefill, video UniPC; no action expert | 20 video steps |
| FDM | Text prefill, clean-action condition encoding once when enabled, video UniPC | 20 video steps |
| IDM teacher forcing | Text prefill, video-only partial Euler, final video-cache prefill, independent action UniPC | 4 Euler updates from sigma 1 to 0.9, then 20 action steps |
| IDM rollout | IDM action sampling, then complete the same video latent before updating history | The above stages, then 20 video-completion steps from sigma 0.9 to zero |

`--video-steps`, `--action-steps`, `--partial-video-steps`,
`--video-completion-steps` and `--video-stop-sigma` override independent stage
settings. `--video-completion-steps` maps to `sampling.video_completion_steps`
(default 20) and is used only for IDM rollout; `--video-steps` controls the full
video task and FDM. `--steps` remains the
legacy joint-sampler setting. Triple sampling does not accept a distilled joint
fixed-step schedule. `shift` applies to UniPC; partial-video Euler uses a linear
schedule from one to the explicitly selected stop sigma. Video/action noise
generators remain independently seeded. IDM completion uses a fresh schedule
starting at exactly `video_stop_sigma` and ending at zero; `shift` changes its
intermediate spacing without restarting the latent or its noise level.

`--fdm-conditioning text|action|both` (or `sampling.fdm_conditioning`) selects
the FDM condition at inference, defaulting to `both`. The text-only choice
skips action-condition prefill and blocks action/state KV; action-only blocks
text KV throughout video and action/state encoding. All choices retain measured
video history. IDM video supervision, cache encoding and rollout all use only
text/history; its action stage additionally reads state.

IDM zeros unknown future GT before model input construction. Its last Euler
velocity was evaluated before the final latent update, so one additional video
forward encodes the final latent at its actual `video_stop_sigma`. Action starts
from fresh Gaussian noise and runs only the action expert using that fixed KV;
video is not recomputed during action denoising. In teacher forcing, the partial
future remains noisy and is not decoded or labeled as a generated clean video.
IDM teacher forcing saves an
`observation.mp4` containing the initial measured frame, action outputs and
measured targets; no `generated.mp4` or VAE future reconstruction is written.
Pure video saves no action curves. FDM action arrays are marked as conditioned,
rather than predicted, in run metadata.

### Model-only rollout

Start with one measured image, the initial prompt and, for IDM, one measured
state. The reader does not request future video, action/state targets or future
subtask text. For triple IDM/video rollout, the requested `--max-actions`
duration may extend beyond the source episode's end because only the initial
observation is read. The dataset `--start` still must leave at least one
source-frame interval according to the shared window planner; the final source
frame alone is not an accepted dataset start. Legacy policy rollout retains its
episode-bounded horizon. Each block then runs:

1. **IDM:** sample future video noise and perform partial Euler updates from
   sigma 1 to `video_stop_sigma`. Encode the final latent at that actual sigma
   once more and fix its visual KV. Generate actions from independent noise.
2. **IDM:** continue the **same** future video latent from `video_stop_sigma`
   to zero. This continuation does not restart from noise or modify the actions
   already generated. **Video task:** directly run the full video schedule from
   sigma 1 to zero, without an action/state branch.
3. Append the completed clean video latents to the bounded history for the next
   block. The next block uses fresh video/action noise and its own temporary
   caches. Only the initial observation is VAE-encoded; generated pixels are not
   re-encoded for feedback. Decoding produces output video.

IDM uses its final valid decoded absolute action as the next state's
approximation, preserving the legacy ideal-execution assumption. It does not
measure a robot's physical feedback. The video expert continues to read only
text and video history, **not the generated actions**; the generated video is
therefore not an action-conditioned simulation of executing those actions.
Generated history can accumulate errors relative to teacher-forced training.
Pure video rollout never loads action statistics or reads action/state columns.
For RoboTwin's separate state/action columns, this feedback is explicitly an
approximation, not a declaration that the action label is the actual next state.
State and action still require matching physical channel mappings and valid
masks. RoboTwin action NPZ inputs provide `state_mask` explicitly, optionally
with a matching `action_mask`; the configuration does not assume Piper validity.

```bash
python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple_robotwin.yaml \
  --checkpoint /path/to/trained_triple_idm_hf \
  --mode rollout --task inverse_dynamics \
  --dataset /mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__adjust_bottle/clean \
  --episode 0 --start 0 --max-actions 160 \
  --partial-video-steps 4 --video-stop-sigma 0.9 --action-steps 20 \
  --video-completion-steps 20 --output outputs/triple_idm_rollout

python tasks/infer/infer_causal_cosmos3.py \
  --config configs/infer/causal_cosmos3_triple_robotwin.yaml \
  --checkpoint /path/to/trained_triple_hf \
  --mode rollout --task video \
  --dataset /mnt/sfs_turbo/public/datasets/robotwin_processed/RoboTwin__adjust_bottle/clean \
  --episode 0 --start 0 --max-actions 160 \
  --video-steps 20 --output outputs/triple_video_rollout
```

Use a new output directory for each run. Both modes export generated video and
block records; pure video exports no action curves. IDM rollout exports
generated actions and its completed video, unlike noisy-future IDM teacher
forcing. `--task all` is not accepted for rollout. An initial NPZ may be supplied
instead of `--dataset`, with its explicit source FPS through `--fps`; the video
task requires only the configured camera images and prompt, without robot state.
`run.json.evaluation_scope` identifies model-only open-loop generation from an
initial observation, with no robot feedback.

`blocks.jsonl` records `expert_calls`, synchronized `stage_seconds`, final video
sigma and output semantics. `expert_calls` counts actual expert computations,
including internal prefills, rather than just top-level operation dispatches.
Its keys remain `text`, `action_condition`, `video` and `action`. At guidance 1,
IDM teacher forcing computes text once, no action conditions, five video
forwards (four Euler plus final cache encoding), and twenty action forwards.
IDM rollout additionally runs the configured video-completion steps (default
20), for 25 video forwards per block. Pure video computes text once, twenty
video forwards and no action forwards; FDM additionally encodes enabled clean
action/state conditions once.

Stage timings exclude VAE encode/decode. No-gradient staged forwards return
`prefill_seconds` for newly built text and action-condition caches. The sampler
keeps their synchronized `text_prefill` / `action_condition_prefill` timings and
subtracts them from the enclosing video/action phase, so each interval is
counted once. Cached forwards do not repeat the prefill. Training with `full`
does not add these profiling synchronizations. State enters only the action
stage in IDM; wall-clock gains still depend on video token count, prefill cost,
cache memory, device kernels and sharding.

For teacher forcing, `run.json.profile` separates the once-per-pack VAE encode
from per-chunk work:

| Profile field | Scope |
| --- | --- |
| `vae_encode_pack_seconds`, `vae_encode_calls` | One encode of the entire measured pack, shared by its chunks; never charged as a new encode per chunk. |
| `vae_encode_memory` | Independent allocator peak snapshot for that one pack encode, taken before any solver resets its counters. |
| `chunks[*].solver_seconds` | Full sampler duration, including its text/condition/video/action stages and input assembly. |
| `chunks[*].vae_decode_seconds` | The existing generated-video decode for that chunk; `null` for IDM. |
| `chunks[*].chunk_seconds` | Solver, physical-action conversion and that chunk's video decode. |
| `vae_reconstruction_decode_pack_seconds` | The separate, existing GT VAE reconstruction decode of the whole pack; `null` for IDM. |
| `pipeline_seconds` | Teacher-forcing execution including pack encode, chunks and reconstruction; excludes checkpoint loading, dataset reading and artifact export. |
| `chunks[*].memory` / `chunk_memory` | Allocator peak allocated/reserved bytes during the solver, and from solver start through chunk end (including decode when present). |

Device operations synchronize before/after timed regions. CPU memory entries are
`null`; no accelerator-memory value is fabricated. With the default
`sampling.profile_memory: true`, the pack encode and each triple solver reset
that device's peak allocator counters. Set it to `false` to disable these queries/resets when the
caller owns memory profiling. Peaks include resident model/input allocations and
other allocations in the same process, rather than reporting incremental expert
memory. Profiling reuses the actual encode/decode calls; it adds no VAE work and
does not decode teacher-forced IDM's noisy future. Rollout records the initial
observation encode, per-block sampling stages including video completion,
output decoding and total execution; generated history does not add VAE encode
calls. Its `vae_encode_scope` is `initial_observation_once`; the compatibility
field `vae_encode_pack_seconds` measures that initial encode, not a replay of
measured future frames. Each rollout chunk records `partial_video`,
`video_cache_prefill`, `action_denoise` and `video_completion` as applicable.
Completed IDM output has `video_sigma: 0` and `video_is_clean: true`, while
`action_condition_video_sigma` retains the stop sigma used by the action cache.
The history/context counts and feedback description identify generated latent
feedback. Accelerator timings/peaks require a real
GPU/NPU run; CPU checks establish reporting and call-count correctness only.

`--task video` uses a video-only read plan and never loads action statistics or
reads action/state columns. It uses the training VideoDataset convention:
source FPS above 16 selects stride 2, otherwise stride 1, with padding aligned
to the configured latent block. Here `--max-actions` retains the legacy CLI name
but counts source-frame intervals. Native video-only samples can also call
`sample_block` without action fields. Standalone pipeline callers can set
`config["task"] = "video"` before loading to omit the action processor.

## Legacy conditioning modes

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
`rollout(dataset=... | input_npz=..., horizon=..., task=..., ...)`. It requires the final
trained config; action tasks require shared 55D action/state interfaces, while
the triple video task accepts a video-only checkpoint. Missing trained
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
KV caching, graph compilation, EMA, multi-device inference
and environment control are outside this implementation.
