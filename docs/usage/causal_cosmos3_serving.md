# Streaming causal Cosmos3 service

The native online path ports streaming VAE, asynchronous execution and RTC from
Cosmos `gy_branch@9d448c7`. It loads a complete native/HF checkpoint directly and
does not construct a Trainer or import `cosmos_framework`.

## Run

Install the optional transport dependencies in the appropriate environment:

```bash
uv sync --extra npu --extra serving --dev
# An already provisioned conda environment can install only the transport:
python -m pip install 'msgpack>=1.1.1' 'websockets>=15.0,<18.0'
```

From the repository root, run a single model process:

```bash
python -m veomni.models.diffusers.causal_cosmos3.serving.server \
  --checkpoint /path/to/complete_native_hf_checkpoint \
  --processor /path/to/Cosmos3-Edge \
  --vae /path/to/Wan2.2-TI2V-5B-Diffusers/vae \
  --statistics /path/to/measured_causal_action_stats.json \
  --robot-config /path/to/robot_config.json \
  --device npu:0 --attention-backend npu_tnd \
  --history-steps 1024 --action-horizon 32 \
  --execute-steps 16 --handoff-steps 24 \
  --steps 10 --guidance 1 --enable-rtc \
  --host 0.0.0.0 --port 8000
```

The equivalent script is
[`tasks/infer/serve_causal_cosmos3.py`](../../tasks/infer/serve_causal_cosmos3.py).
Use the Diffusers VAE directory already supported by native training. Convert a
legacy Cosmos DCP first using the existing
[checkpoint conversion instructions](causal_cosmos3_inference.md#load-trained-weights).
The loader requires the complete shared 55D action/state configuration.

[`causal_cosmos3_serving.yaml`](../../configs/infer/causal_cosmos3_serving.yaml)
selects training-compatible norm/RoPE operations and optional sampling settings.
For CPU functional checks use `--device cpu --dtype float32
--attention-backend grouped`; the loader selects eager CPU operations.
CPU behavior does not establish BF16/NPU numerical equivalence.

The robot configuration declares exact `state_mask` and `action_mask`, the arm
DOFs and, when EEF fields are active, `eef_pose_frame` and its required transform.
When statistics contain one unambiguous source mask and coordinate convention,
`--robot-config` can be omitted. Physical units must match training statistics.
The native client composes camera images using native training's camera rectangles
and resolution buckets. It does not substitute Cosmos's different resize buckets.

RoboTwin client:

```bash
python -m veomni.models.diffusers.causal_cosmos3.serving.robotwin_client \
  --robotwin-root /path/to/RoboTwin --server-uri ws://SERVER_IP:8000 \
  --task-name adjust_bottle --task-config demo_clean \
  --control-mode joint --state-source measured \
  --initial-execution-hz 30 --servo-hz 30 --enable-rtc \
  --max-waypoints 48 --output outputs/causal_cosmos3/robotwin
```

The client needs a CPU-capable VeOmni environment and the transport extra; model
weights and the VAE stay on the server. Its RoboTwin checkout must provide
`scripts/eval_policy_xpolicylab.py` and the corresponding task helpers.

## Execution contract

`H` is the number of predicted actions, `S` the trigger tick relative to the old
chunk, and `R` its fixed handoff tick. Defaults are H32/S16/R24. The same
[`AsyncRemoteSession`](../../veomni/models/diffusers/causal_cosmos3/serving/async_session.py#L64)
owns ordinary and RTC execution. At S it freezes a fresh measured image/state,
submits inference in the background and continues delivering old commands.
An early result waits for R. A late result pauses at R without advancing the
controller tick. At handoff it starts at new prediction row `R-S`.

RTC is opt-in on both client and server. It adds the old unexecuted suffix as
an external physical action target. With H32/S16/R24, the first eight overlap
rows are committed and the next eight receive decaying guidance. The sampler
rebases targets on the fresh measured anchor, normalizes without clipping
quantile tails, computes action-error VJPs over joint video/action inputs, and
restores committed normalized actions after sampling. Physical export then
restores the exact original committed commands, including quaternion values.
Model parameter gradients are not accumulated. Use `torch.no_grad()` rather
than an enclosing `torch.inference_mode()` for guided inference.

Startup `prepare()` calibrates complete RPCs while the robot remains stationary;
calibration predictions are discarded. Adaptive pacing changes the real execution
rate only after the servo adapter acknowledges it. The model FPS and S/R indexes
remain fixed. `--no-adaptive-execution` selects fixed-rate execution;
`--execution-mode blocking` explicitly selects synchronous delivery without RTC.

## Window and cache ownership

`history_steps=P` counts observed action-time intervals independently of H.
P and H must be divisible by 16 for the stride4/VAE4 recipe; P48/H32 and P0/H16
are valid. `--history` is the legacy alias counting 32-action units.
The window samples `tick-P .. tick` at video stride four. Negative logical ticks
repeat the actual reset image. Missing nonnegative observations cause an error.
Future video latents and action placeholders are zero; future ground truth and
predicted anchors are never used as measurements.

The decision layout contains one measured visual prefix, H future actions and
one current state. Historical actions/states and a second clean-video copy are
absent. Each observed latent is its own causal history group; all future
video/action tokens form one current decision group. State attends only to text
and itself. This optional layout leaves existing teacher-forcing layouts and
offline inference defaults in place.

For new policy training, the optional
[decision-window training recipe](causal_cosmos3_training.md#optional-training-aligned-with-online-decision-windows)
uses the same current anchor, clean history and joint future layout. Match the
training P/H and preprocessing when selecting serving settings. Streaming VAE
does not by itself require asynchronous training or an RTC training loss.

Each VAE state belongs to one finite decision window. Reanchoring disposes of
the old state, preencodes available real history and then appends complete
four-frame groups as observations arrive. A completed prefix is supplied
directly to native sampling. The VAE does not encode synthetic future pixels.

`cache_mode=off` runs the native reference. `plans` reuses attention indexes;
`full` additionally prefills fixed per-layer K/V once, then computes only future
tokens. Ordinary inference uses the prefill output for its first velocity and
updates owned K/V buffers in place: ten steps require one prefill and nine compact
replays. RTC uses functional K/V updates and a fresh autograd graph on every
evaluation, including the first: ten steps require one prefill and ten compact
replays. CFG branches and requests own separate entries. Model weights must be
frozen in eval mode. Full-prefill and compact first evaluations can differ slightly
in BF16 because their matrix shapes differ; the reused first output matches the
uncached full-model first evaluation.
UniPC and matching fixed-step SDE schedules are supported; distilled schedules
resolve guidance to one.

## Optional Ascend replay graph

For ordinary inference, add these server flags to the launch command above:

```bash
--npu-replay-graph --no-enable-rtc --cache-mode full --guidance 1
```

Omit `--enable-rtc` on the client too. The default keeps graphs disabled. The
graph option requires an NPU, `npu_tnd`, full caching and resolved guidance one.
Graph and RTC are mutually exclusive: enabling both raises an error before
loading weights, and a graph server rejects client RTC sessions. Keep the eager
path with `--enable-rtc` when input-VJP guidance is needed.

[`NPUReplayGraphCache`](../../veomni/models/diffusers/causal_cosmos3/serving/npu_replay_graph.py#L51)
retains at most one compact-forward capture per model. Each request still runs
its own full prefill. Compatible requests refresh every fixed K/V and RoPE table;
each replay copies current video/action inputs, times and masks into owned static
buffers. Exact layout/routing, shape, dtype, device, model weight versions,
module replacement or autocast/inference/deterministic-kernel context changes
invalidate the capture.
Weights and layout must remain immutable during a request. Concurrent sampling
through the same graph cache is rejected. Reset/disconnect releases captured
state; policy close also removes the model's cache. Capture/replay failures clear
the cache and raise an error.

The sampler's `inference_cache.cond.npu_replay_graph_stats` reports captures,
replays, compatible-request refreshes, invalidations and capture duration. Measure
cold capture separately from warmed requests. Graphs accelerate compact DiT
replay; historical VAE encoding, the last VAE group and full DiT prefill remain
outside the capture. Fused attention uses its synchronous auxiliary path during
capture so all kernel events belong to the captured stream.

The service admits one client and serializes model calls in one worker outside
the WebSocket event loop. Disconnect cleanup drains accepted work before another
client can use the model. The inherited protocol is `cosmos-causal-action-rtc-v1`;
use the matching native client so image preprocessing matches native training.

## Extension interfaces and validation

- [`Cosmos3VideoVAE.stream_encode_start/step/finish/abort/reset`](../../veomni/models/diffusers/causal_cosmos3/cosmos3_core/vae.py#L105)
  owns isolated convolution state. Start accepts normalized `[B,3,1,H,W]`; step
  accepts complete temporal groups with stable geometry/device/dtype. Finish
  returns the normalized prefix and closes the state; abort is idempotent for
  its owner. Ordinary encode/decode can run between window steps.
- [`ObservationWindowEncoder`](../../veomni/models/diffusers/causal_cosmos3/serving/window_streaming.py#L9)
  owns immutable real RGB observations and a disposable VAE state. `add()` stores
  measured ticks; `prepare(anchor)` starts preencoding; `finish(anchor)` requires
  all real observations and returns the prefix while releasing convolution state.
- [`RTCObservedBatchBuilder`](../../veomni/models/diffusers/causal_cosmos3/serving/rtc_batch.py#L44)
  builds native tensor samples from a measured anchor and a prepared prefix. A
  local controller can supply `vae=` to encode the same real window on demand.
- [`ServingModelPolicy.infer`](../../veomni/models/diffusers/causal_cosmos3/serving/serving_model.py#L249)
  calls the window encoder, batch builder,
  [`sample_decision`](../../veomni/models/diffusers/causal_cosmos3/serving/sampling.py#L19)
  and physical export in that order. `observe` and `prepare_window` use the same
  serialized worker. Client-side sessions own the execution clock and action queue.

Regression coverage resides in
[`tests/models/diffusers/causal_cosmos3_serving/`](../../tests/models/diffusers/causal_cosmos3_serving).
It covers real tiny Wan streaming/interleaving/cleanup, P/H geometry, causal
visibility, cached/reference predictions and RTC VJPs, exact physical committed
prefixes, latest measured anchors, fixed handoffs, pacing acknowledgement,
reset/disconnect races and real localhost WebSocket RPCs.
CPU lifecycle tests cover buffer refresh, invalidation and failure recovery;
Ascend-only tests capture real graphs for P0/P48/P1024, compare graph/eager
predictions exactly, and check TND forward values and shared-K/V gradients.
Ascend fused-attention backward is nondeterministic by default. RTC numerical
comparisons use `torch.use_deterministic_algorithms(True)` and restore the caller's
setting; ordinary serving does not change that global setting.

```bash
pytest tests/models/diffusers/causal_cosmos3_serving/ -q
pytest tests/models/diffusers/test_causal_cosmos3{,_packing,_training,_inference,_vae}.py -q
```

These checks establish implementation contracts. They do not measure real robot
success or full-checkpoint NPU latency and numerical equivalence.
