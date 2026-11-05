# Cosmos3 dataset history for training and validation

Action and Video datasets share metadata, prompting, task RNG state, source
sampling and deterministic validation. Their history contracts differ because
they train different inputs. Implementations live in
[`datasets/dataset.py`](../../../veomni/data/causal_cosmos3/datasets/dataset.py);
these helpers perform no VAE encoding or model forward.

## Action: fixed observed decision history

`ActionDataset` always builds one decision at a measured anchor. Its constructor
uses `history_steps=1024` and `action_horizon=32` by default. P/H count raw action
ticks and must be VAE-group aligned. Action has no `layout_mode`, `full_history`
or `history_range` constructor arguments. The output metadata records the fixed
`layout_mode="decision_window"` for packing and serving.

Every legal anchor is a candidate at `start_stride`, including episode tails.
Candidates must use `frame_window`, `max_actions=H` and overlap zero; the source
factory derives `max_actions` from H when omitted. Subtask text is selected at the
anchor. The Reader adds measured visual history to the future `Window`; only
negative episode history repeats reset, and future tail padding is unsupervised.

`num_video_latents(window)` returns `1+(P+H)/actions_per_latent`, including padded
future latents. `select_history(num_latents, *, explicit=None, rng=None)` always
returns `1+P/actions_per_latent`, the observed latent count. A supplied explicit
history must equal that count or it raises `ValueError`; the method consumes
neither the training task RNG nor a supplied validation RNG. P0/H16 therefore has
one observed latent and one future latent; P1024/H32 has 65 observed and two future.

`make_sample(index, *, mode=None, history=None)` reads a window and selects its
FD/ID/policy task after a successful read. An explicit mode consumes no training
RNG. It attaches the one current state, normalized targets, condition indexes and
final prompt/token budget. Fixed history is exposed in the legacy output key
`history_blocks`, whose unit remains latent frames. Direct calls propagate errors;
`__getitem__` logs/skips ordinary failures and propagates memory exhaustion.

The canonical [Action source YAML](../../../configs/data/causal_cosmos3.yaml)
and [training recipe](../../../configs/dit/causal_cosmos3_edge.yaml) use this
contract directly. There is no parallel Action teacher-forcing training recipe.
Generic block transforms remain for explicit offline diagnostic/rollout readers.

## Video: generic block history

`VideoDataset` reads timestamps, task captions and pixels without action/state
schema, processors or statistics. It retains `latent_block_size` and
`full_history`/`history_range` options. Its planned stride is two above 16 FPS and
one otherwise; interval padding aligns to stride × temporal compression × block.

`num_video_latents(window)` returns the padded video latent count including the
independent first latent:

```python
return 1 + (window.video_frames - 1) // self.temporal_compression
```

The base `select_history(num_latents, *, explicit=None, rng=None)` precedence is:

1. A positive explicit Python integer wins without RNG draws. It is not clipped
   to `num_latents`; invalid explicit values raise `ValueError`.
2. `full_history=True` returns `num_latents` without RNG draws.
3. Otherwise sample the inclusive `history_range` using the supplied independent
   generator, or the dataset's training generator when none is supplied.

The default is random history in `[1,32]` latent frames. `make_sample(index,
history=...)` uses the same selection mechanism. Video supports generic
`frame_window` and `segmented` candidate ranges; this block layout is separate
from the Action decision contract.

## Shared validation and captions

[`iter_validation_batches`](../../../veomni/data/causal_cosmos3/datasets/validation.py)
plans candidates without reading payloads, derives a history seed from identity,
uses the concrete dataset's latent count and history selector, then calls
`__getitem__` with explicit history and (for Action) task mode. Action identities
include P/H/layout. Joint Action validation enumerates FD/ID/policy separately;
Video enumerates one `video` pass. Initialized and uninitialized training RNG
states remain unchanged. Repeated rank-padding windows carry weight zero, and
failed reads preserve coordinated DP rounds as empty zero-weight packs.

Shared `prepare_sample` uses the selected subtask caption or the matching
`task_index` description from `meta/tasks.parquet`. Every actual fallback emits
WARNING with source/root, episode, candidate/start/task index, reason and complete
`fallback_caption`, using `repr` to escape embedded newlines/quotes. This includes
missing/invalid subtasks, gaps and disabled subtask selection. Matched subtasks
produce no fallback warning. Missing nonempty text or an empty custom prompt is
rejected before tokenization.

Contracts are exercised in
[`test_causal_cosmos3_decision.py`](../../../tests/data/test_causal_cosmos3_decision.py)
and [`test_causal_cosmos3_pipeline.py`](../../../tests/data/test_causal_cosmos3_pipeline.py):
fixed Action contexts, task/RNG restore, Video random/full history, source
sampling/packing, episode splits, fallback descriptions, validation seeds and
rank-coordinated failure/padding behavior. VAE, model loss/backward and offline
inference are tested independently.
