# Cosmos3 dataset history and video augmentation

The existing Action and Video datasets share history selection while retaining
their own video geometry. These methods support data enumeration; they do not
enable video condition encoding or model-side validation loss.

## Video latent count

The concrete Dataset owns `num_video_latents(window)`:
[Action implementation](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L434),
[Video implementation](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L518).
It receives a legal Reader `Window` and returns the positive number of video
latent frames **including tail padding**, not the number of latent blocks or
supervised frames. It must use the same geometry as that Dataset's reader plan,
perform no reads, and consume no random state. Empty/illegal windows are rejected
by candidate construction or Reader planning before this method is called.

Training calls it from
[Action `make_sample`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L438)
and [Video `make_sample`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L522).
[Validation](../../../veomni/data/causal_cosmos3/datasets/validation.py#L89)
uses the same count for full history. A subclass changing its temporal geometry
must update this method with its read rules; using the wrong count changes
history and thus the conditioning seen by the model. Metadata-only subclasses
need not implement it unless they participate in sample generation/validation.

For example, the Video implementation uses its planned downsampling and padding:

```python
def num_video_latents(self, window):
    return 1 + (window.video_frames - 1) // self.temporal_compression
```

## History selection

The shared base provides
[`select_history(num_latents, *, explicit=None, rng=None)`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L285).
Both concrete datasets reuse it. Preserve its precedence and units when extending
the datasets:

1. An explicit positive Python integer wins and consumes no RNG. Invalid explicit
   values raise `ValueError`; a value larger than `num_latents` is not clipped.
2. `full_history=True` returns `num_latents` without consuming either RNG.
3. Otherwise draw from the inclusive configured `history_range`, using a supplied
   NumPy Generator or, when omitted, the Dataset's training generator. The range
   is not clipped to the window's latent count.

The returned value is stored in the legacy `history_blocks` field, whose unit is
latent frames. Training callers supply any explicit history and leave `rng`
omitted. Validation supplies an independent generator derived from window
identity, then passes the selected history explicitly to sample construction.

The validation call flow is:

1. [Plan the window](../../../veomni/data/causal_cosmos3/datasets/validation.py#L67).
2. [Derive the history seed and select history](../../../veomni/data/causal_cosmos3/datasets/validation.py#L88)
   through the Dataset's latent count and shared selector.
3. [Call `__getitem__` with explicit options](../../../veomni/data/causal_cosmos3/datasets/validation.py#L104):
   mode/history for Action, history alone for Video. Existing error handling
   converts ordinary sample failures to empty packs; memory exhaustion propagates.

This leaves initialized and uninitialized training RNG state unchanged and
makes history independent of rank, iteration order and previous validation runs.
For an existing VideoDataset, the direct opt-in is:

```python
from veomni.data.causal_cosmos3.datasets.validation import iter_validation_batches

for item in iter_validation_batches(video_dataset, rank=0, world_size=1, seed=73):
    batch, record, weight = item["batch"], item["record"], item["weight"]
```

Coverage resides in
[shared history selection](../../../tests/data/test_causal_cosmos3_pipeline.py#L491),
[Video enumeration and RNG preservation](../../../tests/data/test_causal_cosmos3_pipeline.py#L2241),
[short windows and failed reads](../../../tests/data/test_causal_cosmos3_pipeline.py#L2344),
and [failure scheduling across ranks](../../../tests/data/test_causal_cosmos3_pipeline.py#L2619).

## Video augmentation

The framework provides `build_dataset(config, tokenize, *, training=False, ...)`
in [datasets/pipeline.py](../../../veomni/data/causal_cosmos3/datasets/pipeline.py).
Training callers must pass `training=True` to opt into `config.augmentation`.
The Causal Cosmos3 trainer and `build_action_dataloader` do so; validation uses
`training=False`. The default is safe for inference and standalone inspection.
The factory passes an optional `video_transform` to each concrete Dataset.

The shared `LeRobotDataset.prepare_sample` in
[dataset.py](../../../veomni/data/causal_cosmos3/datasets/dataset.py) owns the
call after reader composition and before prompt tokenization and VAE encoding.
It calls `video_transform(video)` exactly once when the callable and video
exist. Subclasses using this hook must call `prepare_sample` once after their
task fields are populated. A direct Dataset caller may provide its own callable;
that caller is responsible for disabling it for validation/inference.

The callable accepts and returns `uint8 [3,T,H,W]`, preserving shape and camera
layout. It must not modify action/state fields or token geometry. Missing video
payloads (metadata/action-only reads) bypass the transform.
The built-in [VideoColorJitter](../../../veomni/data/causal_cosmos3/transforms/augmentation.py)
transposes to `[T,3,H,W]`, applies Torchvision ColorJitter once to the whole
composite sequence, and returns contiguous `[3,T,H,W]`. Both factors and operation
order are shared across time and views. It uses Torch's existing RNG; no dedicated
RNG state is added. `p=0` or all-zero jitter returns the input without consuming
RNG. Invalid enabled configurations raise during factory construction rather
than becoming per-sample read failures.

For example:

```python
training = build_dataset(config, tokenize, training=True)
validation = build_dataset(config, tokenize)  # augmentation disabled
```

See [training configuration](../../usage/causal_cosmos3_training.md) for YAML
fields and ranges. Tests in
[test_causal_cosmos3_pipeline.py](../../../tests/data/test_causal_cosmos3_pipeline.py)
cover shared factors/order, full-canvas contrast, unchanged non-video fields,
Torch RNG use, worker loading and trainer validation gating for both stages.
