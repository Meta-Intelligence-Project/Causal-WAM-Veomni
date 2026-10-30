# Cosmos3 dataset history for training and validation

The existing Action and Video datasets share history selection while retaining
their own video geometry. These methods support data enumeration; they do not
enable video condition encoding or model-side validation loss.

## Video latent count

The concrete Dataset owns `num_video_latents(window)`:
[Action implementation](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L417),
[Video implementation](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L501).
It receives a legal Reader `Window` and returns the positive number of video
latent frames **including tail padding**, not the number of latent blocks or
supervised frames. It must use the same geometry as that Dataset's reader plan,
perform no reads, and consume no random state. Empty/illegal windows are rejected
by candidate construction or Reader planning before this method is called.

Training calls it from
[Action `make_sample`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L438)
and [Video `make_sample`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L510).
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
[`select_history(num_latents, *, explicit=None, rng=None)`](../../../veomni/data/causal_cosmos3/datasets/dataset.py#L272).
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
[shared history selection](../../../tests/data/test_causal_cosmos3_pipeline.py#L385),
[Video enumeration and RNG preservation](../../../tests/data/test_causal_cosmos3_pipeline.py#L1930),
[short windows and failed reads](../../../tests/data/test_causal_cosmos3_pipeline.py#L2033),
and [failure scheduling across ranks](../../../tests/data/test_causal_cosmos3_pipeline.py#L2075).
