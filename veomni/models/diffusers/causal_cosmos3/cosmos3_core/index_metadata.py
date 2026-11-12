"""Host mirrors for immutable integer indexes used by inference planners."""

from functools import partial

import numpy as np
import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference-mode tensors do not expose a mutation version counter.
        return None


def index_tensor(values, *, device=None, reference=None):
    """Upload host indexes once and retain their original ordered values."""
    values = tuple(values)
    # Convert host integers in bulk, avoiding per-scalar Tensor construction.
    host = torch.from_numpy(np.asarray(values, dtype=np.int64))
    tensor = host.to(device=device, dtype=torch.long) if reference is None else host.to(reference)
    version = _version(tensor)
    if version is not None:
        tensor._cosmos_host_indexes = (version, values)
    return tensor


def index_factory(reference):
    return partial(index_tensor, reference=reference)


def index_values(tensor):
    """Avoid device readback; invalidate mirrors after in-place mutations.

    External and inference-mode tensors retain the original tolist behavior.
    Mirrors belong to individual tensors, not a process-global cache.
    """
    mirror = getattr(tensor, "_cosmos_host_indexes", None)
    if mirror is not None and mirror[0] == _version(tensor):
        return mirror[1]
    return tuple(tensor.tolist())
