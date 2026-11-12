"""Reference and metadata validation for no-grad packed group attention."""

import math

import torch
from torch.nn import functional as F


def validate(query, key, value, query_ends, kv_ends, scale):
    """Validate host cumulative ends and homogeneous TND tensors without device reads."""
    if torch.is_grad_enabled():
        raise RuntimeError("TND inference attention requires disabled gradients")
    if any(tensor.ndim != 3 for tensor in (query, key, value)):
        raise ValueError("TND attention inputs must be [tokens, heads, head_dim]")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("TND attention requires matching K/V shapes and Q/K head dimensions")
    if key.shape[1] < 1 or query.shape[1] < 1 or query.shape[1] % key.shape[1]:
        raise ValueError("Q head count must be a positive multiple of KV heads")
    if not query.is_floating_point() or any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
        raise ValueError("TND Q/K/V must have the same floating dtype and device")
    for ends, count in ((query_ends, len(query)), (kv_ends, len(key))):
        if not isinstance(ends, (list, tuple)) or any(type(x) is not int for x in ends):
            raise ValueError("TND cumulative sequence ends must be host integer lists/tuples")
        if not ends or ends[-1] != count or any(a >= b for a, b in zip((0, *ends), ends)):
            raise ValueError("TND ends must be strictly increasing and end at the tensor token count")
    if len(query_ends) != len(kv_ends):
        raise ValueError("TND Q and KV must contain the same number of groups")
    if not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")


def wrapper(query, key, value, *, query_ends, kv_ends, scale):
    """Compute independent noncausal GQA groups, returning rows in packed Q order."""
    validate(query, key, value, query_ends, kv_ends, scale)
    outputs = []
    qs = ks = 0
    repeats = query.shape[1] // key.shape[1]
    for qe, ke in zip(query_ends, kv_ends, strict=True):
        q = query[qs:qe].transpose(0, 1)
        k = key[ks:ke].repeat_interleave(repeats, dim=1).transpose(0, 1)
        v = value[ks:ke].repeat_interleave(repeats, dim=1).transpose(0, 1)
        outputs.append(
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale).transpose(0, 1)
        )
        qs, ks = qe, ke
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
