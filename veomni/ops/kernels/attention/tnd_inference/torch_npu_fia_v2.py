"""Ascend FIA v2 adapter for dense-within-group, no-grad TND attention."""

import torch

from . import eager


def wrapper(query, key, value, *, query_ends, kv_ends, scale):
    """Run FIA v2 with native GQA, unchanged ordered groups and no LSE output.

    This adapter accepts contiguous BF16/FP16 inputs with head_dim=128. It does
    not silently pad, transpose, copy, quantize, change masks or select a fallback.
    """
    eager.validate(query, key, value, query_ends, kv_ends, scale)
    if query.dtype not in (torch.float16, torch.bfloat16) or query.shape[-1] != 128:
        raise ValueError("FIA v2 TND adapter requires FP16/BF16 and head_dim=128")
    if not all(t.is_contiguous() for t in (query, key, value)):
        raise ValueError("FIA v2 requires contiguous Q/K/V; prepare layout outside decoder layers")
    import torch_npu

    api = getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)
    if not callable(api):
        raise RuntimeError(
            "torch_npu lacks npu_fused_infer_attention_score_v2; install a compatible torch_npu/CANN pair"
        )
    output, _ = api(
        query,
        key,
        value,
        num_query_heads=query.shape[1],
        num_key_value_heads=key.shape[1],
        softmax_scale=scale,
        input_layout="TND",
        actual_seq_qlen=list(query_ends),
        actual_seq_kvlen=list(kv_ends),
        sparse_mode=0,
        return_softmax_lse=False,
    )
    return output
