# SPDX-License-Identifier: OpenMDW-1.1
"""Causal Cosmos3 input preparation configuration."""

from transformers import PretrainedConfig


class CausalCosmos3ConditionConfig(PretrainedConfig):
    model_type = "causal_cosmos3_condition"

    def __init__(
        self,
        vae_path=None,
        seed=42,
        train_time_distribution="waver",
        shift=None,
        payload_dtype="bfloat16",
        patch_size=2,
        base_fps=24.0,
        temporal_compression=4,
        temporal_modality_margin=15000,
        attention_backend="dense",
        **kwargs,
    ):
        super().__init__(**kwargs)
        payload_dtype = str(payload_dtype).removeprefix("torch.")
        if payload_dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("payload_dtype must be float32, bfloat16 or float16")
        self.vae_path = str(vae_path) if vae_path is not None else None
        self.seed = seed
        self.train_time_distribution = train_time_distribution
        self.shift = {"256": 3, "480": 5} if shift is None else shift
        self.payload_dtype = payload_dtype
        self.patch_size = patch_size
        self.base_fps = base_fps
        self.temporal_compression = temporal_compression
        self.temporal_modality_margin = temporal_modality_margin
        self.attention_backend = attention_backend
