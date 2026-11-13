# SPDX-License-Identifier: OpenMDW-1.1
"""Causal Cosmos3 input preparation configuration."""

import math

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
        stage="action",
        expert_layout="legacy",
        video_stop_sigma=0.9,
        partial_video_steps=4,
        rollout_cache_max_probability=0.5,
        rollout_cache_warmup_fraction=0.2,
        rollout_cache_ramp_fraction=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if stage not in ("action", "video"):
            raise ValueError("stage must be action or video")
        self.stage = stage
        if expert_layout == "dual":
            expert_layout = "legacy"
        if expert_layout not in ("legacy", "triple"):
            raise ValueError("expert_layout must be dual, legacy or triple")
        if not 0 < video_stop_sigma < 1:
            raise ValueError("video_stop_sigma must lie strictly between zero and one")
        self.expert_layout = expert_layout
        self.video_stop_sigma = float(video_stop_sigma)
        if type(partial_video_steps) is not int or partial_video_steps < 1:
            raise ValueError("partial_video_steps must be a positive integer")
        fractions = (rollout_cache_max_probability, rollout_cache_warmup_fraction, rollout_cache_ramp_fraction)
        if not all(math.isfinite(value) for value in fractions):
            raise ValueError("Rollout-cache probabilities and fractions must be finite")
        if not 0 <= rollout_cache_max_probability <= 1:
            raise ValueError("rollout_cache_max_probability must lie between zero and one")
        if not 0 <= rollout_cache_warmup_fraction < 1 or not 0 < rollout_cache_ramp_fraction <= 1:
            raise ValueError("Rollout-cache warmup must be in [0,1) and ramp must be in (0,1]")
        if rollout_cache_warmup_fraction + rollout_cache_ramp_fraction > 1:
            raise ValueError("Rollout-cache warmup and ramp fractions must sum to at most one")
        self.partial_video_steps = partial_video_steps
        self.rollout_cache_max_probability = float(rollout_cache_max_probability)
        self.rollout_cache_warmup_fraction = float(rollout_cache_warmup_fraction)
        self.rollout_cache_ramp_fraction = float(rollout_cache_ramp_fraction)
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
