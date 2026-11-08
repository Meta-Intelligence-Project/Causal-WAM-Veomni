# SPDX-License-Identifier: OpenMDW-1.1
"""Native Cosmos3 network configuration."""

from transformers import PretrainedConfig


TBPTT_STEPS_PER_LATENT = 16
_LEGACY_TBPTT_LENGTHS = (
    "tbptt_keep_history_steps",
    "tbptt_keep_history_latents",
    "tbptt_video_chunk_latents",
    "tbptt_history_latents",
)


def _reject_legacy_tbptt_override(name):
    raise ValueError(f"{name} was removed; use tbptt_segment_steps (steps, a positive multiple of 16)")


class Cosmos3Config(PretrainedConfig):
    model_type = "cosmos3"
    condition_model_type = "causal_cosmos3_condition"

    def __init__(
        self,
        variant="edge",
        vocab_size=131072,
        hidden_size=2048,
        intermediate_size=9216,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="relu2",
        attention_bias=False,
        rms_norm_eps=1e-5,
        qk_norm_for_text=False,
        use_und_k_norm_for_gen=True,
        norm_scale_in_fp32=True,
        rope_theta=100000000.0,
        mrope_section=None,
        latent_channels=48,
        patch_size=2,
        timestep_scale=0.001,
        action_interface="domain",
        action_dim=64,
        state_dim=0,
        num_embodiment_domains=32,
        initializer_range=0.02,
        video_loss_weight=10.0,
        action_loss_weight=10.0,
        weight_loading=None,
        train_generation_only=False,
        enable_tbptt=False,
        tbptt_segment_steps=256,
        history_encoding="window",
        training_sampling_unit="independent_window",
        fixed_step_sampler_config=None,
        **kwargs,
    ):
        for name in _LEGACY_TBPTT_LENGTHS:
            if name in kwargs:
                _reject_legacy_tbptt_override(name)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("architectures", ["Cosmos3Model"])
        super().__init__(**kwargs)
        self.variant = variant
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm_for_text = qk_norm_for_text
        self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
        self.norm_scale_in_fp32 = norm_scale_in_fp32
        self.rope_theta = rope_theta
        self.mrope_section = list(mrope_section if mrope_section is not None else [24, 20, 20])
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.timestep_scale = timestep_scale
        self.action_interface = action_interface
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_embodiment_domains = num_embodiment_domains
        self.initializer_range = initializer_range
        self.video_loss_weight = video_loss_weight
        self.action_loss_weight = action_loss_weight
        self.train_generation_only = train_generation_only
        self.enable_tbptt = enable_tbptt
        self.tbptt_segment_steps = tbptt_segment_steps
        self.history_encoding = history_encoding
        self.training_sampling_unit = training_sampling_unit
        self.validate_tbptt()
        self.fixed_step_sampler_config = fixed_step_sampler_config
        # 本次加载的筛选与补初始化规则，由 _WeightAudit 消费。
        self.weight_loading = {} if weight_loading is None else weight_loading
        if (
            min(
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                latent_channels,
                patch_size,
                vocab_size,
            )
            <= 0
        ):
            raise ValueError("Network dimensions must be positive")
        if num_attention_heads % num_key_value_heads or head_dim % 2:
            raise ValueError("GQA requires divisible head counts and an even head_dim")
        if len(self.mrope_section) != 3 or sum(self.mrope_section) != head_dim // 2:
            raise ValueError("mrope_section must contain three sections summing to head_dim / 2")
        if hidden_act not in ("relu2", "silu") or action_interface not in ("none", "domain", "shared"):
            raise ValueError("Supported activations: relu2/silu; action interfaces: none/domain/shared")
        if state_dim and action_interface != "shared":
            raise ValueError("Causal state projections require the shared action interface")
        if use_und_k_norm_for_gen and qk_norm_for_text:
            raise ValueError("Separate UND K normalization is only used with unnormalized text Q/K")

    def validate_tbptt(self):
        """Validate serialized settings and later HuggingFace keyword overrides."""
        if self.history_encoding not in ("window", "episode_stream"):
            raise ValueError("history_encoding must be window or episode_stream")
        if self.training_sampling_unit not in ("independent_window", "episode_lane_fifo"):
            raise ValueError("training_sampling_unit must be independent_window or episode_lane_fifo")
        for name in _LEGACY_TBPTT_LENGTHS:
            if hasattr(self, name):
                _reject_legacy_tbptt_override(name)
        if type(self.enable_tbptt) is not bool:
            raise ValueError("enable_tbptt must be a bool")
        if type(self.tbptt_segment_steps) is not int or self.tbptt_segment_steps < 1:
            raise ValueError("tbptt_segment_steps must be a positive integer")
        if self.tbptt_segment_steps % TBPTT_STEPS_PER_LATENT:
            raise ValueError("tbptt_segment_steps must be a multiple of 16")

    @property
    def tbptt_segment_latents(self):
        """Internal segment span; public steps use 16 equivalent ticks per latent."""
        self.validate_tbptt()
        return self.tbptt_segment_steps // TBPTT_STEPS_PER_LATENT

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        for name in _LEGACY_TBPTT_LENGTHS:
            if name in config_dict or name in kwargs:
                _reject_legacy_tbptt_override(name)
        result = super().from_dict(config_dict, **kwargs)
        # PretrainedConfig applies recognized kwargs after constructing config.
        config = result[0] if isinstance(result, tuple) else result
        config.validate_tbptt()
        return result

    def to_dict(self):
        # 导出的模型不应继承首次训练的跳过或缺失参数范围限制。
        self.validate_tbptt()
        config = super().to_dict()
        config.pop("weight_loading", None)
        return config

    @property
    def patch_dim(self):
        return self.latent_channels * self.patch_size**2
