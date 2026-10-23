"""Serializable structure only; task geometry and attention backend are inputs."""

from transformers import PretrainedConfig


class Cosmos3Config(PretrainedConfig):
    model_type = "cosmos3"

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
        **kwargs,
    ):
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

    @property
    def patch_dim(self):
        return self.latent_channels * self.patch_size**2
