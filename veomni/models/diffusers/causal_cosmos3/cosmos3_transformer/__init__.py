"""Register the native Cosmos3 network config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("cosmos3")
def register_config():
    from .configuration_cosmos3_transformer import Cosmos3Config

    return Cosmos3Config


@MODELING_REGISTRY.register("cosmos3")
def register_model(architecture):
    from .modeling_cosmos3_transformer import Cosmos3Model

    return Cosmos3Model
