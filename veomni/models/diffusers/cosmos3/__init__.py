"""Native Cosmos3 networks; condition models and data pipelines are separate."""

from ...registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("cosmos3")
def register_config():
    from .configuration_cosmos3 import Cosmos3Config

    return Cosmos3Config


@MODELING_REGISTRY.register("cosmos3")
def register_model(architecture):
    from .modeling_cosmos3 import Cosmos3Model

    return Cosmos3Model
