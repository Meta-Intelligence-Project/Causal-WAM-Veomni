"""Native Cosmos3 networks; condition models and data pipelines are separate."""

from ...registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("cosmos3")
def register_config():
    from .configuration_causal_cosmos3 import Cosmos3Config

    return Cosmos3Config


@MODELING_REGISTRY.register("cosmos3")
def register_model(architecture):
    from .modeling_causal_cosmos3 import Cosmos3Model

    return Cosmos3Model


@MODEL_CONFIG_REGISTRY.register("causal_cosmos3_condition")
def register_condition_config():
    from .configuration_causal_cosmos3 import CausalCosmos3ConditionConfig

    return CausalCosmos3ConditionConfig


@MODELING_REGISTRY.register("causal_cosmos3_condition")
def register_condition_model(architecture=None):
    from .training import CausalCosmos3ConditionModel

    return CausalCosmos3ConditionModel
