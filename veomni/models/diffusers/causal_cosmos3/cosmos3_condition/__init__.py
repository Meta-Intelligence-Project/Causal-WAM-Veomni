"""Register the causal Cosmos3 condition config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("causal_cosmos3_condition")
def register_condition_config():
    from .configuration_cosmos3_condition import CausalCosmos3ConditionConfig

    return CausalCosmos3ConditionConfig


@MODELING_REGISTRY.register("causal_cosmos3_condition")
def register_condition_model(architecture=None):
    from .modeling_cosmos3_condition import CausalCosmos3ConditionModel

    return CausalCosmos3ConditionModel
