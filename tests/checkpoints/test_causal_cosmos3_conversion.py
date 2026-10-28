"""Legacy Cosmos parameter mapping for the standalone checkpoint conversion tool."""

import pytest

from scripts.causal_cosmos3.convert_inference_checkpoint import cosmos_parameter_name
from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize(
    "source,dest",
    [
        ("net.state2llm.weight", "state_proj_in.weight"),
        ("net.language_model.model.layers.0.self_attn.q_proj_moe_gen.weight", "layers.0.self_attn.add_q_proj.weight"),
        ("net.language_model.model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.to_q.weight"),
        ("net.time_embedder.mlp.2.bias", "time_embedder.linear_2.bias"),
    ],
)
def test_cosmos_mapping_is_explicit(source, dest):
    assert cosmos_parameter_name(source) == dest
