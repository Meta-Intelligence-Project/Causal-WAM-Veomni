"""Action-private capacity and cross-expert gradient isolation."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config, ops_config_scope
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.models.diffusers.causal_cosmos3.cosmos3_core.action_expert import Cosmos3ActionExpert


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def action_config():
    return tiny_config(action_hidden_size=16, action_intermediate_size=32, action_num_hidden_layers=1)


def test_action_layer_detaches_sources_but_trains_its_private_bridge(action_config):
    torch.manual_seed(11)
    with ops_config_scope(eager_ops_config()):
        expert = Cosmos3ActionExpert(action_config)
    layer = expert.layers[0]
    hidden = torch.randn(2, 16, requires_grad=True)
    text_key = torch.randn(2, 2, 8, requires_grad=True)
    text_value = torch.randn(2, 2, 8, requires_grad=True)
    video_key = torch.randn(2, 2, 8, requires_grad=True)
    video_value = torch.randn(2, 2, 8, requires_grad=True)
    text_indexes, video_indexes, action_indexes = (torch.tensor(ids) for ids in ([0, 1], [2, 3], [4, 5]))
    plan = SimpleNamespace(
        backend="dense",
        gen_indexes=action_indexes,
        modality_indexes=(text_indexes, video_indexes, action_indexes),
        mask=torch.ones(2, 6, dtype=torch.bool),
    )
    tables = (torch.ones(2, 8), torch.zeros(2, 8))
    output, own_key, own_value = layer(
        hidden,
        tables,
        plan,
        action_indexes,
        6,
        (text_key, text_value),
        tables,
        video_kv=(video_indexes, video_key, video_value),
    )
    output.square().mean().backward()
    assert own_key.requires_grad and own_value.requires_grad
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert all(source.grad is None for source in (text_key, text_value, video_key, video_value))
    assert layer.text_k_norm.weight.grad is not None and layer.text_k_norm.weight.grad.abs().sum() > 0
    assert layer.to_k.weight.grad is not None and layer.to_k.weight.grad.abs().sum() > 0
    assert layer.mlp.up_proj.weight.grad is not None and layer.mlp.up_proj.weight.grad.abs().sum() > 0


def test_action_expert_has_private_width_and_initialized_embeddings(action_config):
    with ops_config_scope(eager_ops_config()):
        expert = Cosmos3ActionExpert(action_config)
        other = Cosmos3ActionExpert(action_config)
        with torch.device("meta"):
            meta = Cosmos3ActionExpert(action_config)
    assert expert.proj_in.weight.shape == (16, 4)
    assert expert.proj_out.weight.shape == (4, 16)
    assert expert.layers[0].mlp.up_proj.weight.shape == (32, 16)
    assert expert.layers[0].to_q.weight.shape == (32, 16)
    assert expert.modality_embed.isfinite().all() and expert.modality_embed.abs().sum() > 0
    assert expert.state_modality_embed.isfinite().all() and expert.state_modality_embed.abs().sum() > 0
    assert meta.modality_embed.is_meta and meta.layers[0].to_q.weight.is_meta
    assert not {id(p) for p in expert.parameters()}.intersection(id(p) for p in other.parameters())
