"""Live schema validation using a native tiny network and a streaming VAE double."""

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.causal_cosmos3_serving.test_window_streaming import Tokenizer
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import Cosmos3Model
from veomni.models.diffusers.causal_cosmos3.serving.robotwin import RobotwinAdapter
from veomni.models.diffusers.causal_cosmos3.serving.serving_model import ServingModelPolicy, prepare_camera_observation
from veomni.models.diffusers.causal_cosmos3.serving.statistics import BlockStatistics, Quantiles


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def policy(*, enable_rtc=True, **kwargs):
    template = ActionTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[:6] = mask[7:13] = mask[28:30] = True
    stats = BlockStatistics(
        Quantiles(-torch.ones(55), torch.ones(55), mask),
        Quantiles(-torch.ones(55), torch.ones(55), mask),
        dict(geometry=dict(actions_per_block=32, video_stride=4, temporal_compression_factor=4)),
    )
    adapter = RobotwinAdapter(template, state_mask=mask, action_mask=mask)
    model = Cosmos3Model(tiny_config(latent_channels=1, action_dim=55, state_dim=55)).eval().requires_grad_(False)
    return ServingModelPolicy(
        model,
        vae=Tokenizer(),
        statistics=stats,
        adapter=adapter,
        tokenize=lambda _: [1, 2],
        enable_rtc=enable_rtc,
        num_steps=2,
        **kwargs,
    )


@pytest.mark.parametrize("resolution,shape", [("256", (3, 384, 320)), ("480", (3, 720, 640))])
def test_online_camera_composition_matches_native_training_geometry(resolution, shape):
    cameras = {
        key: np.full((480, 640, 3), value, dtype=np.uint8)
        for key, value in (("cam_high", 10), ("cam_left_wrist", 20), ("cam_right_wrist", 30))
    }
    image, size = prepare_camera_observation(cameras, resolution)
    assert image.shape == shape
    assert size.tolist() == [shape[1], shape[2], shape[1], shape[2]]
    assert image[:, 0, 0].tolist() == [10] * 3
    assert image[:, -1, 0].tolist() == [20] * 3
    assert image[:, -1, -1].tolist() == [30] * 3


@pytest.mark.parametrize(
    "options",
    [
        dict(streaming_vae=False),
        dict(history_padding="available"),
        dict(history_steps=17),
        dict(history_steps=32, history_blocks=1),
        dict(action_horizon=17),
        dict(controller_hz=0),
        dict(guidance=float("nan")),
        dict(execute_steps=33),
        dict(handoff_steps=33),
        dict(npu_replay_graph=1),
        dict(npu_replay_graph=True),
        dict(npu_replay_graph=True, enable_rtc=False),
    ],
)
def test_online_policy_rejects_inconsistent_configuration(options):
    with pytest.raises(ValueError):
        policy(**options)


def test_graph_and_rtc_cli_conflict_is_rejected_before_loading_weights():
    from veomni.models.diffusers.causal_cosmos3.serving.server import parser
    from veomni.models.diffusers.causal_cosmos3.serving.serving_model import load_policy

    args = parser().parse_args(
        [
            "--checkpoint",
            "unused",
            "--processor",
            "unused",
            "--vae",
            "unused",
            "--statistics",
            "unused",
            "--npu-replay-graph",
            "--enable-rtc",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_policy(args)
