"""Fixed-input forward/backward and optional reference fixture export.

Run as `python -m scripts.cosmos3.validate_network --help` from the repository.
This exercises network tensors, not a training recipe or data pipeline.
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_config, build_foundation_model
from veomni.models.checkpoint.weights import load_model_weights
from veomni.models.diffusers.cosmos3.attention import CausalLayout


def make_inputs(config, task, device, dtype):
    torch.manual_seed(20261019)
    action = task != "video"
    roles = [-1, -1, 0, 0, 0, 1 if not action else 2, 2 if task == "id" else 1, 2 if task == "id" else 1]
    blocks, modalities = [-1, -1, -1, 0, 1, -1, 0, 1], [0, 0, 1, 1, 1, 1, 1, 1]
    if action:
        roles += [2 if task == "fd" else 1] * 2 + [3, 3]
        blocks += [0, 1, 0, 1]
        modalities += [2, 2, 3, 3]
    layout = CausalLayout(tuple([0] * len(roles)), tuple(blocks), tuple(roles), tuple(modalities), (2,))
    inputs = dict(
        input_ids=torch.tensor([1, 2], device=device),
        video_tokens=torch.randn(6, config.patch_dim, device=device, dtype=dtype),
        video_timesteps=torch.tensor([0, 0, 0, 250, 400, 700], device=device),
        position_ids=torch.tensor(
            [
                [0, 1] + [15000 + b * 4 for b in blocks[2:]],
                [0, 1] + [2] * (len(blocks) - 2),
                [0, 1] + [3] * (len(blocks) - 2),
            ],
            device=device,
        ),
        layout=layout,
    )
    if action:
        inputs.update(
            action_tokens=torch.randn(2, config.action_dim, device=device, dtype=dtype),
            action_timesteps=torch.tensor([400, 700], device=device),
            action_mask=torch.arange(config.action_dim, device=device)[None, :].expand(2, -1) < config.action_dim - 1,
            state_tokens=torch.randn(2, config.state_dim, device=device, dtype=dtype),
            state_mask=torch.arange(config.state_dim, device=device)[None, :].expand(2, -1) < config.state_dim - 1,
        )
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_configs/cosmos3/Cosmos3-Edge.json")
    parser.add_argument("--weights", help="Original transformer component or a complete native safetensors directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--backend", choices=["dense", "grouped", "npu_tnd"], default="dense")
    parser.add_argument("--task", choices=["video", "fd", "id", "policy"], default="video")
    parser.add_argument("--tiny", action="store_true", help="Use two small layers; not real checkpoint validation")
    parser.add_argument("--pretrained-initialization", action="store_true")
    parser.add_argument(
        "--export-reference", type=Path, help="Export a small standalone fixture for the old environment"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.export_reference and not args.tiny and (not args.weights or args.pretrained_initialization):
        parser.error(
            "Full-size fixture export requires a complete checkpoint; new randomly initialized heads need --tiny"
        )
    if args.weights and args.tiny:
        parser.error("--tiny cannot load production checkpoint shapes")
    torch.manual_seed(42)
    config = build_config(args.config)
    if args.tiny:
        config.update(
            dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=8 if config.variant == "super" else 4,
                num_key_value_heads=2,
                head_dim=8,
                mrope_section=[2, 1, 1],
                vocab_size=32,
                latent_channels=2,
            )
        )
    if args.task != "video":
        config.update(
            dict(action_interface="shared", action_dim=4 if args.tiny else 55, state_dim=4 if args.tiny else 55)
        )
    ops = OpsImplementationConfig(
        attn_implementation="eager",
        rms_norm_implementation="torch_npu" if args.device.startswith("npu") and config.variant == "edge" else "eager",
        rotary_pos_emb_implementation="torch_npu"
        if args.device.startswith("npu") and config.variant == "edge"
        else "eager",
    )
    model = build_foundation_model(config, torch_dtype=args.dtype, init_device="meta", ops_implementation=ops)
    if args.weights:
        if args.pretrained_initialization:
            model.prepare_pretrained_initialization()
        load_model_weights(model, args.weights, init_device=args.device)
    else:
        model.to_empty(device=args.device)
        model.apply(model._init_weights)
    inputs = make_inputs(config, args.task, args.device, getattr(torch, args.dtype))
    inputs["video_tokens"].requires_grad_()
    output = model(**inputs, attention_backend=args.backend, output_hidden_states=True)
    video_target = torch.tensor(
        [r == 1 for r, m in zip(inputs["layout"].roles, inputs["layout"].modalities) if m == 1], device=args.device
    )
    loss = output.video[video_target].float().square().sum()
    if output.action is not None:
        action_target = torch.tensor(
            [r == 1 for r, m in zip(inputs["layout"].roles, inputs["layout"].modalities) if m == 2], device=args.device
        )
        loss = loss + output.action[action_target].float().square().sum()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ):
        raise RuntimeError("Nonfinite loss/parameter gradients")
    clean_grad_norm = inputs["video_tokens"].grad[:3].float().norm().item()
    if clean_grad_norm == 0:
        raise RuntimeError("Clean history lost its gradient")
    report = dict(
        variant=config.variant,
        tiny=args.tiny,
        task=args.task,
        dtype=args.dtype,
        backend=args.backend,
        device=args.device,
        parameters=sum(p.numel() for p in model.parameters()),
        loss=loss.item(),
        clean_input_grad_norm=clean_grad_norm,
        loading=model.weight_loading_report,
        old_package_discoverable=importlib.util.find_spec("cosmos_framework") is not None,
        old_modules_imported=any(name.startswith("cosmos_framework") for name in sys.modules),
    )
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.export_reference:
        fixture_inputs = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else dataclasses.asdict(v) for k, v in inputs.items()
        }
        fixture = dict(
            config=config.to_dict(),
            inputs=fixture_inputs,
            weights={k: v.detach().cpu() for k, v in model.state_dict().items()} if args.tiny else None,
            checkpoint=args.weights,
            dtype=args.dtype,
            outputs={k: v.detach().cpu() for k, v in output.items()},
            gradients={
                k: p.grad.cpu()
                for k, p in model.named_parameters()
                if p.grad is not None
                and (
                    args.tiny
                    or k
                    in (
                        "proj_in.weight",
                        "proj_out.weight",
                        "time_embedder.linear_1.weight",
                        "layers.0.self_attn.add_k_proj.weight",
                        "layers.0.self_attn.to_q.weight",
                        "layers.0.self_attn.to_k.weight",
                        "layers.0.self_attn.to_v.weight",
                        "layers.0.self_attn.to_out.weight",
                    )
                )
            },
            video_input_gradient=inputs["video_tokens"].grad.cpu(),
            loss=loss.item(),
        )
        args.export_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, args.export_reference)


if __name__ == "__main__":
    main()
