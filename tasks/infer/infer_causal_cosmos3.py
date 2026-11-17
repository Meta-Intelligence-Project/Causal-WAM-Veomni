"""Teacher forcing or Policy/IDM pixel-feedback rollout."""

import argparse
import copy
import sys
from pathlib import Path

import yaml


if not __package__:
    # Keep direct script execution working without a repository PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tasks.common.causal_cosmos3_args import add_inference_arguments, apply_cli_overrides  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/infer/causal_cosmos3_edge_action_policy.yaml")
    parser.add_argument(
        "--checkpoint", required=True, help="Complete VeOmni model/ckpt DCP or native/HF weights directory"
    )
    parser.add_argument("--mode", choices=("teacher_forcing", "rollout"), required=True)
    parser.add_argument("--task", choices=("policy", "forward_dynamics", "inverse_dynamics", "all"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset")
    inputs.add_argument("--input-npz")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--prompt")
    add_inference_arguments(parser)
    parser.add_argument("--output", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    apply_cli_overrides(config, args)
    args.task = args.task if args.task is not None else config.get("task", "policy")
    if args.task not in ("policy", "forward_dynamics", "inverse_dynamics", "all"):
        parser.error("task must be policy, forward_dynamics, inverse_dynamics or all")
    config["task"] = args.task
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.mode == "rollout" and args.task not in ("policy", "inverse_dynamics"):
        parser.error("Initial-observation-only rollout supports policy or inverse_dynamics")
    if args.mode == "teacher_forcing" and args.input_npz:
        parser.error("Teacher forcing requires a dataset with measured future states/actions")
    if config.get("sampling", {}).get("kv_cache", True) and args.task in ("forward_dynamics", "all"):
        parser.error("Forward dynamics requires --no-kv-cache (including --task all)")
    tasks = ("forward_dynamics", "inverse_dynamics", "policy") if args.task == "all" else (args.task,)
    if config.get("sampling", {}).get("npu_graph", False) and args.task != "policy":
        parser.error("NPU graph currently supports --task policy only")
    outputs = [Path(args.output) / task if len(tasks) > 1 else Path(args.output) for task in tasks]
    for output in outputs:
        if (output / "run.json").exists():
            parser.error(f"Inference output already exists: {output}")
    import torch

    device_type = config["device"].split(":")[0]
    if device_type == "npu":
        import torch_npu  # noqa: F401
    if device_type != "cpu":
        torch.get_device_module(config["device"]).set_device(config["device"])
    from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline, save_result

    pipe = CausalCosmos3Pipeline.from_pretrained(args.checkpoint, config)
    for task, output in zip(tasks, outputs, strict=True):
        identity = dict(
            dataset=args.dataset, input_npz=args.input_npz, episode=args.episode, start=args.start, task=task
        )
        if args.mode == "teacher_forcing":
            result = pipe.teacher_forcing(
                args.dataset, args.episode, args.start, args.max_actions, task=task, prompt=args.prompt
            )
        else:
            result = pipe.rollout(
                dataset=args.dataset,
                input_npz=args.input_npz,
                episode=args.episode,
                start=args.start,
                horizon=args.max_actions,
                fps=config.get("fps"),
                prompt=args.prompt,
                task=task,
            )
            identity.update(pipe.last_input_identity)
        effective = copy.deepcopy(pipe.config)
        effective.update(mode=args.mode, task=task, requested_horizon=args.max_actions, prompt=args.prompt)
        save_result(
            result,
            output,
            config=effective,
            checkpoint=args.checkpoint,
            identity=identity,
            loading_report=pipe.loading_report,
        )
        print(
            f"Saved {result.semantics}: {len(result.actions)} actions, {result.video.shape[1]} frames -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
