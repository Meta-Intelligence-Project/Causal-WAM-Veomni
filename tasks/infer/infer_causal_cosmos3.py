"""Explicit single-NPU teacher forcing or causal policy pixel-feedback rollout."""

import argparse
import copy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/infer/causal_cosmos3.yaml")
    parser.add_argument("--checkpoint", required=True, help="Complete native/HF trained checkpoint directory")
    parser.add_argument("--mode", choices=("teacher_forcing", "rollout"), required=True)
    parser.add_argument("--task", choices=("policy", "forward_dynamics", "inverse_dynamics", "all"), default="policy")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset")
    inputs.add_argument("--input-npz")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--prompt")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--attention-backend", choices=("dense", "grouped", "npu_tnd"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--history", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--guidance-interval", nargs=2, type=float, metavar=("LO", "HI"))
    parser.add_argument("--normalize-cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-step-sigmas", nargs="+", type=float, metavar="SIGMA")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ("device", "dtype", "fps", "attention_backend"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    for name in ("steps", "history", "seed", "guidance", "guidance_interval", "normalize_cfg"):
        value = getattr(args, name)
        if value is not None:
            config.setdefault("sampling", {})[name] = value
    if args.fixed_step_sigmas is not None:
        config.setdefault("sampling", {})["fixed_step_sampler_config"] = dict(
            t_list=args.fixed_step_sigmas, sample_type="sde"
        )
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.mode == "rollout" and args.task != "policy":
        parser.error("Initial-observation-only rollout supports policy")
    if args.mode == "teacher_forcing" and args.input_npz:
        parser.error("Teacher forcing requires a dataset with measured future states/actions")
    tasks = ("forward_dynamics", "inverse_dynamics", "policy") if args.task == "all" else (args.task,)
    outputs = [Path(args.output) / task if len(tasks) > 1 else Path(args.output) for task in tasks]
    for output in outputs:
        if (output / "run.json").exists():
            parser.error(f"Inference output already exists: {output}")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(config["device"])
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
