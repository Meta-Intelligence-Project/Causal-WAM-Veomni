"""Single-NPU video/action teacher forcing or causal pixel-feedback rollout."""

import argparse
import copy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Defaults to the matching action/video inference YAML")
    parser.add_argument("--checkpoint", required=True, help="Complete native/HF trained checkpoint directory")
    parser.add_argument("--stage", choices=("action", "video"), help="Defaults to config stage, or action")
    parser.add_argument("--mode", choices=("teacher_forcing", "rollout"), required=True)
    parser.add_argument("--task", choices=("policy", "forward_dynamics", "inverse_dynamics", "all", "video"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset")
    inputs.add_argument("--input-npz")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-actions", type=int, help="Action horizon (default: 32)")
    parser.add_argument("--num-frames", type=int, help="Video output frame count, including the initial frame")
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
    config_path = args.config or (
        "configs/infer/causal_cosmos3_video.yaml" if args.stage == "video" else "configs/infer/causal_cosmos3.yaml"
    )
    config = yaml.safe_load(Path(config_path).read_text())
    stage = args.stage or config.get("stage", "action")
    if stage not in ("action", "video"):
        parser.error("stage must be action or video")
    config["stage"] = stage
    task = args.task or ("video" if stage == "video" else "policy")
    if (stage == "video" and task != "video") or (stage == "action" and task == "video"):
        parser.error("Video stage requires task=video; action stage requires an action task")
    if args.num_frames is not None and stage != "video":
        parser.error("--num-frames requires --stage video")
    if args.max_actions is not None and stage == "video":
        parser.error("--max-actions requires --stage action; use --num-frames for video")
    horizon = (
        (args.num_frames if args.num_frames is not None else 65)
        if stage == "video"
        else (args.max_actions if args.max_actions is not None else 32)
    )
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
    if horizon < 1:
        parser.error("Requested action/frame count must be positive")
    if args.mode == "rollout" and task not in ("policy", "video"):
        parser.error("Initial-observation-only rollout supports policy or video")
    if args.mode == "teacher_forcing" and args.input_npz:
        parser.error("Teacher forcing requires a dataset with measured future observations")
    tasks = ("forward_dynamics", "inverse_dynamics", "policy") if task == "all" else (task,)
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
                args.dataset, args.episode, args.start, horizon, task=task, prompt=args.prompt
            )
        else:
            result = pipe.rollout(
                dataset=args.dataset,
                input_npz=args.input_npz,
                episode=args.episode,
                start=args.start,
                horizon=horizon,
                fps=config.get("fps"),
                prompt=args.prompt,
            )
            identity.update(pipe.last_input_identity)
        effective = copy.deepcopy(pipe.config)
        effective.update(mode=args.mode, task=task, requested_horizon=horizon, prompt=args.prompt)
        save_result(
            result,
            output,
            config=effective,
            checkpoint=args.checkpoint,
            identity=identity,
            loading_report=pipe.loading_report,
        )
        summary = f"{result.video.shape[1]} frames"
        if result.actions is not None:
            summary = f"{len(result.actions)} actions, " + summary
        print(f"Saved {result.semantics}: {summary} -> {output}", flush=True)


if __name__ == "__main__":
    main()
