"""Explicit single-NPU teacher forcing or causal policy pixel-feedback rollout."""

import argparse
import copy
import json
import time
from contextlib import nullcontext
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
    parser.add_argument(
        "--residual-reuse-full-steps",
        nargs="+",
        type=int,
        help="Refresh count (0, 3, 6, 9), or increasing refresh indices starting at 0",
    )
    parser.add_argument("--history", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--guidance-interval", nargs=2, type=float, metavar=("LO", "HI"))
    parser.add_argument("--normalize-cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-step-sigmas", nargs="+", type=float, metavar="SIGMA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true", help="Warm rollout timing and allocator peaks")
    parser.add_argument(
        "--profile-warmup-runs", type=int, default=1, help="Full unmeasured rollouts before measurement"
    )
    parser.add_argument(
        "--profile-npu-trace", action="store_true", help="Also collect an intrusive Ascend operator trace"
    )
    parser.add_argument("--profile-trace-start", type=int, default=17, help="First traced chunk, zero-based (>=1)")
    parser.add_argument("--profile-trace-chunks", type=int, default=2)
    args = parser.parse_args()
    if args.residual_reuse_full_steps is not None and len(args.residual_reuse_full_steps) == 1:
        if args.residual_reuse_full_steps[0] != 0:
            args.residual_reuse_full_steps = args.residual_reuse_full_steps[0]
    config = yaml.safe_load(Path(args.config).read_text())
    for name in ("device", "dtype", "fps", "attention_backend"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    for name in (
        "steps",
        "history",
        "seed",
        "guidance",
        "guidance_interval",
        "normalize_cfg",
        "residual_reuse_full_steps",
    ):
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
    if args.profile and (args.mode != "rollout" or args.profile_warmup_runs < 1):
        parser.error("--profile requires rollout and at least one full warmup run")
    if args.profile_npu_trace and not args.profile:
        parser.error("--profile-npu-trace requires --profile")
    chunk_size = config.get("geometry", {}).get("actions_per_block", 32)
    chunks = (args.max_actions + chunk_size - 1) // chunk_size
    if args.profile_npu_trace and (
        args.profile_trace_start < 1
        or args.profile_trace_chunks < 1
        or args.profile_trace_start + args.profile_trace_chunks > chunks
    ):
        parser.error("Trace window must fit inside the rollout and start after chunk 0")
    tasks = ("forward_dynamics", "inverse_dynamics", "policy") if args.task == "all" else (args.task,)
    outputs = [Path(args.output) / task if len(tasks) > 1 else Path(args.output) for task in tasks]
    for output in outputs:
        if (output / "run.json").exists():
            parser.error(f"Inference output already exists: {output}")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(config["device"])
    from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline, save_result

    load_begin = time.perf_counter()
    pipe = CausalCosmos3Pipeline.from_pretrained(args.checkpoint, config)
    if args.profile:
        torch.npu.synchronize(pipe.device)
    load_seconds = time.perf_counter() - load_begin
    for task, output in zip(tasks, outputs, strict=True):
        identity = dict(
            dataset=args.dataset, input_npz=args.input_npz, episode=args.episode, start=args.start, task=task
        )
        if args.mode == "teacher_forcing":
            result = pipe.teacher_forcing(
                args.dataset, args.episode, args.start, args.max_actions, task=task, prompt=args.prompt
            )
        else:
            rollout_options = dict(
                dataset=args.dataset,
                input_npz=args.input_npz,
                episode=args.episode,
                start=args.start,
                horizon=args.max_actions,
                fps=config.get("fps"),
                prompt=args.prompt,
            )
            if args.profile:
                from veomni.models.diffusers.causal_cosmos3.rollout_profiler import RolloutProfiler, npu_trace

                warmups = []
                for index in range(args.profile_warmup_runs):
                    warmup = RolloutProfiler(pipe.device, label=f"warmup {index + 1}")
                    # Discard all generated history and results, retaining model/kernel/allocator caches.
                    pipe.rollout(**rollout_options, profiler=warmup)
                    warmups.append(warmup.summary())
                trace = (
                    npu_trace(output / "npu_trace", start=args.profile_trace_start, active=args.profile_trace_chunks)
                    if args.profile_npu_trace
                    else None
                )
                with trace if trace is not None else nullcontext():
                    profiler = RolloutProfiler(pipe.device, trace=trace)
                    rollout_begin = time.perf_counter()
                    result = pipe.rollout(**rollout_options, profiler=profiler)
                    profiler.synchronize()
                    rollout_seconds = time.perf_counter() - rollout_begin
                report = profiler.summary()
                report.update(
                    model_load_seconds=load_seconds,
                    rollout_wall_seconds=rollout_seconds,
                    warmup_runs=warmups,
                    requested_actions=args.max_actions,
                    actions_per_chunk=pipe.geometry.actions_per_block,
                    device=str(pipe.device),
                    device_name=torch.npu.get_device_name(pipe.device),
                    torch_version=torch.__version__,
                    torch_npu_version=torch_npu.__version__,
                    trace_start=args.profile_trace_start if trace is not None else None,
                    trace_chunks=args.profile_trace_chunks if trace is not None else 0,
                    chunks_detail=profiler.records,
                    exclusions="model load, initial observation read, result assembly and artifact export excluded from chunk mean",
                )
                output.mkdir(parents=True, exist_ok=True)
                (output / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(
                    f"Mean chunk latency: {report['mean_chunk_seconds']:.3f} s -> {output / 'profile.json'}",
                    flush=True,
                )
            else:
                result = pipe.rollout(**rollout_options)
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
