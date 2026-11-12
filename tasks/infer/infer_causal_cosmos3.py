"""Explicit single-NPU teacher forcing or causal policy pixel-feedback rollout."""

import argparse
import copy
import json
import math
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
    parser.add_argument(
        "--gen-attention-implementation",
        choices=("eager", "torch_npu_fia_v2"),
        help="Optional no-grad GEN FIA v2 replacement; eager keeps the original backend",
    )
    parser.add_argument(
        "--add-rms-norm-implementation", choices=("eager", "torch_npu"), help="Opt-in residual RMSNorm fusion"
    )
    parser.add_argument(
        "--fuse-qkv", action=argparse.BooleanOptionalAction, default=None, help="Pack Q/K/V projections for inference"
    )
    parser.add_argument(
        "--compact-denoise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep warm hidden compact between layers and prepack attention KV once per block",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--residual-reuse-full-steps",
        nargs="+",
        type=int,
        help="Refresh count, or increasing refresh indices starting at 0; count 4 with 10 steps maps to 0,3,6,9",
    )
    parser.add_argument("--history", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--guidance-interval", nargs=2, type=float, metavar=("LO", "HI"))
    parser.add_argument("--normalize-cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fixed-step-sigmas", nargs="+", type=float, metavar="SIGMA")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile-sampling-details",
        action="store_true",
        help="Synchronize and time packing, model, scheduler and finite checks separately; diagnostic overhead",
    )
    parser.add_argument("--profile", action="store_true", help="Warm rollout timing and allocator peaks")
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=1,
        help="Unmeasured rollouts or fixed denoising invocations before measurement",
    )
    parser.add_argument(
        "--profile-npu-trace", action="store_true", help="Also collect an intrusive Ascend operator trace"
    )
    parser.add_argument(
        "--compare-profile-wall",
        action="store_true",
        help="Compare uninstrumented and stack-recording NPU-profiled wall time (rollout or denoise-only)",
    )
    parser.add_argument("--profile-trace-start", type=int, default=17, help="First traced chunk, zero-based (>=1)")
    parser.add_argument("--profile-trace-chunks", type=int, default=2)
    parser.add_argument(
        "--denoise-only",
        action="store_true",
        help="Benchmark one prepared denoising workload, without rollout or VAE timing",
    )
    parser.add_argument(
        "--denoise-block",
        type=int,
        default=0,
        help="Global target block from START; 0 is cold, e.g. 17 exercises history reuse",
    )
    parser.add_argument(
        "--denoise-runs", type=int, default=10, help="Unprofiled repeats of the same prepared denoising workload"
    )
    parser.add_argument(
        "--denoise-history-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefill clean history once outside timing and fork the same snapshots for every repeat",
    )
    parser.add_argument(
        "--check-compact-accuracy",
        action="store_true",
        help="Compare compact off/on on the same prepared denoising input and noise",
    )
    parser.add_argument("--accuracy-atol", type=float, default=1e-3)
    parser.add_argument("--accuracy-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--check-fia-v2-accuracy",
        action="store_true",
        help="Compare original GEN attention with FIA v2 on identical QKV and denoising noise",
    )
    args = parser.parse_args()
    accuracy_mode = args.check_compact_accuracy or args.check_fia_v2_accuracy
    if args.residual_reuse_full_steps is not None and len(args.residual_reuse_full_steps) == 1:
        if args.residual_reuse_full_steps[0] != 0:
            args.residual_reuse_full_steps = args.residual_reuse_full_steps[0]
    config = yaml.safe_load(Path(args.config).read_text())
    if args.add_rms_norm_implementation is not None:
        config.setdefault("ops_implementation", {})["add_rms_norm_implementation"] = args.add_rms_norm_implementation
    if args.gen_attention_implementation is not None:
        config.setdefault("ops_implementation", {})["gen_attention_implementation"] = args.gen_attention_implementation
    if args.fuse_qkv is not None:
        config.setdefault("inference_optimizations", {})["fuse_qkv"] = args.fuse_qkv
    if args.compact_denoise is not None:
        config.setdefault("inference_optimizations", {})["compact_denoise"] = args.compact_denoise
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
    if args.profile_sampling_details and not (args.profile or args.denoise_only):
        parser.error("--profile-sampling-details requires --profile or --denoise-only")
    if args.profile_npu_trace and not args.profile:
        parser.error("--profile-npu-trace requires --profile")
    if args.compare_profile_wall and not args.profile_npu_trace:
        parser.error("--compare-profile-wall requires --profile-npu-trace and --profile")
    if args.check_compact_accuracy and args.check_fia_v2_accuracy:
        parser.error("Select only one accuracy comparison")
    if accuracy_mode and (args.denoise_only or args.profile_npu_trace or args.compare_profile_wall):
        parser.error("Accuracy checking cannot be combined with denoise-only or trace comparison")
    if accuracy_mode and not all(math.isfinite(x) and x >= 0 for x in (args.accuracy_atol, args.accuracy_rtol)):
        parser.error("Accuracy tolerances must be finite and nonnegative")
    if (args.denoise_only or accuracy_mode) and (
        args.mode != "rollout"
        or args.task != "policy"
        or args.dataset is None
        or args.denoise_block < 0
        or args.denoise_runs < 1
        or args.profile_warmup_runs < 1
    ):
        parser.error(
            "Denoise-only/accuracy modes require dataset policy rollout arguments, a nonnegative block and positive runs/warmups"
        )
    chunk_size = config.get("geometry", {}).get("actions_per_block", 32)
    chunks = (args.max_actions + chunk_size - 1) // chunk_size
    if (
        args.profile_npu_trace
        and not args.denoise_only
        and (
            args.profile_trace_start < 1
            or args.profile_trace_chunks < 1
            or args.profile_trace_start + args.profile_trace_chunks > chunks
        )
    ):
        parser.error("Trace window must fit inside the rollout and start after chunk 0")
    tasks = ("forward_dynamics", "inverse_dynamics", "policy") if args.task == "all" else (args.task,)
    outputs = [Path(args.output) / task if len(tasks) > 1 else Path(args.output) for task in tasks]
    for output in outputs:
        if any(
            (output / name).exists()
            for name in ("run.json", "denoise_profile.json", "compact_accuracy.json", "fia_v2_accuracy.json")
        ):
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
    if args.denoise_only or accuracy_mode:
        from veomni.models.diffusers.causal_cosmos3.denoise_benchmark import (
            benchmark_denoising,
            check_compact_denoise_accuracy,
            check_fia_v2_accuracy,
            prepare_denoise_sample,
        )

        output = outputs[0]
        sample, local_block, identity = prepare_denoise_sample(
            pipe, args.dataset, args.episode, args.start, args.denoise_block, prompt=args.prompt
        )
        if accuracy_mode:
            check = check_fia_v2_accuracy if args.check_fia_v2_accuracy else check_compact_denoise_accuracy
            report = check(
                pipe,
                sample,
                local_block,
                seed_block=args.denoise_block,
                atol=args.accuracy_atol,
                rtol=args.accuracy_rtol,
                reuse_history=args.denoise_history_cache,
            )
        else:
            report = benchmark_denoising(
                pipe,
                sample,
                local_block,
                seed_block=args.denoise_block,
                warmup_runs=args.profile_warmup_runs,
                runs=args.denoise_runs,
                sampling_details=args.profile_sampling_details,
                trace_directory=output / "npu_trace" if args.profile_npu_trace else None,
                reuse_history=args.denoise_history_cache,
            )
        report.update(
            checkpoint=str(Path(args.checkpoint).resolve()),
            identity=identity,
            device=str(pipe.device),
            torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__,
            model_load_seconds=load_seconds,
        )
        output.mkdir(parents=True, exist_ok=True)
        filename = (
            "fia_v2_accuracy.json"
            if args.check_fia_v2_accuracy
            else "compact_accuracy.json"
            if args.check_compact_accuracy
            else "denoise_profile.json"
        )
        (output / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if accuracy_mode:
            print(f"Accuracy {'PASS' if report['passed'] else 'FAIL'} -> {output / filename}", flush=True)
            if not report["passed"]:
                raise SystemExit(1)
            return
        print(
            f"Denoising without profiler: median {report['without_profiling']['median_ms']:.3f} ms -> {output / 'denoise_profile.json'}",
            flush=True,
        )
        if "with_profiling" in report:
            print(f"Denoising with profiler: {report['with_profiling']['mean_ms']:.3f} ms", flush=True)
        if report.get("without_profiling_event") is not None:
            print(f"Denoising NPU event: median {report['without_profiling_event']['median_ms']:.3f} ms", flush=True)
        return
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
                from veomni.models.diffusers.causal_cosmos3.rollout_profiler import (
                    RolloutProfiler,
                    npu_trace,
                    timed_rollout,
                )

                warmups = []
                for index in range(args.profile_warmup_runs):
                    warmup = RolloutProfiler(
                        pipe.device, label=f"warmup {index + 1}", sampling_details=args.profile_sampling_details
                    )
                    # Discard all generated history and results, retaining model/kernel/allocator caches.
                    pipe.rollout(**rollout_options, profiler=warmup)
                    warmups.append(warmup.summary())
                baseline = None
                if args.compare_profile_wall:
                    baseline, baseline_seconds = timed_rollout(pipe, rollout_options)
                torch.npu.synchronize(pipe.device)
                profile_context_begin = time.perf_counter()
                trace = (
                    npu_trace(output / "npu_trace", start=args.profile_trace_start, active=args.profile_trace_chunks)
                    if args.profile_npu_trace
                    else None
                )
                with trace if trace is not None else nullcontext():
                    profiler = RolloutProfiler(
                        pipe.device, trace=trace, sampling_details=args.profile_sampling_details
                    )
                    result, rollout_seconds = timed_rollout(pipe, rollout_options, profiler=profiler)
                torch.npu.synchronize(pipe.device)
                profile_context_seconds = time.perf_counter() - profile_context_begin
                report = profiler.summary()
                report.update(
                    model_load_seconds=load_seconds,
                    rollout_wall_seconds=rollout_seconds,
                    profile_context_wall_seconds=profile_context_seconds,
                    warmup_runs=warmups,
                    requested_actions=args.max_actions,
                    actions_per_chunk=pipe.geometry.actions_per_block,
                    device=str(pipe.device),
                    device_name=torch.npu.get_device_name(pipe.device),
                    torch_version=torch.__version__,
                    torch_npu_version=torch_npu.__version__,
                    inference_optimizations=copy.deepcopy(pipe.config.get("inference_optimizations", {})),
                    ops_implementation=copy.deepcopy(pipe.config.get("ops_implementation", {})),
                    packed_qkv_bytes=pipe.model.packed_qkv_bytes(),
                    trace_start=args.profile_trace_start if trace is not None else None,
                    trace_chunks=args.profile_trace_chunks if trace is not None else 0,
                    trace_with_stack=trace is not None,
                    chunks_detail=profiler.records,
                    exclusions="model load, initial observation read, result assembly and artifact export excluded from chunk mean",
                )
                output.mkdir(parents=True, exist_ok=True)
                if baseline is not None:
                    if baseline.actions.shape != result.actions.shape or baseline.video.shape != result.video.shape:
                        raise ValueError("Profiled and unprofiled rollout output shapes differ")
                    action_difference = (result.actions.float() - baseline.actions.float()).abs()
                    comparison = dict(
                        without_profiling_wall_seconds=baseline_seconds,
                        with_profiling_rollout_wall_seconds=rollout_seconds,
                        with_profiling_context_wall_seconds=profile_context_seconds,
                        rollout_overhead_seconds=rollout_seconds - baseline_seconds,
                        rollout_overhead_ratio=rollout_seconds / baseline_seconds - 1,
                        context_overhead_seconds=profile_context_seconds - baseline_seconds,
                        context_overhead_ratio=profile_context_seconds / baseline_seconds - 1,
                        action_max_abs_difference=action_difference.max().item(),
                        action_mean_abs_difference=action_difference.mean().item(),
                        video_identical=torch.equal(result.video, baseline.video),
                        measured_actions=len(result.actions),
                        trace_start=args.profile_trace_start,
                        trace_chunks=args.profile_trace_chunks,
                        with_stack=True,
                        order="full warmups, uninstrumented rollout, profiled rollout; no cache flush",
                        scope="same process/model/input/seeds; fresh episode history per rollout; model load and final artifact export excluded",
                        context_scope="includes profiler context entry/exit and any trace export; rollout scope includes in-loop trace export",
                    )
                    report["wall_comparison"] = comparison
                    (output / "wall_comparison.json").write_text(
                        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
                    )
                    print(
                        f"Wall without profiling: {baseline_seconds:.3f} s; "
                        f"profiled rollout: {rollout_seconds:.3f} s; "
                        f"including profiler context: {profile_context_seconds:.3f} s",
                        flush=True,
                    )
                    del baseline
                (output / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(
                    f"Mean chunk latency: {report['mean_chunk_seconds']:.3f} s -> {output / 'profile.json'}",
                    flush=True,
                )
                print(
                    f"Total denoise time: {report['total_denoise_seconds']:.3f} s "
                    f"across {report['denoise_block_count']} blocks (measured rollout only)",
                    flush=True,
                )
                if report.get("total_denoise_event_seconds") is not None:
                    print(f"Total denoise event time: {report['total_denoise_event_seconds']:.3f} s", flush=True)
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
