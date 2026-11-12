# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-input denoising wall/trace benchmark, outside the VAE and rollout loop."""

import copy
import math
from pathlib import Path

import numpy as np
import torch

from veomni.data.causal_cosmos3.readers.observation import read_teacher_window

from .cosmos3_core.inference_cache import HistoryCache
from .inference import noise_seed, sample_block
from .rollout_profiler import DenoiseTimer, denoise_trace


def prepare_denoise_sample(pipe, dataset, episode, start, block, *, prompt=None):
    """Read measured history for one target block and VAE-encode it exactly once.

    Crop the input prefix to the same complete-block history limit as rollout.
    This uses measured pixels/states, not generated pixel-feedback history.
    The resulting target is local to the cropped window; its seed index remains
    the requested global block. No future block is appended after the target.
    """
    if type(block) is not int or block < 0:
        raise ValueError("denoise block must be a nonnegative integer")
    size = pipe.geometry.actions_per_block
    latent_block = size // pipe.geometry.actions_per_latent
    context = min(block, math.ceil(pipe.history / latent_block))
    window_start = start + (block - context) * size
    horizon = (context + 1) * size
    sample = read_teacher_window(
        dataset,
        episode,
        window_start,
        horizon,
        source=pipe.source,
        processor=pipe.processor,
        geometry=pipe.geometry,
        tokenize=pipe.tokenize,
        task="policy",
        history=pipe.history,
        prompt=prompt,
    )
    if sample["real_action_steps"] <= context * size:
        raise ValueError("The selected denoise block is beyond the episode; choose an earlier block")
    encoded = pipe.vae.encode_pack([sample])[0]
    for key in ("video_latents", "action", "states", "action_mask", "state_mask"):
        encoded[key] = encoded[key].to(pipe.device)
    return (
        encoded,
        context,
        dict(
            episode=episode,
            start=start,
            global_block=block,
            local_block=context,
            window_start=window_start,
            requested_window_actions=horizon,
            valid_target_actions=min(size, sample["real_action_steps"] - context * size),
            input_semantics="measured-history policy window, not generated rollout feedback",
            latent_shape=list(encoded["video_latents"].shape),
        ),
    )


def _statistics(seconds):
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1000
    return dict(
        runs=len(seconds),
        mean_ms=float(milliseconds.mean()),
        median_ms=float(np.median(milliseconds)),
        min_ms=float(milliseconds.min()),
        max_ms=float(milliseconds.max()),
        p95_ms=float(np.percentile(milliseconds, 95)),
        samples_ms=milliseconds.tolist(),
    )


@torch.no_grad()
def benchmark_denoising(
    pipe,
    sample,
    block,
    *,
    seed_block=None,
    warmup_runs=3,
    runs=10,
    trace_directory=None,
    reuse_history=True,
    sampling_details=False,
):
    """Repeat only the native denoiser on fixed latents and fixed initial noise.

    Each invocation rebuilds its block-local cache. With enabled history reuse,
    prepare the preceding clean snapshot once outside measurement and fork its
    immutable layer records for every invocation. A single traced invocation
    follows the unprofiled repeats; its wall time is measured over exactly the
    same denoising boundaries. No VAE decode or physical-action export runs.
    """
    if type(warmup_runs) is not int or warmup_runs < 1 or type(runs) is not int or runs < 1:
        raise ValueError("denoise warmup_runs and runs must be positive integers")
    if type(reuse_history) is not bool:
        raise ValueError("reuse_history must be a boolean")
    if pipe.model.training or torch.is_inference_mode_enabled() or torch.is_autocast_enabled(pipe.device.type):
        raise ValueError(
            "Denoise benchmarking requires explicit-precision eval outside autocast/inference-mode contexts"
        )
    sid = block if seed_block is None else seed_block
    if type(block) is not int or block < 0 or type(sid) is not int or sid < block:
        raise ValueError("denoise requires a nonnegative block and seed_block >= block")
    options = dict(pipe.sampling_options)
    block_cache_enabled = options.get("prune_attention", True) and options.get("cache_fixed_attention", True)
    seed = options.get("seed", 42)
    noise = {
        name: torch.randn(
            tensor.shape,
            dtype=torch.float32,
            device=tensor.device,
            generator=torch.Generator(device=tensor.device).manual_seed(noise_seed(seed, sid, name)),
        )
        for name, tensor in (("video", sample["video_latents"]), ("action", sample["action"]))
    }
    history = None
    if reuse_history and block > 0 and block_cache_enabled:
        history = HistoryCache()
        # Clean rows cannot read noisy targets: one preceding-block evaluation
        # is sufficient to seed a repeatable history-hit workload. Its sampler
        # output is discarded; it is not a generated rollout prefix.
        prefix_options = dict(
            options, steps=1, residual_reuse_full_steps=None, fixed_step_sampler_config=None, guidance_interval=None
        )
        sample_block(
            pipe.model,
            sample,
            block - 1,
            seed_block=sid - 1,
            initial_noise=noise,
            history_cache=history,
            **prefix_options,
        )

    def evaluate(timer):
        cache = None
        if history is not None:
            cache = HistoryCache()
            # bind() replaces branch records and copies the previous layer
            # dictionary. Old snapshot tensors are immutable and remain owned
            # by the template; measurements never consume one another's cache.
            cache.branches = dict(history.branches)
        result = sample_block(
            pipe.model,
            sample,
            block,
            seed_block=sid,
            initial_noise=noise,
            history_cache=cache,
            profiler=timer,
            **options,
        )
        if timer.seconds is None:
            raise ValueError("The selected block has no active denoising work")
        return result

    warmups = []
    for index in range(warmup_runs):
        timer = DenoiseTimer(pipe.device, sampling_details=sampling_details)
        evaluate(timer)
        warmups.append(timer.seconds)
        print(f"[denoise warmup {index + 1}/{warmup_runs}] {timer.seconds * 1000:.3f} ms", flush=True)
    elapsed, event_elapsed = [], []
    reference = None
    details = []
    for index in range(runs):
        timer = DenoiseTimer(pipe.device, sampling_details=sampling_details)
        result = evaluate(timer)
        elapsed.append(timer.seconds)
        if sampling_details:
            details.append(dict(summary=timer.detail_summary(), records=list(timer.detail_records)))
        if timer.event_seconds is not None:
            event_elapsed.append(timer.event_seconds)
        reference = {key: result[key].detach().cpu() for key in ("video_latents", "action")}
        denoising = result["denoising"]
        del result
        event = "n/a" if timer.event_seconds is None else f"{timer.event_seconds * 1000:.3f} ms"
        print(f"[denoise unprofiled {index + 1}/{runs}] wall={timer.seconds * 1000:.3f} ms, event={event}", flush=True)
    report = dict(
        scope="scheduler setup and the complete denoising loop, including lazy packing and cache binding",
        excluded="model/data loading, VAE encode/decode, device transfer, initial noise/masks, output assembly and export",
        timing="device-synchronized wall at denoising boundaries only; no added per-step synchronization",
        warmup=_statistics(warmups),
        without_profiling=_statistics(elapsed),
        sampling_details_enabled=sampling_details,
        sampling_details=details,
        without_profiling_event=_statistics(event_elapsed) if event_elapsed else None,
        denoising=denoising,
        global_block=sid,
        local_block=block,
        latent_shape=list(sample["video_latents"].shape),
        action_shape=list(sample["action"].shape),
        history_cache="prefilled immutable snapshot per repeat"
        if history is not None
        else "cold; no cross-block reuse",
        block_cache_enabled=block_cache_enabled,
        history_latent_frames=pipe.history,
        text_tokens=len(sample["text_token_ids"]),
        dtype=str(next(pipe.model.parameters()).dtype),
        attention_backend=options.get("attention_backend", "grouped"),
        sampling=copy.deepcopy(pipe.config.get("sampling", {})),
        inference_optimizations=copy.deepcopy(pipe.config.get("inference_optimizations", {})),
        ops_implementation=copy.deepcopy(pipe.config.get("ops_implementation", {})),
        packed_qkv_bytes=pipe.model.packed_qkv_bytes(),
        seeds={name: noise_seed(seed, sid, name) for name in ("video", "action")},
    )
    if trace_directory is not None:
        timer = DenoiseTimer(
            pipe.device, trace_factory=lambda: denoise_trace(Path(trace_directory)), sampling_details=sampling_details
        )
        result = evaluate(timer)
        differences = {key: (result[key].detach().cpu() - reference[key]).abs() for key in ("video_latents", "action")}
        report["with_profiling"] = _statistics([timer.seconds])
        report["with_profiling_event"] = (
            _statistics([timer.event_seconds]) if timer.event_seconds is not None else None
        )
        report["profiler_context_ms"] = timer.context_seconds * 1000
        report["profiling_overhead_ratio"] = timer.seconds / float(np.median(elapsed)) - 1
        report["profiled_output_max_abs_difference"] = {key: value.max().item() for key, value in differences.items()}
        report["trace_with_stack"] = True
        report["trace_scope"] = "one complete denoising loop; no rollout chunk schedule, VAE or noise initialization"
    return report


@torch.no_grad()
def _check_denoise_accuracy(
    pipe, sample, block, *, seed_block=None, atol=1e-3, rtol=1e-3, reuse_history=True, kind="compact"
):
    """Compare compact or GEN-kernel modes on identical measured inputs and noise.

    Each side independently builds history/caches. Report current-block latent
    and valid normalized-action errors; these are numerical parity checks, not
    a task-quality threshold in physical units.
    """
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("Accuracy tolerances must be finite and nonnegative")
    if pipe.model.training or torch.is_inference_mode_enabled() or torch.is_autocast_enabled(pipe.device.type):
        raise ValueError("Accuracy checking requires explicit-precision no-grad eval outside inference_mode")
    options = dict(pipe.sampling_options)
    if not all(
        options.get(key, True) for key in ("reuse_denoise_packing", "prune_attention", "cache_fixed_attention")
    ):
        raise ValueError("Accuracy checking requires prepared query-pruned fixed-attention caching")
    sid = block if seed_block is None else seed_block
    if type(block) is not int or block < 0 or type(sid) is not int or sid < block:
        raise ValueError("Accuracy checking requires nonnegative block and seed_block >= block")
    seed = options.get("seed", 42)
    noise = {
        name: torch.randn(
            tensor.shape,
            dtype=torch.float32,
            device=tensor.device,
            generator=torch.Generator(device=tensor.device).manual_seed(noise_seed(seed, sid, name)),
        )
        for name, tensor in (("video", sample["video_latents"]), ("action", sample["action"]))
    }
    previous = pipe.model.compact_denoise
    previous_op = pipe.model.gen_attention_op
    checked = None
    target_fia_calls = 0
    outputs = []
    warm_evaluations = 0

    def count_warm(layer, args, kwargs):
        nonlocal warm_evaluations
        warm_evaluations += int(kwargs.get("compact_gen", False))

    if not pipe.model.layers:
        raise ValueError("Compact accuracy checking requires a decoder layer")
    handle = pipe.model.layers[0].register_forward_pre_hook(count_warm, with_kwargs=True)
    try:
        for enabled in (False, True):
            warm_evaluations = 0
            if kind == "compact":
                pipe.model.set_compact_denoise(enabled)
            else:
                pipe.model.set_gen_attention_implementation("torch_npu_fia_v2" if enabled else "eager")
                if enabled:
                    checked = _CheckedFIA(pipe.model.gen_attention_op, atol, rtol)
                    pipe.model.gen_attention_op = checked
            history = HistoryCache() if reuse_history and block > 0 else None
            if history is not None:
                prefix_options = dict(
                    options,
                    steps=1,
                    residual_reuse_full_steps=None,
                    fixed_step_sampler_config=None,
                    guidance_interval=None,
                )
                sample_block(
                    pipe.model,
                    sample,
                    block - 1,
                    seed_block=sid - 1,
                    initial_noise=noise,
                    history_cache=history,
                    **prefix_options,
                )
            before_target = 0 if checked is None else checked.calls
            result = sample_block(
                pipe.model, sample, block, seed_block=sid, initial_noise=noise, history_cache=history, **options
            )
            if kind == "fia_v2" and enabled:
                target_fia_calls = checked.calls - before_target
                if target_fia_calls < 1:
                    raise ValueError("Accuracy check did not execute FIA v2 on the target block")
            # CFG counts both cold branches as full evaluations. Observe actual
            # compact execution instead of accepting that aggregate counter.
            if kind == "compact" and enabled and warm_evaluations == 0:
                raise ValueError(
                    "Accuracy checking requires at least two full evaluations on a branch to exercise warm compact layers"
                )
            outputs.append({key: result[key].float().cpu().double() for key in ("video_latents", "action")})
            del result, history
    except ValueError as error:
        if kind == "fia_v2" and checked is not None and not checked.finite:
            return dict(
                passed=False,
                atol=atol,
                rtol=rtol,
                comparison="original GEN attention versus FIA v2",
                attention=checked.summary(),
                metrics=None,
                completed_denoising=False,
                error=str(error),
                local_block=block,
                global_block=sid,
            )
        raise
    finally:
        handle.remove()
        pipe.model.set_compact_denoise(previous)
        pipe.model.gen_attention_op = previous_op
    start = 1 + block * sample["block_size"]
    stop = min(start + sample["block_size"], sample["video_latents"].shape[1])
    video_stop = min(stop, sample.get("num_valid_video_latents", stop))
    frames = torch.as_tensor(sample["action_frame_ids"], device="cpu")
    action_rows = (frames >= start) & (frames < stop)
    if "action_valid_steps" in sample:
        action_rows &= sample["action_valid_steps"].cpu().bool()
    mask = sample["action_mask"].bool().expand_as(sample["action"]).cpu()[action_rows]
    pairs = {
        "video_latents": (
            outputs[0]["video_latents"][:, start:video_stop],
            outputs[1]["video_latents"][:, start:video_stop],
        ),
        "action": (outputs[0]["action"][action_rows][mask], outputs[1]["action"][action_rows][mask]),
    }
    metrics = {}
    for name, (reference, actual) in pairs.items():
        finite = torch.isfinite(reference) & torch.isfinite(actual)
        difference = (actual - reference).abs()
        violations = (~finite) | (difference > atol + rtol * reference.abs())
        all_finite = bool(finite.all())
        metrics[name] = dict(
            elements=reference.numel(),
            finite=all_finite,
            max_abs=float(difference.max()) if all_finite and reference.numel() else None,
            mean_abs=float(difference.mean()) if all_finite and reference.numel() else None,
            rmse=float((difference.square().mean()).sqrt()) if all_finite and reference.numel() else None,
            outside_tolerance=int(violations.sum()),
            passed=not bool(violations.any()),
        )
    attention = None if checked is None else checked.summary()
    return dict(
        passed=all(item["passed"] for item in metrics.values()) and (attention is None or attention["passed"]),
        atol=atol,
        rtol=rtol,
        criterion="abs(candidate - reference) <= atol + rtol * abs(reference); nonfinite values fail",
        scope="current block video latents and valid normalized action channels",
        comparison="compact off/on" if kind == "compact" else "original GEN attention versus FIA v2",
        attention=attention,
        target_fia_calls=target_fia_calls,
        input_semantics="identical prepared measured-history tensors and actual initial noise; independent caches",
        history_cache=bool(reuse_history and block > 0),
        local_block=block,
        global_block=sid,
        dtype=str(next(pipe.model.parameters()).dtype),
        warm_compact_evaluations=warm_evaluations,
        ops_implementation=copy.deepcopy(pipe.config.get("ops_implementation", {})),
        metrics=metrics,
        sampling=copy.deepcopy(pipe.config.get("sampling", {})),
        attention_backend=options.get("attention_backend", "grouped"),
        seeds={name: noise_seed(seed, sid, name) for name in ("video", "action")},
    )


class _CheckedFIA:
    """Accuracy-only handle comparing each actual FIA call with FAS on the same QKV."""

    impl = "torch_npu_fia_v2"

    def __init__(self, op, atol, rtol):
        self.op, self.atol, self.rtol = op, atol, rtol
        self.calls = self.elements = self.outside = 0
        self.total_abs = self.total_square = self.maximum = 0.0
        self.finite = True

    def __call__(self, query, key, value, *, query_ends, kv_ends, scale):
        """Return FIA output after recording numerical error against the unchanged FAS kernel."""
        from .cosmos3_core.attention import packed_generation_attention

        actual = self.op(query, key, value, query_ends=query_ends, kv_ends=kv_ends, scale=scale)
        reference = packed_generation_attention(query, key, value, "npu_tnd", query_ends, kv_ends)
        expected, candidate = reference.float().cpu().double(), actual.float().cpu().double()
        finite = torch.isfinite(expected) & torch.isfinite(candidate)
        difference = (candidate - expected).abs()
        self.calls += 1
        self.elements += expected.numel()
        self.outside += int(((~finite) | (difference > self.atol + self.rtol * expected.abs())).sum())
        all_finite = bool(finite.all())
        self.finite &= all_finite
        if all_finite:
            self.maximum = max(self.maximum, float(difference.max()))
            self.total_abs += float(difference.sum())
            self.total_square += float(difference.square().sum())
        else:
            raise ValueError("Nonfinite FIA v2 or reference FAS attention output")
        return actual

    def summary(self):
        """Summarize per-call raw-attention errors without nonfinite JSON values."""
        valid = self.finite and self.elements > 0
        return dict(
            calls=self.calls,
            elements=self.elements,
            finite=self.finite,
            max_abs=self.maximum if valid else None,
            mean_abs=self.total_abs / self.elements if valid else None,
            rmse=math.sqrt(self.total_square / self.elements) if valid else None,
            outside_tolerance=self.outside,
            passed=valid and self.outside == 0,
            scope="FIA v2 versus FAS for every candidate call on identical QKV and cumulative group ends",
        )


def check_compact_denoise_accuracy(pipe, sample, block, *, seed_block=None, atol=1e-3, rtol=1e-3, reuse_history=True):
    """Compare compact off/on on identical measured denoising inputs and actual noise."""
    return _check_denoise_accuracy(
        pipe, sample, block, kind="compact", seed_block=seed_block, atol=atol, rtol=rtol, reuse_history=reuse_history
    )


def check_fia_v2_accuracy(pipe, sample, block, *, seed_block=None, atol=1e-3, rtol=1e-3, reuse_history=True):
    """Compare FAS/FIA v2 raw attention and complete denoising outputs; restore model handles."""
    return _check_denoise_accuracy(
        pipe, sample, block, kind="fia_v2", seed_block=seed_block, atol=atol, rtol=rtol, reuse_history=reuse_history
    )
