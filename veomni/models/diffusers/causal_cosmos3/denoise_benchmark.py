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
        timer = DenoiseTimer(pipe.device)
        evaluate(timer)
        warmups.append(timer.seconds)
        print(f"[denoise warmup {index + 1}/{warmup_runs}] {timer.seconds * 1000:.3f} ms", flush=True)
    elapsed = []
    reference = None
    for index in range(runs):
        timer = DenoiseTimer(pipe.device)
        result = evaluate(timer)
        elapsed.append(timer.seconds)
        reference = {key: result[key].detach().cpu() for key in ("video_latents", "action")}
        denoising = result["denoising"]
        del result
        print(f"[denoise unprofiled {index + 1}/{runs}] {timer.seconds * 1000:.3f} ms", flush=True)
    report = dict(
        scope="scheduler setup and the complete denoising loop, including lazy packing and cache binding",
        excluded="model/data loading, VAE encode/decode, device transfer, initial noise/masks, output assembly and export",
        timing="device-synchronized wall at denoising boundaries only; no added per-step synchronization",
        warmup=_statistics(warmups),
        without_profiling=_statistics(elapsed),
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
        timer = DenoiseTimer(pipe.device, trace_factory=lambda: denoise_trace(Path(trace_directory)))
        result = evaluate(timer)
        differences = {key: (result[key].detach().cpu() - reference[key]).abs() for key in ("video_latents", "action")}
        report["with_profiling"] = _statistics([timer.seconds])
        report["profiler_context_ms"] = timer.context_seconds * 1000
        report["profiling_overhead_ratio"] = timer.seconds / float(np.median(elapsed)) - 1
        report["profiled_output_max_abs_difference"] = {key: value.max().item() for key, value in differences.items()}
        report["trace_with_stack"] = True
        report["trace_scope"] = "one complete denoising loop; no rollout chunk schedule, VAE or noise initialization"
    return report
