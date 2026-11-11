# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronized rollout timings, allocator peaks and optional Ascend traces."""

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


class RolloutProfiler:
    """One instance per rollout; sampling sub-stages are nested inside disjoint top-level stages."""

    def __init__(self, device, *, label="measured", trace=None):
        self.device = torch.device(device)
        self.backend = None if self.device.type == "cpu" else getattr(torch, self.device.type)
        self.label, self.trace = label, trace
        self.records = []
        self.stages = {}
        self.sampling_stages = {}

    def synchronize(self):
        if self.backend is not None:
            self.backend.synchronize(self.device)

    def start_chunk(self):
        self.synchronize()
        if self.backend is not None:
            self.backend.reset_peak_memory_stats(self.device)
        self.stages = {}
        self.sampling_stages = {}
        self.begin = time.perf_counter()

    @contextmanager
    def stage(self, name, *, sampling=False):
        self.synchronize()
        begin = time.perf_counter()
        path = f"rollout/sampling/{name}" if sampling else f"rollout/{name}"
        annotation = torch.profiler.record_function(path) if self.trace is not None else nullcontext()
        with annotation:
            yield
            self.synchronize()
        stages = self.sampling_stages if sampling else self.stages
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - begin

    def finish_chunk(self, index):
        self.synchronize()
        elapsed = time.perf_counter() - self.begin
        memory = {}
        if self.backend is not None:
            memory = {
                "peak_allocated_bytes": self.backend.max_memory_allocated(self.device),
                "peak_reserved_bytes": self.backend.max_memory_reserved(self.device),
            }
        record = dict(
            chunk=index,
            seconds=elapsed,
            stages_seconds=dict(self.stages),
            sampling_stages_seconds=dict(self.sampling_stages),
            **memory,
        )
        self.records.append(record)
        print(f"[{self.label}] chunk {index + 1}: {elapsed:.3f} s", flush=True)
        if self.trace is not None:
            self.trace.step()
        return record

    def summary(self):
        if not self.records:
            raise ValueError("Cannot summarize an empty rollout")
        count = len(self.records)
        total = sum(row["seconds"] for row in self.records)
        result = dict(
            chunks=count,
            total_chunk_seconds=total,
            mean_chunk_seconds=total / count,
            mean_stages_seconds={
                key: sum(row["stages_seconds"].get(key, 0.0) for row in self.records) / count
                for key in self.records[0]["stages_seconds"]
            },
            mean_sampling_stages_seconds={
                key: sum(row["sampling_stages_seconds"].get(key, 0.0) for row in self.records) / count
                for key in sorted({key for row in self.records for key in row["sampling_stages_seconds"]})
            },
            min_chunk_seconds=min(row["seconds"] for row in self.records),
            max_chunk_seconds=max(row["seconds"] for row in self.records),
            trace_enabled=self.trace is not None,
            timing="device-synchronized wall time; includes stage synchronization overhead",
            feedback="feedback_decode is VAE RGB decoding, a proxy for client feedback, not network or robot latency",
            memory_scope="process tensor allocator; reserved includes warmup cache; excludes driver/external allocations",
        )
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if key in self.records[0]:
                result[key] = max(row[key] for row in self.records)
        return result


def timed_rollout(pipe, options, *, profiler=None):
    """Time a whole rollout with only boundary device synchronization.

    With ``profiler=None``, no stage timers, peak resets, record_function
    annotations or vendor profiler are introduced by this helper. Both modes
    include input reading and result assembly, excluding model loading and
    final artifact export. Each pipeline call starts a fresh episode history.
    """
    backend = None if pipe.device.type == "cpu" else getattr(torch, pipe.device.type)
    if backend is not None:
        backend.synchronize(pipe.device)
    begin = time.perf_counter()
    result = pipe.rollout(**options, profiler=profiler)
    if backend is not None:
        backend.synchronize(pipe.device)
    return result, time.perf_counter() - begin


class DenoiseTimer:
    """Measure only sample_block's denoising region, optionally tracing that region.

    Noise initialization and result assembly are deliberately ignored. There
    are no extra per-step synchronizations, timers or allocator peak resets.
    """

    def __init__(self, device, *, trace_factory=None):
        self.device = torch.device(device)
        self.backend = None if self.device.type == "cpu" else getattr(torch, self.device.type)
        self.trace_factory = trace_factory
        self.seconds = None
        self.context_seconds = None

    def synchronize(self):
        """Wait for the selected device at a denoising boundary."""
        if self.backend is not None:
            self.backend.synchronize(self.device)

    @contextmanager
    def stage(self, name, *, sampling=False):
        """Time/trace denoising and leave every other sampler region untouched."""
        if name != "denoising" or not sampling:
            yield
            return
        if self.seconds is not None:
            raise ValueError("A DenoiseTimer belongs to one sample_block invocation")
        self.synchronize()
        context_begin = time.perf_counter()
        if self.trace_factory is None:
            yield
            self.synchronize()
            self.seconds = self.context_seconds = time.perf_counter() - context_begin
            return
        trace = self.trace_factory()
        with trace:
            self.synchronize()
            begin = time.perf_counter()
            with torch.profiler.record_function("denoise"):
                yield
            self.synchronize()
            self.seconds = time.perf_counter() - begin
        self.synchronize()
        self.context_seconds = time.perf_counter() - context_begin


def _npu_trace(directory, **options):
    import torch_npu

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trace_handler = torch_npu.profiler.tensorboard_trace_handler(str(directory))

    def export_trace(profiler):
        trace_handler(profiler)
        profiler.export_stacks(str(directory / "stacks_cpu.txt"), metric="self_cpu_time_total")

    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=export_trace,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        **options,
    )


def npu_trace(directory, *, start=17, active=2):
    """Capture CPU/NPU operators with Python stacks in a bounded rollout window."""
    import torch_npu

    schedule = torch_npu.profiler.schedule(wait=start - 1, warmup=1, active=active, repeat=1)
    return _npu_trace(directory, schedule=schedule)


def denoise_trace(directory):
    """Record an entire denoising invocation, without rollout-chunk scheduling."""
    return _npu_trace(directory)
