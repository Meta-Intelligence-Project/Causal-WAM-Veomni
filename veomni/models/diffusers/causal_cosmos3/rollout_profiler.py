# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronized rollout timings, allocator peaks and optional Ascend traces."""

import time
from contextlib import contextmanager, nullcontext

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


def npu_trace(directory, *, start=17, active=2):
    """Capture a bounded late-rollout window, after one profiler preparation chunk."""
    import torch_npu

    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=start - 1, warmup=1, active=active, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
