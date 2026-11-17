"""Serve one warmed RoboTwin profiling/Event measurement window on Ascend."""

import argparse
import functools
import inspect
import json
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "event"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, server_args = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "module_specs.json").exists():
        parser.error("Choose a fresh output directory; existing measurements must be preserved")

    import torch
    import torch_npu

    from tasks.client_server.server import session as sm
    from tasks.client_server.server.server import main as serve
    from veomni.models.diffusers.causal_cosmos3 import inference as inf
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import inference_cache as kv
    from veomni.models.diffusers.causal_cosmos3.cosmos3_core import vae

    active = False
    count = 0
    records = []
    specs = []
    phase = ""
    profiler = None
    owner_thread = None
    repo = Path(__file__).resolve().parents[3]

    def install(owner, attr, name):
        fn = getattr(owner, attr)
        raw = inspect.unwrap(fn)
        lines, start = inspect.getsourcelines(raw)
        specs.append(
            dict(
                module=name,
                file=str(Path(inspect.getsourcefile(raw)).relative_to(repo)),
                function=raw.__name__,
                start_line=start,
                end_line=start + len(lines) - 1,
            )
        )
        if args.mode == "profile":
            return

        @functools.wraps(fn)
        def timed(*a, **kw):
            if not active:
                return fn(*a, **kw)
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                cpu_ms = (time.perf_counter() - started) * 1000
                end.record()
                records.append(dict(module=name, phase=phase, begin=begin, end=end, cpu_ms=cpu_ms))

        setattr(owner, attr, timed)

    for owner, attr, name in [
        (sm.CausalInferenceSession, "infer", "session.infer"),
        (sm.CausalInferenceSession, "_observe", "session.observe"),
        (sm.CausalInferenceSession, "_compose", "preprocess.compose"),
        (vae.StreamingVideoEncoder, "encode", "vae.stream_encode"),
        (inf.ObservationSequence, "prepare", "observation.prepare"),
        (kv.ObservationKVCache, "_prefill_text", "kv.prefill_text"),
        (kv.ObservationKVCache, "_append_clean", "kv.append_clean"),
        (inf, "sample_rollout_block", "denoise.sample"),
        (kv.PolicyBlockKVCache, "predict", "denoise.predict"),
        (kv.PolicyBlockKVCache, "_forward_layers", "transformer.layers"),
        (kv.ObservationKVCache, "attend", "attention"),
    ]:
        install(owner, attr, name)
    sm.sample_rollout_block = inf.sample_rollout_block
    (args.output / "module_specs.json").write_text(json.dumps(specs, indent=2))
    original = sm.CausalInferenceSession.infer

    @functools.wraps(original)
    def infer(self, request):
        nonlocal active, count, phase, profiler, owner_thread
        is_infer = isinstance(request, dict) and "state" in request
        if is_infer:
            count += 1
        if is_infer and count == 2:
            torch.npu.synchronize()
            owner_thread = threading.get_ident()
            active = True
            if args.mode == "profile":
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                    record_shapes=True,
                    with_stack=True,
                    with_modules=True,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, data_simplification=False
                    ),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output / "profile"), async_mode=False
                    ),
                )
                profiler.start()
        if not active:
            return original(self, request)
        if threading.get_ident() != owner_thread:
            raise RuntimeError("Measurement must finish on its original connection thread")
        phase = f"robotwin_{'infer' if is_infer else 'observe'}_step_{request.get('observation_step')}"
        scope = torch.profiler.record_function(phase) if profiler is not None else nullcontext()
        try:
            with scope:
                result = original(self, request)
                if profiler is not None:
                    torch.npu.synchronize()
        except BaseException:
            active = False
            if profiler is not None:
                profiler.stop()
                profiler = None
            raise
        if is_infer and count == 3:
            active = False
            if profiler is not None:
                profiler.step()
                profiler.stop()
                profiler.export_stacks(str(args.output / "cpu_stacks.log"), metric="self_cpu_time_total")
                profiler.export_stacks(str(args.output / "npu_stacks.log"), metric="self_npu_time_total")
                profiler = None
            else:
                torch.npu.synchronize()
                rows = [
                    dict(
                        module=r["module"],
                        phase=r["phase"],
                        event_ms=r["begin"].elapsed_time(r["end"]),
                        cpu_ms=r["cpu_ms"],
                    )
                    for r in records
                ]
                (args.output / "event_calls.json").write_text(json.dumps(rows, indent=2))
            (args.output / "completed.json").write_text(json.dumps(dict(mode=args.mode, inference_requests=3)))
            print(f"{args.mode} measurement complete: {args.output}", flush=True)
        return result

    sm.CausalInferenceSession.infer = infer
    sys.argv = [sys.argv[0], *server_args]
    serve()


if __name__ == "__main__":
    main()
