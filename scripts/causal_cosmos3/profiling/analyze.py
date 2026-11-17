"""Compare unprofiled NPU Events with source-attributed profiled kernel durations."""

import argparse
import collections
import csv
import heapq
import json
import re
from decimal import Decimal
from pathlib import Path


def analyze(trace, measurement, output):
    ROOT = output
    ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(trace.read_text())
    es = data if isinstance(data, list) else data["traceEvents"]
    specs = json.loads((measurement / "module_specs.json").read_text())

    def ns(x):
        return int(Decimal(str(x)) * 1000)

    def interval(e):
        a = ns(e["ts"])
        return a, a + ns(e.get("dur", 0))

    phases = [e for e in es if e.get("name", "").startswith("robotwin_") and e.get("ph") == "X"]

    def phase_at(t):
        return next((e["name"] for e in phases if interval(e)[0] <= t <= interval(e)[1]), None)

    frame_re = re.compile(r"([^;\r\n]+\.py)\((\d+)\): ([^;\r\n]+)")

    def frame_modules(stack):
        frames = frame_re.findall(stack)
        return [
            s["module"]
            for s in specs
            if any(
                p.strip().endswith(s["file"])
                and fn.strip() == s["function"]
                and s["start_line"] <= int(line) <= s["end_line"]
                for p, line, fn in frames
            )
        ]

    # CPU op -> module scopes from its actual code stack. Cache repeated stacks.
    stack_cache = {}
    cpus = collections.defaultdict(list)
    for e in es:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            a, b = interval(e)
            cpus[(e["pid"], e["tid"])].append((a, b, e))
    starts = {}
    ends = {}
    for e in es:
        if e.get("cat") == "async_npu":
            key = str(e["id"])
            if e.get("ph") == "s":
                starts[key] = e
            elif e.get("ph") == "f":
                ends[key] = e
    # Resolve each CPU launch to the innermost CPU operator using interval queries.
    queries = collections.defaultdict(list)
    for key, s in starts.items():
        if key in ends:
            queries[(s["pid"], s["tid"])].append((ns(s["ts"]), key))
    origin = {}
    for thread, qs in queries.items():
        ops = sorted(cpus[thread], key=lambda x: x[0])
        index = 0
        heap = []
        for t, key in sorted(qs):
            while index < len(ops) and ops[index][0] <= t:
                a, b, e = ops[index]
                heapq.heappush(heap, (b - a, b, index))
                index += 1
            while heap and heap[0][1] < t:
                heapq.heappop(heap)
            if heap:
                origin[key] = ops[heap[0][2]][2]
    # Map flow finish to exact device task. Do not infer attribution by wall overlap.
    device = {}
    for e in es:
        if e.get("ph") == "X" and "Task Type" in e.get("args", {}):
            device[(e["pid"], e["tid"], ns(e["ts"]))] = e
    bytask = collections.defaultdict(list)
    for key, end in ends.items():
        taskkey = (end["pid"], end["tid"], ns(end["ts"]))
        if taskkey in device:
            bytask[taskkey].append(key)
    compute = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU", "DSA_SQE"}
    dma = {"SDMA_SQE", "PCIE_DMA_SQE"}
    rows = []
    grouped = collections.defaultdict(lambda: [0, 0.0])
    totals = collections.defaultdict(lambda: collections.Counter())
    for taskkey, e in device.items():
        typ = e["args"]["Task Type"]
        if typ not in compute | dma:
            continue
        keys = bytask.get(taskkey, [])
        ops = [origin[k] for k in keys if k in origin]
        op = min(ops, key=lambda x: float(x.get("dur", 0))) if ops else None
        source_time = interval(op)[0] if op else None
        phase = phase_at(source_time) if op else None
        if phase is None:
            continue
        stack = op.get("args", {}).get("Call stack", "")
        if stack not in stack_cache:
            stack_cache[stack] = frame_modules(stack)
        modules = stack_cache[stack]
        kind = "compute" if typ in compute else "dma"
        row = dict(
            kernel=e["name"],
            task_type=typ,
            kind=kind,
            duration_us=float(e["dur"]),
            device_pid=e["pid"],
            stream=e["tid"],
            device_start_us=e["ts"],
            task_id=e["args"].get("Task Id"),
            flow_ids=";".join(keys),
            cpu_op=op["name"],
            cpu_start_us=op["ts"],
            phase=phase,
            modules=";".join(modules),
        )
        rows.append(row)
        for m in modules:
            totals[m][kind + "_us"] += row["duration_us"]
            totals[m][kind + "_count"] += 1
            g = grouped[(m, e["name"], typ, kind)]
            g[0] += 1
            g[1] += row["duration_us"]
    # Retain all device tasks without a CPU flow/stack so coverage remains auditable.
    unmapped = [
        dict(
            kernel=e["name"],
            task_type=e["args"]["Task Type"],
            duration_us=e["dur"],
            pid=e["pid"],
            stream=e["tid"],
            ts=e["ts"],
        )
        for key, e in device.items()
        if e["args"]["Task Type"] in compute | dma and not any(k in origin for k in bytask.get(key, []))
    ]
    (ROOT / "unmapped_device_tasks.json").write_text(json.dumps(unmapped, indent=2))
    with (ROOT / "kernel_occurrences.csv").open("w", newline="") as f:
        fields = [
            "kernel",
            "task_type",
            "kind",
            "duration_us",
            "device_pid",
            "stream",
            "device_start_us",
            "task_id",
            "flow_ids",
            "cpu_op",
            "cpu_start_us",
            "phase",
            "modules",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (ROOT / "module_kernel_totals.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "kernel", "task_type", "kind", "count", "total_us"])
        for key, val in sorted(grouped.items(), key=lambda x: (x[0][0], -x[1][1])):
            w.writerow([*key, *val])
    calls = json.loads((measurement / "event_calls.json").read_text())
    summary = []
    for spec in specs:
        m = spec["module"]
        cs = [c for c in calls if c["module"] == m]
        # Python function scopes verify that equivalent call counts were executed.
        pc = []
        for e in es:
            if e.get("cat") != "python_function":
                continue
            frames = frame_re.findall(e["name"])
            if any(
                spec["file"].endswith(p.strip())
                and fn.strip() == spec["function"]
                and spec["start_line"] <= int(line) <= spec["end_line"]
                for p, line, fn in frames
            ) and phase_at(interval(e)[0]):
                pc.append(e)
        t = totals[m]
        summary.append(
            dict(
                module=m,
                event_calls=len(cs),
                profile_calls=len(pc),
                event_total_ms=sum(c["event_ms"] for c in cs),
                event_mean_ms=sum(c["event_ms"] for c in cs) / len(cs) if cs else 0,
                cpu_total_ms=sum(c["cpu_ms"] for c in cs),
                kernel_count=t["compute_count"],
                kernel_total_ms=t["compute_us"] / 1000,
                dma_count=t["dma_count"],
                dma_total_ms=t["dma_us"] / 1000,
            )
        )
    with (ROOT / "module_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    report = dict(
        source_trace=str(trace),
        matched_device_tasks=len(rows),
        matched_compute_kernels=sum(r["kind"] == "compute" for r in rows),
        unmapped_device_tasks=len(unmapped),
        unmapped_device_time_ms=sum(float(e["duration_us"]) for e in unmapped) / 1000,
        unique_compute_kernel_sum_ms=sum(r["duration_us"] for r in rows if r["kind"] == "compute") / 1000,
        summary=summary,
    )
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2))
    if {c["phase"] for c in calls} != {e["name"] for e in phases}:
        raise ValueError("Event/profile measurement phases differ")
    if not all(r["event_calls"] == r["profile_calls"] for r in summary):
        raise ValueError("Event/profile module call counts differ; see module_summary.csv")
    if not rows:
        raise ValueError("No CPU-to-device tasks were matched")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Attribute torch_npu kernels to Event-timed code scopes using CPU/NPU flows."
    )
    parser.add_argument("--trace", type=Path, required=True, help="trace_view.json from the profile run")
    parser.add_argument(
        "--measurement",
        type=Path,
        required=True,
        help="Event output containing module_specs.json and event_calls.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.measurement, args.output), indent=2))


if __name__ == "__main__":
    main()
