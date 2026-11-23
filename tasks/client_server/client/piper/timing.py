"""Client-clock timing for inference availability and real RTC execution windows."""

import json
import threading
import time
from pathlib import Path

import numpy as np


class PiperTimings:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.origin = clock()
        self.events = []
        self.lock = threading.Lock()

    def now(self):
        return self.clock() - self.origin

    def record(self, event, **values):
        row = dict(event=event, at_s=self.now(), **values)
        with self.lock:
            self.events.append(row)
        return row

    def snapshot(self):
        with self.lock:
            events = sorted(self.events, key=lambda row: row["at_s"])
        requests = {}
        for row in events:
            if "observation_step" in row:
                requests.setdefault(row["observation_step"], {})[row["event"]] = row
        handoffs = []
        for step, rows in sorted(requests.items()):
            if not step or not {"request_submitted", "request_completed", "handoff_due"} <= rows.keys():
                continue
            submitted = rows["request_submitted"]["at_s"]
            completed = rows["request_completed"]["at_s"]
            due = rows["handoff_due"]["at_s"]
            handoffs.append(
                dict(
                    observation_step=step,
                    request_available_ms=(completed - submitted) * 1000,
                    execution_overlap_ms=(due - submitted) * 1000,
                    slack_ms=(due - completed) * 1000,
                    uncovered_inference_ms=max(0, completed - due) * 1000,
                    ready_before_handoff=completed <= due,
                )
            )

        def stats(values):
            return (
                None
                if not values
                else dict(
                    count=len(values),
                    mean=float(np.mean(values)),
                    p50=float(np.percentile(values, 50)),
                    p95=float(np.percentile(values, 95)),
                    max=float(max(values)),
                )
            )

        completed = [r for r in events if r["event"] == "request_completed"]
        starts = {r["round"]: r["at_s"] for r in events if r["event"] == "execution_start"}
        execution_ms = [
            (r["at_s"] - starts[r["round"]]) * 1000
            for r in events
            if r["event"] == "execution_end" and r["round"] in starts
        ]
        commands = [r for r in events if r["event"] == "command"]
        summary = dict(
            completed_requests=len(completed),
            execution_block_ms=stats(execution_ms),
            arrival_wait_ms=stats([r["duration_ms"] for r in events if r["event"] == "arrival_wait"]),
            command_interval_ms=stats([(b["at_s"] - a["at_s"]) * 1000 for a, b in zip(commands, commands[1:])]),
            evaluated_handoffs=len(handoffs),
            late_handoffs=sum(not r["ready_before_handoff"] for r in handoffs),
            client_round_trip_ms=stats([r["client_round_trip_ms"] for r in completed]),
            server_infer_ms=stats([r["server_infer_ms"] for r in completed if r["server_infer_ms"] is not None]),
            execution_overlap_ms=stats([r["execution_overlap_ms"] for r in handoffs]),
            uncovered_inference_ms=stats([r["uncovered_inference_ms"] for r in handoffs]),
        )
        return dict(
            schema_version=1,
            clock="client perf_counter seconds since protocol construction",
            summary=summary,
            handoffs=handoffs,
            events=events,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)
        return data["summary"]
