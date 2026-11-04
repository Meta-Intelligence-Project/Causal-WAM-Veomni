# SPDX-License-Identifier: OpenMDW-1.1
"""WebSocket planner with a local RTC clock and measured request round trips.

Execution uses CPU Torch in the native VeOmni environment; checkpoints stay on
the server. Ordinary and RTC requests share the same publication clock.
"""

import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from queue import Full, Queue

import numpy as np

from veomni.models.diffusers.causal_cosmos3.serving.wire import (
    HISTORY_TRANSPORT,
    MAX_MESSAGE_BYTES,
    PROTOCOL,
    packb,
    unpackb,
)


class RemotePolicyError(RuntimeError):
    """A matched server response reports an inference or protocol error."""


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class CausalPolicyClient:
    def __init__(self, uri="ws://localhost:8000", *, control_mode="joint", request_timeout=120.0):
        from websockets.sync.client import connect

        if isinstance(request_timeout, bool) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive and finite")
        if control_mode not in ("joint", "eef"):
            raise ValueError("control_mode must be joint or eef")
        self.request_timeout = float(request_timeout)
        self.control_mode = control_mode
        self._lock = threading.RLock()
        self._closed = False
        self._session = None
        self._episode_id = None
        self._request_id = 0
        self._image_size = None
        self._cached_history_ticks = set()
        self.last_timing = None
        self.request_timings = []
        self._transport_active = False
        self._last_window_hint = None
        self._window_generation = 0
        self._last_observation_tick = None
        self.connection = connect(
            uri, compression=None, max_size=MAX_MESSAGE_BYTES, ping_interval=None, close_timeout=1, proxy=None
        )
        try:
            self.metadata = unpackb(self.connection.recv(timeout=self.request_timeout))
            if not isinstance(self.metadata, dict) or self.metadata.get("protocol") != PROTOCOL:
                raise ValueError(f"Expected a {PROTOCOL} server")
            modes = self.metadata.get("available_control_modes")
            if modes is not None and control_mode not in modes:
                raise ValueError(f"Server does not support control_mode={control_mode!r}")
            if self.metadata.get("streaming_vae", False):
                self._transport_active = True
                self._outbox = Queue(maxsize=128)
                self._pending = {}
                self._transport_error = None
                self._sender = threading.Thread(target=self._send_loop, name="cosmos-observation-send", daemon=True)
                self._receiver = threading.Thread(
                    target=self._receive_loop, name="cosmos-response-receive", daemon=True
                )
                self._sender.start()
                self._receiver.start()
        except BaseException:
            self.connection.close()
            self._closed = True
            raise

    def infer(self, observation):
        """One serialized RPC, including encoding, transport and decoding timing."""
        if self._transport_active:
            return self._infer_background(observation)
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy connection is closed")
            operation = observation.get("operation", "predict")
            episode = str(uuid.uuid4()) if operation == "reset" else self._episode_id
            if episode is None:
                raise ValueError("Reset with a first measurement before requesting predictions")
            request = dict(observation)
            request.update(operation=operation, episode_id=episode, request_id=self._request_id)
            request.setdefault("tick", 0)
            request.setdefault("revision", 0)
            self._request_id += 1
            started = time.perf_counter()
            try:
                encoded = packb(request)
                self.connection.send(encoded)
                received = self.connection.recv(timeout=self.request_timeout)
                response = unpackb(received)
                if not isinstance(response, dict) or any(
                    response.get(key) != request[key] for key in ("request_id", "episode_id", "tick", "revision")
                ):
                    raise ValueError("Response identity does not match the submitted plan")
            except BaseException:
                # An unread late reply must never become the next RPC's result.
                self._closed = True
                self.connection.close()
                raise
            self.last_timing = {
                "round_trip_seconds": time.perf_counter() - started,
                "request_bytes": len(encoded),
                "response_bytes": len(received),
                "history_frames_sent": sum("image" in record for record in request.get("history", [])),
                "history_frames_reused": sum("image" not in record for record in request.get("history", [])),
                **response.get("server_timing", {}),
            }
            self.request_timings.append(
                dict(
                    operation=operation,
                    request_id=request["request_id"],
                    episode_id=episode,
                    tick=request["tick"],
                    revision=request["revision"],
                    **self.last_timing,
                )
            )
            if "error" in response:
                raise RemotePolicyError(response["error"])
            if self.metadata.get("history_transport") == HISTORY_TRANSPORT:
                ticks = response.get("cached_history_ticks")
                if (
                    not isinstance(ticks, list)
                    or any(type(index) is not int or not 0 <= index < request["tick"] for index in ticks)
                    or len(set(ticks)) != len(ticks)
                ):
                    self._closed = True
                    self.connection.close()
                    raise ValueError("Invalid historical image cache acknowledgement")
                self._cached_history_ticks = set(ticks)
            if operation == "reset":
                self._episode_id = episode
            return response

    def _fail_transport(self, error):
        with self._lock:
            self._closed = True
            self._transport_error = error
            pending, self._pending = self._pending, {}
        for _, future in pending.values():
            if not future.done():
                future.set_exception(error)
        try:
            self._outbox.put_nowait(None)
        except Full:
            pass
        self.connection.close()

    def _submit(self, observation):
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy connection is closed") from self._transport_error
            if len(self._pending) >= 256:
                error = RuntimeError("Too many pending observation messages; stop instead of dropping frames")
                self._fail_transport(error)
                raise error
            operation = observation.get("operation", "predict")
            episode = str(uuid.uuid4()) if operation == "reset" else self._episode_id
            if episode is None:
                raise ValueError("Reset with a first measurement before requesting predictions")
            request = dict(observation)
            request.update(operation=operation, episode_id=episode, request_id=self._request_id)
            request.setdefault("tick", 0)
            request.setdefault("revision", 0)
            self._request_id += 1
            future = Future()
            self._pending[request["request_id"]] = (request, future)
            try:
                self._outbox.put_nowait((request, future))
            except Full:
                error = RuntimeError("Observation transport queue is full; stop and reset instead of dropping frames")
                self._fail_transport(error)
                raise error from None
            return request, future

    def _send_loop(self):
        try:
            while True:
                item = self._outbox.get()
                if item is None:
                    return
                request, future = item
                encoded = packb(request)
                future.request_bytes = len(encoded)
                self.connection.send(encoded)
        except BaseException as error:
            self._fail_transport(error)

    def _receive_loop(self):
        try:
            while True:
                received = self.connection.recv()
                response = unpackb(received)
                if not isinstance(response, dict):
                    raise ValueError("Response identity does not match a submitted request")
                with self._lock:
                    entry = self._pending.pop(response.get("request_id"), None)
                if entry is None:
                    raise ValueError("Response identity does not match a submitted request")
                request, future = entry
                if any(response.get(key) != request[key] for key in ("request_id", "episode_id", "tick", "revision")):
                    error = ValueError("Response identity does not match the submitted request")
                    future.set_exception(error)
                    raise error
                if request["operation"] in ("observe", "prepare_window") and "error" in response:
                    error = RemotePolicyError(response["error"])
                    future.set_exception(error)
                    raise error
                future.set_result((response, len(received)))
        except BaseException as error:
            self._fail_transport(error)

    def _infer_background(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")
        started = time.perf_counter()
        request, future = self._submit(observation)
        try:
            response, response_bytes = future.result(timeout=self.request_timeout)
        except TimeoutError:
            self._fail_transport(TimeoutError("Policy request timed out"))
            raise
        self.last_timing = dict(
            round_trip_seconds=time.perf_counter() - started,
            request_bytes=future.request_bytes,
            response_bytes=response_bytes,
            history_frames_sent=sum("image" in row for row in request.get("history", [])),
            history_frames_reused=sum("image" not in row for row in request.get("history", [])),
            **response.get("server_timing", {}),
            **response.get("streaming_timing", {}),
        )
        self.request_timings.append(
            dict(
                operation=request["operation"],
                request_id=request["request_id"],
                episode_id=request["episode_id"],
                tick=request["tick"],
                revision=request["revision"],
                **self.last_timing,
            )
        )
        if "error" in response:
            raise RemotePolicyError(response["error"])
        ticks = response.get("cached_history_ticks", [])
        if not isinstance(ticks, list) or any(type(t) is not int or not 0 <= t < request["tick"] for t in ticks):
            error = ValueError("Invalid historical image cache acknowledgement")
            self._fail_transport(error)
            raise error
        if request["operation"] == "reset" or request["episode_id"] == self._episode_id:
            self._cached_history_ticks = set(ticks)
        if request["operation"] == "reset":
            self._episode_id = request["episode_id"]
            self._last_window_hint = None
            self._last_observation_tick = None
            self._window_generation = 0
        return response

    def observe(self, *, tick, image, revision=0):
        """Enqueue one real control observation without waiting for inference."""
        if self._transport_active:
            with self._lock:
                if tick != self._last_observation_tick:
                    self._submit(dict(operation="observe", tick=tick, revision=revision, image=_numpy(image).copy()))
                    self._last_observation_tick = tick

    def prepare_window(self, *, anchor_tick, revision=0):
        """A speculative anchor; the server rebuilds if the actual trigger differs."""
        if self._transport_active:
            hint = (self._episode_id, revision, anchor_tick)
            with self._lock:
                if hint != self._last_window_hint:
                    self._window_generation += 1
                    self._submit(
                        dict(
                            operation="prepare_window",
                            tick=0,
                            revision=revision,
                            anchor_tick=anchor_tick,
                            window_generation=self._window_generation,
                        )
                    )
                    self._last_window_hint = hint

    def _close_session(self):
        if self._session is not None:
            session, self._session = self._session, None
            session.close()

    def reset(self, *, prompt, measurement):
        """Drain the old episode and obtain an ordinary, unexecuted bootstrap."""
        self._close_session()
        self._image_size = _numpy(measurement["image_size"]).astype(np.float32, copy=True)
        return self.infer(
            dict(
                operation="reset",
                prompt=prompt,
                state=_numpy(measurement["state"]).astype(np.float32, copy=True),
                image=_numpy(measurement["image"]).copy(),
                image_size=self._image_size,
                tick=0,
                revision=0,
                rtc=None,
            )
        )

    def _actions(self, response):
        import torch

        actions = np.asarray(response.get("actions"))
        if (
            actions.dtype.kind != "f"
            or actions.shape
            != (
                self.metadata.get(
                    "returned_action_steps", self.metadata.get("action_horizon", self.metadata["actions_per_block"])
                ),
                55,
            )
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Server returned an invalid physical unified55 action chunk")
        return torch.from_numpy(actions.astype(np.float32, copy=True))

    def plan(
        self, *, state, state_mask, image, action_mask, history=(), tick, revision, rtc, executed_observations=None
    ):
        """RPC callback for RTCSession; upload only genuinely required history."""
        for name, mask in (("state_mask", state_mask), ("action_mask", action_mask)):
            if not np.array_equal(_numpy(mask), np.asarray(self.metadata[name], dtype=bool)):
                raise ValueError(f"{name} differs from the server's configured robot")
        if self.metadata.get("simulation_sync", False):
            if rtc is not None:
                raise ValueError("Synchronous simulation does not accept RTC prefixes")
            if executed_observations is None:
                raise ValueError("Synchronous simulation requires the executed observation batch")
            response = self.infer(
                dict(
                    state=_numpy(state).astype(np.float32, copy=True),
                    image=_numpy(image).copy(),
                    executed_observations=[
                        dict(tick=record.tick, image=_numpy(record.image).copy()) for record in executed_observations
                    ],
                    tick=tick,
                    revision=revision,
                    rtc=None,
                )
            )
            return self._actions(response)
        if executed_observations is not None:
            raise ValueError("Executed observation batches require synchronous simulation")
        from veomni.models.diffusers.causal_cosmos3.serving.history_window import build_history_window

        window = build_history_window(
            tick,
            **(
                dict(history_steps=self.metadata["history_steps"])
                if self.metadata.get("layout_mode") == "decision_window"
                else dict(history_blocks=self.metadata["history_blocks"])
            ),
            actions_per_block=32,  # legacy alias单位固定32步，与独立预测H无关。
            video_stride=self.metadata["video_stride"],
            history_padding=self.metadata.get("history_padding", "available"),
        )
        selected = tuple(index for index in window.required_observation_ticks if index < tick)
        observed = {record.tick: record.image for record in history}
        if any(index not in observed for index in selected):
            raise ValueError("Missing real historical image required by the remote batch builder")
        prefix = None
        if rtc is not None:
            prefix = dict(
                previous_actions=_numpy(rtc.previous_actions).astype(np.float32, copy=True),
                delay_steps=rtc.delay_steps,
                beta=rtc.beta,
            )
        response = self.infer(
            dict(
                state=_numpy(state).astype(np.float32, copy=True),
                image=_numpy(image).copy(),
                history=[
                    dict(tick=index)
                    if self.metadata.get("history_transport") == HISTORY_TRANSPORT
                    and index in self._cached_history_ticks
                    else dict(tick=index, image=_numpy(observed[index]).copy())
                    for index in selected
                ],
                tick=tick,
                revision=revision,
                rtc=prefix,
            )
        )
        return self._actions(response)

    def create_session(
        self,
        first_measurement,
        *,
        prompt,
        enable_rtc=False,
        execution_mode="auto",
        execution_hz=None,
        execution_policy=None,
        execution_rate_callback=None,
        adaptive_execution=None,
        **session_options,
    ):
        """Return a bootstrapped local session; prepare before enabling motion."""
        from veomni.models.diffusers.causal_cosmos3.serving.rtc_pacing import RTCExecutionPolicy

        if execution_mode not in ("auto", "blocking", "async"):
            raise ValueError("execution_mode must be auto, blocking or async")
        simulation_sync = self.metadata.get("simulation_sync", False)
        if simulation_sync and (enable_rtc or execution_mode == "async"):
            raise ValueError("Synchronous simulation requires blocking execution without RTC")
        if enable_rtc and execution_mode == "blocking":
            raise ValueError("Blocking execution_mode conflicts with enable_rtc")
        if enable_rtc and not self.metadata.get("rtc_enabled", False):
            raise ValueError("Enable RTC on the server before creating an RTC client session")
        if adaptive_execution is not None and type(adaptive_execution) is not bool:
            raise ValueError("adaptive_execution must be a boolean or None")
        if adaptive_execution is False and execution_policy is not None:
            raise ValueError("execution_policy conflicts with adaptive_execution=False")
        adaptive = (
            callable(execution_rate_callback) or execution_policy is not None
            if adaptive_execution is None
            else adaptive_execution
        )
        if adaptive and not callable(execution_rate_callback):
            raise ValueError("Adaptive async execution needs a client execution_rate_callback")
        # RTC仅添加引导；两种模式共用S/R调度，旧服务保留显式blocking基线。
        selected_mode = execution_mode
        if selected_mode == "auto":
            selected_mode = (
                "async"
                if not simulation_sync and (enable_rtc or self.metadata.get("streaming_vae", False))
                else "blocking"
            )
        if selected_mode == "async":
            from veomni.models.diffusers.causal_cosmos3.serving.async_session import (
                AsyncRemoteSession,
                resolve_async_schedule,
            )

            # 只有实际选择异步调度时才校验S/R；direct RPC仍允许S=H。
            resolve_async_schedule(
                self.metadata.get("action_horizon", self.metadata["actions_per_block"]),
                self.metadata["execute_steps"],
                self.metadata.get("handoff_steps"),
            )
            if adaptive and execution_policy is None:
                execution_policy = RTCExecutionPolicy()
        elif session_options:
            raise ValueError("Async session options require asynchronous execution")
        initial = self._actions(self.reset(prompt=prompt, measurement=first_measurement))
        frequency = self.metadata["controller_hz"]
        rate = frequency if execution_hz is None else execution_hz
        if selected_mode == "async":
            session = AsyncRemoteSession(
                self,
                initial,
                execution_hz=rate,
                enable_rtc=enable_rtc,
                execution_policy=execution_policy if adaptive else None,
                execution_rate_callback=execution_rate_callback,
                **session_options,
            )
        else:
            session = BlockingRemoteSession(self, initial, execution_hz=rate)
        self._session = session
        return session

    def actions(self, response):
        from veomni.models.diffusers.causal_cosmos3.serving.serving_model import adapter_from_metadata

        return adapter_from_metadata(self.metadata).split_actions(_numpy(self._actions(response)))[self.control_mode]

    def close(self):
        if not self._closed:
            if self._transport_active:
                self._fail_transport(RuntimeError("Policy connection is closed"))
                try:
                    self._outbox.put_nowait(None)
                except Full:
                    pass
            else:
                self._closed = True
                self.connection.close()
        self._close_session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BlockingRemoteSession:
    """Ordinary baseline: predict H, execute s, then block for a new chunk."""

    def __init__(self, client, initial_actions, *, execution_hz):
        from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55

        if isinstance(execution_hz, bool) or not math.isfinite(execution_hz) or execution_hz <= 0:
            raise ValueError("execution_hz must be positive and finite")
        self.client, self._actions = client, initial_actions
        self.template = ActionTemplate55()
        self.controller_hz = client.metadata["controller_hz"]
        self.execution_hz = float(execution_hz)
        self.horizon = client.metadata.get("action_horizon", len(initial_actions))
        self.steps = client.metadata["execute_steps"]
        if type(self.steps) is not int or not 0 < self.steps <= len(initial_actions):
            raise ValueError("Invalid server execution horizon")
        self.simulation_sync = client.metadata.get("simulation_sync", False)
        self._executed_observations = []
        self._history = deque(
            maxlen=max(1, client.metadata.get("history_steps", client.metadata["history_blocks"] * 32))
        )
        self._tick = self._origin = self._requests = 0
        self._closed = False

    def prepare(self, **measurement):
        if self._closed:
            raise RuntimeError("Policy session is closed")
        if not self.simulation_sync:
            self.client.prepare_window(anchor_tick=self._origin + self.steps)
        return self.status()

    def get_action(self, *, state, state_mask, image, action_mask):
        from veomni.models.diffusers.causal_cosmos3.serving.rtc_session import RTCObservation

        self.prepare()
        if self.simulation_sync:
            if self._tick > self._origin:
                self._executed_observations.append(RTCObservation(self._tick, image.detach().cpu().clone()))
        else:
            self.client.observe(tick=self._tick, image=image)
        if self._tick - self._origin >= self.steps:
            extra = dict(executed_observations=tuple(self._executed_observations)) if self.simulation_sync else {}
            self._actions = self.client.plan(
                state=state,
                state_mask=state_mask,
                image=image,
                action_mask=action_mask,
                history=tuple(self._history),
                tick=self._tick,
                revision=0,
                rtc=None,
                **extra,
            )
            self._origin = self._tick
            self._requests += 1
            self._executed_observations.clear()
            if not self.simulation_sync:
                self.client.prepare_window(anchor_tick=self._origin + self.steps)
        action = self._actions[self._tick - self._origin].clone()
        self._history.append(RTCObservation(self._tick, image.detach().cpu().clone()))
        self._tick += 1
        return action

    def status(self):
        return dict(
            tick=self._tick,
            chunk_start_tick=self._origin,
            remaining_actions=max(0, self.steps - (self._tick - self._origin)),
            prediction_horizon=self.horizon,
            minimum_execution_steps=self.steps,
            controller_hz=self.controller_hz,
            execution_hz=self.execution_hz,
            rtc_enabled=False,
            execution_mode="simulation_sync" if self.simulation_sync else "blocking_non_rtc",
            prepared=True,
            closed=self._closed,
            error=None,
            ordinary_requests=self._requests,
            published_requests=0,
            inference_pending=False,
            adaptive_execution=False,
        )

    def close(self, *, wait=True):
        self._closed = True
