# SPDX-License-Identifier: OpenMDW-1.1
"""One model-local, bounded NPU graph for ordinary compact prefix replay.

Adapted from Cosmos gy_branch@9d448c7. Each request still owns its prefill.
Compatible requests refresh every fixed K/V and RoPE buffer before replay;
RTC retains the separate functional input-gradient path.
"""

import copy
import hashlib
import threading
import time
import weakref
from contextlib import contextmanager

import torch


_MODEL_ATTRIBUTE = "_causal_cosmos3_npu_replay_graph"


def _description(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device)


class _NPUBackend:
    def supports(self, tensor):
        return tensor.device.type == "npu"

    def synchronize(self, device):
        torch.npu.synchronize(device)

    def capture(self, forward, device):
        import torch_npu  # noqa: F401

        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            for _ in range(3):
                forward()
        torch.npu.current_stream(device).wait_stream(stream)
        self.synchronize(device)
        started = time.perf_counter()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = forward()
        self.synchronize(device)
        return graph, output, time.perf_counter() - started


class NPUReplayGraphCache:
    """Capture one exact native layout and retain every captured tensor owner.

    Enter ``request()`` for the whole sampler call. Weights, fixed inputs and
    routing are immutable within that request. Signature checks and device-index
    reads happen once per new request, never on every denoising step. The private
    backend argument is a CPU lifecycle-test seam, not a production fallback.
    """

    def __init__(self, model, *, _backend=None):
        self._model = weakref.ref(model)
        self._backend = _backend if _backend is not None else _NPUBackend()
        self._lock = threading.Lock()
        self._owner = None
        self._closed = False
        self._graph = self._signature = self._input_signature = self._entry = None
        self._capture_entry = self._inputs = self._layers = self._output = self._device = None
        self._stats = dict(
            captures=0,
            replays=0,
            requests=0,
            refreshes=0,
            invalidations=0,
            failures=0,
            capture_seconds=0.0,
            last_capture_seconds=0.0,
            last_invalidation=None,
            signature=None,
        )

    @property
    def stats(self):
        return {**self._stats, "has_graph": self._graph is not None, "closed": self._closed}

    @contextmanager
    def request(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or reentrant NPU graph requests are unsupported")
        try:
            if self._closed:
                raise RuntimeError("NPU replay graph cache is closed")
            self._owner = threading.get_ident()
            try:
                yield self
            except BaseException:
                if self._device is not None:
                    self._clear("request_failed")
                raise
        finally:
            self._owner = None
            self._lock.release()

    def _validate(self, entry, inputs):
        if self._owner != threading.get_ident():
            raise RuntimeError("Enter the NPU graph request scope before replay")
        model = self._model()
        if self._closed or model is None or model.training:
            raise ValueError("NPU replay graphs require a live eval-mode model")
        if entry.model is not model:
            raise ValueError("The prefix entry belongs to a different model")
        if entry.differentiable or torch.is_grad_enabled():
            raise ValueError("NPU replay graphs and RTC input gradients are mutually exclusive")
        if entry.plan.backend != "npu_tnd":
            raise ValueError("NPU replay graphs require npu_tnd attention")
        chunks = entry.compact.chunks
        if len(chunks) != 1 or len(chunks[0].query_ends) != 1:
            raise ValueError("NPU replay graphs require one current attention group")
        expected = {"video_tokens", "video_timesteps", "action_tokens", "action_timesteps", "action_mask"}
        if set(inputs) != expected or any(not self._backend.supports(value) for value in inputs.values()):
            raise ValueError("NPU replay requires the native compact input tensors on NPU")
        device = inputs["video_tokens"].device
        if any(value.device != device or value.requires_grad for value in inputs.values()):
            raise ValueError("NPU graph inputs must share a device and have no input gradients")
        if not entry.layers or any(
            key.device != device
            or layer["value"].device != device
            or key.dtype != layer["value"].dtype
            or key.shape != layer["value"].shape
            for layer in entry.layers.values()
            for key in (layer["key"],)
        ):
            raise ValueError("NPU replay requires complete prefilled native K/V")

    def _input_description(self, entry, inputs):
        device = inputs["video_tokens"].device.type
        return (
            tuple((name, _description(value)) for name, value in sorted(inputs.items())),
            tuple(_description(value) for value in entry.tables),
            id(entry.compact),
            torch.is_autocast_enabled(device),
            str(torch.get_autocast_dtype(device)),
            torch.is_inference_mode_enabled(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    def _entry_signature(self, entry, inputs):
        model = self._model()
        weights = tuple(
            (kind, name, _description(value), value.data_ptr(), value._version)
            for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers()))
            for name, value in values
        )
        routing = tuple(
            (
                tuple(chunk.query_indexes.detach().cpu().tolist()),
                tuple(chunk.kv_indexes.detach().cpu().tolist()),
                chunk.query_ends,
                chunk.kv_ends,
                chunk.query_span,
                chunk.kv_span,
            )
            for chunk in entry.compact.chunks
        )
        config = model.config
        return (
            entry.inputs["layout"],
            routing,
            tuple(entry.packed.detach().cpu().tolist()),
            tuple(entry.video_gen.detach().cpu().tolist()),
            tuple(entry.action_gen.detach().cpu().tolist()),
            self._input_description(entry, inputs)[:2],
            self._input_description(entry, inputs)[3:],
            tuple(
                (index, _description(layer["key"]), _description(layer["value"]))
                for index, layer in entry.layers.items()
            ),
            weights,
            tuple((name, id(module)) for name, module in model.named_modules()),
            (config.timestep_scale, config.hidden_size, config.action_interface),
        )

    def _clear(self, reason):
        error = None
        try:
            if self._device is not None:
                self._backend.synchronize(self._device)
        except Exception as exc:
            error = exc
        finally:
            self._graph = self._signature = self._input_signature = self._entry = None
            self._capture_entry = self._inputs = self._layers = self._output = self._device = None
            self._stats["signature"] = None
            self._stats["last_invalidation"] = reason
        return error

    def invalidate(self, reason="explicit_reset", *, _close=False):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot invalidate an NPU graph during sampling")
        try:
            if self._graph is not None:
                self._stats["invalidations"] += 1
            error = self._clear(reason)
            if _close:
                self._closed = True
            if error is not None:
                raise RuntimeError("Could not synchronize while clearing the NPU graph") from error
        finally:
            self._lock.release()

    def close(self):
        self.invalidate("closed", _close=True)

    def run(self, entry, inputs):
        self._validate(entry, inputs)
        description = self._input_description(entry, inputs)
        previous = None if self._entry is None else self._entry()
        new_entry = previous is not entry or description != self._input_signature
        signature = self._entry_signature(entry, inputs) if new_entry else self._signature
        if new_entry:
            self._stats["requests"] += 1
        if self._graph is not None and signature != self._signature:
            self._stats["invalidations"] += 1
            error = self._clear("layout_or_weights_changed")
            if error is not None:
                raise RuntimeError("Could not synchronize while replacing the NPU graph") from error
        try:
            if self._graph is None:
                self._device = inputs["video_tokens"].device
                self._signature, self._input_signature = signature, description
                self._entry = weakref.ref(entry)
                self._layers = {
                    index: {name: value.detach().clone() for name, value in layer.items()}
                    for index, layer in entry.layers.items()
                }
                static = self._capture_entry = copy.copy(entry)
                static.model = weakref.proxy(self._model())
                static.reference = static.inputs = static.graph_cache = None
                static.layers = self._layers
                static.tables = tuple(value.detach().clone() for value in entry.tables)
                self._inputs = {name: value.detach().clone() for name, value in inputs.items()}
                entry.layers = self._layers
                self._graph, self._output, elapsed = self._backend.capture(
                    lambda: static.current_forward(self._inputs), self._device
                )
                self._stats["captures"] += 1
                self._stats["capture_seconds"] += elapsed
                self._stats["last_capture_seconds"] = elapsed
                self._stats["signature"] = hashlib.sha256(repr(signature).encode()).hexdigest()
            elif new_entry:
                for index, layer in entry.layers.items():
                    for name, value in layer.items():
                        self._layers[index][name].copy_(value)
                for target, value in zip(self._capture_entry.tables, entry.tables, strict=True):
                    target.copy_(value)
                entry.layers = self._layers
                self._entry = weakref.ref(entry)
                self._input_signature = description
                self._stats["refreshes"] += 1
            for name, value in inputs.items():
                self._inputs[name].copy_(value)
            self._graph.replay()
            self._stats["replays"] += 1
            return self._output
        except Exception as exc:
            self._stats["failures"] += 1
            cleanup_error = self._clear("capture_or_replay_failed")
            detail = " Cleanup synchronization also failed." if cleanup_error is not None else ""
            raise RuntimeError(f"NPU graph capture or replay failed; cache cleared.{detail}") from exc


@contextmanager
def graph_request(model, enabled):
    """Serialize graph sampling while retaining a bounded graph across requests."""
    if not enabled:
        yield None
        return
    if next(model.parameters()).device.type != "npu":
        raise ValueError("NPU replay graphs require an Ascend NPU")
    cache = getattr(model, _MODEL_ATTRIBUTE, None)
    if cache is None or cache.stats["closed"]:
        cache = NPUReplayGraphCache(model)
        setattr(model, _MODEL_ATTRIBUTE, cache)
    with cache.request():
        yield cache


def clear_model_graph(model, *, close=False):
    cache = getattr(model, _MODEL_ATTRIBUTE, None)
    if cache is not None:
        if close:
            cache.close()
            delattr(model, _MODEL_ATTRIBUTE)
        else:
            cache.invalidate("episode_reset")
