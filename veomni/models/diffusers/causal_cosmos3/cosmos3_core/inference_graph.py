# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded, session-owned graph replay for warm Policy denoising steps."""

import copy
import weakref

import torch


def _tensor_spec(tensor):
    return tensor.shape, tensor.dtype, tensor.device


class PolicyGraphReplay:
    """One exact-shape slot. Captured inputs never alias measured history.

    The first denoising step remains eager and binds the current state. Later
    steps copy that pass's packed KV, RoPE and residual into persistent buffers.
    Full and residual-only graphs use separate pools; their outputs stay alive.
    No callable or pass is retained, avoiding a history -> graph -> history cycle.
    """

    @staticmethod
    def signature(cache, inputs, refresh):
        tensors = (*inputs, cache.action_mask, *cache.tables)
        return (
            tuple(_tensor_spec(tensor) for tensor in tensors),
            tuple((_tensor_spec(layer.key), layer.start, layer.stop) for layer in cache.layers),
            refresh is not None,
        )

    def __init__(self, cache, inputs, refresh):
        from .inference_cache import ResidualCache

        self.spec = self.signature(cache, inputs, refresh)
        self.inputs = tuple(tensor.clone() for tensor in inputs)
        static = copy.copy(cache)
        static.npu_graph = False
        static.history = copy.copy(cache.history)
        static.history._policy_graph = None
        static.history._graph_capture = True
        static.action_mask = cache.action_mask.clone()
        static.tables = tuple(table.clone() for table in cache.tables)
        static.layers = [copy.copy(layer) for layer in cache.layers]
        for layer in static.layers:
            layer.key, layer.value = layer.key.clone(), layer.value.clone()
        static.residual = ResidualCache()
        self.action_mask, self.tables, self.layers = static.action_mask, static.tables, static.layers
        self.graphs, self.outputs = {}, {}
        self.residual = None

        # Each mode warms up and captures on the same side stream. The matching
        # warmup also supplies the first prediction, avoiding a redundant replay.
        stream = torch.npu.Stream(device=cache.history.device)
        for mode in [None] if refresh is None else [True, False]:
            if mode is False:
                # Reuse consumes either the caller's cold residual or the full
                # warmup's result. Capture alone does not populate this buffer.
                residual = cache.residual.residual if refresh is False else self.initial[1]
                self.residual.copy_(residual)
            stream.wait_stream(torch.npu.current_stream(cache.history.device))
            with torch.npu.stream(stream):
                outputs = static.predict(*self.inputs, residual_refresh=mode)
            if mode is refresh:
                self.initial = outputs, static.residual.residual
            torch.npu.current_stream(cache.history.device).wait_stream(stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                self.outputs[mode] = static.predict(*self.inputs, residual_refresh=mode)
            self.graphs[mode] = graph
            if mode is True:
                self.residual = static.residual.residual
        self.bound = weakref.ref(cache)

    def replay(self, cache, inputs, refresh):
        signature = None
        if self.residual is not None:
            signature = (id(cache), cache.revision, self.residual.shape, cache.history.dtype, cache.history.device)
            if refresh is False and (cache.residual.residual is None or cache.residual.signature != signature):
                raise ValueError("Residual cache is empty or does not match this block layout/model/precision")
        if self.initial is not None:
            outputs, residual = self.initial
            self.initial = None
            if refresh is True:
                cache.residual.residual, cache.residual.signature = residual, signature
            return outputs
        if self.bound() is not cache:
            self.action_mask.copy_(cache.action_mask)
            for target, source in zip(self.tables, cache.tables, strict=True):
                target.copy_(source)
            for target, source in zip(self.layers, cache.layers, strict=True):
                target.key.copy_(source.key)
                target.value.copy_(source.value)
            if self.residual is not None and cache.residual.residual is not None:
                self.residual.copy_(cache.residual.residual)
            self.bound = weakref.ref(cache)
        for target, source in zip(self.inputs, inputs, strict=True):
            target.copy_(source)
        self.graphs[refresh].replay()
        if refresh is True:
            # Pass ownership matters even if two callers alternate on this slot.
            cache.residual.residual = self.residual.clone()
            cache.residual.signature = signature
        # Preserve eager output ownership when callers retain previous predictions.
        return tuple(output.clone() for output in self.outputs[refresh])


def replay_policy(cache, video, action, timestep, refresh):
    if refresh is False and cache.residual.residual is None:
        raise ValueError("Residual cache is empty or does not match this block layout/model/precision")
    history = cache.history
    inputs = (video, action, timestep)
    spec = PolicyGraphReplay.signature(cache, inputs, refresh)
    slot = history._policy_graph
    if slot is None or slot.spec != spec:
        if slot is not None:
            # Drop the previous shape before allocating another graph pool.
            torch.npu.synchronize(history.device)
            history._policy_graph = None
            del slot
        slot = PolicyGraphReplay(cache, inputs, refresh)
        history._policy_graph = slot
        history.policy_graph_captures += 1
    return slot.replay(cache, inputs, refresh)
