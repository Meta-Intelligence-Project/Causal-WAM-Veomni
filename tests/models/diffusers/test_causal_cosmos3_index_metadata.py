"""CPU checks for host-side index selection and cache planning."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def core(monkeypatch):
    # Load only the core modules, without model registration or vendor imports.
    name = "_cosmos_index_test"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).parents[3] / "veomni/models/diffusers/causal_cosmos3/cosmos3_core")]
    monkeypatch.setitem(sys.modules, name, package)
    utils = ModuleType("veomni.utils.import_utils")
    utils.is_torch_npu_available = lambda: False
    monkeypatch.setitem(sys.modules, "veomni.utils.import_utils", utils)
    modules = [
        importlib.import_module(f"{name}.{part}") for part in ("index_metadata", "attention", "inference_cache")
    ]
    yield modules
    for key in list(sys.modules):
        if key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key)


def layout(attention):
    # Text, clean frames -1/0, current frames -1/0, two actions and one state.
    return attention.CausalLayout(
        sample_ids=(0,) * 9,
        block_ids=(-1, -1, 0, -1, 0, -1, 0, 0, 0),
        roles=(-1, 0, 0, 2, 1, 0, 1, 1, 3),
        modalities=(0, 1, 1, 1, 1, 1, 2, 2, 3),
        history_blocks=(2,),
        frame_ids=(-1, 0, 1, 0, 1, 0, -1, -1, -1),
        block_sizes=(1,),
    )


def test_mirror_invalidates_on_mutation_and_view_write(core):
    metadata, _, _ = core
    tensor = metadata.index_tensor([3, 1, 3])
    assert metadata.index_values(tensor) == (3, 1, 3)
    tensor[1:].add_(2)
    assert metadata.index_values(tensor) == (3, 3, 5)
    assert metadata.index_values(torch.tensor([4, 2])) == (4, 2)
    with torch.inference_mode():
        tensor = metadata.index_tensor([1, 2])
        tensor.add_(1)
        assert metadata.index_values(tensor) == (2, 3)


@pytest.mark.parametrize("limit", [3, 131072])
@pytest.mark.parametrize("partial", [False, True])
def test_query_selection_preserves_order_and_visibility(core, limit, partial):
    _, attention, _ = core
    source = layout(attention)
    plan = attention.prepare_attention(source, "cpu", "grouped", max_kv_tokens=limit)
    keep = [True] * len(plan.gen_indexes)
    if partial:
        keep[-2] = False
    selected = attention.select_attention_queries(plan, keep, allow_partial=partial)
    mask = attention.dense_generation_mask(source, "cpu")
    seen = []
    for chunk in selected.chunks:
        qs = ks = 0
        for qe, ke in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            queries = chunk.query_indexes[qs:qe].tolist()
            keys = chunk.kv_indexes[ks:ke].tolist()
            for query in queries:
                assert keep[query]
                assert keys == torch.where(mask[query])[0].tolist()
            seen.extend(queries)
            qs, ks = qe, ke
    assert seen == selected.query_selection.tolist()
    assert sorted(seen) == [i for i, use in enumerate(keep) if use]


def test_block_and_boundary_planning_need_no_index_readbacks(core, monkeypatch):
    metadata, attention, cache = core
    source = layout(attention)
    plan = attention.prepare_attention(source, "cpu", "grouped")
    original = torch.Tensor.tolist

    def guarded(tensor):
        raise AssertionError("Unexpected index readback during block planning")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "tolist", guarded)
        tables = (torch.zeros(len(plan.gen_indexes), 128), torch.ones(len(plan.gen_indexes), 128))
        result = cache.BlockCache.prepare(plan, source, 0, tables, compact=True)
        cache.BoundaryRows.prepare(((0, 1, 2, 3, 4), (5, 6), (7,)), result.dynamic)
    assert metadata.index_values(result.dynamic) == tuple(original(result.dynamic))
    assert result.direct_attention is not None


def test_partial_group_requires_explicit_opt_in(core):
    _, attention, _ = core
    plan = attention.prepare_attention(layout(attention), "cpu", "grouped")
    keep = [True] * len(plan.gen_indexes)
    keep[-2] = False
    with pytest.raises(ValueError, match="complete attention groups"):
        attention.select_attention_queries(plan, keep)


def test_history_binding_and_prefill_preserve_host_indexes(core, monkeypatch):
    metadata, attention, cache = core
    source = layout(attention)
    plan = attention.prepare_attention(source, "cpu", "grouped")
    layer = object()
    model = SimpleNamespace(
        config=SimpleNamespace(patch_size=2), layers=(layer,), parameters=lambda: iter((torch.empty(0),))
    )
    sample = dict(
        video_latents=torch.zeros(48, 2, 2, 2),
        text_token_ids=metadata.index_tensor([1]),
        block_size=1,
        history_blocks=2,
        conditioning_fps=6.0,
        conditioning_fps_action=24.0,
    )
    history = cache.HistoryCache()
    previous = history.bind(model, sample, 0, 0, True, plan, source, None)
    previous.following[layer] = object()
    with monkeypatch.context() as context:

        def fail(tensor):
            raise AssertionError("Unexpected index readback in history planning")

        context.setattr(torch.Tensor, "tolist", fail)
        current = history.bind(model, sample, 1, 1, True, plan, source, None)
        assert len(current.reused_rows) > 0
        tables = (torch.zeros(len(plan.gen_indexes), 128), torch.ones(len(plan.gen_indexes), 128))
        cache.BlockCache.prepare(plan, source, 1, tables, history=current, compact=True)
