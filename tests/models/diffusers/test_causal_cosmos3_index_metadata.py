"""CPU checks for host-side index selection and cache planning."""

import ast
import importlib
import sys
from dataclasses import fields, is_dataclass, replace
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


@pytest.mark.parametrize("size", [0, 1, 9, 64, 1000])
def test_bulk_row_assembly_matches_scatter_and_owns_storage(core, size):
    metadata, _, cache = core
    generator = torch.Generator().manual_seed(341 + size)
    destinations = torch.randperm(size, generator=generator)
    placements, sources = [], []
    expected = torch.zeros(size, 3)
    for source in range(3):
        dests = destinations[source::4].tolist()
        rows = torch.randperm(len(dests), generator=generator).tolist()
        values = torch.randn(len(dests), 3, generator=generator)
        placements.append((dests, rows))
        sources.append(values)
        expected[dests] = values[rows]
    assembly = cache._RowAssembly.prepare(size, placements, metadata.index_tensor)
    result = assembly.assemble(*sources)
    assert torch.equal(result, expected)
    if size:
        result.add_(1)
        for actual, values in zip(sources, placements, strict=True):
            dests, rows = values
            assert torch.equal(expected[dests], actual[rows])


@pytest.mark.parametrize(
    "placements",
    [
        [([0, 0], [0, 1])],
        [([0], [0]), ([0], [0])],
        [([-1], [0])],
        [([4], [0])],
        [([0], [])],
    ],
)
def test_bulk_row_assembly_rejects_invalid_placements(core, placements):
    metadata, _, cache = core
    with pytest.raises(ValueError):
        cache._RowAssembly.prepare(4, placements, metadata.index_tensor)


def test_structure_cache_is_bounded_and_keys_layout_device_backend(core, monkeypatch):
    _, attention, cache = core
    owner, other = SimpleNamespace(), SimpleNamespace()
    original = cache.prepare_attention
    calls = []

    def prepare(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(cache, "prepare_attention", prepare)
    source = layout(attention)
    first = cache.prepare_denoise_structure(owner, source, "cpu", "grouped")
    assert first is cache.prepare_denoise_structure(owner, replace(source), torch.device("cpu"), "grouped")
    assert len(calls) == 1
    assert cache.prepare_denoise_structure(other, source, "cpu", "grouped") is not first
    dense = cache.prepare_denoise_structure(owner, source, "cpu", "dense")
    assert dense is not first and dense.attention.mask is not None
    changed = replace(source, history_blocks=(1,))
    current = cache.prepare_denoise_structure(owner, changed, "cpu", "grouped")
    assert owner._denoise_structure is current and current is not first
    assert not hasattr(current, "tables") and not hasattr(current, "history")
    device_changed = cache.prepare_denoise_structure(owner, changed, "meta", "grouped")
    assert device_changed.device == torch.device("meta") and device_changed is not current


def test_model_preparation_reuses_only_structure_and_recomputes_rope(core):
    _, attention, cache = core
    path = (
        Path(__file__).parents[3]
        / "veomni/models/diffusers/causal_cosmos3/cosmos3_transformer/modeling_cosmos3_transformer.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "prepare_denoise_packing"
    )
    calls = []

    def rotary(positions, config, dtype):
        calls.append(positions)
        values = positions[0, :, None].expand(-1, 128).to(dtype)
        return values, values + 1

    namespace = dict(
        torch=torch,
        replace=replace,
        SimpleNamespace=SimpleNamespace,
        prepare_denoise_structure=cache.prepare_denoise_structure,
        rotary_tables=rotary,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    owner = SimpleNamespace(training=False, config=SimpleNamespace())
    source = layout(attention)
    positions = torch.zeros(3, len(source.roles))
    inputs = dict(layout=source, position_ids=positions, video_tokens=torch.zeros(1, 192))
    with torch.no_grad():
        first = namespace["prepare_denoise_packing"](owner, inputs, "grouped")
        inputs = dict(inputs, position_ids=positions + 5)
        second = namespace["prepare_denoise_packing"](owner, inputs, "grouped")
    assert len(calls) == 2
    assert first.rows is second.rows and first.gen_assembly is second.gen_assembly
    assert first.attention is not second.attention
    first.attention.backend = "dense"
    assert owner._denoise_structure.attention.backend == "grouped"
    assert second.fixed_attention is None
    assert second.positions is inputs["position_ids"]
    assert torch.equal(second.tables[1][0], first.tables[1][0] + 5)


def assert_same_plan(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif is_dataclass(actual):
        for item in fields(actual):
            assert_same_plan(getattr(actual, item.name), getattr(expected, item.name))
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_same_plan(a, b)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_same_plan(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
def test_block_template_matches_fresh_plan_and_isolates_state(core, monkeypatch, compact, with_history, boundary):
    metadata, attention, cache = core
    source = layout(attention)
    plan = attention.prepare_attention(source, "cpu", "grouped")
    owner = SimpleNamespace()
    modality_rows = ((0, 1, 2, 3, 4), (5, 6), (7,)) if boundary else None

    def history(reused=(0,)):
        return (
            None
            if not with_history
            else cache.HistoryBinding(
                {}, {}, metadata.index_tensor([0, 1, 4]), metadata.index_tensor(reused), metadata.index_tensor(reused)
            )
        )

    tables = (torch.zeros(8, 128), torch.ones(8, 128))
    expected = cache.BlockCache.prepare(plan, source, 0, tables, history=history(), compact=compact)
    first, bounds = cache.prepare_block_cache(
        owner, plan, source, 0, tables, history=history(), compact=compact, modality_rows=modality_rows
    )
    assert_same_plan(first, expected)
    template = owner._denoise_block_template
    assert template.cache.layers == {} and template.cache.history is None
    assert template.cache.dynamic_tables == () and template.cache.prefill.tables == ()
    first.layers["layer"] = torch.ones(1)
    first.compact_plan.chunks = ()
    first.prefill.plan.chunks = ()
    if first.history is not None:
        first.history.previous["old"] = torch.ones(1)
    current_tables = tuple(table + 7 for table in tables)
    expected = cache.BlockCache.prepare(plan, source, 0, current_tables, history=history(), compact=compact)
    original = cache.BlockCache.prepare

    def fail(*args, **kwargs):
        raise AssertionError("Repeated block structure should not be rebuilt")

    with monkeypatch.context() as context:
        context.setattr(cache.BlockCache, "prepare", fail)
        current_history = history()
        second, next_bounds = cache.prepare_block_cache(
            owner,
            replace(plan),
            source,
            0,
            current_tables,
            history=current_history,
            compact=compact,
            modality_rows=modality_rows,
        )
    assert_same_plan(second, expected)
    assert second.history is current_history and second.layers == {}
    assert next_bounds is bounds
    assert second is not first and second.prefill is not first.prefill
    assert cache.BlockCache.prepare == original


def test_block_template_invalidates_on_history_mapping_and_source_plan(core):
    metadata, attention, cache = core
    source = layout(attention)
    plan = attention.prepare_attention(source, "cpu", "grouped")
    owner = SimpleNamespace()
    tables = (torch.zeros(8, 128), torch.ones(8, 128))
    cache.prepare_block_cache(owner, plan, source, 0, tables)
    first = owner._denoise_block_template
    history = cache.HistoryBinding(
        {}, {}, metadata.index_tensor([0, 1, 4]), metadata.index_tensor([0]), metadata.index_tensor([0])
    )
    cache.prepare_block_cache(owner, plan, source, 0, tables, history=history)
    second = owner._denoise_block_template
    assert second is not first
    history = cache.HistoryBinding(
        {}, {}, metadata.index_tensor([0, 1, 4]), metadata.index_tensor([0]), metadata.index_tensor([1])
    )
    cache.prepare_block_cache(owner, plan, source, 0, tables, history=history)
    assert owner._denoise_block_template is not second
    old = owner._denoise_block_template
    fresh_plan = attention.prepare_attention(source, "cpu", "grouped")
    cache.prepare_block_cache(owner, fresh_plan, source, 0, tables, history=history)
    assert owner._denoise_block_template is not old
