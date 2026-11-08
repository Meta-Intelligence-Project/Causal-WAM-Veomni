"""Serialized episode contracts and rejection by independent-window entrypoints."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.diffusers.test_causal_cosmos3 import tiny_config
from tests.models.diffusers.test_causal_cosmos3_inference import ConstantVelocity
from tests.models.diffusers.test_causal_cosmos3_packing import payload_sample
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import Cosmos3Config
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import (
    Cosmos3Model,
    PreTrainedModel,
)
from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline, sample_block


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class UnreadPayload:
    def __getitem__(self, key):
        raise AssertionError("Rejected checkpoint must not read the sample/config")

    def __deepcopy__(self, memo):
        raise AssertionError("Rejected checkpoint must not copy the sample/config")


class UntouchedModel:
    def __init__(self, encoding):
        self.config = SimpleNamespace(history_encoding=encoding)

    def eval(self):
        raise AssertionError("Rejected checkpoint must not mutate the model")

    def parameters(self):
        raise AssertionError("Rejected checkpoint must not inspect model parameters")


@pytest.mark.parametrize("entry", ["pipeline", "sample_block"])
@pytest.mark.parametrize("encoding", ["episode_stream", "unknown", None, False])
def test_offline_window_entries_reject_before_reading_inputs_or_mutating_model(entry, encoding):
    model, payload = UntouchedModel(encoding), UnreadPayload()
    with pytest.raises(ValueError, match="Offline window inference requires a window checkpoint"):
        if entry == "pipeline":
            CausalCosmos3Pipeline(model, payload, payload, payload, payload)
        else:
            # Even invalid sampler settings must not bypass the model contract.
            sample_block(model, payload, 0, steps=0)


@pytest.mark.parametrize("entry", ["pipeline", "sample_block"])
@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_offline_window_entries_preserve_old_and_explicit_window_checkpoints(entry, explicit_metadata):
    model = ConstantVelocity()
    if explicit_metadata:
        model.config.history_encoding = "window"
    if entry == "pipeline":
        result = CausalCosmos3Pipeline(model, None, None, lambda _: [1, 2], dict(source={}))
        assert result.model is model and not model.training
        assert not next(model.parameters()).requires_grad
    else:
        sample = payload_sample("policy", actions=32)
        before = copy.deepcopy(sample)
        result = sample_block(model, sample, 0, steps=1)
        assert result["video_latents"].shape == sample["video_latents"].shape
        assert result["action"].shape == sample["action"].shape
        assert model.times
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)


INVALID_METADATA = [
    ("history_encoding", "unknown"),
    ("history_encoding", None),
    ("history_encoding", False),
    ("training_sampling_unit", "unknown"),
    ("training_sampling_unit", None),
    ("training_sampling_unit", False),
]


@pytest.mark.parametrize("field,value", INVALID_METADATA)
@pytest.mark.parametrize("source", ["json", "hf_override"])
def test_checkpoint_metadata_rejects_invalid_serialized_values_and_hf_overrides(tmp_path, field, value, source):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    if source == "json":
        path = tmp_path / "config.json"
        saved = json.loads(path.read_text())
        saved[field] = value
        path.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match=field):
            Cosmos3Config.from_json_file(path)
    else:
        with pytest.raises(ValueError, match=field):
            Cosmos3Config.from_pretrained(tmp_path, **{field: value})


@pytest.mark.parametrize("field,value", INVALID_METADATA)
def test_model_constructor_revalidates_late_metadata_before_allocating_or_mutating_model(monkeypatch, field, value):
    config = tiny_config()
    setattr(config, field, value)

    def reject_base_init(*args, **kwargs):
        raise AssertionError("Invalid metadata must fail before constructing the model")

    monkeypatch.setattr(PreTrainedModel, "__init__", reject_base_init)
    with pytest.raises(ValueError, match=field):
        Cosmos3Model(config)


@pytest.mark.parametrize(
    "encoding,unit,enabled", [("window", "independent_window", False), ("episode_stream", "episode_lane_fifo", True)]
)
def test_checkpoint_history_and_training_sampling_metadata_roundtrip(tmp_path, encoding, unit, enabled):
    config = tiny_config(history_encoding=encoding, training_sampling_unit=unit, enable_tbptt=enabled)
    config.save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    restored = Cosmos3Config.from_pretrained(tmp_path)
    for key, value in (
        ("history_encoding", encoding),
        ("training_sampling_unit", unit),
        ("enable_tbptt", enabled),
        ("tbptt_segment_steps", 256),
    ):
        assert raw[key] == getattr(restored, key) == value
    assert "weight_loading" not in raw


def test_old_checkpoint_without_episode_metadata_preserves_independent_window_defaults():
    raw = tiny_config().to_dict()
    raw.pop("history_encoding")
    raw.pop("training_sampling_unit")
    original = copy.deepcopy(raw)
    config = Cosmos3Config.from_dict(raw)
    assert config.history_encoding == "window"
    assert config.training_sampling_unit == "independent_window"
    assert raw == original


@pytest.mark.parametrize("return_unused_kwargs", [False, True])
def test_valid_hf_overrides_apply_and_revalidate_episode_metadata_without_mutating_saved_dict(return_unused_kwargs):
    raw = tiny_config().to_dict()
    original = copy.deepcopy(raw)
    result = Cosmos3Config.from_dict(
        raw,
        history_encoding="episode_stream",
        training_sampling_unit="episode_lane_fifo",
        enable_tbptt=True,
        tbptt_segment_steps=128,
        return_unused_kwargs=return_unused_kwargs,
    )
    config = result[0] if return_unused_kwargs else result
    assert config.history_encoding == "episode_stream" and config.training_sampling_unit == "episode_lane_fifo"
    assert config.enable_tbptt is True and config.tbptt_segment_steps == 128
    if return_unused_kwargs:
        assert result[1] == {}
    assert raw == original
