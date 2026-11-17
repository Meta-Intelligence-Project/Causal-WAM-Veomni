"""CLI configuration contracts without loading model weights or accelerator runtimes."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tasks.client_server.server import server
from tasks.infer import infer_causal_cosmos3 as offline


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize(
    "entry",
    [
        ["tasks/infer/infer_causal_cosmos3.py"],
        ["-m", "tasks.infer.infer_causal_cosmos3"],
        ["-m", "tasks.client_server.server.server"],
    ],
)
def test_help_without_pythonpath(entry):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--npu-graph" in result.stdout and "--residual-reuse-full-steps" in result.stdout


@pytest.fixture(params=["offline", "server"])
def entrypoint(request, tmp_path):
    if request.param == "offline":
        return (
            offline,
            offline.build_parser(),
            ["--mode", "rollout", "--dataset", "unused", "--output", str(tmp_path / "output")],
        )
    return server, server.parser(), []


def capture_config(entrypoint, tmp_path, monkeypatch, config, flags):
    module, _, required = entrypoint
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(config))
    captured = {}

    class ConfigCaptured(Exception):
        pass

    def load(checkpoint, resolved):
        assert checkpoint == "unused"
        captured.update(copy.deepcopy(resolved))
        raise ConfigCaptured

    monkeypatch.setitem(
        sys.modules,
        "veomni.models.diffusers.causal_cosmos3.inference",
        SimpleNamespace(CausalCosmos3Pipeline=SimpleNamespace(from_pretrained=load), save_result=None),
    )
    monkeypatch.setitem(
        sys.modules, "tasks.client_server.server.session", SimpleNamespace(CausalInferenceSession=None)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: SimpleNamespace(type="cpu")))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", ["inference", "--config", str(path), "--checkpoint", "unused", *required, *flags])
    with pytest.raises(ConfigCaptured):
        module.main()
    return captured


def test_omitted_overrides_preserve_yaml(entrypoint, tmp_path, monkeypatch):
    config = dict(
        task="policy",
        device="cpu",
        dtype="bfloat16",
        fps=30,
        source=dict(name="custom"),
        sampling=dict(
            steps=10,
            history=32,
            seed=7,
            kv_cache=False,
            npu_graph=False,
            normalize_cfg=True,
            residual_reuse_full_steps=[0, 1, 6, 8],
            shift=2.0,
            guidance=3.0,
            negative_prompt="original",
            fixed_step_sampler_config=dict(t_list=[0.8], sample_type="sde"),
        ),
    )
    assert capture_config(entrypoint, tmp_path, monkeypatch, config, []) == config


def test_cli_overrides_reach_model_loader(entrypoint, tmp_path, monkeypatch):
    flags = [
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--attention-backend",
        "dense",
        "--fps",
        "60",
        "--statistics",
        "stats.json",
        "--tokenizer",
        "tokenizer",
        "--vae",
        "vae",
        "--base-fps",
        "25",
        "--temporal-modality-margin",
        "12000",
        "--steps",
        "12",
        "--history",
        "16",
        "--seed",
        "0",
        "--observation-mode",
        "window",
        "--no-kv-cache",
        "--no-npu-graph",
        "--residual-reuse-full-steps",
        "null",
        "--shift",
        "2",
        "--guidance",
        "3",
        "--guidance-interval",
        "200",
        "800",
        "--no-normalize-cfg",
        "--negative-prompt",
        "",
        "--fixed-step-sigmas",
        "0.9",
        "0.5",
    ]
    config = dict(
        task="policy",
        device="original",
        fps=30,
        source=dict(name="custom"),
        sampling=dict(
            kv_cache=True,
            npu_graph=True,
            normalize_cfg=True,
            residual_reuse_full_steps=[0, 1, 6, 8],
            negative_prompt="original",
        ),
    )
    resolved = capture_config(entrypoint, tmp_path, monkeypatch, config, flags)
    assert resolved == dict(
        task="policy",
        device="cpu",
        dtype="float32",
        attention_backend="dense",
        fps=60.0,
        statistics="stats.json",
        tokenizer="tokenizer",
        vae="vae",
        base_fps=25.0,
        temporal_modality_margin=12000,
        source=dict(name="custom"),
        sampling=dict(
            steps=12,
            history=16,
            seed=0,
            observation_mode="window",
            kv_cache=False,
            npu_graph=False,
            residual_reuse_full_steps=None,
            shift=2.0,
            guidance=3.0,
            guidance_interval=[200.0, 800.0],
            normalize_cfg=False,
            negative_prompt="",
            fixed_step_sampler_config=dict(t_list=[0.9, 0.5], sample_type="sde"),
        ),
    )


@pytest.mark.parametrize("value, expected", [("4", 4), ("[0, 1, 6, 8]", [0, 1, 6, 8]), ("null", None)])
def test_residual_schedule_overrides(entrypoint, tmp_path, monkeypatch, value, expected):
    config = dict(task="policy", device="cpu", fps=30, sampling=dict(residual_reuse_full_steps=[0, 2]))
    resolved = capture_config(entrypoint, tmp_path, monkeypatch, config, ["--residual-reuse-full-steps", value])
    assert resolved["sampling"]["residual_reuse_full_steps"] == expected


@pytest.mark.parametrize("override", [[], ["--task", "policy"]])
def test_task_uses_yaml_unless_overridden(entrypoint, tmp_path, monkeypatch, override):
    config = dict(device="cpu", fps=30, task="inverse_dynamics")
    resolved = capture_config(entrypoint, tmp_path, monkeypatch, config, override)
    assert resolved["task"] == ("policy" if override else "inverse_dynamics")


@pytest.mark.parametrize("flag", ["kv-cache", "npu-graph", "normalize-cfg"])
@pytest.mark.parametrize("enabled", [True, False])
def test_boolean_overrides(entrypoint, tmp_path, monkeypatch, flag, enabled):
    key = flag.replace("-", "_")
    config = dict(task="policy", device="cpu", fps=30, sampling={key: not enabled})
    resolved = capture_config(entrypoint, tmp_path, monkeypatch, config, [f"--{'' if enabled else 'no-'}{flag}"])
    assert resolved["sampling"][key] is enabled


@pytest.mark.parametrize(
    "flags",
    [
        ["--steps", "0"],
        ["--history", "-1"],
        ["--dtype", "int8"],
        ["--npu-graph", "false"],
        ["--residual-reuse-full-steps", "true"],
        ["--residual-reuse-full-steps", "0"],
        ["--residual-reuse-full-steps", "[1, 2]"],
        ["--residual-reuse-full-steps", "[0, 0]"],
        ["--residual-reuse-full-steps", "[0, true]"],
        ["--residual-reuse-full-steps", "["],
    ],
)
def test_invalid_shared_arguments_fail_during_parsing(entrypoint, flags):
    _, parser, required = entrypoint
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--checkpoint", "unused", *required, *flags])
    assert error.value.code == 2
