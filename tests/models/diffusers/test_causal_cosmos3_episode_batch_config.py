"""Packed-episode opt-in, native singleton wiring and reader shutdown boundaries."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.causal_cosmos3.datasets import pipeline
from veomni.trainer import causal_cosmos3 as module
from veomni.trainer.base import BackgroundPrefetcher, VeOmniIter
from veomni.trainer.dit_trainer import DiTTrainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@dataclass
class WorkerOptions:
    type: str = "native"
    use_background_prefetcher: bool = True
    num_workers: int = 2
    in_order: bool = True
    persistent_workers: bool = False


def trainer_args(*, batch=2, accum=1, stage="action", tbptt=True, workers=2, global_batch=None):
    args = SimpleNamespace(
        model=SimpleNamespace(config_path="unused", model_config={}, condition_model_cfg={}),
        data=SimpleNamespace(max_seq_len=54000, dyn_bsz_buffer_size=8, dataloader=WorkerOptions(num_workers=workers)),
        train=SimpleNamespace(
            training_task="online_training",
            micro_batch_size=batch,
            gradient_accumulation_steps=accum,
            global_batch_size=batch * accum * 2 if global_batch is None else global_batch,
            global_rank=0,
            dataloader_batch_size=batch * accum,
            max_steps=30,
            dyn_bsz_runtime="main",
            seed=42,
            checkpoint=SimpleNamespace(save_steps=0, output_dir="unused"),
        ),
        train_steps=30,
    )
    config = SimpleNamespace(
        action_interface="shared" if stage == "action" else "none",
        action_dim=55 if stage == "action" else 0,
        state_dim=55 if stage == "action" else 0,
        enable_tbptt=tbptt,
        tbptt_segment_steps=256,
        tbptt_segment_latents=16,
    )
    trainer = module.CausalCosmos3Trainer.__new__(module.CausalCosmos3Trainer)
    trainer.tokenize = lambda text: [1, 2]
    trainer.dataset_config = dict(stage=stage, geometry={}, sample=dict(action_horizon=32))
    trainer.segment_memories = {}
    trainer._episode_batch_loader = None
    trainer.base = SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            model=SimpleNamespace(config=config), parallel_state=SimpleNamespace(dp_size=2, dp_group=None)
        ),
        train_dataset=SimpleNamespace(episodes=(0, 1, 2)),
    )
    return trainer, config


def mock_setup(monkeypatch, config):
    calls = []
    monkeypatch.setattr(module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(module, "build_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(sp_size=1, dp_size=2, dp_group=None))
    monkeypatch.setattr(DiTTrainer, "_setup", lambda self: calls.append(True))
    return calls


@pytest.mark.parametrize("batch,accum", [(1, 2), (2, 1), (3, 2)])
def test_action_tbptt_batch_size_opt_in_preserves_segment_accounting(monkeypatch, batch, accum):
    trainer, config = trainer_args(batch=batch, accum=accum, workers=4)
    calls = mock_setup(monkeypatch, config)
    trainer._setup()
    args = trainer.base.args
    assert calls == [True] and trainer.episode_training
    assert args.train.global_batch_size == batch * accum * 2
    assert args.train.dataloader_batch_size == (accum if batch == 1 else 1)
    assert not args.train.dyn_bsz
    assert args.model.model_config["history_encoding"] == "episode_stream"


@pytest.mark.parametrize("stage,tbptt", [("action", False), ("video", False), ("video", True)])
def test_other_training_paths_still_reject_multiple_packs(monkeypatch, stage, tbptt):
    trainer, config = trainer_args(stage=stage, tbptt=tbptt)
    calls = mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        trainer._setup()
    assert calls == []


@pytest.mark.parametrize("batch", [True, 0, -1, 1.5])
def test_episode_microbatch_requires_positive_integer(monkeypatch, batch):
    trainer, config = trainer_args(batch=batch)
    calls = mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="positive integer"):
        trainer._setup()
    assert calls == []


def test_global_batch_must_cover_complete_microbatches(monkeypatch):
    trainer, config = trainer_args(global_batch=6)
    mock_setup(monkeypatch, config)
    with pytest.raises(ValueError, match="micro_batch_size.*dp_size"):
        trainer._setup()


@pytest.mark.parametrize("setting", [None, "0"])
def test_npu_episode_batch_rejects_matmul_k_shift_before_model_build(monkeypatch, setting):
    trainer, config = trainer_args()
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(module, "get_device_type", lambda: "npu")
    if setting is None:
        monkeypatch.delenv("CLOSE_MATMUL_K_SHIFT", raising=False)
    else:
        monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", setting)
    with pytest.raises(ValueError, match="CLOSE_MATMUL_K_SHIFT=1 before starting Python"):
        trainer._setup()


@pytest.mark.parametrize("batch,setting", [(1, None), (2, "1")])
def test_npu_single_stream_or_stable_packed_matmul_setup(monkeypatch, batch, setting):
    trainer, config = trainer_args(batch=batch)
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(module, "get_device_type", lambda: "npu")
    if setting is None:
        monkeypatch.delenv("CLOSE_MATMUL_K_SHIFT", raising=False)
    else:
        monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", setting)
    trainer._setup()
    assert trainer.episode_training


@pytest.mark.parametrize("batch,accum", [(1, 2), (2, 1)])
def test_worker_loader_reads_singletons_before_independent_lane_grouping(monkeypatch, batch, accum):
    trainer, _ = trainer_args(batch=batch, accum=accum)
    trainer.episode_training = True
    captured = {}
    source = SimpleNamespace(set_epoch=lambda epoch: None)

    def build(**kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr(module, "build_native_dataloader", build)
    trainer._build_dataloader()
    assert captured["micro_batch_size"] == 1 and not captured["dyn_bsz"]
    if batch == 1:
        assert captured["global_batch_size"] == 4 and captured["dataloader_batch_size"] == 2
        assert trainer._episode_batch_loader is None
        assert trainer.base.train_dataloader.dataloader is source
    else:
        assert captured["global_batch_size"] == 2 and captured["dataloader_batch_size"] == 1
        assert trainer._episode_batch_loader.batch_size == 2
        assert trainer._episode_batch_loader.num_micro_batches == 1
        assert trainer._episode_batch_loader.lane_ids == (0, 1)
        assert trainer.base.train_dataloader.dataloader is trainer._episode_batch_loader
    assert trainer.base.train_dataloader.state_dict() is None


@pytest.mark.parametrize("workers,episodes", [(0, 3), (1, 3), (4, 1)])
def test_insufficient_rank_lanes_fail_before_worker_iteration(monkeypatch, workers, episodes):
    trainer, config = trainer_args(workers=workers)
    trainer.episode_training = True
    trainer.base.data_transform = trainer.tokenize
    mock_setup(monkeypatch, config)
    monkeypatch.setattr(
        pipeline, "build_episode_dataset", lambda *args, **kwargs: SimpleNamespace(episodes=tuple(range(episodes)))
    )
    with pytest.raises(ValueError, match="active lanes per rank"):
        trainer._build_dataset()


def test_epoch_end_stops_prefetch_before_closing_owned_workers(monkeypatch):
    trainer, _ = trainer_args()
    order = []
    trainer.base.data_iterator = SimpleNamespace(stop=lambda timeout=None: order.append(("stop", timeout)))
    trainer._episode_batch_loader = SimpleNamespace(close=lambda: order.append(("close", None)))
    monkeypatch.setattr(DiTTrainer, "on_epoch_end", lambda self: order.append(("callbacks", None)))
    trainer.on_epoch_end()
    assert order == [("callbacks", None), ("stop", None), ("stop", None), ("close", None)]


def test_slow_read_is_joined_and_late_rgb_update_drained_before_loader_close(monkeypatch):
    started, release, reading, closed = Event(), Event(), Event(), Event()

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            reading.set()
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test did not release its finite read")
            reading.clear()
            return {"raw_rgb_placeholder": object()}

    trainer, _ = trainer_args()
    iterator = VeOmniIter(Source(), use_background_prefetcher=True)
    assert started.wait(timeout=2)
    trainer.base.data_iterator = iterator
    timeouts, errors = [], []
    original_stop = BackgroundPrefetcher.stop

    def stop(self, timeout=5.0):
        timeouts.append(timeout)
        # Make a mistaken timed stop fail quickly without a five-second test.
        return original_stop(self, timeout=None if timeout is None else 0)

    def close():
        assert not reading.is_set()
        assert not iterator.iterator.thread.is_alive()
        assert iterator.iterator.queue.empty()
        closed.set()

    def shutdown():
        try:
            trainer._close_episode_loader()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(BackgroundPrefetcher, "stop", stop)
    trainer._episode_batch_loader = SimpleNamespace(close=close)
    thread = Thread(target=shutdown)
    thread.start()
    try:
        assert iterator.iterator.stop_event.wait(timeout=1)
        assert thread.is_alive() and not closed.is_set()
        release.set()
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert closed.is_set() and timeouts == [None, None]
    finally:
        release.set()
        thread.join(timeout=3)
        original_stop(iterator.iterator, timeout=None)


@pytest.fixture
def policy_launcher(tmp_path):
    """Exercise Bash and real argument assembly with a Python stub, never torchrun."""
    repo = Path(__file__).resolve().parents[3]
    model = tmp_path / "model root"
    model.mkdir()
    source = tmp_path / "source data.yaml"
    source.write_text("sampling: sample_uniform\nsources:\n  - name: fixture\n    roots: [/not-read-by-launcher]\n")
    conda = tmp_path / "conda"
    profile = conda / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    prefix = conda / "envs/mock-env"
    profile.write_text(
        f"conda() {{ export POLICY_MOCK_CONDA=1 CONDA_PREFIX={shlex.quote(str(prefix))}; "
        f'export PATH={shlex.quote(str(prefix / "bin"))}:"$PATH"; }}\n'
    )
    ascend = tmp_path / "ascend.sh"
    ascend.write_text("export POLICY_MOCK_ASCEND=1\n")
    python = prefix / "bin/mock_python"
    python.parent.mkdir(parents=True)
    record = tmp_path / "called.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['POLICY_MOCK_RECORD']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'conda': os.environ.get('POLICY_MOCK_CONDA'), "
        "'ascend': os.environ.get('POLICY_MOCK_ASCEND'), 'kshift': os.environ.get('CLOSE_MATMUL_K_SHIFT'), "
        "'wandb_mode': os.environ.get('WANDB_MODE'), 'wandb_dir': os.environ.get('WANDB_DIR')}))\n"
        "print('POLICY_MOCK_STDOUT node=' + os.environ.get('NODE_RANK', '0'), flush=True)\n"
        "print('POLICY_MOCK_STDERR', file=sys.stderr, flush=True)\n"
        "sys.exit(int(os.environ.get('POLICY_MOCK_EXIT', '0')))\n"
    )
    python.chmod(0o755)
    env = dict(os.environ)
    for name in (
        "NPROC_PER_NODE",
        "NNODES",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "DP_SHARD_SIZE",
        "MA_NUM_GPUS",
        "MA_NUM_HOSTS",
        "VC_WORKER_HOSTS",
        "VC_TASK_INDEX",
        "ASCEND_RT_VISIBLE_DEVICES",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
        "LOG_DIR",
        "LOG_TIME",
        "DRY_RUN",
        "CLOSE_MATMUL_K_SHIFT",
        "WANDB_DIR",
        "WANDB_MODE",
        "POLICY_MOCK_EXIT",
        "CONDA_PREFIX",
        "TRAIN_CONFIG",
    ):
        env.pop(name, None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        MODEL_PATH=str(model),
        DATA_CONFIG=str(source),
        OUTPUT_DIR=str(tmp_path / "output"),
        CONDA_HOME=str(conda),
        CONDA_ENV="mock-env",
        ASCEND_ENV=str(ascend),
        PYTHON_BIN=str(python),
        POLICY_MOCK_RECORD=str(record),
        LOG_TIME="launcher-test",
    )

    def launch(options=(), **overrides):
        child = {**env, **{key: str(value) for key, value in overrides.items()}}
        return subprocess.run(
            [
                "bash",
                str(repo / "scripts/causal_cosmos3/training_scripts/train_policy_robotwin_tbptt_modelarts.sh"),
                *options,
            ],
            env=child,
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=20,
        )

    return SimpleNamespace(repo=repo, env=env, run=launch, record=record, output=tmp_path / "output")


def _policy_dry_command(result):
    assert result.returncode == 0, result.stderr + result.stdout
    for line in result.stdout.splitlines():
        words = shlex.split(line)
        if "-m" in words:
            index = words.index("-m")
            if words[index + 1 : index + 2] == ["torch.distributed.run"]:
                return words[index - 1 :]
    raise AssertionError(f"No shell-quoted torchrun command: {result.stdout}")


def _policy_option(command, option):
    option = option.replace("_", "-")
    for index, word in enumerate(command):
        name, separator, value = word.partition("=")
        if name.replace("_", "-") == "--" + option:
            return value if separator else command[index + 1]
    raise AssertionError(f"Missing {option}: {command}")


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"MA_NUM_GPUS": "4"}, "4"),
        ({"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", "MA_NUM_GPUS": "8"}, "4"),
        ({"NPROC_PER_NODE": "2", "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3", "MA_NUM_GPUS": "8"}, "2"),
        ({}, "8"),
    ],
)
def test_policy_modelarts_launcher_local_rank_precedence_and_dry_run(policy_launcher, overrides, expected):
    result = policy_launcher.run(
        DRY_RUN=1, CONDA_HOME="/must-not-be-sourced", ASCEND_ENV="/must-not-be-sourced", **overrides
    )
    command = _policy_dry_command(result)
    assert _policy_option(command, "nproc_per_node") == expected
    assert "--standalone" in command and command[0] == policy_launcher.env["PYTHON_BIN"]
    assert not policy_launcher.output.exists() and not policy_launcher.record.exists()


def test_policy_modelarts_launcher_resolves_multinode_platform_metadata(policy_launcher):
    result = policy_launcher.run(
        DRY_RUN=1, MA_NUM_GPUS=4, MA_NUM_HOSTS=2, VC_TASK_INDEX=1, VC_WORKER_HOSTS="worker0,worker1"
    )
    command = _policy_dry_command(result)
    assert _policy_option(command, "nproc_per_node") == "4"
    assert _policy_option(command, "nnodes") == "2" and _policy_option(command, "node_rank") == "1"
    assert _policy_option(command, "master_addr") == "worker0"
    assert _policy_option(command, "master_port") == "50012" and "--standalone" not in command


def test_policy_modelarts_launcher_explicit_cluster_values_and_cli_win(policy_launcher):
    options = ["--train.max_steps", "3", "--train.micro_batch_size", "1", "--enable-tbptt", "false"]
    result = policy_launcher.run(
        options,
        DRY_RUN=1,
        NPROC_PER_NODE=4,
        NNODES=2,
        NODE_RANK=1,
        MASTER_ADDR="explicit-master",
        MA_NUM_GPUS=8,
        MA_NUM_HOSTS=3,
        VC_TASK_INDEX=0,
        VC_WORKER_HOSTS="worker0,worker1,worker2",
    )
    command = _policy_dry_command(result)
    assert _policy_option(command, "nnodes") == "2" and _policy_option(command, "node_rank") == "1"
    assert _policy_option(command, "master_addr") == "explicit-master"
    assert command[-len(options) :] == options


@pytest.mark.parametrize(
    "overrides",
    [
        {"NNODES": "2", "MASTER_ADDR": "worker0"},
        {"NNODES": "2", "NODE_RANK": "0"},
        {"NNODES": "2", "NODE_RANK": "0", "MASTER_ADDR": "127.0.0.1"},
        {"NNODES": "2", "NODE_RANK": "0", "MASTER_ADDR": "localhost"},
    ],
)
def test_policy_modelarts_launcher_rejects_incomplete_multinode_topology(policy_launcher, overrides):
    result = policy_launcher.run(DRY_RUN=1, NPROC_PER_NODE=4, **overrides)
    assert result.returncode != 0 and not policy_launcher.record.exists()
    assert not policy_launcher.output.exists()


@pytest.mark.parametrize(
    "world,local,topology",
    [
        (4, 4, {"NPROC_PER_NODE": "4"}),
        (32, 8, {"NPROC_PER_NODE": "8", "NNODES": "4", "NODE_RANK": "2", "MASTER_ADDR": "worker0"}),
    ],
)
def test_policy_modelarts_launcher_command_reaches_real_typed_policy_recipe(
    policy_launcher, monkeypatch, world, local, topology
):
    from tasks.train_causal_cosmos3 import parse_causal_cosmos3_args

    # Keep CPU backend validation real, matching existing native-recipe parser tests.
    cpu_ops = [
        "--model.ops_implementation.rms_norm_implementation",
        "eager",
        "--model.ops_implementation.rotary_pos_emb_implementation",
        "eager",
    ]
    command = _policy_dry_command(policy_launcher.run(cpu_ops, DRY_RUN=1, **topology))
    task_index = next(i for i, value in enumerate(command) if value.endswith("tasks/train_causal_cosmos3.py"))
    assert Path(command[task_index + 1]).name == "causal_cosmos3_edge_policy_tbptt.yaml"
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local))
    args, dataset = parse_causal_cosmos3_args(command[task_index + 1 :])
    assert dataset["stage"] == "action" and dataset["sample"]["mode"] == "policy"
    assert (dataset["sample"]["history_steps"], dataset["sample"]["action_horizon"]) == (1024, 32)
    assert args.model.model_config["enable_tbptt"] is True
    assert args.model.model_config["tbptt_segment_steps"] == 256
    assert args.model.model_config["video_loss_weight"] == 1
    assert args.model.model_config["action_loss_weight"] == 1
    assert args.train.wandb.enable is True
    assert args.train.micro_batch_size == 2 and args.train.gradient_accumulation_steps == 1
    assert args.model.accelerator.dp_size == world
    assert args.model.accelerator.dp_shard_size == local
    assert args.model.accelerator.dp_replicate_size == world // local
    assert args.train.global_batch_size == world * 2
    assert args.model.accelerator.gradient_checkpointing.selective_recompute_layers == 28
    assert args.data.dataloader.num_workers == 4 and args.data.train_path == policy_launcher.env["DATA_CONFIG"]


@pytest.mark.parametrize("exit_code", [0, 23])
def test_policy_modelarts_launcher_normal_logs_and_preserves_python_exit(policy_launcher, exit_code):
    result = policy_launcher.run(NPROC_PER_NODE=4, POLICY_MOCK_EXIT=exit_code)
    assert result.returncode == exit_code, result.stdout + result.stderr
    recorded = json.loads(policy_launcher.record.read_text())
    assert recorded["argv"][:2] == ["-m", "torch.distributed.run"]
    assert recorded["conda"] == recorded["ascend"] == "1"
    assert recorded["kshift"] == "1"
    assert recorded["wandb_mode"] == "offline"
    assert recorded["wandb_dir"] == policy_launcher.env["OUTPUT_DIR"]
    logs = [path for path in (policy_launcher.output / "logs").iterdir() if path.is_file()]
    assert len(logs) == 1
    assert "launcher-test" in logs[0].name
    assert "POLICY_MOCK_STDOUT" in logs[0].read_text() and "POLICY_MOCK_STDERR" in logs[0].read_text()


def test_policy_modelarts_launcher_each_node_owns_a_separate_log(policy_launcher):
    for rank in (0, 1):
        result = policy_launcher.run(NPROC_PER_NODE=4, NNODES=2, NODE_RANK=rank, MASTER_ADDR="worker0")
        assert result.returncode == 0, result.stdout + result.stderr
    logs = [path for path in (policy_launcher.output / "logs").iterdir() if path.is_file()]
    assert len(logs) == 2 and len({path.name for path in logs}) == 2
    assert any("POLICY_MOCK_STDOUT node=0" in path.read_text() for path in logs)
    assert any("POLICY_MOCK_STDOUT node=1" in path.read_text() for path in logs)


def test_policy_modelarts_launcher_rejects_python_from_another_conda_prefix(policy_launcher, tmp_path):
    foreign = tmp_path / "foreign-env/bin/python"
    foreign.parent.mkdir(parents=True)
    foreign.write_text(Path(policy_launcher.env["PYTHON_BIN"]).read_text())
    foreign.chmod(0o755)
    result = policy_launcher.run(NPROC_PER_NODE=4, PYTHON_BIN=str(foreign))
    assert result.returncode != 0 and "PYTHON_BIN" in result.stderr
    assert not policy_launcher.record.exists() and not policy_launcher.output.exists()


@pytest.mark.parametrize("environment", ["base", "prefix", "named"])
def test_policy_modelarts_dry_run_infers_python_candidate_without_activation(policy_launcher, tmp_path, environment):
    if environment == "prefix":
        conda_env = str(tmp_path / "absolute-prefix")
        expected = Path(conda_env) / "bin/python"
    elif environment == "base":
        conda_env = "base"
        expected = Path(policy_launcher.env["CONDA_HOME"]) / "bin/python"
    else:
        conda_env = "named-env"
        expected = Path(policy_launcher.env["CONDA_HOME"]) / "envs/named-env/bin/python"
    result = policy_launcher.run(DRY_RUN=1, NPROC_PER_NODE=4, PYTHON_BIN="", CONDA_ENV=conda_env)
    assert _policy_dry_command(result)[0] == str(expected)
    assert not policy_launcher.record.exists() and not policy_launcher.output.exists()


def test_policy_modelarts_normal_default_python_comes_from_activated_prefix(policy_launcher):
    # Conda may resolve a named environment outside the guessed CONDA_HOME/envs path.
    resolved = Path(policy_launcher.env["PYTHON_BIN"]).with_name("python")
    resolved.write_text(Path(policy_launcher.env["PYTHON_BIN"]).read_text())
    resolved.chmod(0o755)
    result = policy_launcher.run(NPROC_PER_NODE=4, PYTHON_BIN="", CONDA_ENV="resolved-elsewhere")
    assert _policy_dry_command(result)[0] == str(resolved)
    assert json.loads(policy_launcher.record.read_text())["conda"] == "1"


@pytest.mark.parametrize("variable", ["OUTPUT_DIR", "WANDB_DIR", "LOG_DIR", "TMPDIR"])
def test_policy_modelarts_launcher_directory_creation_failure_prevents_training(policy_launcher, tmp_path, variable):
    paths = {name: tmp_path / name.lower() for name in ("OUTPUT_DIR", "WANDB_DIR", "LOG_DIR", "TMPDIR")}
    blocked_parent = tmp_path / "not a directory"
    blocked_parent.write_text("Preserve this existing file.")
    paths[variable] = blocked_parent / variable.lower()

    result = policy_launcher.run(NPROC_PER_NODE=4, **paths)

    assert result.returncode != 0 and not policy_launcher.record.exists()
    assert variable in result.stderr and str(paths[variable]) in result.stderr
    assert "OUTPUT_DIR" in result.stderr and "writ" in result.stderr.lower() and "shared" in result.stderr.lower()
    assert blocked_parent.read_text() == "Preserve this existing file."


@pytest.mark.parametrize("variable", ["OUTPUT_DIR", "WANDB_DIR", "LOG_DIR", "TMPDIR"])
def test_policy_modelarts_launcher_existing_directory_requires_write_probe(policy_launcher, tmp_path, variable):
    paths = {name: tmp_path / name.lower() for name in ("OUTPUT_DIR", "WANDB_DIR", "LOG_DIR", "TMPDIR")}
    for path in paths.values():
        path.mkdir()
        (path / "keep.txt").write_text("Do not delete unrelated files.")
    real_mktemp = shutil.which("mktemp")
    assert real_mktemp is not None
    commands = tmp_path / "fault commands"
    commands.mkdir()
    injected = commands / "mktemp"
    injected.write_text(
        "#!/bin/bash\n"
        'for argument in "$@"; do\n'
        '    case "$argument" in\n'
        '        "$POLICY_MOCK_UNWRITABLE"|"$POLICY_MOCK_UNWRITABLE"/*)\n'
        "            printf 'simulated permission denied\\n' >&2; exit 13 ;;\n"
        "    esac\n"
        "done\n"
        f'exec {shlex.quote(real_mktemp)} "$@"\n'
    )
    injected.chmod(0o755)

    result = policy_launcher.run(
        NPROC_PER_NODE=4,
        PATH=f"{commands}{os.pathsep}{policy_launcher.env['PATH']}",
        POLICY_MOCK_UNWRITABLE=paths[variable],
        **paths,
    )

    assert result.returncode != 0 and not policy_launcher.record.exists()
    assert variable in result.stderr and str(paths[variable]) in result.stderr
    assert "OUTPUT_DIR" in result.stderr and "writ" in result.stderr.lower() and "shared" in result.stderr.lower()
    for path in paths.values():
        assert sorted(child.name for child in path.iterdir()) == ["keep.txt"]
        assert (path / "keep.txt").read_text() == "Do not delete unrelated files."


def test_policy_modelarts_launcher_absolute_output_prepares_wandb_logs_and_removes_probes(policy_launcher, tmp_path):
    output = tmp_path / "shared output absolute"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("Existing output must be preserved.")
    scratch = tmp_path / "scratch space"

    result = policy_launcher.run(NPROC_PER_NODE=4, OUTPUT_DIR=output, TMPDIR=scratch)

    assert result.returncode == 0, result.stderr + result.stdout
    recorded = json.loads(policy_launcher.record.read_text())
    assert _policy_option(recorded["argv"], "train.checkpoint.output_dir") == str(output)
    assert recorded["wandb_dir"] == str(output) and recorded["wandb_mode"] == "offline"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt", "logs"]
    logs = list((output / "logs").iterdir())
    assert len(logs) == 1 and logs[0].name == "train_node0_launcher-test.log"
    assert "POLICY_MOCK_STDOUT" in logs[0].read_text()
    assert keep.read_text() == "Existing output must be preserved."
    assert scratch.is_dir() and not list(scratch.iterdir())


def test_policy_modelarts_launcher_dry_run_creates_no_output_or_scratch_directories(policy_launcher, tmp_path):
    paths = {name: tmp_path / name.lower() for name in ("OUTPUT_DIR", "WANDB_DIR", "LOG_DIR", "TMPDIR")}

    result = policy_launcher.run(DRY_RUN=1, NPROC_PER_NODE=4, **paths)

    assert _policy_dry_command(result)
    assert not policy_launcher.record.exists()
    assert not any(path.exists() for path in paths.values())


@pytest.fixture
def generic_launcher(policy_launcher, tmp_path):
    """Run the real wrappers in a temporary checkout with no hardware commands."""
    repo = tmp_path / "generic checkout"
    repo.mkdir()
    action = Path("scripts/causal_cosmos3/training_scripts/train_action.sh")
    (repo / action).parent.mkdir(parents=True)
    for script in (Path("train.sh"), action):
        shutil.copy2(policy_launcher.repo / script, repo / script)

    commands = tmp_path / "generic commands"
    commands.mkdir()
    # A restricted PATH excludes cnmon and real accelerator probes/launchers.
    for name in ("bash", "dirname", "mkdir", "tee", "date", "tr", "wc"):
        executable = shutil.which(name)
        assert executable is not None
        (commands / name).symlink_to(executable)
    for name in ("nvidia-smi", "rocm-smi"):
        probe = commands / name
        probe.write_text("#!/bin/sh\nexit 1\n")
        probe.chmod(0o755)
    shutil.copy2(policy_launcher.env["PYTHON_BIN"], commands / "torchrun")

    env = {
        **policy_launcher.env,
        "PATH": str(commands),
        "NNODES": "1",
        "NPROC_PER_NODE": "2",
        "NODE_RANK": "0",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    }
    for name in ("LOG_FILE", "additional_args"):
        env.pop(name, None)
    log = repo / "log.txt"
    log.write_text("Preserve the working-directory log.\n")

    def launch(options=(), *, wrapper=False, **overrides):
        return subprocess.run(
            ["bash", str(repo / (action if wrapper else "train.sh")), *options],
            env={**env, **{key: str(value) for key, value in overrides.items()}},
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=20,
        )

    return SimpleNamespace(
        run=launch, repo=repo, record=policy_launcher.record, env=env, log=log, output=policy_launcher.output
    )


def test_generic_launcher_preserves_spaced_arguments_and_literal_globs(generic_launcher, tmp_path):
    for name in ("one.literal", "two.literal"):
        (generic_launcher.repo / name).touch()
    options = ["tasks/train_causal_cosmos3.py", "config with spaces.yaml", "--prompt", "pick blue cup", "*.literal"]

    result = generic_launcher.run(options, LOG_FILE=tmp_path / "selected log.txt")

    assert result.returncode == 0, result.stderr + result.stdout
    argv = json.loads(generic_launcher.record.read_text())["argv"]
    assert argv[-len(options) :] == options
    assert "--standalone" in argv


def test_generic_launcher_writes_selected_log_without_touching_working_directory(generic_launcher, tmp_path):
    log = tmp_path / "selected logs" / "worker output.log"

    result = generic_launcher.run(["tasks/train_causal_cosmos3.py"], LOG_FILE=log)

    assert result.returncode == 0, result.stderr + result.stdout
    text = log.read_text()
    assert "POLICY_MOCK_STDOUT" in text and "POLICY_MOCK_STDERR" in text
    assert generic_launcher.log.read_text() == "Preserve the working-directory log.\n"


@pytest.mark.parametrize("blocked", ["parent", "file"])
def test_generic_launcher_log_failure_prevents_workers(generic_launcher, tmp_path, blocked):
    if blocked == "parent":
        parent = tmp_path / "not a directory"
        parent.write_text("Preserve this file.")
        log = parent / "worker.log"
    else:
        log = tmp_path / "directory instead of log"
        log.mkdir()

    result = generic_launcher.run(["tasks/train_causal_cosmos3.py"], LOG_FILE=log)

    assert result.returncode != 0 and not generic_launcher.record.exists()
    assert str(log if blocked == "file" else log.parent) in result.stderr
    assert generic_launcher.log.read_text() == "Preserve the working-directory log.\n"


def test_generic_launcher_preserves_worker_failure_through_tee(generic_launcher, tmp_path):
    result = generic_launcher.run(
        ["tasks/train_causal_cosmos3.py"], LOG_FILE=tmp_path / "failure.log", POLICY_MOCK_EXIT=23
    )

    assert generic_launcher.record.exists()
    assert result.returncode == 23, result.stderr + result.stdout
    assert "POLICY_MOCK_STDERR" in result.stdout


def test_action_wrapper_places_default_log_under_output(generic_launcher):
    result = generic_launcher.run(wrapper=True)

    assert result.returncode == 0, result.stderr + result.stdout
    argv = json.loads(generic_launcher.record.read_text())["argv"]
    assert _policy_option(argv, "train.checkpoint.output_dir") == str(generic_launcher.output)
    log = generic_launcher.output / "logs/train_node0_launcher-test.log"
    assert "POLICY_MOCK_STDOUT" in log.read_text()
    assert generic_launcher.log.read_text() == "Preserve the working-directory log.\n"
