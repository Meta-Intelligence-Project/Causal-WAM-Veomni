"""Cosmos process control; hardware and model imports stay in child processes."""

import copy
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import inference_web as lingbot
import yaml
from flask import Blueprint, abort, jsonify, request, send_file


ROOT = Path("/home/agilex/models/causal-cosmos")
REPO = ROOT / "cosmos-framework"
HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "cosmos_inference_config.json"
LOGS = HERE.parent / "logs" / "cosmos"
RESULTS = Path("/home/agilex/models/logs/cosmos")
LEGACY_RESULTS = Path("/home/agilex/models/logs")
RUN_NAME = re.compile(r"\d{8}_\d{6}(?:_\d{2,})?")
DEFAULT_CHECKPOINT = "1104-command-32delta"
PORT = 8000
VEOMNI_REPO = Path("/home/agilex/models/Causal-WAM-Veomni")
SERVER_RANGES = {
    "steps": (1, 100, int),
    "guidance": (1, 20, float),
    "shift": (0.1, 20, float),
    "history": (1, 128, int),
    "seed": (0, 2147483647, int),
}
GRIPPER_DEFAULTS = dict(left_gripper_scale=0.9, right_gripper_scale=0.9)
SERVER_PYTHON = "/home/agilex/miniconda3/envs/infer-wam-veomni/bin/python"
MODEL_CONFIGS = {"1104-command-32delta": "configs/server/causal_cosmos3_server_poc.yaml"}
cosmos_api = Blueprint("cosmos", __name__)


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else VEOMNI_REPO / path


def model_config(relative):
    if relative not in MODEL_CONFIGS:
        raise ValueError("请选择已绑定推理配置的 VeOmni checkpoint")
    return yaml.safe_load(repo_path(MODEL_CONFIGS[relative]).read_text())


def geometry_from_config(config):
    geometry = dict(config["geometry"])
    geometry.update(
        execute_steps=geometry["actions_per_block"],
        fps=config["fps"],
        resolution=config["source"]["resolution"],
        **config.get("async_infer", {}),
    )
    return geometry


def checkpoint(relative):
    config = model_config(relative)
    folder = (VEOMNI_REPO / "ckpts" / relative).resolve()
    model = json.loads((folder / "config.json").read_text())
    if model.get("model_type") != "cosmos3":
        raise ValueError("Checkpoint 不是 Cosmos3 模型")
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    if not index.get("weight_map"):
        raise ValueError("Checkpoint 缺少权重索引")
    for name in set(index["weight_map"].values()):
        path = (folder / name).resolve()
        if folder not in path.parents or not path.is_file():
            raise ValueError("Checkpoint 权重分片缺失或路径无效")
    for key in ("statistics", "tokenizer", "vae"):
        if not repo_path(config[key]).exists():
            raise ValueError("模型资源缺失：" + key)
    return folder, config, geometry_from_config(config)


def discover_checkpoints():
    items = []
    for relative in MODEL_CONFIGS:
        try:
            folder, _, geometry = checkpoint(relative)
            items.append(dict(relative=relative, name=relative + " / VeOmni", path=str(folder), geometry=geometry))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return items


def validate_rtc_execution(client, server):
    if server.get("rtc_enabled", False) and (
        client.get("smooth_method", "none") != "none" or client.get("control_mode") == "eef"
    ):
        raise ValueError("RTC 模式要求轨迹平滑为 none、使用关节控制；如需平滑或 EEF，请关闭 RTC 并重启服务")


def validate_config(data):
    if not isinstance(data, dict) or set(data) - {"checkpoint", "server", "client", "prompt"}:
        raise ValueError("Cosmos 配置格式错误")
    relative = data.get("checkpoint", DEFAULT_CHECKPOINT)
    _, base, geometry = checkpoint(relative)
    server = {key: base["sampling"][key] for key in SERVER_RANGES}
    server.update(base.get("async_infer", dict(rtc_enabled=False, rtc_delay=16)))
    values = data.get("server", {})
    if not isinstance(values, dict):
        raise ValueError("server 必须是配置对象")
    values = dict(values)
    # Accept the old saved schema only when it has identical action semantics.
    if values.pop("action_representation", "relative") != "relative":
        raise ValueError("当前 checkpoint 使用相对动作，不支持 absolute")
    if values.pop("execute_steps", geometry["actions_per_block"]) != geometry["actions_per_block"]:
        raise ValueError("VeOmni 不支持旧版截短执行步数，请使用完整动作块")
    if set(values) - set(server):
        raise ValueError("未知 Cosmos 服务参数")
    server.update(values)
    if type(server["rtc_enabled"]) is not bool:
        raise ValueError("rtc_enabled 必须为布尔值")
    delay = server["rtc_delay"]
    if type(delay) is not int or not 1 <= delay <= geometry["actions_per_block"] // 2:
        raise ValueError("RTC 延迟须为 1 到半个动作块之间的整数")
    for key, (low, high, kind) in SERVER_RANGES.items():
        value = server[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
            or (kind is int and type(value) is not int)
        ):
            raise ValueError("{} 必须在 {}–{} 范围内".format(key, low, high))
    # Reuse the existing control and smoothing validation, without its checkpoint binding.
    client = data.get("client", {})
    if not isinstance(client, dict):
        raise ValueError("client 必须是配置对象")
    client = dict(client)
    if server["rtc_enabled"]:
        client.setdefault("smooth_method", "none")
    save_prediction_video = client.pop("save_prediction_video", False)
    if type(save_prediction_video) is not bool:
        raise ValueError("save_prediction_video 必须为布尔值")
    grippers = dict(GRIPPER_DEFAULTS)
    for key in grippers:
        value = client.pop(key, grippers[key])
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3:
            raise ValueError("{} 必须在 0–3 范围内".format(key))
        grippers[key] = float(value)
    eef = client.get("control_mode") == "eef"
    if eef:
        client["control_mode"] = "movej"
    control = lingbot.validate_config({"client": client, "prompt": data.get("prompt", "")})
    if eef:
        control["client"]["control_mode"] = "eef"
    control["client"].update(grippers, save_prediction_video=save_prediction_video)
    validate_rtc_execution(control["client"], server)
    result = dict(checkpoint=relative, server=server, client=control["client"], prompt=control["prompt"])
    server_signature(result)
    return result


def read_config():
    if CONFIG_FILE.is_file():
        return validate_config(json.loads(CONFIG_FILE.read_text()))
    return validate_config(dict(prompt="", client=dict(max_chunks=1)))


def scan_processes(kind):
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [x.decode(errors="replace") for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if not args or not any("python" in x or "pt_elastic" in x for x in args[:2]):
                continue
            cwd = (proc / "cwd").resolve()
            module = "tasks.client_server.client.piper" if kind == "client" else "tasks.client_server.server.server"
            if cwd == VEOMNI_REPO.resolve() and module in args:
                if kind == "server":
                    port = args[args.index("--port") + 1] if "--port" in args else "8000"
                    if port != str(PORT):
                        continue
                found.append(dict(pid=int(proc.name), args=args, implementation="veomni"))
                continue
            if cwd != REPO.resolve():
                continue
            scripts = {Path(x).name for x in args}
            match = (
                "cosmos_framework.inference.causal_action.piper_client" in args
                if kind == "client"
                else bool(scripts & {"launch_checkpoint_server.py", "cuda_server_bootstrap.py"})
                or "cosmos_framework.inference.causal_action.server" in args
            )
            if match:
                found.append(dict(pid=int(proc.name), args=args))
        except (OSError, RuntimeError):
            pass
    return found


def connections(state):
    sockets = set()
    for name in ("tcp", "tcp6"):
        try:
            for line in Path("/proc/net/" + name).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == state and int(fields[1].rsplit(":", 1)[1], 16) == PORT:
                    sockets.add(fields[9])
        except OSError:
            pass
    return sockets


def ready():
    # The server binds only after model load. Match the actual listener to a Cosmos
    # process; probing WebSocket would reset the model cache or occupy its only slot.
    sockets = connections("0A")
    for proc in scan_processes("server"):
        try:
            for fd in Path("/proc/{}/fd".format(proc["pid"])).iterdir():
                target = os.readlink(str(fd))
                if target.startswith("socket:[") and target[8:-1] in sockets:
                    return True
        except OSError:
            pass
    return False


def process_identity(pid):
    # Include boot identity and start ticks so a recycled PID is never owned.
    stat = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start=stat[19], boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def owned_server(proc):
    try:
        saved = json.loads((LOGS / "server-owner.json").read_text())
        return saved == process_identity(proc["pid"])
    except (OSError, ValueError, IndexError):
        return False


def active_server():
    return next((p for p in scan_processes("server") if p.get("implementation") == "veomni"), None)


def external_server():
    proc = active_server()
    return proc if proc and not owned_server(proc) else None


def active_signature():
    proc = active_server()
    if not proc:
        return None
    try:
        args = proc["args"]
        config = yaml.safe_load(repo_path(args[args.index("--config") + 1]).read_text())
        # Account for supported scalar command-line overrides on external servers.
        for key in SERVER_RANGES:
            flag = "--" + key.replace("_", "-")
            if flag in args:
                config["sampling"][key] = SERVER_RANGES[key][2](args[args.index(flag) + 1])
        if "--fps" in args:
            config["fps"] = float(args[args.index("--fps") + 1])
        return dict(checkpoint=str(repo_path(args[args.index("--checkpoint") + 1]).resolve()), config=config)
    except (OSError, ValueError, IndexError, KeyError):
        return None


def server_signature(config):
    folder, base, _ = checkpoint(config["checkpoint"])
    effective = copy.deepcopy(base)
    for key in ("statistics", "tokenizer", "vae"):
        effective[key] = str(repo_path(effective[key]).resolve())
    effective["sampling"].update({key: config["server"][key] for key in SERVER_RANGES})
    effective["async_infer"] = {key: config["server"][key] for key in ("rtc_enabled", "rtc_delay")}
    effective["prediction_debug_root"] = str(RESULTS)
    schedule = effective["sampling"].get("residual_reuse_full_steps")
    if isinstance(schedule, list) and any(step >= effective["sampling"]["steps"] for step in schedule):
        raise ValueError("去噪步数必须大于 residual_reuse_full_steps 中的最大步号；请调整服务 YAML")
    return dict(checkpoint=str(folder), config=effective)


def active_geometry():
    signature = active_signature()
    return geometry_from_config(signature["config"]) if signature else None


def prediction_path(directory):
    paths = sorted(path for path in directory.glob("*.mp4") if re.fullmatch(r"\d{8}_\d{6}_\d{2,}\.mp4", path.name))
    return paths[-1] if paths else directory / "prediction.mp4"


def create_result_directory():
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    number = 0
    while True:
        name = stamp if number == 0 else stamp + f"_{number:02d}"
        directory = RESULTS / name
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


def video_directory(run):
    if re.fullmatch(r"run-\d+", run):
        if (RESULTS / run).is_dir():
            return RESULTS / run, RESULTS
        if (LEGACY_RESULTS / run).is_dir():
            return LEGACY_RESULTS / run, LEGACY_RESULTS
        return LOGS / run, LOGS
    if RUN_NAME.fullmatch(run):
        if (RESULTS / run).is_dir():
            return RESULTS / run, RESULTS
        return LEGACY_RESULTS / run, LEGACY_RESULTS
    abort(404)


def prediction_videos():
    videos = []
    directories = [(directory, LOGS) for directory in LOGS.glob("run-*")]
    if RESULTS.is_dir():
        directories.extend(
            (directory, RESULTS)
            for directory in RESULTS.iterdir()
            if RUN_NAME.fullmatch(directory.name) or re.fullmatch(r"run-\d+", directory.name)
        )
    if LEGACY_RESULTS.is_dir():
        directories.extend(
            (directory, LEGACY_RESULTS)
            for directory in LEGACY_RESULTS.iterdir()
            if RUN_NAME.fullmatch(directory.name) or re.fullmatch(r"run-\d+", directory.name)
        )
    for directory, base in directories:
        path = prediction_path(directory)
        if not path.is_file() or path.is_symlink() or base.resolve() not in path.resolve().parents:
            continue
        videos.append(
            dict(
                run=path.parent.name,
                name=path.name,
                bytes=path.stat().st_size,
                url="/api/cosmos/videos/" + path.parent.name,
                modified=path.stat().st_mtime,
            )
        )
    return sorted(videos, key=lambda item: item["modified"], reverse=True)[:20]


@cosmos_api.route("/api/cosmos/videos/<run>")
def prediction_video(run):
    directory, base = video_directory(run)
    path = prediction_path(directory)
    if not path.is_file() or path.is_symlink() or base.resolve() not in path.resolve().parents:
        abort(404)
    return send_file(
        path,
        mimetype="video/mp4",
        conditional=True,
        as_attachment=request.args.get("download") == "1",
        download_name=path.name if path.name != "prediction.mp4" else run + "-prediction.mp4",
    )


def prediction_recordings():
    manifests = sorted(RESULTS.glob("*/predictions*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recordings = []
    for path in manifests[:5]:
        if not RUN_NAME.fullmatch(path.parent.parent.name) or RESULTS.resolve() not in path.resolve().parents:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        run, name = path.parent.parent.name, path.parent.name
        prefix = f"/api/cosmos/predictions/{run}/{name}/"
        for clip in data["clips"]:
            if clip["status"] == "complete":
                clip["prediction_url"] = prefix + clip["prediction"]
                clip["observation_url"] = prefix + clip["observation"]
        recordings.append(dict(data, run=run, name=name, key=run + "/" + name, directory=str(path.parent)))
    return recordings


@cosmos_api.route("/api/cosmos/predictions/<run>/<recording>/<filename>")
def prediction_artifact(run, recording, filename):
    if (
        not RUN_NAME.fullmatch(run)
        or not re.fullmatch(r"predictions(?:-[a-f0-9]{32})?", recording)
        or not re.fullmatch(r"(?:\d{4,}(?:-observation\.png|\.mp4|\.npz)|manifest\.json)", filename)
    ):
        abort(404)
    path = RESULTS / run / recording / filename
    if not path.is_file() or path.is_symlink() or RESULTS.resolve() not in path.resolve().parents:
        abort(404)
    return send_file(path, conditional=True, as_attachment=request.args.get("download") == "1")


class CosmosManager:
    def __init__(self):
        self.processes = {}
        self.message = ""
        self.output = None
        self.requested_stops = set()

    def running(self, kind):
        proc = self.processes.get(kind)
        return bool(proc and proc.poll() is None) or bool(scan_processes(kind))

    def busy(self):
        return self.running("client")

    def spawn(self, kind, args, env=None):
        self.requested_stops.discard(kind)
        LOGS.mkdir(parents=True, exist_ok=True)
        with (LOGS / (kind + ".log")).open("w") as stream:
            proc = subprocess.Popen(
                args,
                cwd=str(VEOMNI_REPO),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.processes[kind] = proc
        return proc.pid

    def start_server(self):
        if self.running("server") or connections("0A"):
            raise RuntimeError("Cosmos 服务已运行或 8000 端口被占用")
        config = read_config()
        signature = server_signature(config)
        path = LOGS / ("server-{}.json".format(time.time_ns()))
        # JSON is valid YAML; preserve an immutable effective configuration per launch.
        lingbot.write_json(path, signature["config"])
        env = dict(
            os.environ, PYTHON_BIN=SERVER_PYTHON, INFER_CONFIG=str(path), CHECKPOINT_PATH=signature["checkpoint"]
        )
        pid = self.spawn(
            "server",
            [
                "bash",
                str(VEOMNI_REPO / "scripts/causal_cosmos3/server_scripts/server.sh"),
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            env=env,
        )
        lingbot.write_json(LOGS / "server-owner.json", process_identity(pid))
        self.message = "Cosmos 服务端启动中，等待模型加载"
        return pid

    def start_client(self, single=False):
        config = read_config()
        if not config["prompt"].strip():
            raise ValueError("请填写 Cosmos 任务指令")
        _, _, geometry = checkpoint(config["checkpoint"])
        # Apply hardware-side gripper scaling and closing threshold.
        client = dict(config["client"])
        client.update(gripper_close_threshold=0.02)
        geometry = active_geometry() or geometry
        validate_rtc_execution(client, geometry)
        client.update(model_fps=geometry["fps"], server_host="127.0.0.1", server_port=PORT)
        path = LOGS / ("client-{}.json".format(time.time_ns()))
        lingbot.write_json(path, dict(client=client, prompt=config["prompt"]))
        self.output = create_result_directory()
        lingbot.write_json(self.output / "inference_config.json", dict(config, effective_server=active_signature()))
        stop_file = LOGS / "runtime_config.json"
        lingbot.write_json(stop_file, dict(emergency_stop=False))
        launcher = VEOMNI_REPO / "scripts/causal_cosmos3/client_scripts/piper.sh"
        pid = self.spawn(
            "client",
            [
                "bash",
                str(launcher),
                "--config-file",
                str(path),
                "--max-chunks",
                str(1 if single else client["max_chunks"]),
                "--output",
                str(self.output),
                "--stop-file",
                str(stop_file),
            ],
        )
        self.message = "Cosmos 客户端启动中；推理并执行一次" if single else "Cosmos 客户端启动中"
        return pid

    def stop(self, kind):
        if kind == "server" and external_server():
            raise RuntimeError("VeOmni 服务由外部终端管理，请在启动终端停止")
        if kind == "server" and connections("01"):
            raise RuntimeError("Cosmos 客户端仍已连接，请先结束客户端")
        if kind == "client":
            lingbot.write_json(LOGS / "runtime_config.json", dict(emergency_stop=True))
        self.requested_stops.add(kind)
        pids = {p["pid"] for p in scan_processes(kind)}
        proc = self.processes.get(kind)
        if proc and proc.poll() is None:
            pids.add(proc.pid)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGINT if kind == "client" else signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.message = (
            "已请求结束 Cosmos 推理，停止发送后续动作，不触发电子急停"
            if kind == "client"
            else "已发送 Cosmos 服务停止请求，等待进程退出"
        )

    def status(self):
        config = read_config()
        server, client = self.running("server"), self.running("client")
        logs = {key: lingbot.tail(LOGS / (key + ".log")) for key in ("server", "client")}
        if external_server():
            logs["server"] = lingbot.tail(LOGS / "veomni_server.log")
        progress = {}
        for line in logs["client"]:
            if line.startswith("CLIENT_STATUS "):
                try:
                    progress = json.loads(line[len("CLIENT_STATUS ") :])
                except ValueError:
                    pass
        exits = {kind: proc.poll() for kind, proc in self.processes.items()}
        message = self.message
        server_ready = server and ready()
        if server_ready and (not message or message.startswith("Cosmos 服务端启动中")):
            message = (
                "VeOmni 外部服务已就绪，将使用新版 Piper 客户端"
                if external_server()
                else "VeOmni 服务端已就绪，可启动新版 Piper 客户端执行任务"
            )
        if not server and "server" not in self.requested_stops and exits.get("server") not in (None, 0):
            message = "Cosmos 服务端已退出，请查看服务日志（退出码 {}）".format(exits["server"])
        if not client and "client" not in self.requested_stops and exits.get("client") not in (None, 0, 130):
            message = "Cosmos 客户端已退出，请查看执行日志（退出码 {}）".format(exits["client"])
        phase = progress.get("status", "starting") if client else progress.get("status", "stopped")
        if not client and phase in ("starting", "inferring", "executing", "updating_cache", "saving_video"):
            phase = (
                "failed"
                if "client" not in self.requested_stops and exits.get("client") not in (None, 0, 130)
                else "stopped"
            )
        return dict(
            ok=True,
            server=dict(
                running=server,
                ready=server_ready,
                external=bool(external_server()),
                geometry=active_geometry(),
                checkpoint=(active_signature() or {}).get("checkpoint"),
                restart_required=server_ready
                and not external_server()
                and active_signature() != server_signature(config),
                pids=[p["pid"] for p in scan_processes("server")],
                port=PORT,
                exit_code=exits.get("server"),
            ),
            client=dict(
                running=client,
                phase=phase,
                exit_code=exits.get("client"),
                progress=progress,
                pids=[p["pid"] for p in scan_processes("client")],
            ),
            checkpoints=discover_checkpoints(),
            message=message,
            logs=logs,
            videos=prediction_videos(),
            prediction_recordings=prediction_recordings(),
            collection_running=lingbot.manager.collection_busy(),
        )


manager = CosmosManager()


@cosmos_api.route("/api/cosmos/config", methods=["GET", "POST"])
def configuration():
    try:
        with lingbot.manager.lock:
            if request.method == "GET":
                return jsonify(
                    ok=True, config=read_config(), checkpoints=discover_checkpoints(), presets=lingbot.presets()
                )
            if manager.busy() or lingbot.manager.busy():
                raise RuntimeError("请先停止推理后保存 Cosmos 配置")
            config = validate_config(request.get_json(silent=True))
            if manager.running("server"):
                saved = read_config()
                if config["checkpoint"] != saved["checkpoint"] or config["server"] != saved["server"]:
                    raise RuntimeError("Cosmos 服务运行中，模型与服务参数已锁定，请先停止服务")
            lingbot.write_json(CONFIG_FILE, config)
            return jsonify(ok=True, config=config, message="Cosmos 配置已保存")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(ok=False, error=str(exc)), 409


@cosmos_api.route("/api/cosmos/status")
def status():
    try:
        with lingbot.manager.lock:
            return jsonify(manager.status())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return jsonify(ok=False, error=str(exc)), 400


@cosmos_api.route("/api/cosmos/<target>/<action>", methods=["POST"])
def action(target, action):
    if (target, action) not in {
        ("server", "start"),
        ("server", "stop"),
        ("client", "start"),
        ("client", "test"),
        ("client", "stop"),
    }:
        return jsonify(ok=False, error="未知操作"), 404
    from rtc_web import client_pids
    from rtc_web import processes as rtc_processes

    with lingbot.manager.lock:
        try:
            if action == "stop":
                if target == "server" and manager.busy():
                    raise RuntimeError("请先停止 Cosmos 客户端")
                manager.stop(target)
                return jsonify(ok=True, message=manager.message)
            if manager.busy() or lingbot.manager.busy() or client_pids() or lingbot.manager.collection_busy():
                raise RuntimeError("请先停止推理或退出数据采集会话")
            if lingbot.manager.running("server") or rtc_processes():
                raise RuntimeError("请先停止 Lingbot / RTC 模型服务以释放显存")
            if target == "server":
                pid = manager.start_server()
            else:
                if not active_server() or not ready():
                    raise RuntimeError("Cosmos 服务端尚未就绪")
                if not external_server() and active_signature() != server_signature(read_config()):
                    raise RuntimeError("Cosmos 服务参数与保存配置不同，请先重启服务")
                pid = manager.start_client(single=action == "test")
            return jsonify(ok=True, pid=pid, message=manager.message)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except (RuntimeError, OSError) as exc:
            return jsonify(ok=False, error=str(exc)), 409


@cosmos_api.before_app_request
def guard_cosmos_ownership():
    # Use the same lock as collection/arm guards, including RTC starts.
    guarded = {
        "/api/collection/start",
        "/api/piper/start",
        "/api/piper/go-zero",
        "/api/piper/enable",
        "/api/piper/clear-estop",
        "/api/piper/stop",
        "/api/camera/start",
        "/api/camera/stop",
        "/api/system/activate",
        "/api/inference/server/start",
        "/api/inference/client/start",
        "/api/inference/client/test",
        "/api/inference/text/prepare",
        "/api/rtc/start",
        "/api/rtc/client/start",
    }
    if request.method == "POST" and request.path in guarded:
        # web_server retains this lock until teardown on arm/collection routes;
        # Lingbot endpoints acquire it again when performing their actions.
        from flask import g

        lingbot.manager.lock.acquire()
        g.cosmos_guard = True
        if manager.busy():
            return jsonify(ok=False, error="Cosmos 真机执行中，请先停止 Cosmos 客户端"), 409
        if request.path in {
            "/api/inference/server/start",
            "/api/inference/text/prepare",
            "/api/rtc/start",
            "/api/rtc/client/start",
        } and manager.running("server"):
            return jsonify(ok=False, error="请先停止 Cosmos 模型服务以释放显存"), 409


@cosmos_api.teardown_app_request
def release_cosmos_ownership(_error):
    from flask import g

    if getattr(g, "cosmos_guard", False):
        g.cosmos_guard = False
        lingbot.manager.lock.release()
