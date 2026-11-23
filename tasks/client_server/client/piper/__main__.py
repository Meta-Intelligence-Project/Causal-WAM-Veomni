"""Piper hardware execution based on LingBot's real_robot_inference_merged.py.

Only this explicit executable connects hardware. The RoboTwin simulator client
remains available separately; Cosmos real-robot runs use this module.
"""

import argparse
import fcntl
import importlib.util
import json
import logging
import math
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from tasks.client_server.client.piper.eef import send_eef_action
from tasks.client_server.client.piper.protocol import (
    CosmosProtocol,
    RunReport,
    run_loop,
)
from tasks.client_server.client.runtime import CausalPolicyClient


class PiperPolicyClient(CausalPolicyClient):
    """Cosmos wire protocol with Lingbot's connection and receive waits."""

    def __init__(self, uri, *, control_mode="joint", stop_check=None):
        while True:
            if stop_check is not None and stop_check():
                raise KeyboardInterrupt("Stop requested")
            try:
                super().__init__(uri, control_mode=control_mode, metadata_stop_check=stop_check)
                return
            except ValueError:
                # A wire-protocol mismatch cannot be fixed by retrying a socket.
                raise
            except BaseException as exc:
                connection = getattr(self, "connection", None)
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        logging.exception("Failed to close the previous Cosmos connection")
                    self.connection = None
                if not isinstance(exc, Exception):
                    raise
                logging.info("Waiting for Cosmos server... (%s)", exc)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if stop_check is not None and stop_check():
                        raise KeyboardInterrupt("Stop requested") from None
                    time.sleep(0.1)


def load_lingbot(directory):
    directory = Path(directory).resolve()
    sys.path.insert(0, str(directory))
    name = "_cosmos_lingbot_hardware"
    spec = importlib.util.spec_from_file_location(name, directory / "real_robot_inference_merged.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stop_requested(path):
    try:
        return bool(json.loads(Path(path).read_text()).get("emergency_stop", False))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def configuration(args, hardware):
    raw = json.loads(Path(args.config_file).read_text()) if args.config_file else {}
    values = dict(raw.get("client", raw))
    if {"async_infer", "rtc_enabled", "rtc_delay", "rtc_mode", "rtc_stride"} & values.keys():
        raise ValueError("Move RTC settings from client JSON to the server YAML async_infer section")
    aliases = {"left_can": "can_left", "right_can": "can_right", "fps": "model_fps"}
    values = {aliases.get(k, k): v for k, v in values.items()}
    mode = args.control_mode or values.get("control_mode", "movej")
    hardware_mode = "movej" if mode == "eef" else mode
    if hardware_mode not in hardware.CONFIG_MAP:
        raise ValueError("Invalid control mode")
    cfg = SimpleNamespace(**vars(hardware.CONFIG_MAP[hardware_mode]()))
    for key in ("rtc_enabled", "rtc_delay", "rtc_mode", "rtc_stride", "async_infer"):
        vars(cfg).pop(key, None)
    cfg.__dict__.update(
        model_family="causal-cosmos",
        smooth_method="none",
        action_dt=1 / 30,
        max_chunks=15,
        model_fps=30.0,
        report_dir="",
        left_gripper_scale=0.8,
        right_gripper_scale=0.9,
        gripper_close_threshold=0.02,
        save_prediction_video=False,
        eef_position_scale=1000.0,
        server_port=8000,
        cpv_acc=0.5,
        cpv_dcc=0.5,
        cpv_cv=0.5,
    )
    if not hasattr(cfg, "movej_speed_percent"):
        cfg.movej_speed_percent = 0.06
    allowed = set(vars(cfg))
    if set(values) - allowed:
        raise ValueError("Unknown client configuration: " + str(sorted(set(values) - allowed)))
    cfg.__dict__.update(values)
    if type(cfg.save_prediction_video) is not bool:
        raise ValueError("save_prediction_video must be boolean")
    cfg.control_mode = mode
    cfg.prompt = args.prompt if args.prompt is not None else raw.get("prompt", "")
    if not isinstance(cfg.prompt, str) or not cfg.prompt.strip():
        raise ValueError("A nonempty prompt is required")
    if args.max_chunks is not None:
        cfg.max_chunks = args.max_chunks
    if type(cfg.max_chunks) is not int or cfg.max_chunks < 0:
        raise ValueError("max_chunks must be a nonnegative integer (one chunk = one Server response)")
    if args.host:
        cfg.server_host = args.host
    if args.port:
        cfg.server_port = args.port
    cfg.report_dir = args.output or cfg.report_dir
    for key in ["action_dt", "model_fps", "movej_speed_percent", "wait_tol_deg", "eef_position_scale"]:
        value = getattr(cfg, key, None)
        if value is None and key == "movej_speed_percent" and mode == "cpv":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    if hasattr(cfg, "movej_speed_percent") and cfg.movej_speed_percent > 1:
        raise ValueError("movej_speed_percent must be <= 1")
    if cfg.can_left == cfg.can_right:
        raise ValueError("Left/right CAN interfaces must differ")
    for key in ["left_gripper_scale", "right_gripper_scale", "gripper_close_threshold"]:
        value = getattr(cfg, key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
    return cfg


def prediction_video_path(directory, started_at):
    """Name by local run start time; number recordings sharing the same second."""
    directory = Path(directory)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    prefix = stamp + "_"
    numbers = []
    paths = list(directory.parent.glob("run-*/" + prefix + "*.mp4"))
    paths.extend(directory.parent.glob("[0-9]*/" + prefix + "*.mp4"))
    paths.extend(directory.glob(prefix + "*.mp4"))
    for path in paths:
        suffix = path.stem[len(prefix) :]
        if suffix.isdigit():
            numbers.append(int(suffix))
    number = max(numbers, default=0) + 1
    return directory / (prefix + f"{number:02d}.mp4")


def run(cfg, hardware, stop_file, loop=None):
    started_at = time.time()
    report = RunReport(cfg)
    piper = camera = client = logger = protocol = None
    handlers = {}
    code, error = 0, None
    logger_started = False
    lock = open("/tmp/piper-real-robot.lock", "a+")
    interrupted_flag = False

    def check_stop():
        return interrupted_flag or stop_requested(stop_file)

    def interrupted(signum, frame):
        nonlocal interrupted_flag
        interrupted_flag = True
        raise KeyboardInterrupt("Stop requested")

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        if check_stop():
            raise KeyboardInterrupt("Stop requested")
        directory = cfg.report_dir or str(Path(cfg.save_dir) / time.strftime("%Y%m%d_%H%M%S"))
        logger = hardware.InferenceLogger(directory, cfg.prompt)
        cfg.prediction_run_id = Path(directory).name
        client = PiperPolicyClient(
            f"ws://{cfg.server_host}:{cfg.server_port}",
            control_mode="eef" if cfg.control_mode == "eef" else "joint",
            stop_check=check_stop,
        )
        if any(client.metadata.get(k) != 6 for k in ("left_arm_dof", "right_arm_dof")):
            raise ValueError("Piper requires six joints per arm")
        protocol = CosmosProtocol(client, cfg)
        report.update(
            async_infer=dict(rtc_enabled=protocol.rtc_enabled, rtc_delay=protocol.rtc_delay),
            observation_upload_mode="rtc_async"
            if protocol.rtc_stride
            else "async_groups"
            if protocol.stream_observations
            else "full_block",
            observation_chunk_steps=protocol.observation_group if protocol.stream_observations else None,
        )
        piper = hardware.PiperController(can_left=cfg.can_left, can_right=cfg.can_right)
        piper.connect()
        piper.enable()
        if cfg.control_mode == "eef":
            for robot in (piper.robot_l, piper.robot_r):
                robot.set_speed_percent(int(cfg.movej_speed_percent * 100))
                robot.set_auto_set_motion_mode_enabled(True)
                robot.set_motion_mode("p")
        elif cfg.control_mode in ("origin", "movej", "movejs"):
            for robot in (piper.robot_l, piper.robot_r):
                robot.set_speed_percent(int(cfg.movej_speed_percent * 100))
                hardware.restore_movej_mode(robot)
        else:
            for robot in (piper.robot_l, piper.robot_r):
                hardware.setup_cpv_mode(robot, cfg.cpv_acc, cfg.cpv_dcc, cfg.cpv_cv)
            joints = piper.get_joint_angles()
            hardware.send_cpv_joints(piper.robot_l, joints[:6])
            hardware.send_cpv_joints(piper.robot_r, joints[6:12])
            time.sleep(0.2)
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("cosmos_piper_client", anonymous=True, disable_signals=True)
        camera = hardware.CameraManager(
            cam_high_topic=cfg.cam_high,
            cam_left_topic=cfg.cam_left,
            cam_right_topic=cfg.cam_right,
            width=cfg.cam_width,
            height=cfg.cam_height,
            fps=cfg.cam_fps,
        )
        camera.start()
        smoother = None
        if cfg.smooth_method != "none" and cfg.control_mode != "eef":
            smoother = hardware.JointTrajectorySmoother(
                smooth_method=cfg.smooth_method,
                smooth_factor=cfg.smooth_factor,
                v_max=cfg.v_max,
                dt=cfg.action_dt,
                outlier_threshold_deg=cfg.outlier_threshold,
                savgol_window=cfg.savgol_window,
                savgol_order=cfg.savgol_order,
                butter_cutoff=cfg.butter_cutoff,
            )
        logger.start(cfg.control_mode)
        logger_started = True
        if cfg.control_mode == "eef":
            info_path = Path(directory) / "task_info.json"
            if info_path.is_file():
                info = json.loads(info_path.read_text())
                info.update(action_space="eef_14_xyzrpy_metres_radians", quaternion_order="xyzw")
                info_path.write_text(json.dumps(info, indent=2) + "\n")
        (loop or run_loop)(
            piper,
            camera,
            protocol,
            cfg,
            logger,
            smoother,
            send_eef_action if cfg.control_mode == "eef" else hardware.send_action,
            check_stop,
            report,
        )
        if getattr(cfg, "save_prediction_video", False) and protocol.predicted_chunks and not protocol.rtc_enabled:
            previous_status = report.data["status"]
            report.update(status="saving_video")
            try:
                destination = prediction_video_path(directory, started_at)
                temporary = destination.with_name("." + destination.name)
                metrics = protocol.save_prediction_video(temporary, stop_check=check_stop)
                temporary.replace(destination)
                metrics["prediction_video_path"] = str(destination)
                report.update(**metrics)
                print("模型生成视频已保存：" + metrics["prediction_video_path"], flush=True)
            except Exception as exc:
                report.update(prediction_video_error=str(exc))
                traceback.print_exc()
            finally:
                report.update(status=previous_status)
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        error = str(exc) or type(exc).__name__
        report.update(status="stopped" if code == 130 else "failed", error=error)
        if code != 130:
            traceback.print_exc()
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if piper is not None:
            # Match Lingbot: ordinary exceptions close connections without electronic estop.
            if cfg.control_mode == "cpv":
                for robot in (piper.robot_l, piper.robot_r):
                    try:
                        hardware.restore_movej_mode(robot)
                    except Exception:
                        traceback.print_exc()
        # The worker shares the inference socket: no video/close RPC may overlap it.
        if protocol is not None:
            if protocol.prediction_debug_dir:
                report.update(prediction_debug_dir=protocol.prediction_debug_dir, prediction_decode="on_disconnect")
                print("RTC 预测已记录，断开连接后服务端解码：" + protocol.prediction_debug_dir, flush=True)
            try:
                protocol.close(abort=bool(code))
            except Exception as exc:
                if not code:
                    code, error = 1, str(exc)
                    report.update(status="failed", error=error)
                else:
                    report.update(observation_upload_error=str(exc))
                traceback.print_exc()
        if protocol is not None and logger is not None:
            try:
                timing_path = logger.save_dir / "piper_timings.json"
                summary = protocol.timings.save(timing_path)
                # Populate the legacy logger too, so it does not report a misleading zero.
                if hasattr(logger, "timings"):
                    events = protocol.timings.snapshot()["events"]
                    logger.timings["server_infer_s"] = [
                        row["server_infer_ms"] / 1000
                        for row in events
                        if row["event"] == "request_completed" and row["server_infer_ms"] is not None
                    ]
                    starts = {row["round"]: row["at_s"] for row in events if row["event"] == "execution_start"}
                    logger.timings["action_exec_s"] = [
                        row["at_s"] - starts[row["round"]]
                        for row in events
                        if row["event"] == "execution_end" and row["round"] in starts
                    ]
                report.update(timing_path=str(timing_path), timing_summary=summary)
                print("PIPER_TIMINGS " + json.dumps(summary), flush=True)
            except Exception:
                traceback.print_exc()
        for resource, method, args in [
            (client, "close", ()),
            (camera, "stop", ()),
            (piper, "close", ()),
            (logger if logger_started else None, "finish", (report.data["robot_commands_sent"], not code, error)),
        ]:
            if resource is not None:
                try:
                    getattr(resource, method)(*args)
                except Exception:
                    traceback.print_exc()
        lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return code


def main(runner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-client-dir", default="/home/agilex/models/lingbot-va-piper/client")
    parser.add_argument("--config-file", help="Flat client JSON or archived Cosmos website configuration")
    parser.add_argument("--prompt")
    parser.add_argument("--control-mode", "--control_mode", choices=["origin", "movej", "movejs", "cpv", "eef"])
    parser.add_argument("--max-chunks", "--max_chunks", type=int)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--output")
    parser.add_argument("--stop-file", default="/home/agilex/models/lingbot-va-piper/client/runtime_config.json")
    parser.add_argument(
        "--check", action="store_true", help="Validate dependencies/config; no socket or hardware connections"
    )
    args = parser.parse_args()
    hardware = load_lingbot(args.lingbot_client_dir)
    cfg = configuration(args, hardware)
    if args.check:
        print("Cosmos Piper client imports/config OK; no connections opened")
        return 0
    return (runner or run)(cfg, hardware, args.stop_file)


if __name__ == "__main__":
    raise SystemExit(main())
