"""Serve Policy/IDM inference to one unified55 client."""

import argparse
import logging
import math
import os
import threading
import time
from pathlib import Path

import yaml

from tasks.client_server.wire import packb, unpackb


logger = logging.getLogger(__name__)


class InferenceServer:
    """Own the listening socket, connected clients and the model session lifetime."""

    def __init__(self, policy, *, host, port):
        # Keep the legacy keyword for callers; the object is a task-neutral session.
        from websockets.sync.server import serve

        self.session = policy
        self._reservation = threading.Lock()
        self._lifecycle = threading.Condition()
        self._connections = set()
        self._stopping = False
        self._server = serve(
            self._handler,
            host,
            port,
            compression=None,
            max_size=128 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        )

    @property
    def socket(self):
        return self._server.socket

    def _handler(self, connection):
        from websockets.exceptions import ConnectionClosed

        with self._lifecycle:
            stopping = self._stopping
            if not stopping:
                self._connections.add(connection)
        if stopping:
            connection.close(code=1001, reason="Server shutting down")
            return
        try:
            if not self._reservation.acquire(blocking=False):
                connection.close(code=1013, reason="An inference client is already connected")
                return
            try:
                # Device selection is thread-local on accelerators.
                if self.session.pipe.device.type != "cpu":
                    import torch

                    torch.get_device_module(self.session.pipe.device).set_device(self.session.pipe.device)
                self.session.reset()
                connection.send(packb(self.session.metadata))
                for message in connection:
                    started = time.perf_counter()
                    try:
                        result = self.session.infer(unpackb(message))
                        result["server_timing"] = dict(infer_ms=(time.perf_counter() - started) * 1000)
                    except Exception as error:
                        logger.exception("Inference request failed")
                        result = dict(error=str(error), reset_required=self.session.needs_reset)
                    connection.send(packb(result))
            except ConnectionClosed:
                pass
            finally:
                try:
                    self.session.reset()
                finally:
                    self._reservation.release()
        finally:
            with self._lifecycle:
                self._connections.discard(connection)
                self._lifecycle.notify_all()

    def serve_forever(self):
        self._server.serve_forever()

    def shutdown(self):
        # websockets.sync.Server.shutdown() only closes its listener. Explicitly
        # close live clients, then wait until any in-flight model call finishes.
        with self._lifecycle:
            first_shutdown = not self._stopping
            self._stopping = True
            connections = list(self._connections)
        if first_shutdown:
            self._server.shutdown()
        for connection in connections:
            connection.close(code=1001, reason="Server shutting down")
        with self._lifecycle:
            self._lifecycle.wait_for(lambda: not self._connections)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def create_server(policy, *, host="0.0.0.0", port=8000):
    return InferenceServer(policy, host=host, port=port)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/server/causal_cosmos3_server.yaml")
    p.add_argument(
        "--checkpoint", required=True, help="Native/HF checkpoint directory or DCP global_step_N/model/ckpt"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device")
    p.add_argument("--statistics")
    p.add_argument("--adapter-config", help="JSON with canonical unified55 masks and optional arm DOFs")
    p.add_argument("--fps", type=float, help="Training-aligned action/control FPS")
    p.add_argument("--steps", type=int)
    p.add_argument("--history", type=int, help="History budget in latent frames, as in native inference")
    p.add_argument("--seed", type=int)
    p.add_argument("--observation-mode", choices=("streaming", "window"))
    p.add_argument("--task", choices=("policy", "inverse_dynamics"), help="Override YAML task (default: policy)")
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse Transformer K/V with the selected observation strategy (default: enabled)",
    )
    return p


def main():
    args = parser().parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Inference serving requires one process; launch with python, not multi-rank torchrun")
    logging.basicConfig(level=logging.INFO)
    config_path = Path(args.config).expanduser().resolve()
    logger.info("Inference configuration: %s", config_path)
    config = yaml.safe_load(config_path.read_text())
    task = args.task if args.task is not None else config.get("task", "policy")
    if task not in ("policy", "inverse_dynamics"):
        raise ValueError("Server task must be policy or inverse_dynamics")
    config["task"] = task
    if args.observation_mode is not None:
        config.setdefault("sampling", {})["observation_mode"] = args.observation_mode
    if args.kv_cache is not None:
        config.setdefault("sampling", {})["kv_cache"] = args.kv_cache
    for name in ("device", "statistics", "fps"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    for name in ("steps", "history", "seed"):
        value = getattr(args, name)
        if value is not None:
            if name in ("steps", "history") and value < 1:
                raise ValueError(f"{name} must be positive")
            config.setdefault("sampling", {})[name] = value
    if args.adapter_config:
        import json

        config["adapter"] = json.loads(Path(args.adapter_config).read_text())
    fps = config.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"{config_path}: fps must be a finite positive action/control rate. "
            "Set fps in this YAML or pass --fps matching the training-aligned action/control rate. "
            "The launcher reads INFER_CONFIG, which may select an offline inference YAML without fps."
        )
    logger.info("Inference fps=%s, task=%s, kv_cache=%s", fps, task, config.get("sampling", {}).get("kv_cache", True))
    import torch

    if str(config["device"]).split(":")[0] == "npu":
        import torch_npu  # noqa: F401

    device = torch.device(config["device"])
    if device.type != "cpu":
        torch.get_device_module(device).set_device(device)
    from tasks.client_server.server.session import CausalInferenceSession
    from veomni.models.diffusers.causal_cosmos3.inference import CausalCosmos3Pipeline

    pipe = CausalCosmos3Pipeline.from_pretrained(args.checkpoint, config)
    session = CausalInferenceSession(pipe, fps=fps, task=task)
    with create_server(session, host=args.host, port=args.port) as server:
        logger.info("Causal inference listening on ws://%s:%s", args.host, server.socket.getsockname()[1])
        server.serve_forever()


if __name__ == "__main__":
    main()


# Backward-compatible class name; new code uses InferenceServer.
PolicyServer = InferenceServer
