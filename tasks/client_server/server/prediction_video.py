# SPDX-License-Identifier: OpenMDW-1.1
"""Durable, independent RTC forecasts; decode only after execution disconnects."""

import json
import logging
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from veomni.models.diffusers.causal_cosmos3.inference import write_video


logger = logging.getLogger(__name__)


class PredictionRecording:
    def __init__(self, root, run, *, fps, video_stride, horizon, delay, prompt):
        run = run or ("debug-" + uuid4().hex)
        if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run):
            raise ValueError("Invalid prediction run identifier")
        root = Path(root).expanduser().resolve()
        self.directory = root / run / "predictions"
        if root not in self.directory.resolve().parents:
            raise ValueError("Prediction directory escapes configured root")
        # A reset must never overwrite a previous recording.
        if self.directory.exists():
            self.directory = self.directory.with_name("predictions-" + uuid4().hex)
        self.directory.mkdir(parents=True)
        self.data = dict(
            status="recording",
            mode="rtc_independent_forecasts",
            fps=fps,
            video_stride=video_stride,
            horizon=horizon,
            delay=delay,
            prompt=prompt,
            clips=[],
        )
        self._manifest()

    def _manifest(self):
        temporary = self.directory / ".manifest.json"
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self.directory / "manifest.json")

    def capture(self, *, step, latents, pixels, state, actions):
        """Copy tensors to disk before replying; keep no GPU outputs alive."""
        index = len(self.data["clips"])
        stem = f"{index:04d}"
        torch.save(latents.detach().cpu().clone(), self.directory / (stem + ".pt"))
        np.savez(
            self.directory / (stem + ".npz"),
            observed_rgb=pixels.cpu().numpy(),
            state=state.cpu().numpy(),
            **{"action_" + key: np.asarray(value) for key, value in actions.items() if value is not None},
        )
        stride = self.data["video_stride"]
        self.data["clips"].append(
            dict(
                index=index,
                observation_step=step,
                latent_file=stem + ".pt",
                data_file=stem + ".npz",
                prediction=stem + ".mp4",
                observation=stem + "-observation.png",
                # Frame zero reconstructs the measured conditioning latent.
                frame_control_steps=list(range(step, step + self.data["horizon"] + 1, stride)),
                observation_frame_control_steps=[step]
                if index == 0
                else list(range(step - pixels.shape[1] * stride + stride, step + 1, stride)),
                status="recorded",
            )
        )
        self._manifest()

    @torch.no_grad()
    def finish(self, vae):
        from PIL import Image

        self.data["status"] = "decoding"
        self._manifest()
        fps = self.data["fps"] / self.data["video_stride"]
        for clip in self.data["clips"]:
            try:
                with np.load(self.directory / clip["data_file"]) as data:
                    observed = data["observed_rgb"][:, -1].transpose(1, 2, 0)
                    Image.fromarray(observed).save(self.directory / clip["observation"])
                latents = torch.load(self.directory / clip["latent_file"], map_location="cpu", weights_only=True)
                pixels = vae.decode(latents).cpu()
                if pixels.shape[1] != len(clip["frame_control_steps"]):
                    raise ValueError("Decoded prediction length does not match control steps")
                target = self.directory / clip["prediction"]
                temporary = target.with_name("." + target.name)
                write_video(temporary, pixels, fps)
                temporary.replace(target)
                clip.update(status="complete", frames=pixels.shape[1])
            except Exception as exc:
                clip.update(status="failed", error=str(exc))
                logger.exception("Prediction decode failed: %s", self.directory / clip["latent_file"])
            self._manifest()
        self.data["status"] = "failed" if any(c["status"] == "failed" for c in self.data["clips"]) else "complete"
        self._manifest()
        logger.info("Prediction recording %s: %s", self.data["status"], self.directory)
