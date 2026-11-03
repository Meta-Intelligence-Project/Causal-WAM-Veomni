# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cosmos DMD2 fixed-sigma SDE sampling with the local scheduler interface.

The source predicts x0 at each discrete sigma, then adds fresh noise at the
next sigma. The step interface lets both samplers reuse block conditioning,
CFG and FP32 state updates. A caller-owned Generator isolates fresh noise.
"""

import math

import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput


class FixedStepSampler:
    def __init__(self, t_list, sample_type="sde", num_train_timesteps=1000):
        self.sigmas = [float(t) for t in t_list]
        if not self.sigmas or any(not math.isfinite(t) or not 0 <= t <= 1 for t in self.sigmas):
            raise ValueError("t_list must contain finite sigmas in [0, 1]")
        if self.sigmas[-1] != 0:
            self.sigmas.append(0.0)
        if len(self.sigmas) < 2 or any(a <= b for a, b in zip(self.sigmas, self.sigmas[1:])):
            raise ValueError("t_list must be strictly descending to zero")
        if sample_type != "sde":
            raise ValueError("FixedStepSampler supports only sample_type='sde'")
        self.num_train_timesteps = num_train_timesteps
        self.set_timesteps()

    def set_timesteps(self, num_inference_steps=None, device=None, shift=None):
        """Use the checkpoint's sigma list; steps and shift do not override it."""
        self.timesteps = torch.tensor(
            [sigma * self.num_train_timesteps for sigma in self.sigmas[:-1]], device=device, dtype=torch.float32
        )
        self._step_index = 0

    def step(self, model_output, timestep, sample, *, generator=None, return_dict=False):
        sigma, sigma_next = self.sigmas[self._step_index : self._step_index + 2]
        x0 = sample.float() - sigma * model_output.float()
        if sigma_next > 0:
            noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
            result = (1.0 - sigma_next) * x0 + sigma_next * noise
        else:
            result = x0
        self._step_index += 1
        return SchedulerOutput(prev_sample=result) if return_dict else (result,)
