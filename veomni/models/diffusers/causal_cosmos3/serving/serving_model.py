# SPDX-License-Identifier: OpenMDW-1.1
"""Native model worker, measured observation windows and physical action export."""

import math
import os
from functools import partial

import numpy as np


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def prepare_camera_observation(cameras, resolution="256"):
    """Compose RGB observations using the native training camera rectangles."""
    import av
    import torch
    from av.video.reformatter import Interpolation

    from veomni.data.causal_cosmos3.transforms.layout import camera_layout

    values = {key: np.asarray(cameras[key]) for key in CAMERA_KEYS}
    for value in values.values():
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 2:
            raise ValueError("Camera observations must be real RGB uint8 HWC images")
    info = dict(
        features={CAMERA_KEYS[0]: dict(names=["height", "width", "channels"], shape=values[CAMERA_KEYS[0]].shape)}
    )
    layout = camera_layout(
        info, dict(zip(("head", "left", "right"), CAMERA_KEYS, strict=True)), resolution, "concat_view"
    )
    height = max(camera.top + camera.height for camera in layout)
    width = max(camera.left + camera.width for camera in layout)
    canvas = torch.empty((3, height, width), dtype=torch.uint8)
    for camera in layout:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(values[camera.key]), format="rgb24")
        resized = frame.reformat(
            width=camera.width, height=camera.height, format="rgb24", interpolation=Interpolation.BICUBIC
        )
        canvas[:, camera.top : camera.top + camera.height, camera.left : camera.left + camera.width] = (
            torch.from_numpy(resized.to_ndarray()).permute(2, 0, 1)
        )
    return canvas, torch.tensor([height, width, height, width], dtype=torch.float32)


def adapter_from_metadata(metadata):
    from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55

    from .robotwin import RobotwinAdapter

    template = ActionTemplate55()
    if metadata.get("template_id") != template.template_id:
        raise ValueError("The remote model does not use the unified55 template")
    return RobotwinAdapter(
        template, state_mask=metadata["state_mask"], action_mask=metadata["action_mask"], **metadata["adapter"]
    )


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tensor(value, *, dtype=None):
    """Own NumPy wire storage before exposing it to a writable Tensor API."""
    import torch

    return torch.as_tensor(value.copy() if isinstance(value, np.ndarray) else value, dtype=dtype)


class ServingModelPolicy:
    """One serialized online episode using complete native/HF model assets."""

    def __init__(
        self,
        model,
        *,
        vae,
        tokenize,
        statistics,
        template=None,
        adapter=None,
        history_steps=None,
        history_blocks=None,
        action_horizon=32,
        execute_steps=16,
        handoff_steps=None,
        controller_hz=30.0,
        resolution="256",
        num_steps=10,
        guidance=1.0,
        cache_mode="full",
        enable_rtc=False,
        npu_replay_graph=False,
        reuse_reset_prefix=True,
        history_padding="repeat_first",
        streaming_vae=True,
        attention_backend="grouped",
        sampling_options=None,
        seed=42,
    ):
        from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55

        from .episode_prefix import EpisodePrefixCache
        from .robotwin import RobotwinAdapter
        from .window_streaming import ObservationWindowEncoder

        if streaming_vae is not True or history_padding != "repeat_first":
            raise ValueError("Online inference requires streaming_vae=True and repeat_first history")
        if history_steps is not None and history_blocks is not None:
            raise ValueError("Use history_steps or legacy history_blocks, not both")
        if history_blocks is not None:
            history_blocks = _integer(history_blocks, "history_blocks", 1)
        self.history_steps = _integer(
            1024
            if history_blocks is None and history_steps is None
            else history_blocks * 32
            if history_steps is None
            else history_steps,
            "history_steps",
        )
        self.action_horizon = _integer(action_horizon, "action_horizon", 1)
        if self.history_steps % 16 or self.action_horizon % 16:
            raise ValueError("history_steps and action_horizon must be divisible by 16")
        self.execute_steps = _integer(execute_steps, "execute_steps", 1)
        if self.execute_steps > self.action_horizon:
            raise ValueError("execute_steps must not exceed the action horizon")
        self.handoff_steps = (
            min(self.action_horizon, self.execute_steps + max(1, self.execute_steps // 2))
            if handoff_steps is None and self.execute_steps < self.action_horizon
            else handoff_steps
        )
        if self.handoff_steps is not None:
            _integer(self.handoff_steps, "handoff_steps", 1)
            if self.handoff_steps > self.action_horizon:
                raise ValueError("handoff_steps must not exceed the action horizon")
        if isinstance(controller_hz, bool) or not math.isfinite(controller_hz) or controller_hz <= 0:
            raise ValueError("controller_hz must be finite and positive")
        if cache_mode not in ("off", "plans", "full"):
            raise ValueError("cache_mode must be off, plans or full")
        if type(npu_replay_graph) is not bool:
            raise ValueError("npu_replay_graph must be a bool")
        if type(reuse_reset_prefix) is not bool:
            raise ValueError("reuse_reset_prefix must be a bool")
        if npu_replay_graph and enable_rtc:
            raise ValueError("NPU replay graphs and enable_rtc are mutually exclusive; disable RTC")
        if npu_replay_graph and (
            cache_mode != "full" or attention_backend != "npu_tnd" or next(model.parameters()).device.type != "npu"
        ):
            raise ValueError("NPU replay graphs require an NPU model, full cache and npu_tnd attention")
        if isinstance(guidance, bool) or not math.isfinite(guidance) or guidance <= 0:
            raise ValueError("guidance must be positive and finite")
        self.model = model.eval().requires_grad_(False)
        self.vae, self.tokenize, self.statistics = vae, tokenize, statistics
        self.template = template or ActionTemplate55()
        self.adapter = adapter or RobotwinAdapter.from_statistics(self.template, statistics)
        statistics.validate(self.template.width, self.adapter.state_mask, self.adapter.action_mask)
        geometry = statistics.provenance.get("geometry", {})
        if (
            geometry.get("actions_per_block"),
            geometry.get("video_stride"),
            geometry.get("temporal_compression_factor", geometry.get("temporal_compression")),
        ) != (32, 4, 4):
            raise ValueError("Serving statistics must declare the trained 32-action/stride4/VAE4 geometry")
        self.controller_hz, self.resolution = float(controller_hz), str(resolution)
        self.num_steps, self.guidance, self.seed = (
            _integer(num_steps, "num_steps", 1),
            float(guidance),
            _integer(seed, "seed"),
        )
        self.cache_mode, self.enable_rtc, self.attention_backend = cache_mode, bool(enable_rtc), attention_backend
        self.sampling_options = dict(sampling_options or {})
        if self.sampling_options.get("fixed_step_sampler_config") is not None:
            self.guidance = 1.0
        if npu_replay_graph and self.guidance != 1:
            raise ValueError("NPU replay graphs require guidance=1")
        self.npu_replay_graph = npu_replay_graph
        self.episode_prefix = (
            EpisodePrefixCache() if reuse_reset_prefix and cache_mode == "full" and self.history_steps else None
        )
        self.window_encoder = ObservationWindowEncoder(
            vae,
            history_steps=self.history_steps,
            action_horizon=self.action_horizon,
            history_padding="repeat_first",
            reuse_reset_prefix=reuse_reset_prefix,
        )
        self.clear()

    @property
    def metadata(self):
        return dict(
            actions_per_block=self.action_horizon,
            action_horizon=self.action_horizon,
            history_steps=self.history_steps,
            history_blocks=max(1, (self.history_steps + 31) // 32),
            layout_mode="decision_window",
            execute_steps=self.execute_steps,
            handoff_steps=self.handoff_steps,
            controller_hz=self.controller_hz,
            fps=self.controller_hz,
            history_padding="repeat_first",
            streaming_vae=True,
            video_stride=4,
            resolution=self.resolution,
            rtc_enabled=self.enable_rtc,
            npu_replay_graph=self.npu_replay_graph,
            reuse_reset_prefix=self.window_encoder.reuse_reset_prefix,
            state_mask=self.adapter.state_mask.numpy().copy(),
            action_mask=self.adapter.action_mask.numpy().copy(),
            adapter=dict(
                left_arm_dof=self.adapter.dofs["left"],
                right_arm_dof=self.adapter.dofs["right"],
                eef_pose_frame=self.adapter.eef_pose_frame,
                T_world_base=self.adapter.T_world_base,
            ),
            template_id=self.template.template_id,
            available_control_modes=list(self.adapter.available_control_modes),
            seed=self.seed,
            num_steps=self.num_steps,
            guidance=self.guidance,
        )

    def clear(self):
        from .npu_replay_graph import clear_model_graph

        clear_model_graph(self.model)
        if self.episode_prefix is not None:
            self.episode_prefix.clear()
        self.window_encoder.clear()
        self.builder = self.first_image = None

    def close(self):
        from .npu_replay_graph import clear_model_graph

        self.clear()
        clear_model_graph(self.model, close=True)

    def _measurement(self, request):
        import torch

        state = _tensor(request["state"], dtype=torch.float32).detach().cpu().clone()
        if state.shape != (self.template.width,):
            raise ValueError("State must be one physical unified55 measurement")
        image = _tensor(request["image"]).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError("Image must be one prepared real CPU uint8 CHW observation")
        return self.template.sanitize(state, self.adapter.state_mask), image.clone()

    def observe(self, request):
        if self.builder is None:
            raise ValueError("Reset before sending observations")
        self.window_encoder.add(_integer(request["tick"], "tick"), _tensor(request["image"]))
        return {}

    def prepare_window(self, request):
        if self.builder is None:
            raise ValueError("Reset before preparing an observation window")
        self.window_encoder.prepare(_integer(request["anchor_tick"], "anchor_tick"))
        return {}

    def infer(self, request):
        import torch

        from veomni.data.causal_cosmos3.transforms.layout import VIEW_DESCRIPTION, format_prompt

        from .rtc import RTCPrefix
        from .rtc_batch import RTCObservation, RTCObservedBatchBuilder
        from .sampling import sample_decision

        device = next(self.model.parameters()).device
        if device.type != "cpu":
            getattr(torch, device.type).set_device(device)
        operation = request.get("operation")
        tick, revision = _integer(request.get("tick", 0), "tick"), _integer(request.get("revision", 0), "revision")
        if operation not in ("reset", "predict"):
            raise ValueError("operation must be reset or predict")
        state, image = self._measurement(request)
        if operation == "reset":
            self.clear()
            prompt = request.get("prompt")
            if tick or revision or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Reset requires tick/revision zero and a nonempty prompt")
            size = _tensor(request["image_size"])
            if (
                size.shape != (4,)
                or not torch.isfinite(size).all()
                or not torch.equal(size, size.round())
                or tuple(size[:2].tolist()) != tuple(image.shape[1:])
                or (size[2:] <= 0).any()
                or (size[2:] > size[:2]).any()
            ):
                raise ValueError("image_size must describe the prepared canvas and valid content")
            self.first_image = image.clone()
            text = format_prompt(
                dict(
                    caption=prompt,
                    mode="policy",
                    image_size=image.shape[1:],
                    conditioning_fps=self.controller_hz / 4,
                    viewpoint="concat_view",
                    additional_view_description=VIEW_DESCRIPTION["concat_view"],
                )
            )
            self.builder = RTCObservedBatchBuilder(
                template=self.template,
                statistics=self.statistics,
                tokenize=self.tokenize,
                prompt=text,
                image_shape=image.shape,
                controller_hz=self.controller_hz,
                history_steps=self.history_steps,
                action_horizon=self.action_horizon,
            )
        elif self.builder is None:
            raise ValueError("Reset with a fresh observation before requesting predictions")
        prefix = request.get("rtc")
        if prefix is not None and (not self.enable_rtc or operation == "reset"):
            raise ValueError("RTC requires an enabled server and an ordinary bootstrap first")
        rtc = (
            None
            if prefix is None
            else RTCPrefix(
                _tensor(prefix["previous_actions"], dtype=torch.float32).clone(),
                prefix["delay_steps"],
                prefix.get("beta", 5.0),
            )
        )
        history = tuple(
            RTCObservation(_integer(item["tick"], "history tick"), _tensor(item["image"]))
            for item in request.get("history", [])
        )
        prepared = self.window_encoder.count if self.window_encoder.anchor == tick else 0
        try:
            for record in history:
                self.window_encoder.add(record.tick, record.image, prune=False)
            self.window_encoder.add(tick, image, prune=False)
            latent, timing = self.window_encoder.finish(tick)
        finally:
            self.window_encoder.prune()
        timing["sampled_frames_encoded_before_predict"] = prepared
        sample = self.builder(
            state=state,
            state_mask=self.adapter.state_mask,
            action_mask=self.adapter.action_mask,
            image=image,
            tick=tick,
            history=history,
            first_image=self.first_image,
            revision=revision,
            prepared_vision_prefix=latent,
        )
        options = dict(self.sampling_options)
        if self.guidance != 1:
            options["negative_text_ids"] = torch.as_tensor(self.tokenize(""), dtype=torch.long)
        if self.episode_prefix is not None:
            if tick >= self.history_steps:
                self.episode_prefix.clear()
            else:
                options["episode_cache"] = self.episode_prefix
        generated = sample_decision(
            self.model,
            sample,
            steps=self.num_steps,
            guidance=self.guidance,
            seed=self.seed,
            cache_mode=self.cache_mode,
            rtc=rtc,
            attention_backend=self.attention_backend,
            npu_replay_graph=self.npu_replay_graph,
            **options,
        )
        absolute = self.statistics.action.denormalize(generated["action"].cpu(), self.adapter.action_mask)
        actions = self.template.decode_action_delta(absolute, state, self.adapter.action_mask)
        prepared_prefix = generated["prepared_prefix"]
        if prepared_prefix is not None and prepared_prefix.delay_steps:
            actions[: prepared_prefix.delay_steps] = prepared_prefix.absolute_prefix[
                : prepared_prefix.delay_steps
            ].cpu()
        if not torch.isfinite(actions).all():
            raise ValueError("Model produced nonfinite physical actions")
        return dict(
            actions=actions.numpy(),
            tick=tick,
            revision=revision,
            streaming_timing=timing,
            inference_cache=generated["inference_cache"],
            rtc_vjp_calls=generated["rtc_vjp_calls"],
        )


def load_policy(args):
    """Load the native model, Diffusers Wan VAE and training tokenizer without a Trainer."""
    import torch
    import yaml

    from veomni.data.causal_cosmos3.transforms.action_template import ActionTemplate55
    from veomni.data.causal_cosmos3.transforms.tokenization import build_prompt_tokenizer, tokenize_prompt

    from ..cosmos3_core.vae import Cosmos3VideoVAE
    from ..inference import load_trained_model
    from .robotwin import RobotwinAdapter
    from .statistics import BlockStatistics

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The model service supports one model rank")
    if getattr(args, "npu_replay_graph", False) and args.enable_rtc:
        raise ValueError("--npu-replay-graph and --enable-rtc are mutually exclusive")
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401
    if device.type != "cpu":
        getattr(torch, device.type).set_device(device)
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    dtype = getattr(torch, args.dtype)
    ops = dict(config.get("ops_implementation", {}))
    if device.type == "cpu":
        ops.update(attn_implementation="eager", rms_norm_implementation="eager", rotary_pos_emb_implementation="eager")
    model = load_trained_model(args.checkpoint, device=device, dtype=dtype, ops_implementation=ops)
    vae = Cosmos3VideoVAE(args.vae, device=device, dtype=dtype)
    tokenize = partial(tokenize_prompt, tokenizer=build_prompt_tokenizer(args.processor))
    statistics = BlockStatistics.load(args.statistics)
    template = ActionTemplate55()
    adapter = RobotwinAdapter.from_statistics(template, statistics, args.robot_config)
    options = {
        key: config.get("sampling", {})[key]
        for key in ("shift", "guidance_interval", "normalize_cfg", "fixed_step_sampler_config")
        if config.get("sampling", {}).get(key) is not None
    }
    if "fixed_step_sampler_config" not in options and getattr(model.config, "fixed_step_sampler_config", None):
        options["fixed_step_sampler_config"] = model.config.fixed_step_sampler_config
    if options.get("fixed_step_sampler_config"):
        args.guidance = 1.0
    return ServingModelPolicy(
        model,
        vae=vae,
        tokenize=tokenize,
        statistics=statistics,
        template=template,
        adapter=adapter,
        history_steps=args.history_steps,
        history_blocks=args.history,
        action_horizon=args.action_horizon,
        execute_steps=args.execute_steps,
        handoff_steps=args.handoff_steps,
        controller_hz=args.fps,
        resolution=args.resolution,
        num_steps=args.steps,
        guidance=args.guidance,
        cache_mode=args.cache_mode,
        enable_rtc=args.enable_rtc,
        npu_replay_graph=getattr(args, "npu_replay_graph", False),
        reuse_reset_prefix=getattr(args, "reuse_reset_prefix", True),
        seed=args.seed,
        attention_backend=args.attention_backend,
        history_padding=args.history_padding,
        streaming_vae=args.streaming_vae,
        sampling_options=options,
    )
