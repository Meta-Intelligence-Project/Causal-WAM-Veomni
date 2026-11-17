"""Shared configuration overrides for offline and served Causal Cosmos3 inference."""

import argparse

import yaml


_ROOT_OPTIONS = (
    "device",
    "dtype",
    "attention_backend",
    "fps",
    "statistics",
    "tokenizer",
    "vae",
    "base_fps",
    "temporal_modality_margin",
)
_SAMPLING_OPTIONS = (
    "steps",
    "history",
    "seed",
    "observation_mode",
    "kv_cache",
    "npu_graph",
    "residual_reuse_full_steps",
    "shift",
    "guidance",
    "guidance_interval",
    "normalize_cfg",
    "negative_prompt",
)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _residual_schedule(value):
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as error:
        raise argparse.ArgumentTypeError("expected a refresh count, a list of indices, or null") from error
    if (
        parsed is None
        or (type(parsed) is int and parsed > 0)
        or (
            isinstance(parsed, list)
            and parsed
            and all(type(index) is int and index >= 0 for index in parsed)
            and parsed[0] == 0
            and all(a < b for a, b in zip(parsed, parsed[1:]))
        )
    ):
        return parsed
    raise argparse.ArgumentTypeError("expected a positive count, increasing indices starting at 0, or null")


def add_inference_arguments(parser):
    """Register shared overrides; omitted arguments leave YAML values untouched."""
    root = parser.add_argument_group("model and preprocessing overrides")
    root.add_argument("--device")
    root.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    root.add_argument("--attention-backend", choices=("dense", "grouped", "npu_tnd"))
    root.add_argument("--fps", type=float, help="Training-aligned action/control FPS")
    root.add_argument("--statistics")
    root.add_argument("--tokenizer")
    root.add_argument("--vae")
    root.add_argument("--base-fps", type=float)
    root.add_argument("--temporal-modality-margin", type=int)

    sampling = parser.add_argument_group("sampling overrides")
    sampling.add_argument("--steps", type=_positive_int)
    sampling.add_argument("--history", type=_positive_int, help="History budget in latent frames")
    sampling.add_argument("--seed", type=int)
    sampling.add_argument("--observation-mode", choices=("streaming", "window"))
    sampling.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reuse Transformer K/V (default: use YAML setting)",
    )
    sampling.add_argument(
        "--npu-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay NPU graphs for streaming Policy with KV and npu_tnd (default: use YAML setting)",
    )
    sampling.add_argument(
        "--residual-reuse-full-steps",
        type=_residual_schedule,
        default=argparse.SUPPRESS,
        metavar="COUNT|INDICES|null",
        help="Full decoder refresh count or quoted list, e.g. '[0, 1, 6, 8]'; null disables residual reuse",
    )
    sampling.add_argument("--shift", type=float)
    sampling.add_argument("--guidance", type=float)
    sampling.add_argument("--guidance-interval", nargs=2, type=float, metavar=("LO", "HI"))
    sampling.add_argument("--normalize-cfg", action=argparse.BooleanOptionalAction, default=None)
    sampling.add_argument("--negative-prompt")
    sampling.add_argument("--fixed-step-sigmas", nargs="+", type=float, metavar="SIGMA")


def apply_cli_overrides(config, args):
    """Apply explicit CLI values in place, including an explicit null residual schedule."""
    for name in _ROOT_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    for name in _SAMPLING_OPTIONS:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is not None or name == "residual_reuse_full_steps":
            config.setdefault("sampling", {})[name] = value
    if args.fixed_step_sigmas is not None:
        config.setdefault("sampling", {})["fixed_step_sampler_config"] = dict(
            t_list=args.fixed_step_sigmas, sample_type="sde"
        )
