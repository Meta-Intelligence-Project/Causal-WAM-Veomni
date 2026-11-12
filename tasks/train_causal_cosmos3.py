"""解析 Cosmos3 checkpoint 资源并接入 video/action 训练数据和文本编码。"""

import argparse
from functools import partial
from pathlib import Path

import yaml

from veomni.arguments.parser import _add_arguments_recursive, _deep_update, _instantiate_recursive, _string_to_bool
from veomni.data.causal_cosmos3.transforms.tokenization import build_prompt_tokenizer, tokenize_prompt
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer
from veomni.trainer.dit_trainer import VeOmniDiTArguments


def configure_gradient_accumulation(args, grad_accum_iter):
    """Convert an explicit accumulation count using the standard batch setup."""
    train, accelerator = args.train, args.model.accelerator
    if grad_accum_iter is not None:
        if isinstance(grad_accum_iter, bool) or not isinstance(grad_accum_iter, int) or grad_accum_iter < 1:
            raise ValueError("train.grad_accum_iter must be a positive integer")
        train.global_batch_size = train.micro_batch_size * accelerator.dp_size * grad_accum_iter
        train._derive_batch_config(accelerator)


def parse_causal_cosmos3_args(argv=None):
    """Assemble experiment/source settings while retaining typed TBPTT overrides."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("config_file", nargs="?", help="Path to YAML config file")
    _add_arguments_recursive(parser, VeOmniDiTArguments)
    parser.add_argument("--train.grad_accum_iter", type=int, default=argparse.SUPPRESS)
    # model_config is a free-form dict, so its typed leaves need explicit CLI options.
    parser.add_argument(
        "--enable-tbptt",
        "--model.model_config.enable_tbptt",
        dest="model.model_config.enable_tbptt",
        type=_string_to_bool,
        default=argparse.SUPPRESS,
        help="Override the YAML TBPTT switch without modifying its file (true/false).",
    )
    parser.add_argument(
        "--tbptt-segment-steps",
        "--model.model_config.tbptt_segment_steps",
        dest="model.model_config.tbptt_segment_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Override the TBPTT temporal segment span in steps; stage alignment still applies.",
    )
    parser.add_argument(
        "--decision-stride-steps",
        "--data.sample.decision_stride_steps",
        dest="data.sample.decision_stride_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Action episode decision spacing in control steps; leaves action_horizon unchanged.",
    )
    parsed = vars(parser.parse_args(argv))
    config_file = parsed.pop("config_file")
    config = {}
    if config_file and config_file.endswith((".yaml", ".yml")):
        with open(config_file) as stream:
            config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Cosmos3 training YAML must be a mapping")
    overrides = {}
    for key, value in parsed.items():
        keys = key.split(".")
        target = overrides
        for part in keys[:-1]:
            target = target.setdefault(part, {})
        target[keys[-1]] = value
    config = _deep_update(config, overrides)
    data_config = config.setdefault("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("data must be a mapping")
    stage = data_config.pop("stage", None)
    if stage not in ("action", "video"):
        raise ValueError("data.stage must be action or video")
    dataset_config = {"stage": stage}
    for section in ("geometry", "candidates", "sample"):
        values = data_config.pop(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"data.{section} must be a mapping")
        dataset_config[section] = values
    train_config = config.setdefault("train", {})
    steps = train_config.pop("grad_accum_iter", None)
    if steps is not None:
        # The explicit accumulation count takes precedence over a saved batch size.
        train_config.pop("global_batch_size", None)
    args = _instantiate_recursive(VeOmniDiTArguments, config)
    configure_gradient_accumulation(args, steps)
    with open(args.data.train_path) as stream:
        sources = yaml.safe_load(stream)
    if not isinstance(sources, dict):
        raise ValueError("Cosmos3 sources YAML must be a mapping")
    unknown = set(sources) - {"sampling", "validation", "sources"}
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(
            "Sources YAML supports only sampling, validation and sources; "
            f"unexpected fields: {names}. Put experiment fields under data in the training YAML."
        )
    if not isinstance(sources.get("sources"), list) or not sources["sources"]:
        raise ValueError("Sources YAML must contain a nonempty sources list")
    if "validation" in sources and not isinstance(sources["validation"], dict):
        raise ValueError("Sources YAML validation must be a mapping")
    dataset_config.update(sources)
    condition_options = args.model.condition_model_cfg
    if not isinstance(condition_options, dict):
        raise ValueError("model.condition_model_cfg must be a mapping")
    if "stage" in condition_options and condition_options["stage"] != stage:
        raise ValueError("Condition stage must match the data stage")
    condition_options["stage"] = stage
    return args, dataset_config


def configure_args(args):
    """由完整预训练根目录解析组件路径，结构配置和 DCP 恢复参数独立保留。"""
    if args.model.model_path is None:
        raise ValueError("model.model_path must point to the pretrained Cosmos3 checkpoint root")
    root = Path(args.model.model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cosmos3 checkpoint root does not exist: {root}")
    if Path(args.model.config_path).resolve() == root:
        raise ValueError(
            "Set model.config_path to a native Cosmos3 architecture config separately from model.model_path"
        )
    for component in ("text_tokenizer", "vae"):
        if not (root / component).is_dir():
            raise FileNotFoundError(f"Missing Cosmos3 checkpoint component: {root / component}")
    # 此处只解析资源路径，权重加载与完整 checkpoint 恢复仍交给模型 runtime。
    args.model.model_path = str(root / "transformer")
    args.model.tokenizer_path = str(root / "text_tokenizer")
    args.model.condition_model_cfg["vae_path"] = str(root / "vae")
    return args


def main():
    args, dataset_config = parse_causal_cosmos3_args()
    args = configure_args(args)
    tokenizer = build_prompt_tokenizer(args.model.tokenizer_path)
    # Dataset 使用最终 IDs 统计长度；模型和 condition 不再追加文本边界 token。
    tokenize = partial(tokenize_prompt, tokenizer=tokenizer)
    trainer = CausalCosmos3Trainer(args, dataset_config=dataset_config, tokenize=tokenize)
    trainer.train()


if __name__ == "__main__":
    main()
