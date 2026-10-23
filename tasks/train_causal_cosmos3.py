"""解析 Cosmos3 checkpoint 资源并接入 action 训练数据和文本编码。"""

from functools import partial
from pathlib import Path

from veomni.arguments import parse_args
from veomni.data.causal_cosmos3.tokenization import build_prompt_tokenizer, tokenize_prompt
from veomni.trainer.causal_cosmos3 import CausalCosmos3Trainer
from veomni.trainer.dit_trainer import VeOmniDiTArguments


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
    args = configure_args(parse_args(VeOmniDiTArguments))
    tokenizer = build_prompt_tokenizer(args.model.tokenizer_path)
    # Dataset 使用最终 IDs 统计长度；模型和 condition 不再追加文本边界 token。
    tokenize = partial(tokenize_prompt, tokenizer=tokenizer)
    trainer = CausalCosmos3Trainer(args, tokenize=tokenize)
    trainer.train()


if __name__ == "__main__":
    main()
