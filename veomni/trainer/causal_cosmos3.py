"""适配 action 数据构建，复用 DiT 的标准训练循环。"""

from dataclasses import asdict

import yaml

from ..data.causal_cosmos3.pipeline import ActionBatchingStrategy, build_action_dataset, collate_action_pack
from ..data.data_loader import build_native_dataloader
from ..distributed.parallel_state import get_parallel_state
from .dit_trainer import DiTTrainer, VeOmniDiTArguments


class CausalCosmos3Trainer(DiTTrainer):
    """通过父类的数据构建钩子接入 action pack。

    一个 microbatch 对应一个 pack，每个 pack 包含的样本数量可以不同。
    每次 optimizer 更新，每个 DP rank 消费 global_batch_size / dp_size 个 pack。
    模型 runtime 构建、前向反向和训练循环均继承父类。
    """

    def __init__(self, args: VeOmniDiTArguments, *, tokenize):
        # 父类构造函数会调用子类的构建方法，因此需要先保存 tokenize 回调。
        self.tokenize = tokenize
        super().__init__(args)

    def _setup(self):
        """校验 packing 配置，并复用父类的分布式初始化。"""
        args = self.base.args
        if not callable(self.tokenize):
            raise ValueError("Pass an explicit tokenize(prompt) callback")
        if args.train.training_task != "online_training":
            raise ValueError("The action pipeline currently requires online_training")
        if args.train.micro_batch_size != 1:
            raise ValueError("Action packing requires micro_batch_size=1: one microbatch is one pack")
        if args.train.max_steps is None or args.train.max_steps < 1:
            raise ValueError("Action packing requires positive train.max_steps (optimizer steps per epoch)")
        if args.data.dataloader.type != "native" or args.train.dyn_bsz_runtime != "main":
            raise ValueError("Action packing requires the native dataloader with dyn_bsz_runtime='main'")
        if min(args.data.max_seq_len, args.data.dyn_bsz_buffer_size) < 1:
            raise ValueError("Action token budget and packing buffer size must be positive")
        super()._setup()
        parallel = get_parallel_state()
        if parallel.sp_size != 1:
            raise ValueError("The action pipeline currently supports DP with SP/CP=1")
        if args.train.global_batch_size < parallel.dp_size or args.train.global_batch_size % parallel.dp_size:
            raise ValueError("global_batch_size must be a positive multiple of dp_size")
        # 父类 setup 默认设置固定 batch，这里再启用按 token 预算动态组包。
        args.train.dyn_bsz = True
        # 底层 DataLoader 每次读取一个候选样本，再交给主进程的 packing 缓冲区组合。
        args.train.dataloader_batch_size = 1

    def _build_data_transform(self):
        """提供最终 prompt 的 tokenize 回调，供 Dataset 准确统计文本 token 数量。"""
        self.base.data_transform = self.tokenize

    def _build_dataset(self):
        """构建 action 数据来源，并显式设置每轮的 optimizer 步数。"""
        args = self.base.args
        with open(args.data.train_path) as stream:
            config = yaml.safe_load(stream)
        # 来源 YAML 提供采样和几何配置；随机种子统一采用训练配置。
        self.base.train_dataset = build_action_dataset(
            {**config, "seed": args.train.seed}, tokenize=self.base.data_transform
        )
        # 每个 pack 的样本数可变，不能根据原始候选数量推算 optimizer 步数。
        args._train_steps = args.train.max_steps
        self.base.train_steps = args.train_steps

    def _build_dataloader(self):
        """每个 optimizer step 返回一组 microbatch，单项格式为 {"samples": [...]}。"""
        args = self.base.args
        loader_kwargs = asdict(args.data.dataloader)
        # loader 类型和后台预取由外层控制，不作为原生 loader 的构建参数传入。
        loader_kwargs.pop("type")
        loader_kwargs.pop("use_background_prefetcher")
        self.base.train_dataloader = build_native_dataloader(
            dataset=self.base.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            dyn_bsz=True,
            dyn_bsz_runtime="main",
            bsz_warmup_ratio=0,
            batching_strategy=ActionBatchingStrategy(args.data.max_seq_len, args.data.dyn_bsz_buffer_size),
            collate_fn=collate_action_pack,
            seed=args.train.seed,
            save_steps=args.train.checkpoint.save_steps,
            **loader_kwargs,
        )
