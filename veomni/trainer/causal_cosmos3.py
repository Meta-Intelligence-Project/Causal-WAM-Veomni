"""Cosmos3 video/action training and deterministic loss-only validation."""

import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FSDPModule

from ..checkpoint import layout
from ..data.causal_cosmos3.datasets.dataset import MODES
from ..data.causal_cosmos3.datasets.pipeline import ActionBatchingStrategy, build_dataset, collate_action_pack
from ..data.causal_cosmos3.datasets.split import split_dataset_config
from ..data.causal_cosmos3.datasets.validation import (
    VALIDATION_VERSION,
    canonical_json,
    fingerprint,
    iter_validation_batches,
)
from ..data.data_loader import build_native_dataloader
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..models import build_config
from ..utils import helper
from ..utils.device import get_device_rng_state, set_device_rng_state
from ..utils.dist_utils import all_reduce, any_rank_failed, raise_if_any_rank_failed
from .callbacks.evaluate_callback import EvaluateCallback
from .callbacks.global_state_callback import GlobalStateCallback
from .dit_trainer import DiTTrainer, VeOmniDiTArguments


logger = helper.create_logger(__name__)
MODE_TAGS = {"forward_dynamics": "fd", "inverse_dynamics": "id", "policy": "policy", "video": "video"}


class CausalCosmos3RestartDataLoader:
    """Forward loading and epoch changes without copying or restoring pack state."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def state_dict(self):
        # The shared prefetcher calls this after each pack. Do not snapshot the
        # underlying packer's residual CPU videos, even at checkpoint time.
        return None

    def load_state_dict(self, state):
        # Older checkpoints can contain a data cursor; restart ignores it.
        pass


class CausalCosmos3GlobalStateCallback(GlobalStateCallback):
    """Keep training progress/RNG recovery, but restart the Cosmos3 data stream."""

    def state_dict(self, state):
        payload = super().state_dict(state)
        payload["data_resume_mode"] = "restart"
        return payload

    def _read_current(self, load_path):
        # The shared reader also reads loader/rank_*.pt. Old video-buffer
        # snapshots can be large and are unnecessary for restart recovery.
        extra_path = Path(layout.extra_state_path(load_path, self.rank))
        if not extra_path.exists():
            return None
        return torch.load(extra_path, map_location="cpu", weights_only=False)

    def _restore_position(self, global_state):
        # This hook runs for current and legacy layouts before data restoration.
        global_state["train_dataloader"] = None
        global_state["data_resume_mode"] = "restart"
        super()._restore_position(global_state)


class CausalCosmos3Trainer(DiTTrainer):
    """通过父类的数据构建钩子接入 video/action pack。

    一个 microbatch 对应一个 pack，每个 pack 包含的样本数量可以不同。
    每次 optimizer 更新，每个 DP rank 消费 global_batch_size / dp_size 个 pack。
    沿用父类的模型构建顺序、前向反向和训练循环。
    """

    def __init__(self, args: VeOmniDiTArguments, *, tokenize):
        # 父类构造函数会调用子类的构建方法，因此需要先保存 tokenize 回调。
        self.tokenize = tokenize
        self.segment_memories = {}
        super().__init__(args)
        previous = self.base.global_state_callback
        callback = CausalCosmos3GlobalStateCallback(self.base)
        self.base.global_state_callback = callback
        self.base._callbacks[self.base._callbacks.index(previous)] = callback
        if self.base.validation_dataset is not None:
            if getattr(self, "episode_training", False):
                from .causal_cosmos3_episode import EpisodeValidator

                self.validator = EpisodeValidator(self.base, self.base.validation_dataset)
            else:
                self.validator = CausalCosmos3Validator(self.base, self.base.validation_dataset)
            previous = self.base.evaluate_callback
            callback = CausalCosmos3EvaluateCallback(self.base, self.validator)
            self.base.evaluate_callback = callback
            self.base._callbacks[self.base._callbacks.index(previous)] = callback

    def _setup(self):
        """在模型构建前确定阶段，校验 packing 配置并复用父类初始化。"""
        args = self.base.args
        # Cosmos3 sources use their own YAML schema and sampling metadata. The
        # generic text multisource tracker expects a different `names` schema.
        args.data.enable_multisource = False
        if not callable(self.tokenize):
            raise ValueError("Pass an explicit tokenize(prompt) callback")
        if args.train.training_task != "online_training":
            raise ValueError("The Cosmos3 pipeline currently requires online_training")
        if args.train.micro_batch_size != 1:
            raise ValueError("Cosmos3 packing requires micro_batch_size=1: one microbatch is one pack")
        if args.train.max_steps is None or args.train.max_steps < 1:
            raise ValueError("Cosmos3 packing requires positive train.max_steps (optimizer steps per epoch)")
        if args.data.dataloader.type != "native" or args.train.dyn_bsz_runtime != "main":
            raise ValueError("Cosmos3 packing requires the native dataloader with dyn_bsz_runtime='main'")
        if min(args.data.max_seq_len, args.data.dyn_bsz_buffer_size) < 1:
            raise ValueError("Cosmos3 token budget and packing buffer size must be positive")
        # DiTTrainer builds the model before the dataset, so resolve the data
        # stage here and reuse this same source configuration for both splits.
        with open(args.data.train_path) as stream:
            self.source_config = yaml.safe_load(stream)
        if not isinstance(self.source_config, dict):
            raise ValueError("Cosmos3 data configuration must be a mapping")
        stage = self.source_config.get("stage", "action")
        if stage not in ("action", "video"):
            raise ValueError("Cosmos3 data stage must be action or video")
        condition_options = args.model.condition_model_cfg
        if "stage" in condition_options and condition_options["stage"] != stage:
            raise ValueError("Condition stage must match the data stage")
        # Check the effective architecture, including JSON defaults and YAML
        # overrides, without loading weights or changing the selected heads.
        model_config = build_config(args.model.config_path, **args.model.model_config)
        if stage == "video":
            if model_config.action_interface != "none" or model_config.state_dim != 0:
                raise ValueError("Video stage requires action_interface='none' and state_dim=0")
            block = self.source_config.get("geometry", {}).get("latent_block_size", 2)
            if model_config.enable_tbptt and (
                type(block) is not int or block < 1 or model_config.tbptt_keep_history_latents % block
            ):
                raise ValueError("tbptt_keep_history_latents must be divisible by the video's latent block size")
        elif model_config.action_interface != "shared" or min(model_config.action_dim, model_config.state_dim) <= 0:
            raise ValueError("Action stage requires action_interface='shared' and positive action_dim/state_dim")
        self.episode_training = stage == "action" and model_config.enable_tbptt
        encoding = "episode_stream" if self.episode_training else "window"
        args.model.model_config["history_encoding"] = encoding
        args.model.model_config["training_sampling_unit"] = (
            "episode_lane_fifo" if self.episode_training else "independent_window"
        )
        condition_options["history_encoding"] = encoding
        if self.episode_training and not args.data.dataloader.in_order:
            raise ValueError("Episode TBPTT requires dataloader.in_order=True")
        if self.episode_training:
            from ..data.causal_cosmos3.transforms.processing import Geometry

            geometry = Geometry(**self.source_config.get("geometry", {}))
            horizon = self.source_config.get("sample", {}).get("action_horizon", 32)
            if (
                type(horizon) is not int
                or horizon < 1
                or horizon % geometry.actions_per_latent
                or model_config.tbptt_keep_history_latents % (horizon // geometry.actions_per_latent)
            ):
                raise ValueError("Episode TBPTT K must contain a whole number of VAE-aligned Action horizons")
            if self.source_config.get("sampling", "sample_uniform") != "sample_uniform":
                raise ValueError("Episode TBPTT currently requires sampling: sample_uniform")
        condition_options["stage"] = stage
        super()._setup()
        parallel = get_parallel_state()
        if parallel.sp_size != 1:
            raise ValueError("The Cosmos3 pipeline currently supports DP with SP/CP=1")
        if args.train.global_batch_size < parallel.dp_size or args.train.global_batch_size % parallel.dp_size:
            raise ValueError("global_batch_size must be a positive multiple of dp_size")
        # 父类 setup 默认设置固定 batch，这里再启用按 token 预算动态组包。
        args.train.dyn_bsz = not self.episode_training
        # 底层 DataLoader 每次读取一个候选样本，再交给主进程的 packing 缓冲区组合。
        args.train.dataloader_batch_size = (
            args.train.global_batch_size // parallel.dp_size if self.episode_training else 1
        )

    def _build_data_transform(self):
        """提供最终 prompt 的 tokenize 回调，供 Dataset 准确统计文本 token 数量。"""
        self.base.data_transform = self.tokenize

    def _build_dataset(self):
        """按已确定的阶段构建训练/验证数据，并设置每轮的 optimizer 步数。"""
        args = self.base.args
        error, split_manifest = None, None
        self.base.validation_config = None
        self.base.validation_dataset = None
        try:
            source_config = self.source_config
            train_config = source_config
            validation_config = None
            if "split_ratio" in source_config.get("validation", {}):
                train_config, validation_config, split_manifest = split_dataset_config(source_config)
            self.base.validation_config = validation_config
            if getattr(self, "episode_training", False):
                from ..data.causal_cosmos3.datasets.pipeline import build_episode_dataset

                self.base.train_dataset = build_episode_dataset(
                    {**train_config, "seed": args.train.seed},
                    tokenize=self.base.data_transform,
                    segment_latents=self.base.model.model.config.tbptt_keep_history_latents,
                )
            else:
                self.base.train_dataset = build_dataset(
                    {**train_config, "seed": args.train.seed}, tokenize=self.base.data_transform
                )
            if validation_config is not None:
                # Enumerate all held-out sources regardless of training weights.
                self.base.validation_dataset = build_dataset(
                    {**validation_config, "sampling": "sample_uniform"}, tokenize=self.base.data_transform
                )
        except Exception as caught:
            error = caught
        raise_if_any_rank_failed(
            error, "Cosmos3 dataset construction", group=getattr(get_parallel_state(), "dp_group", None)
        )
        if split_manifest is not None:
            self.base.episode_split_manifest = split_manifest
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
        if getattr(self, "episode_training", False):
            from ..data.causal_cosmos3.datasets.episode import collate_episode_segment
            from .causal_cosmos3_episode import EpisodeLoaderItems

            loader_kwargs["persistent_workers"] = False
            self.base.train_dataloader = CausalCosmos3RestartDataLoader(
                build_native_dataloader(
                    dataset=EpisodeLoaderItems(self.base.train_dataset),
                    micro_batch_size=1,
                    global_batch_size=args.train.global_batch_size,
                    dataloader_batch_size=args.train.dataloader_batch_size,
                    max_seq_len=args.data.max_seq_len,
                    train_steps=args.train_steps,
                    dyn_bsz=False,
                    collate_fn=collate_episode_segment,
                    seed=args.train.seed,
                    save_steps=args.train.checkpoint.save_steps,
                    **loader_kwargs,
                )
            )
            return
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
        self.base.train_dataloader = CausalCosmos3RestartDataLoader(self.base.train_dataloader)

    def forward_backward_step(self, micro_batch):
        if not getattr(self, "episode_training", False):
            return super().forward_backward_step(micro_batch)
        from .causal_cosmos3_episode import episode_forward_backward

        return episode_forward_backward(self, micro_batch)

    def _clear_episode_state(self):
        if getattr(self, "episode_training", False):
            self.segment_memories.clear()
            self.base.model.condition_model.clear_episode_streams()

    def train_step(self, data_iterator):
        try:
            if getattr(self, "episode_training", False):
                from .causal_cosmos3_episode import SynchronizedEpisodeIterator

                data_iterator = SynchronizedEpisodeIterator(data_iterator, self.base.model.parallel_state.dp_group)
            return super().train_step(data_iterator)
        except BaseException:
            self._clear_episode_state()
            raise

    def on_step_begin(self, micro_batches=None):
        if getattr(self, "episode_training", False):
            parallel = self.base.model.parallel_state
            count = sum(len(batch["segment"]["decisions"]) for batch in micro_batches)
            total = torch.tensor(count, dtype=torch.int64, device=self.base.device)
            if dist.is_initialized():
                all_reduce(total, op="sum", group=parallel.dp_group)
            if total.item() < 1:
                raise ValueError("An episode optimizer update needs supervised decision slots")
            self.episode_global_decisions = int(total.item())
        return super().on_step_begin(micro_batches=micro_batches)

    def on_epoch_begin(self):
        self._clear_episode_state()
        if getattr(self, "episode_training", False):
            self.base.train_dataset.set_epoch(self.base.state.epoch)
        return super().on_epoch_begin()

    def on_train_end(self):
        try:
            return super().on_train_end()
        finally:
            self._clear_episode_state()


@contextmanager
def validation_state(runtime):
    """Restore all training RNG streams and individual module flags, even on error."""
    condition = runtime.condition_model
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng, device_rng = torch.get_rng_state(), get_device_rng_state()
    condition_rng = condition.rng_state_dict()
    flags = {module: module.training for root in (runtime.model, condition) for module in root.modules()}
    try:
        runtime.model.eval()
        condition.eval()
        yield
    finally:
        try:
            # FSDP2 keeps root parameters unsharded after forward, expecting a
            # backward to reshard them. Evaluation has no backward: explicitly
            # return every group to its FP32 shards, including on prepare/log errors.
            for module in runtime.model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
        finally:
            # Assign flags individually: train(flag) recurses and would overwrite
            # frozen child modules' original eval settings. A newly loaded VAE stays eval.
            for module, training in flags.items():
                module.training = training
            condition.load_rng_state_dict(condition_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            set_device_rng_state(device_rng)


def fixed_validation_inputs(condition, samples, seeds):
    """Use independent time/video streams and an optional action stream."""
    time_generator = torch.Generator(device="cpu").manual_seed(seeds["time"])
    video_generator = torch.Generator(device=condition.device).manual_seed(seeds["video"])
    sigmas = condition.sample_sigmas(samples, generator=time_generator)
    videos = [
        torch.randn(s["video_latents"].shape, dtype=torch.float32, device=condition.device, generator=video_generator)
        for s in samples
    ]
    actions = None
    if condition.config.stage == "action":
        action_generator = torch.Generator(device=condition.device).manual_seed(seeds["action"])
        actions = [
            torch.randn(s["action"].shape, dtype=torch.float32, device=condition.device, generator=action_generator)
            for s in samples
        ]
    inputs, targets = condition.prepare(samples, sigmas=sigmas, epsilon_video=videos, epsilon_action=actions)
    return {**inputs, "targets": targets, "attention_backend": condition.config.attention_backend}


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


class CausalCosmos3Validator:
    """Evaluate the prepared validation dataset through the current model runtime."""

    def __init__(self, trainer, dataset):
        self.trainer, self.dataset = trainer, dataset
        self.runtime = trainer.model
        self.parallel = self.runtime.parallel_state
        self.output_dir = Path(trainer.args.train.checkpoint.output_dir)
        config = trainer.validation_config
        stage = config.get("stage", "action")
        if stage not in ("action", "video"):
            raise ValueError("Validation stage must be action or video")
        if self.runtime.condition_model.config.stage != stage:
            raise ValueError("Validation and Condition stages must match")
        if stage == "action":
            mode = config.get("sample", {}).get("mode", "joint")
            self.modes = MODES if mode == "joint" else (mode,)
            self.loss_names = ("video_loss", "action_loss")
        else:
            self.modes = ("video",)
            self.loss_names = ("video_loss",)
        self.seed = config.get("validation", {}).get("seed", 42)
        self.window_count = len(dataset)
        self.rounds_per_mode = math.ceil(self.window_count / self.parallel.dp_size)
        condition_config = self.runtime.condition_model.config.to_dict()
        for name in ("_name_or_path", "transformers_version", "seed"):
            condition_config.pop(name, None)
        self.config = json.loads(
            canonical_json(
                dict(
                    data=config,
                    condition=condition_config,
                    model=self.runtime.model.config.to_dict(),
                    tokenizer_path=trainer.args.model.tokenizer_path,
                )
            )
        )
        self.validation_id = fingerprint(
            dict(
                version=VALIDATION_VERSION,
                episode_split_id=getattr(trainer, "episode_split_manifest", {}).get("split_id"),
                config=self.config,
            )
        )
        self.last_metrics = None
        self._wandb_metrics_defined = False

    def _synchronize(self, error, what):
        raise_if_any_rank_failed(error, what, group=self.parallel.dp_group)

    def _publish(self, state, metrics, duration):
        error = None
        try:
            if self.trainer.args.train.global_rank == 0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                result = dict(
                    global_step=state.global_step,
                    epoch=state.epoch,
                    validation_id=self.validation_id,
                    duration_seconds=duration,
                    config=self.config,
                    metrics=metrics,
                )
                with (self.output_dir / "validation.jsonl").open("a") as stream:
                    stream.write(canonical_json(result) + "\n")
                logger.info(f"Validation step={state.global_step} id={self.validation_id}: {metrics}")
                if self.trainer.args.train.wandb.enable:
                    import wandb

                    if wandb.run is not None:
                        if not self._wandb_metrics_defined:
                            wandb.define_metric("eval/global_step")
                            wandb.define_metric("eval/*", step_metric="eval/global_step")
                            self._wandb_metrics_defined = True
                        wandb.log(
                            {**metrics, "eval/global_step": state.global_step},
                            step=max(state.global_step, wandb.run.step),
                            commit=False,
                        )
        except Exception as caught:
            error = caught
        self._synchronize(error, "Validation logging")

    @torch.no_grad()
    def evaluate(self, state):
        with use_parallel_state(self.parallel), validation_state(self.runtime):
            start = time.monotonic()
            iterator = iter_validation_batches(
                self.dataset, self.parallel.dp_rank, self.parallel.dp_size, seed=self.seed
            )
            sums = torch.zeros(len(self.modes), dtype=torch.float32, device=self.trainer.device)
            # Supervised, unsupervised, padding, and globally skipped real windows.
            counts = torch.zeros((len(self.modes), 4), dtype=torch.int64, device=self.trainer.device)
            for turn in range(len(self.modes) * self.rounds_per_mode):
                mode_id, mode_turn = divmod(turn, self.rounds_per_mode)
                error, item = None, None
                try:
                    item = next(iterator)
                    if item["record"]["mode"] != self.modes[mode_id]:
                        raise ValueError("Validation iterator mode does not match the evaluator mode")
                except Exception as caught:
                    error = caught
                self._synchronize(error, "Validation data iteration")
                padded = mode_turn * self.parallel.dp_size + self.parallel.dp_rank >= self.window_count
                # Every rank makes this decision before condition/model forwards.
                if any_rank_failed(not item["batch"]["samples"], group=self.parallel.dp_group):
                    counts[mode_id, 2] += int(padded)
                    counts[mode_id, 3] += int(not padded)
                    continue
                error, inputs = None, None
                try:
                    batch = _to_device(item["batch"], self.trainer.device)
                    encoded = self.runtime.condition_model.get_condition(**batch)
                    inputs = fixed_validation_inputs(
                        self.runtime.condition_model, encoded["samples"], item["record"]["seeds"]
                    )
                    supervised = any(mask.any().item() for mask in inputs["targets"]["loss_masks"].values())
                except Exception as caught:
                    error = caught
                self._synchronize(error, "Validation input preparation")
                with self.trainer.model_fwd_context:
                    outputs = self.runtime(**inputs)
                error = None
                try:
                    loss = sum(outputs.loss[name].float() for name in self.loss_names)
                    if not torch.isfinite(loss).item():
                        raise ValueError("Validation requires finite losses for the selected stage")
                    if not item["weight"]:
                        counts[mode_id, 2] += 1
                    else:
                        if supervised:
                            sums[mode_id] += loss.detach().reshape(())
                            counts[mode_id, 0] += 1
                        else:
                            counts[mode_id, 1] += 1
                except Exception as caught:
                    error = caught
                self._synchronize(error, "Validation loss collection")
                del inputs, outputs, batch, encoded, item
            if dist.is_initialized():
                all_reduce(sums, op="sum", group=self.parallel.dp_group)
                all_reduce(counts, op="sum", group=self.parallel.dp_group)
            metrics = {}
            for mode, total, (count, unsupervised, padded, skipped) in zip(
                self.modes, sums.cpu().tolist(), counts.cpu().tolist(), strict=True
            ):
                if count == 0:
                    raise ValueError(
                        f"Validation mode {mode} has no supervised windows ({unsupervised} unsupervised, {skipped} skipped)"
                    )
                tag = f"eval/{MODE_TAGS[mode]}"
                metrics.update(
                    {
                        f"{tag}/loss": total / count,
                        f"{tag}/count": count,
                        f"{tag}/unsupervised_count": unsupervised,
                        f"{tag}/padding_count": padded,
                        f"{tag}/skipped_count": skipped,
                    }
                )
            self._publish(state, metrics, time.monotonic() - start)
            self.last_metrics = metrics
            return metrics


class CausalCosmos3EvaluateCallback(EvaluateCallback):
    """Reuse evaluation cadence and avoid evaluating twice at the same step."""

    def __init__(self, trainer, validator):
        super().__init__(trainer)
        self.validator = validator
        self._last_step = None

    def _evaluate(self, state):
        if self._last_step != state.global_step:
            self.validator.evaluate(state)
            self._last_step = state.global_step
