"""Episode FIFO loading and one-backward-per-segment Cosmos3 training."""

import copy
import hashlib
import time

import torch
from torch.utils.data import IterableDataset

from ..distributed.parallel_state import use_parallel_state
from ..utils.dist_utils import all_reduce, raise_if_any_rank_failed
from .causal_cosmos3 import MODE_TAGS, CausalCosmos3Validator, validation_state


class EpisodeLoaderItems(IterableDataset):
    """Adapt one ordered segment to the native loader's one-to-N item protocol."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for segment in self.dataset:
            yield [segment]

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def state_dict(self):
        return {"data_resume_mode": "restart"}

    def load_state_dict(self, _state):
        pass


class SynchronizedEpisodeIterator:
    """Coordinate reader failures before any rank enters the model collectives."""

    def __init__(self, iterator, group):
        self.iterator, self.group = iterator, group

    def __next__(self):
        error, batch = None, None
        try:
            batch = next(self.iterator)
        except Exception as caught:
            error = caught
        raise_if_any_rank_failed(error, "Episode data loading", group=self.group)
        return batch


def episode_forward_backward(trainer, micro_batch):
    raw = micro_batch["segment"]
    runtime = trainer.base.model
    condition = runtime.condition_model
    lane = raw["lane_id"]
    memory = None if raw["reset"] else trainer.segment_memories.get(lane)
    try:
        error, encoded, inputs = None, None, None
        try:
            if not raw["reset"] and memory is None:
                raise ValueError("A continued episode segment has no committed model memory")
            with torch.no_grad():
                encoded = condition.get_condition(segment=raw)
                inputs = condition.process_condition(**encoded)
            from ..models.diffusers.causal_cosmos3.cosmos3_core.segment import _prepare

            _prepare(runtime.model, inputs["stream_segment"], memory)
        except Exception as caught:
            error = caught
        raise_if_any_rank_failed(error, "Episode condition preparation", group=runtime.parallel_state.dp_group)
        inputs["stream_memory"] = memory
        with use_parallel_state(runtime.parallel_state), trainer.base.model_fwd_context:
            outputs = runtime(**inputs)
        loss, losses = trainer.postforward(outputs, inputs)
        episode = encoded["episode"]
        # Cancel DiT's per-microbatch mean and FSDP's DP mean, then normalize by
        # this update's actual decisions across all ranks and accumulated segments.
        factor = (
            episode["actual_decisions"]
            * trainer.base.num_micro_batches
            * runtime.parallel_state.dp_size
            / trainer.episode_global_decisions
        )
        loss = loss * factor
        losses = {key: value * factor for key, value in losses.items()}
        with use_parallel_state(runtime.parallel_state), trainer.base.model_bwd_context:
            loss.backward()
        candidate = outputs.segment_memory
        if candidate is None or any(
            value.requires_grad or value.grad_fn is not None for value in candidate.layer_inputs
        ):
            raise ValueError("Episode memory must be a detached, graph-free snapshot")
        trainer.last_episode_statistics = {
            **outputs.segment_stats,
            "actual_decisions": episode["actual_decisions"],
            "nominal_decisions": episode["nominal_decisions"],
            "lane_id": lane,
            "segment_id": raw["segment_id"],
            "reset": raw["reset"],
            "final": raw["final"],
        }
        if raw["final"]:
            trainer.segment_memories.pop(lane, None)
            condition.clear_episode_streams(lane)
        else:
            trainer.segment_memories[lane] = candidate
        return loss, losses
    except BaseException:
        trainer.segment_memories.pop(lane, None)
        condition.clear_episode_streams(lane)
        raise


class EpisodeValidator(CausalCosmos3Validator):
    """Deterministic held-out episode streams with synchronized padding ranks."""

    def __init__(self, trainer, dataset):
        super().__init__(trainer, dataset)
        from ..data.causal_cosmos3.datasets.validation import fingerprint

        self.config = {**self.config, "episode_validation": dict(version=1, phase=0, stride="action_horizon")}
        self.validation_id = fingerprint(self.config)

    def _dataset_for_mode(self, mode, *, rank, world_size):
        from ..data.causal_cosmos3.datasets.dataset import StatefulConcatDataset
        from ..data.causal_cosmos3.datasets.episode import EpisodeDataset, _action_subsets

        subsets = []
        for source in _action_subsets(self.dataset):
            subset = copy.copy(source)
            subset.mode = mode
            subsets.append(subset)
        return EpisodeDataset(
            StatefulConcatDataset(subsets),
            segment_latents=self.runtime.model.config.tbptt_keep_history_latents,
            rank=rank,
            world_size=world_size,
            seed=self.seed,
            repeat=False,
        )

    @torch.no_grad()
    def evaluate(self, state):
        from ..models.diffusers.causal_cosmos3.cosmos3_condition.episode_condition import context_identity

        condition = self.runtime.condition_model
        training_streams = condition.episode_streams
        condition.episode_streams = {}
        started = time.monotonic()
        metrics = {}
        try:
            with use_parallel_state(self.parallel), validation_state(self.runtime):
                for mode in self.modes:
                    condition.clear_episode_streams()
                    memories = {}
                    iterator = iter(
                        self._dataset_for_mode(mode, rank=self.parallel.dp_rank, world_size=self.parallel.dp_size)
                    )
                    # Each rank can build a cold padding item locally. No pixel
                    # payload is broadcast and no padded decision contributes loss.
                    error, fallback = None, None
                    try:
                        fallback = next(iter(self._dataset_for_mode(mode, rank=0, world_size=1)))
                    except Exception as caught:
                        error = caught
                    self._synchronize(error, "Episode validation padding preparation")
                    sums = torch.zeros((), dtype=torch.float32, device=self.trainer.device)
                    counts = torch.zeros(3, dtype=torch.int64, device=self.trainer.device)
                    while True:
                        error, raw = None, None
                        try:
                            raw = next(iterator, None)
                        except Exception as caught:
                            error = caught
                        self._synchronize(error, "Episode validation reading")
                        active = torch.tensor(int(raw is not None), dtype=torch.int64, device=self.trainer.device)
                        if torch.distributed.is_initialized():
                            all_reduce(active, op="sum", group=self.parallel.dp_group)
                        if not active.item():
                            break
                        padded = raw is None
                        if padded:
                            raw = {**fallback, "reset": True, "final": True}
                        lane = raw["lane_id"]
                        memory = None if raw["reset"] else memories.get(lane)
                        error, encoded, inputs = None, None, None
                        try:
                            identity = f"{self.seed}:{mode}:{context_identity(raw)}:{raw['segment_id']}"
                            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (
                                2**63 - 1
                            )
                            condition.time_generator.manual_seed(seed)
                            condition.generator.manual_seed((seed + 1) % (2**63 - 1))
                            encoded = condition.get_condition(segment=raw)
                            inputs = condition.process_condition(**encoded)
                            from ..models.diffusers.causal_cosmos3.cosmos3_core.segment import _prepare

                            _prepare(self.runtime.model, inputs["stream_segment"], memory)
                        except Exception as caught:
                            error = caught
                        self._synchronize(error, "Episode validation preparation")
                        with self.trainer.model_fwd_context:
                            output = self.runtime(**inputs, stream_memory=memory)
                        error = None
                        try:
                            loss = sum(output.loss.values())
                            if not torch.isfinite(loss).item():
                                raise ValueError("Episode validation requires finite losses")
                            count = len(raw["decisions"])
                            if padded:
                                counts[2] += count
                            else:
                                targets = inputs["stream_segment"]["targets"]
                                supervised = sum(
                                    int(
                                        targets["loss_masks"]["video_loss_mask"][span.current_video_rows].any().item()
                                        or targets["loss_masks"]["action_loss_mask"][span.action_rows].any().item()
                                    )
                                    for span in targets["spans"]
                                )
                                sums += loss.float() * count
                                counts[0] += supervised
                                counts[1] += count - supervised
                            if raw["final"]:
                                memories.pop(lane, None)
                                condition.clear_episode_streams(lane)
                            else:
                                memories[lane] = output.segment_memory
                        except Exception as caught:
                            error = caught
                        self._synchronize(error, "Episode validation collection")
                    if torch.distributed.is_initialized():
                        all_reduce(sums, op="sum", group=self.parallel.dp_group)
                        all_reduce(counts, op="sum", group=self.parallel.dp_group)
                    supervised, unsupervised, padded = counts.cpu().tolist()
                    if supervised < 1:
                        raise ValueError(f"Episode validation mode {mode} has no supervised decisions")
                    tag = f"eval/{MODE_TAGS[mode]}"
                    metrics.update(
                        {
                            f"{tag}/loss": float(sums.item()) / supervised,
                            f"{tag}/count": supervised,
                            f"{tag}/unsupervised_count": unsupervised,
                            f"{tag}/padding_count": padded,
                            f"{tag}/skipped_count": 0,
                        }
                    )
                self._publish(state, metrics, time.monotonic() - started)
                self.last_metrics = metrics
                return metrics
        finally:
            condition.clear_episode_streams()
            condition.episode_streams = training_streams
