"""Episode-owned sequential Action segments with one shared raw read per segment."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..readers.episode import read_episode_rows, read_episode_video
from ..readers.lerobot import plan_window
from .dataset import MODES, ActionDataset


@dataclass(frozen=True)
class _Episode:
    subset: int
    row: int
    first: int
    stop: int
    first_candidate: int


def _action_subsets(dataset):
    if isinstance(dataset, ActionDataset):
        return [dataset]
    if hasattr(dataset, "dataset"):
        return _action_subsets(dataset.dataset)
    children = getattr(dataset, "datasets", getattr(dataset, "_datasets", ()))
    return [subset for child in children for subset in _action_subsets(child)]


def collate_episode_segment(samples):
    """Collate exactly one indivisible segment as one DiT microbatch."""
    if len(samples) != 1:
        raise ValueError("Episode training requires one complete segment per microbatch")
    return {"segment": samples[0]}


class EpisodeDataset(IterableDataset):
    """Shuffle complete owned episodes, preserving each lane's segment ordering.

    One lane is one DataLoader worker. Ranks own disjoint episode sets before
    those sets are split across workers, so excess workers cannot empty a rank.
    Each visit advances a legal anchor phase and consumes its episode entirely.
    No raw pixels, latents, decoder state or trainable history is retained here.
    """

    def __init__(self, dataset, *, segment_latents, rank=0, world_size=1, seed=0, repeat=True):
        super().__init__()
        if type(segment_latents) is not int or segment_latents < 1:
            raise ValueError("segment_latents must be a positive integer")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid episode data-parallel ownership")
        self.datasets = _action_subsets(dataset)
        if not self.datasets:
            raise ValueError("Episode training requires Action datasets")
        self.segment_latents, self.rank, self.world_size, self.seed = segment_latents, rank, world_size, seed
        self.repeat = repeat
        self.epoch = 0
        self._visits = {}
        records = []
        for subset_id, subset in enumerate(self.datasets):
            horizon = subset.action_horizon
            latent_horizon = horizon // subset.geometry.actions_per_latent
            if segment_latents % latent_horizon:
                raise ValueError("segment_latents must contain a whole number of Action horizons")
            if horizon % subset.candidates.start_stride:
                raise ValueError("Action horizon must align to the candidate start stride")
            for row in np.unique(subset.windows[:, 0]):
                row = int(row)
                indexes = np.flatnonzero(subset.windows[:, 0] == row)
                length = int(subset.episodes["dataset_to_index"][row] - subset.episodes["dataset_from_index"][row])
                first, stop = subset._episode_bounds(length)
                records.append(_Episode(subset_id, row, first, stop, int(indexes[0])))
        records.sort(
            key=lambda item: (
                self.datasets[item.subset].source,
                self.datasets[item.subset].root,
                int(self.datasets[item.subset].episodes["episode_index"][item.row]),
            )
        )
        self.episodes = records[rank::world_size]
        if not self.episodes and repeat:
            raise ValueError(f"DP rank {rank} owns no eligible episode; reduce DP size or provide more episodes")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._visits = {}

    def state_dict(self):
        # Restart recovery never serializes buffered data or a live stream.
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.set_epoch(state.get("epoch", 0))

    def __iter__(self):
        worker = get_worker_info()
        lane, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        owned = self.episodes[lane::workers]
        if not owned:
            return
        order_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank, lane, 91]))
        task_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank, lane, 92]))
        while True:
            for position in order_rng.permutation(len(owned)):
                episode = owned[int(position)]
                visit_key = (episode.subset, episode.row)
                visit = self._visits.get(visit_key, 0) if self.repeat else 0
                if self.repeat:
                    self._visits[visit_key] = visit + 1
                yield from self._episode_segments(episode, lane=lane, visit=visit, rng=task_rng)
            if not self.repeat:
                return

    def _episode_segments(self, episode, *, lane, visit, rng):
        ds = self.datasets[episode.subset]
        horizon, stride = ds.action_horizon, ds.candidates.start_stride
        legal_phases = min(horizon // stride, (episode.stop - episode.first - 2) // stride + 1)
        phase = ((visit + self.epoch) % legal_phases) * stride
        start = episode.first + phase
        segment_decisions = self.segment_latents // (horizon // ds.geometry.actions_per_latent)
        episode_id = int(ds.episodes["episode_index"][episode.row])
        ep = {key: value[episode.row].item() for key, value in ds.episodes.items()}
        segment_id, context_start = 0, start
        sample_cache = {}
        while start < episode.stop - 1:
            # One extra decision of row metadata reveals a context change at a
            # full segment boundary without decoding any lookahead RGB frames.
            count = min(segment_decisions + 1, (episode.stop - 2 - start) // horizon + 1)
            union = plan_window(
                ds.root,
                episode_id,
                start,
                count * horizon,
                columns=ds.columns,
                cameras=ds.cameras,
                block_size=horizon,
                video_stride=ds.geometry.video_stride,
                info=ds.info,
                ep=ep,
            )
            reset = segment_id == 0
            rows = read_episode_rows(union, history_steps=ds.history_steps if reset else 0)
            decisions = []
            for offset in range(count):
                anchor = start + offset * horizon
                index = episode.first_candidate + (anchor - episode.first) // stride
                window, caption, source_fps = ds.plan_read_window(index)
                if window.start != anchor:
                    raise ValueError("Episode anchor does not match the configured candidate grid")
                if anchor not in sample_cache:
                    mode = str(rng.choice(MODES, p=ds.mode_weights)) if ds.mode == "joint" else ds.mode
                    begin = offset * horizon
                    ticks = anchor + np.arange(-ds.history_steps, horizon + 1, ds.geometry.video_stride)
                    raw = dict(
                        window=window,
                        state=rows["state"][begin : begin + horizon + 1],
                        action=rows["action"][begin : begin + horizon],
                        state_mask=rows["state_masks"][begin].clone().bool(),
                        action_mask=rows["action_masks"][begin].clone().bool(),
                        task_index=int(rows["task_indexes"][begin]),
                        timestamps=rows["timestamps"][begin : begin + horizon + 1].clone(),
                        real_action_steps=window.real_actions,
                        video_frame_ticks=np.clip(ticks, 0, anchor + window.real_actions).tolist(),
                    )
                    sample_cache[anchor] = ds._sample_from_raw(
                        index, raw, mode=mode, caption=caption, source_fps=source_fps
                    )
                sample = dict(
                    sample_cache[anchor],
                    decision_offset_ticks=offset * horizon,
                    decision_offset_latents=offset * horizon // ds.geometry.actions_per_latent,
                )
                decisions.append(sample)
            text = decisions[0]["text_token_ids"]
            actual = 1
            while actual < min(count, segment_decisions) and torch.equal(text, decisions[actual]["text_token_ids"]):
                actual += 1
            context_final = actual < count and not torch.equal(text, decisions[actual]["text_token_ids"])
            final = context_final or start + actual * horizon >= episode.stop - 1
            decisions = decisions[:actual]
            for sample in decisions:
                sample_cache.pop(sample["window"].start)
            end = start + actual * horizon
            actual_window = replace(
                union, actions=actual * horizon, real_actions=min(union.real_actions, actual * horizon)
            )
            item = dict(
                lane_id=lane,
                stream_id=(ds.source, ds.root, episode_id, self.epoch, visit, phase, context_start),
                segment_id=segment_id,
                reset=reset,
                final=final,
                start_tick=start,
                end_tick=end,
                history_steps=ds.history_steps,
                action_horizon=horizon,
                actions_per_latent=ds.geometry.actions_per_latent,
                video_stride=ds.geometry.video_stride,
                segment_latents=self.segment_latents,
                source=ds.source,
                subset=ds.root,
                episode=episode_id,
                source_fps=decisions[0]["source_fps"],
                conditioning_fps=decisions[0]["conditioning_fps"],
                storage_fps=actual_window.fps,
                resolution=ds.resolution,
                viewpoint=ds.viewpoint,
                image_size=(actual_window.height, actual_window.width),
                decisions=decisions,
            )
            if ds.video:
                item.update(
                    read_episode_video(
                        actual_window,
                        history_steps=ds.history_steps,
                        reset=reset,
                        anchor_timestamp=rows["timestamps"][0].item(),
                        tolerance_s=ds.tolerance_s,
                        decode_threads=ds.decode_threads,
                    )
                )
            yield item
            start = end
            if context_final:
                context_start, segment_id = start, 0
            else:
                segment_id += 1
