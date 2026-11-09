# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded rank-local grouping of independent episode lanes after worker reads."""

from collections import deque


def active_episode_lane_ids(dataset, *, num_workers):
    """Return nonempty worker lanes after EpisodeDataset's rank ownership split."""
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    return tuple(range(min(num_workers or 1, len(dataset.episodes))))


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _close_iterator(iterator):
    close = getattr(iterator, "close", None)
    if close is None:
        # Torch's worker iterator has no public close; use its idempotent cleanup.
        close = getattr(iterator, "_shutdown_workers", None)
    if close is not None:
        close()


class EpisodeBatchDataLoader:
    """Make full optimizer updates from a native singleton-segment worker loader.

    The inner loader must yield a list containing one {"segment": raw} item
    (a direct singleton dict is also accepted). Grouping happens in the consuming
    process, so each output microbatch takes one FIFO head from each of
    batch_size different lanes. num_micro_batches such microbatches form an
    update. A lane may reappear in a later microbatch, after its previous
    forward/backward commits.

    Exact finite exhaustion stops iteration; an incomplete update raises instead
    of dropping buffered segments. The buffer cap includes segments selected for
    the update being assembled. Skew that prevents a complete batch before the
    cap stops reads with an error. No tensor-containing resume state is retained.

    Stop the outer background prefetcher before close/set_epoch. B1 callers keep
    their existing loader and collator unchanged.
    """

    def __init__(self, loader, *, batch_size, num_micro_batches, lane_ids, max_buffered_segments=None):
        self.loader = loader
        self.batch_size = _positive(batch_size, "batch_size")
        self.num_micro_batches = _positive(num_micro_batches, "num_micro_batches")
        self.lane_ids = tuple(lane_ids)
        if any(type(lane) is not int or lane < 0 for lane in self.lane_ids):
            raise ValueError("lane_ids must contain nonnegative integer lane IDs")
        if len(set(self.lane_ids)) != len(self.lane_ids):
            raise ValueError("lane_ids must be unique")
        if len(self.lane_ids) < self.batch_size:
            raise ValueError(
                f"Episode batch_size={self.batch_size} requires at least that many active lanes; "
                f"only {len(self.lane_ids)} are available. Increase num_workers/eligible episodes or reduce batch size."
            )
        required = self.batch_size * self.num_micro_batches
        self.max_buffered_segments = _positive(
            required + 2 * len(self.lane_ids) if max_buffered_segments is None else max_buffered_segments,
            "max_buffered_segments",
        )
        if self.max_buffered_segments < required:
            raise ValueError("max_buffered_segments must fit a complete optimizer update")
        self._iterator = None

    def __iter__(self):
        self.close()
        self._iterator = _EpisodeBatchIterator(self)
        return self._iterator

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def close(self):
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None

    def set_epoch(self, epoch):
        self.close()
        self.loader.set_epoch(epoch)

    def state_dict(self):
        # Never request the inner StatefulDataLoader snapshot: it can contain RGB.
        return {"data_resume_mode": "restart"}

    def load_state_dict(self, _state):
        self.close()


class _EpisodeBatchIterator:
    def __init__(self, loader):
        self.source_loader = loader.loader
        self.source = iter(self.source_loader)
        self.batch_size = loader.batch_size
        self.num_micro_batches = loader.num_micro_batches
        self.max_buffered_segments = loader.max_buffered_segments
        self.queues = {lane: deque() for lane in loader.lane_ids}
        self.ready = deque()
        self.last = {}
        self.buffered_segments = self.peak_buffered_segments = 0
        self.closed = False

    def __iter__(self):
        return self

    def close(self):
        if not self.closed:
            self.closed = True
            source, self.source = self.source, None
            try:
                _close_iterator(source)
            finally:
                for queue in self.queues.values():
                    queue.clear()
                self.ready.clear()
                self.last.clear()
                self.buffered_segments = 0
                # StatefulDataLoader caches its iterator even without persistent
                # workers. Shutdown alone retains out-of-order raw payloads in
                # that cache. Detach only our reader, never a newer iterator.
                if getattr(self.source_loader, "_iterator", None) is source:
                    self.source_loader._iterator = None

    def _read(self):
        if self.buffered_segments >= self.max_buffered_segments:
            raise ValueError(
                "Episode lane lookahead reached max_buffered_segments before a full independent-lane batch; "
                "the singleton reader is too skewed or a declared lane is not producing."
            )
        item = next(self.source)
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise ValueError("Episode batch reader must emit exactly one singleton segment per read")
            item = item[0]
        if not isinstance(item, dict) or "segment" not in item or not isinstance(item["segment"], dict):
            raise ValueError("Episode batch reader requires [{'segment': raw}] singleton items")
        raw = item["segment"]
        lane = raw.get("lane_id")
        if type(lane) is not int or lane not in self.queues:
            raise ValueError(f"Episode reader produced an unknown lane: {lane!r}")
        self._check_order(raw)
        if not self.queues[lane]:
            self.ready.append(lane)
        self.queues[lane].append(raw)
        self.buffered_segments += 1
        self.peak_buffered_segments = max(self.peak_buffered_segments, self.buffered_segments)

    def _check_order(self, raw):
        lane = raw["lane_id"]
        try:
            stream, segment, start, end, reset, final = (
                raw[key] for key in ("stream_id", "segment_id", "start_tick", "end_tick", "reset", "final")
            )
        except KeyError as error:
            raise ValueError("Episode segments require stream/order metadata") from error
        if (
            type(segment) is not int
            or segment < 0
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or type(reset) is not bool
            or type(final) is not bool
        ):
            raise ValueError("Episode segment IDs, tick bounds and reset/final flags are invalid")
        previous = self.last.get(lane)
        if reset:
            if segment != 0 or (previous is not None and not previous[-1]):
                raise ValueError("An episode lane can reset only at segment zero after a final segment")
        elif (
            previous is None
            or previous[-1]
            or stream != previous[0]
            or segment != previous[1] + 1
            or start != previous[2]
        ):
            raise ValueError("Episode lane segments must remain FIFO without gaps or missing resets")
        # Metadata only: do not retain a consumed raw tensor payload here.
        self.last[lane] = (stream, segment, end, final)

    def _microbatch(self):
        selected, lanes = [], set()
        while len(selected) < self.batch_size:
            for _ in range(len(self.ready)):
                lane = self.ready.popleft()
                if lane in lanes:
                    self.ready.append(lane)
                    continue
                selected.append(self.queues[lane].popleft())
                lanes.add(lane)
                if self.queues[lane]:
                    self.ready.append(lane)
                break
            else:
                self._read()
        return {"segments": tuple(selected)}

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            result = [self._microbatch() for _ in range(self.num_micro_batches)]
            self.buffered_segments -= self.batch_size * self.num_micro_batches
            return result
        except StopIteration:
            remaining = self.buffered_segments
            self.close()
            if remaining:
                raise ValueError(
                    f"Episode source exhausted with {remaining} unconsumed segment(s); "
                    "cannot form a complete independent-lane optimizer update"
                ) from None
            raise
        except BaseException:
            self.close()
            raise
