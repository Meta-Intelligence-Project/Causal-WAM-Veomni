# SPDX-License-Identifier: OpenMDW-1.1
"""Compute and save independent statistics for every LeRobot child dataset.

Run from the repository root:
    python -m scripts.causal_cosmos3.statistics.compute_causal_action_stats \
        --dataset-root /path/to/robotwin_processed --dataset-processes 4

Each dataset receives meta/causal_action_stats.json. All episodes are used;
no merged output, model loading or video decoding is involved.
"""

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import torch

from veomni.data.causal_cosmos3.readers.lerobot import ActionColumns
from veomni.data.causal_cosmos3.statistics import compute_statistics, discover_dataset_roots, save_statistics
from veomni.data.causal_cosmos3.transforms.processing import Geometry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", nargs="+", required=True, help="Dataset roots or their parent directories")
    parser.add_argument("--dataset-processes", type=int, default=1, help="Concurrent independent dataset processes")
    parser.add_argument("--state-key", default="state_unified")
    parser.add_argument("--action-key", default="action_unified")
    parser.add_argument("--state-mask-key", default="mask_state")
    parser.add_argument("--action-mask-key", default="mask_action")
    parser.add_argument(
        "--action-from-state", action="store_true", help="Use state as target; offset remains explicit"
    )
    parser.add_argument("--action-time-offset-steps", type=int, default=0)
    parser.add_argument("--actions-per-block", type=int, default=32)
    parser.add_argument("--method", choices=("exact", "reservoir"), default="reservoir")
    parser.add_argument("--bounds", choices=("q01_q99", "min_max"), default="q01_q99")
    parser.add_argument("--reservoir-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4, help="Ordered read/encode threads per dataset")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Debug: first N blocks per dataset; output is marked partial")
    args = parser.parse_args(argv)
    if min(args.dataset_processes, args.reservoir_size, args.batch_size, args.num_workers) < 1:
        parser.error("Process count, reservoir size, batch size and worker count must be positive")
    if args.seed < 0 or args.log_every < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Seed/log interval must be nonnegative and limit must be positive")
    try:
        Geometry(actions_per_block=args.actions_per_block)
    except ValueError as error:
        parser.error(str(error))
    return args


def compute_child(job):
    root, options = job
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    options = dict(options)
    options["columns"] = ActionColumns(**options["columns"])
    options["geometry"] = Geometry(**options["geometry"])
    statistics = compute_statistics(root, **options)
    output = root / "meta/causal_action_stats.json"
    save_statistics(statistics, output)
    logging.info("Saved %s; blocks=%d", output, statistics["provenance"]["blocks"])
    return str(output)


def main(argv=None):
    args = parse_args(argv)
    roots = discover_dataset_roots(args.dataset_root)
    columns = ActionColumns(
        state=args.state_key,
        target=args.state_key if args.action_from_state else args.action_key,
        state_mask=args.state_mask_key,
        target_mask=args.state_mask_key if args.action_from_state else args.action_mask_key,
        target_offset=args.action_time_offset_steps,
    )
    options = {
        key: getattr(args, key)
        for key in ("method", "reservoir_size", "seed", "bounds", "limit", "batch_size", "num_workers", "log_every")
    }
    options.update(columns=asdict(columns), geometry=asdict(Geometry(actions_per_block=args.actions_per_block)))
    jobs = [(root, options) for root in roots]
    if args.dataset_processes == 1:
        for job in jobs:
            compute_child(job)
    else:
        with ProcessPoolExecutor(
            max_workers=args.dataset_processes, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for _ in pool.map(compute_child, jobs):
                pass


if __name__ == "__main__":
    main()
