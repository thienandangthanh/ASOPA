#!/usr/bin/env python
"""Baseline comparison launcher.

Loads the validation dataset and dispatches it through `show_speed_performance_dataset`
which runs every non-RL baseline (g/w sort, tabu, Qian heuristic, exhaustive)
and reports per-method utility + wall-time.

Run:
    uv run python run_baseline.py --user_num 10 --val_user_num 8 --no_cuda
"""

from __future__ import annotations

import torch

from configurations import get_options
from utils import load_problem


def main(opts) -> None:
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    problem = load_problem(opts.problem)

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    # Lazy import — show.py builds matplotlib figures, no need to load it
    # until we know we'll use it.
    from show import show_speed_performance_dataset
    print("validation_dataset", val_dataset[0])
    show_speed_performance_dataset(val_dataset)


if __name__ == "__main__":
    main(get_options())
