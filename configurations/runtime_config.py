"""Runtime / I/O configuration.

These knobs describe *how the run is executed and observed*: seed, device,
logging, checkpointing, dataset paths, resumption. They are independent of
both the NOMA problem and the learning algorithm.
"""

from __future__ import annotations

import argparse


def register_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add runtime-and-IO arguments to an existing parser."""
    group = parser.add_argument_group("runtime (I/O, logging, device)")

    # Problem selection
    group.add_argument(
        "--problem", default="noop",
        help="Problem registry key (currently only 'noop' is registered)",
    )
    group.add_argument(
        "--data_distribution", type=str, default=None,
        help="Data distribution to use during training (problem-specific)",
    )

    # RNG
    group.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed",
    )

    # Device
    group.add_argument(
        "--no_cuda", action="store_true",
        help="Disable CUDA even if available (used in this VM, no GPU)",
    )

    # Validation
    group.add_argument(
        "--val_size", type=int, default=1000,
        help="Number of validation samples per epoch",
    )
    group.add_argument(
        "--val_dataset", type=str, default=None,
        help="Path to a pre-built validation dataset (optional)",
    )
    group.add_argument(
        "--eval_only", action="store_true",
        help="Skip training and run validation only",
    )

    # Logging
    group.add_argument(
        "--log_step", type=int, default=50,
        help="Log to TensorBoard every N steps",
    )
    group.add_argument(
        "--log_dir", default="logs",
        help="TensorBoard log directory",
    )
    group.add_argument(
        "--no_tensorboard", action="store_true",
        help="Disable TensorBoard logging",
    )
    group.add_argument(
        "--no_progress_bar", action="store_true",
        help="Disable tqdm progress bars",
    )
    group.add_argument(
        "--run_name", default="run",
        help="Name to identify the run (timestamp is appended automatically)",
    )

    # Output
    group.add_argument(
        "--output_dir", default="outputs",
        help="Directory to write trained models / args.json",
    )
    group.add_argument(
        "--checkpoint_epochs", type=int, default=1,
        help="Save checkpoint every N epochs (0 = never)",
    )

    # Resume / load
    group.add_argument(
        "--load_path",
        help="Path to a checkpoint to load model + optimizer state from",
    )
    group.add_argument(
        "--resume",
        help="Resume training from this checkpoint",
    )

    # Optional epoch index used for resumption math (kept from legacy options.py)
    group.add_argument(
        "--epoch_start", type=int, default=0,
        help="Start at epoch N (relevant for lr-decay scheduling)",
    )


RUNTIME_DEFAULTS = {
    "problem": "noop",
    "data_distribution": None,
    "seed": 1234,
    "no_cuda": False,
    "val_size": 1000,
    "val_dataset": None,
    "eval_only": False,
    "log_step": 50,
    "log_dir": "logs",
    "no_tensorboard": False,
    "no_progress_bar": False,
    "run_name": "run",
    "output_dir": "outputs",
    "checkpoint_epochs": 1,
    "load_path": None,
    "resume": None,
    "epoch_start": 0,
}
