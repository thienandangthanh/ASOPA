"""Environment (NOMA domain) configuration.

These knobs describe the *physical problem*: number of users, distance range,
weight range, channel noise, fairness exponent. They have nothing to do with
the learning algorithm.
"""

from __future__ import annotations

import argparse


def register_env_args(parser: argparse.ArgumentParser) -> None:
    """Add NOMA-environment arguments to an existing parser."""
    group = parser.add_argument_group("environment (NOMA domain)")

    # User count is canonical. `--graph_size` is a back-compat alias kept so
    # existing scripts and the README keep working.
    group.add_argument(
        "--user_num",
        type=int,
        default=10,
        help="Number of users in the training topology (canonical name)",
    )
    group.add_argument(
        "--graph_size",
        type=int,
        default=None,
        help="Alias for --user_num (kept for back-compat with run.py / docs)",
    )
    group.add_argument(
        "--val_user_num",
        type=int,
        default=8,
        help="Number of users in the validation topology",
    )
    group.add_argument(
        "--val_graph_size",
        type=int,
        default=None,
        help="Alias for --val_user_num (kept for back-compat)",
    )

    # Geometry
    group.add_argument(
        "--d_min", type=float, default=20.0,
        help="Minimum distance from a user to the base station (m)",
    )
    group.add_argument(
        "--d_max", type=float, default=100.0,
        help="Maximum distance from a user to the base station (m)",
    )

    # Weights
    group.add_argument(
        "--w_min", type=float, default=1.0,
        help="Minimum throughput weight",
    )
    group.add_argument(
        "--w_max", type=float, default=32.0,
        help="Maximum throughput weight",
    )

    # Variable-user dataset bounds
    group.add_argument(
        "--num_min", type=int, default=5,
        help="Minimum users sampled in the variable-user dataset",
    )
    group.add_argument(
        "--num_max", type=int, default=10,
        help="Maximum users sampled in the variable-user dataset",
    )

    # Channel
    group.add_argument(
        "--noise", type=float, default=3.981e-15,
        help="Gaussian white noise at the BS, default -184 dBm/Hz * 1 MHz",
    )
    group.add_argument(
        "--alpha", type=float, default=1.0,
        help="Fairness exponent: 1 = proportional fair, <1 = closer to throughput max",
    )

    # Validation epoch index used by ASOPA_validation.py to pick the checkpoint.
    group.add_argument(
        "--val_epoch", type=int, default=300,
        help="Epoch index of the checkpoint used for validation",
    )


ENV_DEFAULTS = {
    "user_num": 10,
    "val_user_num": 8,
    "d_min": 20.0,
    "d_max": 100.0,
    "w_min": 1.0,
    "w_max": 32.0,
    "num_min": 5,
    "num_max": 10,
    "noise": 3.981e-15,
    "alpha": 1.0,
    "val_epoch": 300,
}
