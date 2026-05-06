"""Learning-algorithm configuration (RL + neural network hyperparameters).

These knobs describe *how the model learns*: optimizer settings, network
sizes, baseline strategy, REINFORCE bookkeeping. They have nothing to do
with the underlying NOMA problem.
"""

from __future__ import annotations

import argparse


def register_learning_args(parser: argparse.ArgumentParser) -> None:
    """Add learning-algorithm arguments to an existing parser."""
    group = parser.add_argument_group("learning (RL/NN)")

    # Network architecture
    group.add_argument(
        "--model", default="attention",
        help="Model type: 'attention' (default) or 'pointer'",
    )
    group.add_argument(
        "--embedding_dim", type=int, default=128,
        help="Input embedding dim",
    )
    group.add_argument(
        "--hidden_dim", type=int, default=128,
        help="Hidden layer dim in encoder/decoder",
    )
    group.add_argument(
        "--n_encode_layers", type=int, default=3,
        help="Number of encoder layers",
    )
    group.add_argument(
        "--tanh_clipping", type=float, default=10.0,
        help="Clip parameters to +/- this via tanh; 0 disables clipping",
    )
    group.add_argument(
        "--normalization", default="batch",
        help="Normalization type: 'batch' or 'instance'",
    )

    # Optimizer
    group.add_argument(
        "--lr_model", type=float, default=1e-4,
        help="Learning rate for the actor network",
    )
    group.add_argument(
        "--lr_critic", type=float, default=1e-4,
        help="Learning rate for the critic network",
    )
    group.add_argument(
        "--lr_decay", type=float, default=1.0,
        help="Learning-rate decay multiplier per epoch",
    )
    group.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Max L2 norm for gradient clipping (0 disables)",
    )

    # Training schedule
    group.add_argument(
        "--n_epochs", type=int, default=501,
        help="Number of training epochs",
    )
    group.add_argument(
        "--batch_size", type=int, default=64,
        help="Instances per training batch",
    )
    group.add_argument(
        "--epoch_size", type=int, default=1280,
        help="Instances per training epoch (must be a multiple of batch_size)",
    )
    group.add_argument(
        "--eval_batch_size", type=int, default=128,
        help="Batch size used during validation/baseline evaluation",
    )

    # Memory / efficiency
    group.add_argument(
        "--checkpoint_encoder", action="store_true",
        help="Trade compute for memory by checkpointing the encoder",
    )
    group.add_argument(
        "--shrink_size", type=int, default=None,
        help="Shrink batch when at least this many instances are finished",
    )

    # Baseline strategy
    group.add_argument(
        "--baseline", default="rollout",
        help="REINFORCE baseline: 'rollout', 'critic', 'exponential', or None",
    )
    group.add_argument(
        "--exp_beta", type=float, default=0.8,
        help="Exponential moving average decay for the exponential baseline",
    )
    group.add_argument(
        "--bl_alpha", type=float, default=0.05,
        help="Significance threshold for rollout-baseline t-test",
    )
    group.add_argument(
        "--bl_warmup_epochs", type=int, default=None,
        help="Warmup epochs (default 1 for rollout, 0 otherwise)",
    )


LEARNING_DEFAULTS = {
    "model": "attention",
    "embedding_dim": 128,
    "hidden_dim": 128,
    "n_encode_layers": 3,
    "tanh_clipping": 10.0,
    "normalization": "batch",
    "lr_model": 1e-4,
    "lr_critic": 1e-4,
    "lr_decay": 1.0,
    "max_grad_norm": 1.0,
    "n_epochs": 501,
    "batch_size": 64,
    "epoch_size": 1280,
    "eval_batch_size": 128,
    "checkpoint_encoder": False,
    "shrink_size": None,
    "baseline": "rollout",
    "exp_beta": 0.8,
    "bl_alpha": 0.05,
    "bl_warmup_epochs": None,
}
