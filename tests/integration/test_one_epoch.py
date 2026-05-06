"""Integration smoke test: 1 epoch of REINFORCE training runs end-to-end on
CPU and produces finite loss. Tiny epoch_size + batch_size to stay fast.

This is a *liveness* test, not a numeric reproduction test. The training loop
samples random data per epoch and uses non-deterministic decoding, so
asserting an exact loss value across machines is fragile. We just check that
training advances without exceptions and produces finite costs.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.slow
def test_one_epoch_training_smoke():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from nets.attention_model import AttentionModel
    from options import get_options
    from reinforce_baselines import ExponentialBaseline
    from train import train_epoch
    from utils import load_problem

    opts = get_options(
        [
            "--no_cuda",
            "--no_tensorboard",
            "--no_progress_bar",
            "--problem",
            "noop",
            "--graph_size",
            "8",
            "--val_graph_size",
            "8",
            "--val_size",
            "16",
            "--batch_size",
            "8",
            "--epoch_size",
            "16",
            "--eval_batch_size",
            "4",
            "--baseline",
            "exponential",
            "--n_epochs",
            "1",
        ]
    )
    opts.device = torch.device("cpu")
    os.makedirs(opts.save_dir, exist_ok=True)

    problem = load_problem(opts.problem)

    torch.manual_seed(opts.seed)
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
    ).to(opts.device)

    baseline = ExponentialBaseline(opts.exp_beta)

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": opts.lr_model}])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size, num_samples=opts.val_size
    )

    avg_reward, cost = train_epoch(
        model, optimizer, baseline, lr_scheduler, 0,
        val_dataset, problem, None, opts,
    )

    assert torch.isfinite(torch.tensor(avg_reward)), "avg_reward must be finite"
    assert cost.numel() == opts.val_size
    assert torch.all(torch.isfinite(cost)), "all per-sample costs must be finite"
