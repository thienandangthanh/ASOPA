#!/usr/bin/env python
"""Top-level training entry point.

The actual training and validation logic lives in
`attention_model.training_loop`. This script just parses options, wires
the model + baseline + optimizer, and runs `train_epoch` in a loop.

Usage:
    uv run python train.py --user_num 10 --n_epochs 1 --no_cuda
    uv run python train.py --eval_only --load_path output/checkpoints/...
"""

from __future__ import annotations

import json
import os
import pprint
import time

import scipy.io as sio
import torch
import torch.optim as optim

from attention_model.attention_model import AttentionModel
from attention_model.pointer_network import PointerNetwork
from attention_model.training_loop import (
    clip_grad_norms,
    get_inner_model,
    jilu_val_cost,
    rollout,
    train_batch,
    train_epoch,
    validate,
)
from configurations import get_options
from utils import load_problem, torch_load_cpu
from utils.reinforce_baselines import (
    ExponentialBaseline,
    NoBaseline,
    RolloutBaseline,
    WarmupBaseline,
)

# Re-exported so legacy callers can keep importing from `train`.
__all__ = [
    "train_epoch",
    "validate",
    "rollout",
    "train_batch",
    "clip_grad_norms",
    "get_inner_model",
    "jilu_val_cost",
]


def _build_model(opts, problem) -> torch.nn.Module:
    cls = {"attention": AttentionModel, "pointer": PointerNetwork}.get(opts.model)
    assert cls is not None, "Unknown model: {!r}".format(opts.model)
    return cls(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
    ).to(opts.device)


def _build_baseline(opts, model, problem):
    if opts.baseline == "rollout":
        return RolloutBaseline(model, problem, opts)
    if opts.baseline == "exponential":
        return ExponentialBaseline(opts.exp_beta)
    if opts.baseline in (None, "none"):
        return NoBaseline()
    raise ValueError("Unknown baseline: {!r}".format(opts.baseline))


def _save_checkpoint(model, opts, epoch: int) -> None:
    ckpt_dir = "output/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": get_inner_model(model).state_dict(),
            "model_init_args": {
                "embedding_dim": opts.embedding_dim,
                "hidden_dim": opts.hidden_dim,
                "n_heads": getattr(model, "n_heads", 8),
                "n_encode_layers": opts.n_encode_layers,
                "tanh_clipping": opts.tanh_clipping,
                "mask_inner": True,
                "mask_logits": True,
                "normalization": opts.normalization,
                "checkpoint_encoder": opts.checkpoint_encoder,
                "shrink_size": opts.shrink_size,
            },
            "epoch": epoch,
        },
        "%s/variable_user_n%d_epoch%d.pth" % (ckpt_dir, opts.user_num, epoch),
    )


def main(opts) -> None:
    pprint.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    os.makedirs(opts.save_dir, exist_ok=True)
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True, default=str)

    tb_logger = None
    if not opts.no_tensorboard:
        from tensorboard_logger import Logger as TbLogger
        tb_logger = TbLogger(
            os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.user_num), opts.run_name)
        )

    problem = load_problem(opts.problem)
    model = _build_model(opts, problem)

    # Optional checkpoint load.
    load_path = opts.load_path or opts.resume
    if load_path:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)
        get_inner_model(model).load_state_dict(
            {**get_inner_model(model).state_dict(), **load_data.get("model_state_dict", load_data.get("model", {}))}
        )

    baseline = _build_baseline(opts, model, problem)

    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    if opts.eval_only:
        opts.eval_batch_size = 1
        t0 = time.time()
        validate(model, val_dataset, opts)
        print(f"ASOPA average validation time: {(time.time() - t0) / opts.val_size:.4f}s/sample")
        return

    cost_history = []
    best_reward = -float("inf")
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        avg_reward, cost = train_epoch(
            model, optimizer, baseline, lr_scheduler, epoch,
            val_dataset, problem, tb_logger, opts,
        )
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"New best model! Epoch {epoch}, Reward: {avg_reward:.4f}")
            _save_checkpoint(model, opts, epoch)
        cost_history.append(cost.tolist())

    out_dir = "input_data/output"
    os.makedirs(out_dir, exist_ok=True)
    sio.savemat(
        "%s/n%d_performance_value_%d.mat" % (out_dir, opts.user_num, opts.val_size),
        {"performance_percent": cost_history},
    )


if __name__ == "__main__":
    main(get_options())
