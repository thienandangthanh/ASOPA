#!/usr/bin/env python
"""Validation entry point.

Loads a state_dict checkpoint produced by `train.py`, builds a fresh
AttentionModel with the saved init_args, and runs greedy validation on
the held-out dataset for `--val_user_num` users.

Run:
    uv run python ASOPA_validation.py --user_num 10 --val_user_num 8 --no_cuda
"""

from __future__ import annotations

import pprint
import time

import torch

from attention_model.attention_model import AttentionModel
from attention_model.training_loop import validate
from configurations import get_options
from utils import load_problem


CHECKPOINT_FMT = "output/checkpoints/variable_user_n{user_num}_epoch{epoch}.pth"


def _build_model(opts, problem, init_args) -> torch.nn.Module:
    return AttentionModel(
        init_args["embedding_dim"],
        init_args["hidden_dim"],
        problem,
        n_encode_layers=init_args["n_encode_layers"],
        mask_inner=init_args["mask_inner"],
        mask_logits=init_args["mask_logits"],
        normalization=init_args.get("normalization", "batch"),
        tanh_clipping=init_args["tanh_clipping"],
        checkpoint_encoder=init_args.get("checkpoint_encoder", False),
        shrink_size=init_args.get("shrink_size", None),
    ).to(opts.device)


def main(opts) -> None:
    pprint.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    problem = load_problem(opts.problem)

    ckpt_path = opts.load_path or CHECKPOINT_FMT.format(
        user_num=opts.user_num, epoch=opts.val_epoch
    )
    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, weights_only=False, map_location=opts.device)
    init_args = payload["model_init_args"]

    model = _build_model(opts, problem, init_args)
    model.load_state_dict(payload["model_state_dict"])

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    opts.eval_batch_size = 1
    t0 = time.time()
    validate(model, val_dataset, opts)
    print(f"Average inference time: {(time.time() - t0) / opts.val_size:.4f}s/sample")


if __name__ == "__main__":
    main(get_options())
