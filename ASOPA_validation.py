import os
import json
import pprint as pp
import time

import torch
import torch.optim as optim


import numpy as np

from nets.critic_network import CriticNetwork
from configurations import get_options
from train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    validate_epoch = opts.val_epoch
    # Build a fresh AttentionModel and load the converted state_dict checkpoint.
    payload = torch.load(
        "output/checkpoints/variable_user_n10_epoch{}.pth".format(validate_epoch),
        weights_only=False,
        map_location=opts.device,
    )
    init_args = payload["model_init_args"]
    model = AttentionModel(
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
    model.load_state_dict(payload["model_state_dict"])
    val_dataset = problem.load_val_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )
    opts.eval_batch_size = 1
    time_start = time.time()
    validate(model, val_dataset, opts)


if __name__ == "__main__":
    run(get_options())
