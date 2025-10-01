import os
import json
import pprint as pp
import time

import torch
import torch.optim as optim


import numpy as np

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from conf import args


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem =load_problem(opts.problem)

    validate_epoch = args.val_epoch
    # Load model with weights_only=False for compatibility with older PyTorch model files
    # This is safe since we trust the source of this checkpoint
    model= torch.load('Variable_user_n10_epoch{}.pth'.format(validate_epoch), weights_only=False)
    # Move model to the correct device
    model = model.to(opts.device)
    val_dataset = problem.load_val_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    opts.eval_batch_size = 1
    time_start = time.time()
    validate(model, val_dataset, opts)


if __name__ == "__main__":
    run(get_options())