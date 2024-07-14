#!/usr/bin/env python

import os
import json
import pprint as pp
import time
import scipy.io as sio

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


# SEE_SEE_DUIBI=True
SEE_SEE_DUIBI = False
# SEE_SEE_DUIBI = True

# JUST_VAL = True
JUST_VAL = False

def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))


    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)
    print(f'opts.model={opts.model},type(model)={type(model)}')
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)

    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'rollout':
        if SEE_SEE_DUIBI:
            baseline=NoBaseline()
        else:
            baseline = RolloutBaseline(model, problem, opts)
        # baseline = NoBaseline()
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # if opts.bl_warmup_epochs > 0:
    #     baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    # val_dataset = problem.load_val_dataset(
    #     size=opts.val_graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
    #     distribution=opts.data_distribution)
    # val_dataset = problem.make_dataset(
    #     size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    # print(val_dataset.g.squeeze(-1).numpy()[0])
    # print(val_dataset.w.squeeze(-1).numpy()) 正确的随机   val_dataset是随机的
    if SEE_SEE_DUIBI:
        from show import show_speed_performance_dataset,show_speed_performance
        show_speed_performance_dataset(val_dataset)
        # show_speed_performance(val_dataset.g.squeeze(-1).numpy())
        return

    # if opts.resume:
    #     epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
    #
    #     torch.set_rng_state(load_data['rng_state'])
    #     if opts.use_cuda:
    #         torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
    #     # Set the random states
    #     # Dumping of state was done before epoch callback, so do that now (model is loaded)
    #     baseline.epoch_callback(model, epoch_resume)
    #     print("Resuming after {}".format(epoch_resume))
    #     opts.epoch_start = epoch_resume + 1
    max_reward = 44.3
    if opts.eval_only or JUST_VAL:
        # 将验证的batchsize变为1，从而公平地对比时间
        opts.eval_batch_size = 1
        time_start = time.time()
        validate(model, val_dataset, opts)

        time_end = time.time()
        print(f'ASOPA在验证集的平均用时为{(time_end-time_start)/opts.val_size}s')
    else:
        print('ooo')
        # print('validation_dataset',val_dataset[0])
        cost_epoch_his = []
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            avg_reward,cost_epoch = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )
            # print('avg_reward',avg_reward,'max_reward',max_reward)
            if avg_reward > max_reward:
                # max_reward = avg_reward
                torch.save(model,'Variable_user_n%d_epoch%d.pth'%(opts.graph_size,epoch))
            cost_epoch = cost_epoch.tolist()
            cost_epoch_his.append(cost_epoch)
        sio.savemat('./performance_percent/n%d_performance_value_%d.mat' % (opts.graph_size, opts.val_size),
                    {'performance_percent': cost_epoch_his})

if __name__ == "__main__":
    run(get_options())
