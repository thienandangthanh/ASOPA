import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

from conf import args
import csv
import scipy.io as sio


def jilu_val_cost(epoch, val_performance, t_cost, path):
    # path = "1.csv"
    with open(path, "a+") as f:
        csv_write = csv.writer(f)
        data_row = [epoch, val_performance, t_cost]
        csv_write.writerow(data_row)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print("Validating...")
    t1 = time.time()
    cost = rollout(model, dataset, opts)
    t2 = time.time()
    # print('val_cost',cost[0],dataset[0])
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )
    print("Validation time: {}".format(t2 - t1))
    delay = t2 - t1
    # top15_list = sio.loadmat("Top15/n%d_top15_tabu" % (opts.val_graph_size))['performance_list']
    # Hit_top15 = 0
    # Hit_top10 = 0
    # Hit_top5 = 0
    # Hit_top1 = 0
    # for i in range(len(cost)):
    #     # 降序排序
    #     top15_list_i = np.sort(top15_list[i])
    #     # print(i,'轮Top15:',-float(cost[i]),top15_list[i])
    #     if -float(cost[i]) >= top15_list_i[0]:
    #         Hit_top15 += 1
    #     if -float(cost[i]) >= top15_list_i[4]:
    #         Hit_top10 += 1
    #     if -float(cost[i]) >= top15_list_i[9]:
    #         Hit_top5 += 1
    #     if -float(cost[i]) >= top15_list_i[14]:
    #         Hit_top1 += 1
    #
    # Hit_top15_percent = Hit_top15 / len(cost)
    # Hit_top10_percent = Hit_top10 / len(cost)
    # Hit_top5_percent = Hit_top5 / len(cost)
    # Hit_top1_percent = Hit_top1 / len(cost)
    #
    # print('Top15 target percent:', Hit_top15_percent)
    # print('Top10 target percent:', Hit_top10_percent)
    # print('Top5 target percent:', Hit_top5_percent)
    # print('Top1 target percent:', Hit_top1_percent)
    return avg_cost, cost


def rollout(model, dataset, opts):
    # print(dataset)
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    print("model.eval")

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, order = model(move_to(bat, opts.device))
        # print(cost)
        return cost.data.cpu()

    # aaaa = tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ccc = [
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.eval_batch_size),
            disable=opts.no_progress_bar,
        )
    ]
    bbb = torch.cat(ccc, 0)
    # print('ccc',ccc)
    # print('bbb',bbb)
    return bbb


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            (
                max_norm if max_norm > 0 else math.inf
            ),  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model,
    optimizer,
    baseline,
    lr_scheduler,
    epoch,
    val_dataset,
    problem,
    tb_logger,
    opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    # print(val_dataset.w)
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    # Generate new training data for each epoch
    # print('start**************')
    # 在每个epoch中都重新生成新的数据集，且使用baseline网络来对数据进行预测来作为标签    # xxx每次生成新的数据集处  每次由class NOOPDataset(Dataset)产生，当时生成写错，导致一直没变
    xxx = problem.make_allnum_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size,
        distribution=opts.data_distribution,
    )

    # print('xxx_index',xxx[0],xxx[-1],xxx)
    training_dataset = baseline.wrap_dataset(xxx)
    # print('training_dataset',training_dataset[0],training_dataset[-1])
    # print('xxx',xxx)
    # print('middle*********')
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=1
    )
    # print('end**************')
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    # print("111!!!!")

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        # print("22!!!!")

        train_batch(
            model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
        )
        # print("!!!!!\n")
        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    # if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
    #     print('Saving model and state...')
    #     torch.save(
    #         {
    #             'model': get_inner_model(model).state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'rng_state': torch.get_rng_state(),
    #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #             'baseline': baseline.state_dict()
    #         },
    #         os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     )
    t1 = time.time()
    avg_reward, cost = validate(model, val_dataset, opts)
    t2 = time.time()
    t_cost = t2 - t1
    # print('Validation duration',t_cost)
    jilu_val_cost(
        epoch, -avg_reward.item(), t_cost, "%d_n_allnum.csv" % (args.val_user_num)
    )

    # if not opts.no_tensorboard:
    #     tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    # return -avg_reward.item()
    return -avg_reward.item(), cost


def train_batch(
    model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    training_start = time.time()

    x, bl_val = baseline.unwrap_batch(batch)
    # print('bl_val',bl_val)
    # print('x',x)

    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)
    # print(cost,len(cost),log_likelihood)
    # Evaluate baseline, get baseline loss if any (only for critic)

    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    # print('bl_val:',bl_val,'bl_loss:',bl_loss)
    # print('bl_val:',type(bl_val),'bl_loss:',bl_loss)

    # Calculate REINFORCE loss with advantage clipping
    # advantage = cost - baseline, clipped to [-1, 1] for stability
    # This prevents extreme gradients when cost deviates significantly from baseline
    c_reward = 1
    baseline_gap = cost - bl_val
    baseline_gap = torch.clamp(baseline_gap, min=-c_reward, max=c_reward)
    reinforce_loss = (baseline_gap * log_likelihood).mean()
    # reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    training_end = time.time()
    training_cost = training_end - training_start

    # print('training_cost',training_cost)
    # path_ = "%d_n_training_cost.csv"%(args.val_user_num)
    # with open(path_, 'a+') as f:
    #     csv_write = csv.writer(f)
    #     data_row = [epoch, training_cost]
    #     csv_write.writerow(data_row)
    # Logging
    if step % int(opts.log_step) == 0:
        log_values(
            cost,
            grad_norms,
            epoch,
            batch_id,
            step,
            log_likelihood,
            reinforce_loss,
            bl_loss,
            tb_logger,
            opts,
        )
