"""REINFORCE training and validation loop for the attention model.

The high-level entry point is `train_epoch(...)` which:
  1. Generates a fresh variable-user dataset for the epoch.
  2. Wraps it through the baseline (rollout / exponential / no-baseline).
  3. Runs `train_batch(...)` for every minibatch.
  4. Calls `validate(...)` on the held-out val_dataset and logs the result.
  5. Steps the LR scheduler.

Pulled out of the legacy top-level `train.py` so entry-point scripts
(`run.py`, `train.py`, `ASOPA_validation.py`) can stay thin.
"""

from __future__ import annotations

import csv
import math
import time

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_model.attention_model import set_decode_type
from utils import move_to
from utils.log_utils import log_values


def jilu_val_cost(epoch, val_performance, t_cost, path):
    """Append (epoch, validation utility, validation wall-time) to a CSV."""
    with open(path, "a+") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, val_performance, t_cost])


def get_inner_model(model):
    """Unwrap DataParallel if necessary."""
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    """Run greedy decoding on `dataset` and return (avg_cost, per_sample_cost)."""
    print("Validating...")
    t1 = time.time()
    cost = rollout(model, dataset, opts)
    t2 = time.time()
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )
    print("Validation time: {}".format(t2 - t1))
    return avg_cost, cost


def rollout(model, dataset, opts):
    """Greedy-decode the model over `dataset`, return per-sample costs (CPU)."""
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _order = model(move_to(bat, opts.device))
        return cost.data.cpu()

    chunks = [
        eval_model_bat(bat)
        for bat in tqdm(
            DataLoader(dataset, batch_size=opts.eval_batch_size),
            disable=opts.no_progress_bar,
        )
    ]
    return torch.cat(chunks, 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clip each param-group to `max_norm`; return (raw_norms, clipped_norms)."""
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g, max_norm) for g in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_batch(
    model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    """Single REINFORCE step with advantage-clipping."""
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    cost, log_likelihood = model(x)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Advantage clipped to [-1, 1] to prevent extreme gradients when cost
    # deviates significantly from the baseline.
    c_reward = 1
    advantage = torch.clamp(cost - bl_val, min=-c_reward, max=c_reward)
    reinforce_loss = (advantage * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    if step % int(opts.log_step) == 0:
        log_values(
            cost, grad_norms, epoch, batch_id, step,
            log_likelihood, reinforce_loss, bl_loss, tb_logger, opts,
        )


def train_epoch(
    model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem,
    tb_logger, opts,
):
    """One full training epoch + validation. Returns (-avg_reward, per_sample_cost)."""
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    raw_dataset = problem.make_allnum_dataset(
        size=opts.graph_size,
        num_samples=opts.epoch_size,
        distribution=opts.data_distribution,
    )
    training_dataset = baseline.wrap_dataset(raw_dataset)
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=1
    )

    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        train_batch(
            model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    val_t1 = time.time()
    avg_reward, cost = validate(model, val_dataset, opts)
    val_t2 = time.time()
    jilu_val_cost(
        epoch, -avg_reward.item(), val_t2 - val_t1,
        "%d_n_allnum.csv" % (opts.val_user_num),
    )

    baseline.epoch_callback(model, epoch)
    lr_scheduler.step()
    return -avg_reward.item(), cost
