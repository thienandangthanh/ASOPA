"""NOMA Ordering Optimization Problem (NOOP).

Defines the cost function consumed by the attention model's REINFORCE loop:
given a batch of (p_max, weight, channel-gain) tuples and a candidate
decode-order π, return the negative weighted-α throughput per sample.

Module-level `noop_users` and `val_noop_users` are the canonical training
and validation topologies, generated once at import time using defaults
from `configurations.env_config`. Datasets in `sic_ordering.dataset` reuse
these to keep behaviour reproducible across runs.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from configurations import get_default_env_config
from utils.seeding import seed_everything
from power_allocation.topology import get_users_g_hat, get_users_w_hat, set_users_g, set_users_w
from power_allocation import (
    generate_topology,
    generate_val_topology,
    get_max_sum_weighted_alpha_throughput,
    sort_by_decode_order,
)
from sic_ordering.state_noop import StateNOOP


_env = get_default_env_config()

seed_everything(_env.seed)
noop_users = generate_topology(
    _env.user_num, _env.d_min, _env.d_max, _env.w_min, _env.w_max
)
if _env.user_num != _env.val_user_num:
    val_noop_users = generate_val_topology(
        _env.val_user_num, _env.d_min, _env.d_max, _env.w_min, _env.w_max
    )
else:  # 同 user_num 时直接复用 / Reuse training topology when sizes match.
    val_noop_users = noop_users

users_g_hat = get_users_g_hat(noop_users)
user_w_hat = get_users_w_hat(noop_users)
np.random.seed(6741)
g = np.random.rayleigh(1, size=[100, len(noop_users)]) * users_g_hat
w = np.asarray(
    [random.choices([1, 2, 4, 8, 16, 32], k=4) for _ in range(len(noop_users))]
)

# Scale factor that keeps the network's input weights near 1.
w_to_1 = 1e9


class NOOP:
    """NOMA-ordering problem registered under name 'noop'.

    The class is a thin namespace of static methods so the attention model
    can call `problem.get_costs(batch, pi)`, `problem.make_state(g)`, etc.
    """

    NAME = "noop"

    @staticmethod
    def get_costs(dataset, pi):
        """Compute reward = -weighted-α throughput per batch sample.

        :param dataset: tensor of shape (batch, n_users, 3) — [p_max, w, g*w_to_1]
        :param pi: tensor of shape (batch, n_users) — selected decode order
        :returns: (cost, mask) — cost is negative utility, mask is unused
        """
        g = dataset.cpu().numpy()[:, :, -1]
        w = dataset.cpu().numpy()[:, :, -2]
        users = noop_users if len(dataset.cpu().numpy()[0]) == len(noop_users) else val_noop_users
        decode_order = pi.cpu().numpy()

        rewards = []
        for t_g, t_w, t_decode_order in zip(g, w, decode_order):
            set_users_g(users, t_g / w_to_1)
            set_users_w(users, t_w)
            users_order = sort_by_decode_order(users, t_decode_order)
            rewards.append(-get_max_sum_weighted_alpha_throughput(users=users_order))

        return torch.tensor(rewards, device=dataset.device), None

    # Dataset factories — actual classes live in sic_ordering.dataset to
    # avoid a circular import at module-load time.
    @staticmethod
    def make_dataset(*args, **kwargs):
        from sic_ordering.dataset import NOOP_allnum_Dataset
        return NOOP_allnum_Dataset(*args, **kwargs)

    @staticmethod
    def load_val_dataset(*args, **kwargs):
        from sic_ordering.dataset import NOOPValDataset
        return NOOPValDataset(*args, **kwargs)

    @staticmethod
    def make_allnum_dataset(*args, **kwargs):
        from sic_ordering.dataset import NOOP_allnum_Dataset
        return NOOP_allnum_Dataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateNOOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(*args, **kwargs):
        return None
