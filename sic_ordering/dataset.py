"""PyTorch Dataset wrappers for the NOMA SIC-ordering problem.

Three flavours, all consumed by `attention_model.training_loop` via the
NOOP problem class:

  - NOOPDataset           Random training samples (Rayleigh channel × g_hat).
  - NOOPValDataset        Pre-built validation set loaded from .mat file.
  - NOOP_allnum_Dataset   Variable-user training set (mixes user-counts in
                          [num_min, num_max] with zero padding).

The classes intentionally read module-level state from `problem_noop`
(noop_users, val_noop_users, w_to_1) — that's the legacy shape preserved
through the Phase-3 reorganization. A future refactor can take an explicit
config in __init__.
"""

from __future__ import annotations

import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from configurations import get_default_env_config
from my_utils import get_users_g_hat
from power_allocation import generate_topology
from sic_ordering.problem_noop import noop_users, val_noop_users, w_to_1


_env = get_default_env_config()


class NOOPDataset(Dataset):
    """Per-frame training samples with Rayleigh channel and random weights."""

    def __init__(
        self,
        num_samples: int = 1000,
        seed: int = 1234,
        size: int | None = None,
        filename=None,
        distribution=None,
    ):
        if size is None:
            size = len(noop_users)
        print("size", size)
        users = noop_users if size == _env.user_num else val_noop_users
        self.data_num = num_samples
        self.users_g_hat = get_users_g_hat(users)

        if seed:
            np.random.seed(seed)
        g = np.random.exponential(1, size=[num_samples, len(users)]) * self.users_g_hat
        # Keep noisy-channel computation as-is for reproducibility (unused below).
        _ = g + np.random.normal(0, 1, size=[num_samples, len(users)]) * _env.noise

        values_w = [1, 2, 4, 8, 16, 32]
        w = np.asarray(
            [random.choices(values_w, k=len(users)) for _ in range(num_samples)]
        )
        self.g = torch.FloatTensor(g).unsqueeze(-1)
        self.w = torch.FloatTensor(w).unsqueeze(-1)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t = []
        for tuser, tg, tw in zip(noop_users, self.g[idx], self.w[idx]):
            t.append([tuser.p_max, tw, tg * w_to_1])
        return torch.FloatTensor(t)


class NOOPValDataset(Dataset):
    """Validation set loaded from a precomputed .mat file (`Val/n{N}_valdataset.mat`)."""

    def __init__(
        self,
        num_samples: int = 1000,
        seed: int = 1234,
        size: int | None = None,
        filename=None,
        distribution=None,
    ):
        if size is None:
            size = len(noop_users)
        print("size", size)
        users = noop_users if size == _env.user_num else val_noop_users
        self.data_num = num_samples
        self.users_g_hat = get_users_g_hat(users)
        if seed:
            np.random.seed(seed)

        mat = sio.loadmat("Val/n%d_valdataset.mat" % size)
        self.g = torch.FloatTensor(mat["val_g"])
        self.w = torch.FloatTensor(mat["val_w"])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t = []
        for tuser, tg, tw in zip(noop_users, self.g[idx], self.w[idx]):
            t.append([tuser.p_max, tw, tg * w_to_1])
        return torch.FloatTensor(t)


class NOOP_allnum_Dataset(Dataset):
    """Variable-user training set: mixes user-counts in [num_min, num_max]
    with zero padding so a single batch can train on heterogeneous topologies."""

    def __init__(
        self,
        num_samples: int = 1000,
        seed: int = 1234,
        size: int | None = None,
        filename=None,
        distribution=None,
    ):
        users_list = []
        g_list: list = []
        w_list: list = []
        self.data_num = num_samples

        num_list = list(range(_env.num_min, _env.num_max + 1))  # e.g. 5..10
        values_w = [1, 2, 4, 8, 16, 32]
        list_num_random = [213, 213, 213, 213, 214, 214]
        random.shuffle(list_num_random)

        for user_num in num_list:
            users_list.append(
                generate_topology(
                    user_num,
                    _env.d_min, _env.d_max, _env.w_min, _env.w_max,
                    _env.num_max,
                )
            )

        for i, tusers_list in enumerate(users_list):
            self.users_g_hat = get_users_g_hat(tusers_list)
            self.users_g_hat = self.users_g_hat[self.users_g_hat != 0]
            if seed:
                np.random.seed(seed)
            user_num = sum(1 for tusers in tusers_list if tusers.g > 0)
            list_num = list_num_random[i]
            g = np.hstack((
                np.random.exponential(1, size=[list_num, user_num]) * self.users_g_hat,
                np.zeros([list_num, len(tusers_list) - user_num]),
            ))
            w = np.hstack((
                np.array(
                    [random.choices(values_w, k=user_num) for _ in range(list_num)]
                ),
                np.zeros([list_num, len(tusers_list) - user_num]),
            ))
            g_list.extend(list(g))
            w_list.extend(list(w))

        state_list = list(zip(g_list, w_list))
        random.shuffle(state_list)
        g_random = np.array([gr for gr, _ in state_list])
        w_random = np.array([wr for _, wr in state_list])

        self.g = torch.FloatTensor(g_random).unsqueeze(-1)
        self.w = torch.FloatTensor(w_random).unsqueeze(-1)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        t = []
        for tuser, tg, tw in zip(noop_users, self.g[idx], self.w[idx]):
            if tw and tg:
                t.append([tuser.p_max, tw, tg * w_to_1])
            else:
                # padding row
                t.append([0, tw, tg * w_to_1])
        return torch.FloatTensor(t)
