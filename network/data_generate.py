import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from my_utils import *
from resource_allocation_optimization import *


class MyDataset(Dataset):
    def __init__(self, users, data_num=1000, seed=None):
        self.data_num = data_num
        self.users_g_hat = get_users_g_hat(users)
        if seed:
            np.random.seed(seed)
        self.g = np.random.rayleigh(1, size=[data_num, len(users)]) * self.users_g_hat

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return self.g[item]


def get_dataloader(users, data_num=1000, seed=None, batch_size=128, shuffle=False):
    dataset = MyDataset(users, data_num, seed)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader
