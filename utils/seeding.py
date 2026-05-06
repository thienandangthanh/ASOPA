"""Deterministic seeding for reproducible experiments.

`seed_everything(seed)` resets Python's `random`, NumPy's RNG, PyTorch's
CPU/CUDA RNG, and the `PYTHONHASHSEED` env var. Always call before
constructing the model or datasets if you want bit-exact reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 3258) -> None:
    """Re-seed all common RNG sources to `seed`."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
