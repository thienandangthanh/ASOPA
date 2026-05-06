"""Unit tests for `seed_everything` — must produce identical RNG state for a
given seed across repeated calls. After the refactor this lives in
`utils/seeding.py`; the import here is updated then.
"""

from __future__ import annotations

import numpy as np
import torch


def test_seed_everything_is_deterministic_for_numpy(fixed_seed):
    from my_utils import seed_everything

    seed_everything(fixed_seed)
    a = np.random.rand(5)
    seed_everything(fixed_seed)
    b = np.random.rand(5)
    assert np.allclose(a, b), "numpy RNG must be deterministic for the same seed"


def test_seed_everything_is_deterministic_for_torch(fixed_seed):
    from my_utils import seed_everything

    seed_everything(fixed_seed)
    a = torch.rand(5)
    seed_everything(fixed_seed)
    b = torch.rand(5)
    assert torch.allclose(a, b), "torch RNG must be deterministic for the same seed"
