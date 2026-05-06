"""Snapshot tests for the three dataset classes — same seed must produce
identical g/w tensors. Hashes captured pre-refactor in golden_dataset.json."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch


def _tensor_hash(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy().astype(np.float64).tobytes()
    return hashlib.sha256(arr).hexdigest()


@pytest.fixture(autouse=True)
def _seed():
    from my_utils import seed_everything
    seed_everything(1234)


def test_noop_dataset_is_reproducible(fixtures_dir):
    # Import first (may trigger module-level RNG consumption in problem_noop), then re-seed.
    from my_utils import seed_everything
    from sic_ordering.dataset import NOOPDataset

    golden = json.loads((fixtures_dir / "golden_dataset.json").read_text())["NOOPDataset"]
    seed_everything(1234)
    ds = NOOPDataset(num_samples=10, seed=1234)

    assert len(ds) == golden["len"]
    assert _tensor_hash(ds.g) == golden["g_hash"]
    assert _tensor_hash(ds.w) == golden["w_hash"]
    assert _tensor_hash(ds[0]) == golden["first_item_hash"]


def test_noop_val_dataset_is_reproducible(fixtures_dir):
    from my_utils import seed_everything
    from sic_ordering.dataset import NOOPValDataset

    golden = json.loads((fixtures_dir / "golden_dataset.json").read_text())["NOOPValDataset"]
    seed_everything(1234)
    ds = NOOPValDataset(num_samples=5, seed=1234, size=8)

    assert _tensor_hash(ds.g) == golden["g_hash"]
    assert _tensor_hash(ds.w) == golden["w_hash"]
    assert _tensor_hash(ds[0]) == golden["first_item_hash"]


def test_noop_allnum_dataset_is_reproducible(fixtures_dir):
    from my_utils import seed_everything
    from sic_ordering.dataset import NOOP_allnum_Dataset

    golden = json.loads((fixtures_dir / "golden_dataset.json").read_text())["NOOP_allnum_Dataset"]
    seed_everything(1234)
    ds = NOOP_allnum_Dataset(num_samples=10, seed=1234)

    assert _tensor_hash(ds.g) == golden["g_hash"]
    assert _tensor_hash(ds.w) == golden["w_hash"]
    assert _tensor_hash(ds[0]) == golden["first_item_hash"]
