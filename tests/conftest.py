"""Pytest fixtures shared across all ASOPA tests.

Adds the project root to sys.path so test modules can import top-level files
(`resource_allocation_optimization`, `my_utils`, `nets.*`, `problems.*`, etc.)
during the *pre-refactor* phase. After Phase 3, these imports will resolve via
the new package layout (`power_allocation`, `attention_model`, `sic_ordering`).

Also pins random seeds and forces CPU execution for determinism. The VM
running these tests has no GPU; GPU-dependent tests are marked
`@pytest.mark.gpu` and skipped by default (see pyproject.toml).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Ensure project root is importable, regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Force CPU and disable any optional CUDA paths before user code imports torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(autouse=True)
def _isolated_argv(monkeypatch):
    """Clear sys.argv so modules that call `parser.parse_args()` at import
    time (e.g. legacy `conf.py`) don't see pytest's own argv."""
    monkeypatch.setattr(sys, "argv", ["pytest"])


@pytest.fixture
def fixed_seed():
    """Seed numpy / torch / random deterministically. Returns the seed value."""
    seed = 1234
    import random as _random

    import numpy as np
    import torch

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures"
