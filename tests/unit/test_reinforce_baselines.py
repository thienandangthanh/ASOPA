"""Unit tests for the REINFORCE baseline implementations."""

from __future__ import annotations

import pytest
import torch


def test_no_baseline_returns_zero():
    from reinforce_baselines import NoBaseline

    bl = NoBaseline()
    v, loss = bl.eval(torch.zeros(3, 5), torch.tensor([1.0, 2.0, 3.0]))
    assert v == 0
    assert loss == 0


def test_exponential_baseline_initial_value_is_batch_mean():
    from reinforce_baselines import ExponentialBaseline

    bl = ExponentialBaseline(beta=0.8)
    cost = torch.tensor([1.0, 2.0, 3.0])
    v, loss = bl.eval(torch.zeros(3, 5), cost)

    # torch float32 may carry sub-1e-7 noise on simple ops; tolerate it.
    assert float(v) == pytest.approx(2.0, abs=1e-6)
    assert loss == 0


def test_exponential_baseline_decays_with_beta():
    from reinforce_baselines import ExponentialBaseline

    bl = ExponentialBaseline(beta=0.8)
    bl.eval(torch.zeros(3, 5), torch.tensor([1.0, 2.0, 3.0]))  # initial v = 2.0
    v, _ = bl.eval(torch.zeros(3, 5), torch.tensor([4.0, 5.0, 6.0]))  # mean = 5

    # v_new = 0.8 * 2.0 + 0.2 * 5.0 = 2.6 (modulo float32 noise)
    assert float(v) == pytest.approx(2.6, abs=1e-6)


def test_exponential_baseline_state_dict_round_trip():
    from reinforce_baselines import ExponentialBaseline

    bl = ExponentialBaseline(beta=0.8)
    bl.eval(torch.zeros(3, 5), torch.tensor([1.0, 2.0, 3.0]))
    state = bl.state_dict()

    bl2 = ExponentialBaseline(beta=0.8)
    bl2.load_state_dict(state)
    assert torch.allclose(bl2.v, bl.v)
