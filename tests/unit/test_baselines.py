"""Snapshot tests for baseline ordering methods (`duibi_*`).

Validates that the ordering+utility produced for a fixed 5-user topology stays
stable across the refactor. Exhaustive search is the ground-truth optimum.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    from my_utils import seed_everything
    seed_everything(1234)


def _build_users():
    """Recreate the same fixed 5-user setup the capture script used."""
    from my_utils import seed_everything, set_users_g, set_users_w
    from resource_allocation_optimization import generate_topology

    seed_everything(1234)
    users = generate_topology(5, 20, 100, 1, 32)
    real = [u for u in users if u.p_max > 0]
    g_arr = np.array([u.g_hat * 1.5 for u in real])
    w_arr = np.array([u.w_hat for u in real])
    set_users_g(real, g_arr)
    set_users_w(real, w_arr)
    return real


def test_g_descending_ordering_matches_golden(fixtures_dir):
    from resource_allocation_optimization import (
        get_max_sum_weighted_alpha_throughput,
        sort_by_decode_order,
    )

    golden = json.loads((fixtures_dir / "golden_baselines.json").read_text())["g_desc"]
    real = _build_users()

    g_desc = sorted(real, key=lambda u: -u.g)
    for i, u in enumerate(g_desc):
        u.decode_order = i

    assert [int(u.id) for u in g_desc] == golden["order_ids"]

    ordered = sort_by_decode_order(real, [u.id for u in g_desc])
    util = float(get_max_sum_weighted_alpha_throughput(users=ordered))
    assert util == pytest.approx(golden["utility"], rel=1e-6)


def test_w_descending_ordering_matches_golden(fixtures_dir):
    from resource_allocation_optimization import (
        get_max_sum_weighted_alpha_throughput,
        sort_by_decode_order,
    )

    golden = json.loads((fixtures_dir / "golden_baselines.json").read_text())["w_desc"]
    real = _build_users()

    w_desc = sorted(real, key=lambda u: -u.w)
    for i, u in enumerate(w_desc):
        u.decode_order = i

    assert [int(u.id) for u in w_desc] == golden["order_ids"]

    ordered = sort_by_decode_order(real, [u.id for u in w_desc])
    util = float(get_max_sum_weighted_alpha_throughput(users=ordered))
    assert util == pytest.approx(golden["utility"], rel=1e-6)


@pytest.mark.slow
def test_exhaustive_search_finds_optimum(fixtures_dir):
    """120 perms — slow-marked because it's heavier than other unit tests."""
    from resource_allocation_optimization import duibi_exhaustive_search

    golden = json.loads((fixtures_dir / "golden_baselines.json").read_text())["exhaustive"]
    real = _build_users()

    decode_order, max_util, _, _ = duibi_exhaustive_search(real, alpha=1)
    assert list(decode_order) == golden["decode_order"]
    assert float(max_util) == pytest.approx(golden["utility"], rel=1e-6)
