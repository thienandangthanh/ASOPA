"""Snapshot test for `get_max_sum_weighted_alpha_throughput` against a
hand-crafted ordering. Captured value lives in tests/fixtures/golden_throughput.json."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _seed():
    from my_utils import seed_everything
    seed_everything(1234)


def test_get_max_sum_weighted_alpha_throughput_matches_golden(fixtures_dir):
    from power_allocation import (
        generate_topology,
        get_max_sum_weighted_alpha_throughput,
        sort_by_decode_order,
    )

    golden = json.loads((fixtures_dir / "golden_throughput.json").read_text())

    users = generate_topology(5, 20, 100, 1, 32)
    real = [u for u in users if u.p_max > 0]
    for i, u in enumerate(real):
        u.decode_order = i
    ordered = sort_by_decode_order(real, list(range(len(real))))
    util = float(get_max_sum_weighted_alpha_throughput(users=ordered))

    # Throughput is computed via convex optimization; allow modest tolerance.
    assert util == pytest.approx(golden["utility"], rel=1e-6)
    assert len(ordered) == golden["user_count"]
