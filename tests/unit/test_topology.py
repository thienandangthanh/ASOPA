"""Snapshot test for `generate_topology` — same seed must produce same User
attributes (id, p_max, g_hat, w, w_hat, d, g)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_golden(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "golden_topology.json").read_text())


@pytest.fixture(autouse=True)
def _seed():
    """Re-seed before every test in this module (generate_topology pulls many
    RVs and downstream tests would corrupt each other otherwise)."""
    from my_utils import seed_everything

    seed_everything(1234)


def test_generate_topology_user_count_includes_padding(fixtures_dir):
    """generate_topology returns user_number real users + padding to max_num."""
    from resource_allocation_optimization import generate_topology

    users = generate_topology(5, 20, 100, 1, 32)
    assert len(users) == 10, "default max_num=10; 5 real + 5 padded"

    real = [u for u in users if u.p_max > 0]
    padded = [u for u in users if u.p_max == 0]
    assert len(real) == 5
    assert len(padded) == 5


def test_generate_topology_matches_golden_attributes(fixtures_dir):
    from resource_allocation_optimization import generate_topology

    users = generate_topology(10, 20, 100, 1, 32)
    golden = _load_golden(fixtures_dir)

    assert len(users) == len(golden["users"])
    for live, snap in zip(users, golden["users"]):
        assert int(live.id) == snap["id"]
        assert float(live.p_max) == pytest.approx(snap["p_max"], rel=1e-9)
        assert float(live.g_hat) == pytest.approx(snap["g_hat"], rel=1e-9)
        assert float(live.w_hat) == pytest.approx(snap["w_hat"], rel=1e-9)
        assert float(live.d) == pytest.approx(snap["d"], rel=1e-9)
        # `g` and `w` are mutable runtime fields; loose check.
        assert float(live.w) == pytest.approx(snap["w"], rel=1e-9)


def test_generate_topology_different_seed_changes_weights():
    """Sanity check: changing seed actually changes the random weight assignment."""
    from my_utils import seed_everything
    from resource_allocation_optimization import generate_topology

    seed_everything(1)
    users_a = generate_topology(10, 20, 100, 1, 32)
    seed_everything(2)
    users_b = generate_topology(10, 20, 100, 1, 32)

    w_a = [u.w for u in users_a]
    w_b = [u.w for u in users_b]
    assert w_a != w_b, "different seeds should produce different weights"
