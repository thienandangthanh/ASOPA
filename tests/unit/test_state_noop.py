"""Unit tests for StateNOOP — initialize, update, mask, finished semantics."""

from __future__ import annotations

import pytest
import torch


def test_state_noop_initialize_creates_empty_visited():
    from sic_ordering.state_noop import StateNOOP

    g = torch.randn(2, 5, 3)
    state = StateNOOP.initialize(g)

    assert state.g.shape == (2, 5, 3)
    assert state.visited_.shape == (2, 1, 5)
    assert int(state.visited_.sum()) == 0
    assert int(state.i.item()) == 0


def test_state_noop_update_marks_selected_as_visited():
    from sic_ordering.state_noop import StateNOOP

    g = torch.randn(2, 5, 3)
    state = StateNOOP.initialize(g)

    selected = torch.tensor([2, 4], dtype=torch.long)
    state = state.update(selected)

    # After 1 step, exactly one slot per batch item should be visited.
    assert int(state.visited_[0, 0, 2].item()) == 1
    assert int(state.visited_[1, 0, 4].item()) == 1
    assert int(state.visited_[0, 0].sum()) == 1
    assert int(state.i.item()) == 1


def test_state_noop_get_mask_reflects_visited():
    from sic_ordering.state_noop import StateNOOP

    g = torch.randn(1, 4, 3)
    state = StateNOOP.initialize(g)
    state = state.update(torch.tensor([1], dtype=torch.long))

    mask = state.get_mask()
    assert mask.shape == (1, 1, 4)
    assert bool(mask[0, 0, 1].item()) is True  # the visited node
    assert bool(mask[0, 0, 0].item()) is False
    assert bool(mask[0, 0, 2].item()) is False


def test_state_noop_all_finished_after_n_steps():
    from sic_ordering.state_noop import StateNOOP

    g = torch.randn(1, 3, 3)
    state = StateNOOP.initialize(g)
    assert not state.all_finished()

    for sel in [0, 1, 2]:
        state = state.update(torch.tensor([sel], dtype=torch.long))

    assert state.all_finished()
