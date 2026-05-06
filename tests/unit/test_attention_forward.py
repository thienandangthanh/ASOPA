"""Test that loading the converted state_dict checkpoint reproduces the
original full-pickle's behaviour on a fixed input batch.

The decode order on a fixed input must match across CPU runs (deterministic
greedy decoding). Used as the contract that survives the nets/ -> attention_model/
rename in Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


CKPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "checkpoints" / "variable_user_n10_epoch480.pth"
)


@pytest.fixture(scope="module")
def loaded_model():
    if not CKPT_PATH.exists():
        pytest.skip(f"checkpoint missing: {CKPT_PATH}")

    from nets.attention_model import AttentionModel, set_decode_type
    from utils import load_problem

    payload = torch.load(str(CKPT_PATH), weights_only=False, map_location="cpu")
    init_args = payload["model_init_args"]

    problem = load_problem("noop")
    model = AttentionModel(
        init_args["embedding_dim"],
        init_args["hidden_dim"],
        problem,
        n_encode_layers=init_args["n_encode_layers"],
        mask_inner=init_args["mask_inner"],
        mask_logits=init_args["mask_logits"],
        normalization=init_args.get("normalization", "batch"),
        tanh_clipping=init_args["tanh_clipping"],
        checkpoint_encoder=init_args.get("checkpoint_encoder", False),
        shrink_size=init_args.get("shrink_size", None),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    set_decode_type(model, "greedy")
    return model


def test_state_dict_load_matches_validation_golden(loaded_model, fixtures_dir):
    """End-to-end: state_dict checkpoint reproduces avg_cost from golden_n8_validation."""
    from problems.noop.problem_noop import NOOP

    golden = json.loads((fixtures_dir / "golden_n8_validation.json").read_text())
    if golden.get("skipped"):
        pytest.skip(golden.get("reason", "validation golden skipped"))

    val_dataset = NOOP.load_val_dataset(size=8, num_samples=100)
    from torch.utils.data import DataLoader

    costs = []
    with torch.no_grad():
        for batch in DataLoader(val_dataset, batch_size=1):
            cost, _ = loaded_model(batch)
            costs.append(cost.cpu())
    cost = torch.cat(costs, 0)

    avg_cost = float(cost.mean())
    # Floats from convex solvers can drift at ~1e-6; allow a tiny margin.
    assert avg_cost == pytest.approx(golden["avg_cost"], rel=1e-6)


def test_attention_forward_returns_cost_and_log_likelihood(loaded_model):
    """Smoke check on shape contract — model(input) returns (cost, log_likelihood)."""
    from problems.noop.problem_noop import NOOP

    val_dataset = NOOP.load_val_dataset(size=8, num_samples=4)
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(val_dataset, batch_size=2)))
    with torch.no_grad():
        cost, log_likelihood = loaded_model(batch)

    assert cost.shape == (2,)
    assert log_likelihood.shape == (2,)
    assert torch.all(torch.isfinite(cost))
    assert torch.all(torch.isfinite(log_likelihood))
