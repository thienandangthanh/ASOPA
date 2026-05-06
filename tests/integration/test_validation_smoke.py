"""Integration smoke test: full validation pass on n=8 with the converted
state_dict checkpoint. Asserts avg_cost stays within tolerance of the
pre-refactor golden value.
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


@pytest.mark.slow
def test_full_validation_n8_matches_golden(fixtures_dir):
    if not CKPT_PATH.exists():
        pytest.skip(f"checkpoint missing: {CKPT_PATH}")

    golden = json.loads((fixtures_dir / "golden_n8_validation.json").read_text())
    if golden.get("skipped"):
        pytest.skip(golden.get("reason", "validation golden skipped"))

    from attention_model.attention_model import AttentionModel
    from configurations import get_options
    from train import validate
    from utils import load_problem

    opts = get_options(
        [
            "--no_cuda",
            "--no_tensorboard",
            "--no_progress_bar",
            "--graph_size",
            "8",
            "--val_graph_size",
            "8",
            "--val_size",
            str(golden["val_size"]),
            "--eval_batch_size",
            "1",
            "--problem",
            "noop",
        ]
    )
    opts.device = torch.device("cpu")

    problem = load_problem(opts.problem)

    payload = torch.load(str(CKPT_PATH), weights_only=False, map_location="cpu")
    init_args = payload["model_init_args"]
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
    ).to(opts.device)
    model.load_state_dict(payload["model_state_dict"])

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    avg_cost, cost = validate(model, val_dataset, opts)

    # Within 0.5% of golden — accommodates tiny float drift across re-runs.
    assert float(avg_cost) == pytest.approx(golden["avg_cost"], rel=5e-3)

    # First sample's cost must match very tightly (deterministic greedy).
    assert float(cost[0]) == pytest.approx(golden["cost_first10"][0], rel=1e-6)
