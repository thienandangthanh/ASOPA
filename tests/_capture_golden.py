"""Capture pre-refactor golden values that future tests assert against.

Runs against the *current* (unrefactored) code paths. After Phase 3, the
package layout changes; the captured fixtures stay valid because they are
plain numbers/JSON, not pickles tied to import paths.

Outputs:
    tests/fixtures/golden_topology.json      — generate_topology determinism
    tests/fixtures/golden_throughput.json    — get_max_sum_weighted_alpha_throughput on hand-crafted users
    tests/fixtures/golden_dataset.json       — NOOPDataset / NOOPValDataset / NOOP_allnum_Dataset hashes
    tests/fixtures/golden_baselines.json     — duibi_* orderings on a fixed topology
    tests/fixtures/golden_n8_validation.json — full validation pass on n=8 with checkpoint epoch 480 (CPU)

CPU-only. The VM has no GPU; GPU-specific golden capture is the user's job
post-refactor.

Run:
    uv run python tests/_capture_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# CPU-only deterministic environment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONHASHSEED"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# conf.py parses sys.argv at import; make it deterministic.
sys.argv = ["capture-golden"]

import numpy as np
import torch

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def _tensor_hash(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy().astype(np.float64).tobytes()
    return hashlib.sha256(arr).hexdigest()


def _array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def capture_topology() -> dict:
    """Snapshot generate_topology output for fixed seed + args."""
    from my_utils import seed_everything
    from resource_allocation_optimization import generate_topology

    seed_everything(1234)
    users = generate_topology(10, 20, 100, 1, 32)
    snap = []
    for u in users:
        snap.append(
            {
                "id": int(u.id),
                "p_max": float(u.p_max),
                "g_hat": float(u.g_hat),
                "w_hat": float(u.w_hat),
                "g": float(u.g),
                "w": float(u.w),
                "d": float(u.d),
            }
        )
    return {"seed": 1234, "user_number": 10, "users": snap}


def capture_throughput() -> dict:
    """Snapshot get_max_sum_weighted_alpha_throughput on fixed users."""
    from my_utils import seed_everything
    from resource_allocation_optimization import (
        generate_topology,
        get_max_sum_weighted_alpha_throughput,
        sort_by_decode_order,
    )

    seed_everything(1234)
    users = generate_topology(5, 20, 100, 1, 32)
    real_users = [u for u in users if u.p_max > 0]
    for i, u in enumerate(real_users):
        u.decode_order = i
    ordered = sort_by_decode_order(real_users, list(range(len(real_users))))
    util = float(get_max_sum_weighted_alpha_throughput(users=ordered))
    return {
        "user_count": len(real_users),
        "decode_order_identity": list(range(len(real_users))),
        "utility": util,
    }


def capture_dataset() -> dict:
    """Snapshot dataset reproducibility (g/w tensor hashes)."""
    from my_utils import seed_everything
    from problems.noop.problem_noop import (
        NOOP_allnum_Dataset,
        NOOPDataset,
        NOOPValDataset,
    )

    seed_everything(1234)
    ds_train = NOOPDataset(num_samples=10, seed=1234)
    seed_everything(1234)
    ds_val = NOOPValDataset(num_samples=5, seed=1234, size=8)
    seed_everything(1234)
    ds_allnum = NOOP_allnum_Dataset(num_samples=10, seed=1234)

    def first_item(ds):
        item = ds[0]
        return _tensor_hash(item)

    return {
        "NOOPDataset": {
            "g_hash": _tensor_hash(ds_train.g),
            "w_hash": _tensor_hash(ds_train.w),
            "first_item_hash": first_item(ds_train),
            "len": len(ds_train),
        },
        "NOOPValDataset": {
            "g_hash": _tensor_hash(ds_val.g),
            "w_hash": _tensor_hash(ds_val.w),
            "first_item_hash": first_item(ds_val),
            "len": len(ds_val),
        },
        "NOOP_allnum_Dataset": {
            "g_hash": _tensor_hash(ds_allnum.g),
            "w_hash": _tensor_hash(ds_allnum.w),
            "first_item_hash": first_item(ds_allnum),
            "len": len(ds_allnum),
        },
    }


def capture_baselines() -> dict:
    """Snapshot duibi_* baseline orderings + utilities on a 5-user topology."""
    from my_utils import seed_everything, set_users_g, set_users_w
    from resource_allocation_optimization import (
        duibi_exhaustive_search,
        generate_topology,
        get_max_sum_weighted_alpha_throughput,
        sort_by_decode_order,
    )

    seed_everything(1234)
    users = generate_topology(5, 20, 100, 1, 32)
    real = [u for u in users if u.p_max > 0]
    g_arr = np.array([u.g_hat * 1.5 for u in real])
    w_arr = np.array([u.w_hat for u in real])
    set_users_g(real, g_arr)
    set_users_w(real, w_arr)

    out = {}
    # Hand-coded "g desc" — sort users by descending g, decode_order = position
    g_desc = sorted(real, key=lambda u: -u.g)
    for i, u in enumerate(g_desc):
        u.decode_order = i
    ordered = sort_by_decode_order(real, [u.id for u in g_desc])
    out["g_desc"] = {
        "order_ids": [int(u.id) for u in g_desc],
        "utility": float(get_max_sum_weighted_alpha_throughput(users=ordered)),
    }

    # Hand-coded "w desc"
    seed_everything(1234)
    users = generate_topology(5, 20, 100, 1, 32)
    real = [u for u in users if u.p_max > 0]
    set_users_g(real, g_arr)
    set_users_w(real, w_arr)
    w_desc = sorted(real, key=lambda u: -u.w)
    for i, u in enumerate(w_desc):
        u.decode_order = i
    ordered = sort_by_decode_order(real, [u.id for u in w_desc])
    out["w_desc"] = {
        "order_ids": [int(u.id) for u in w_desc],
        "utility": float(get_max_sum_weighted_alpha_throughput(users=ordered)),
    }

    # Exhaustive search ground truth (5 users only — 120 perms, fast)
    seed_everything(1234)
    users = generate_topology(5, 20, 100, 1, 32)
    real = [u for u in users if u.p_max > 0]
    set_users_g(real, g_arr)
    set_users_w(real, w_arr)
    try:
        decode_order, max_util = duibi_exhaustive_search(real)
        out["exhaustive"] = {
            "decode_order": [int(x) for x in decode_order],
            "utility": float(max_util),
        }
    except Exception as e:  # noqa: BLE001
        out["exhaustive"] = {"error": repr(e)}

    return out


def capture_validation_n8(checkpoint_path: Path) -> dict:
    """Run the full ASOPA_validation flow on n=8 against epoch-480 checkpoint, CPU."""
    from options import get_options
    from train import validate
    from utils import load_problem

    if not checkpoint_path.exists():
        return {"skipped": True, "reason": f"checkpoint not found: {checkpoint_path}"}

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
            "100",
            "--eval_batch_size",
            "1",
            "--problem",
            "noop",
        ]
    )
    opts.device = torch.device("cpu")

    problem = load_problem(opts.problem)

    # Full-pickle load of legacy checkpoint (works pre-refactor).
    model = torch.load(str(checkpoint_path), weights_only=False, map_location="cpu")
    model = model.to(opts.device)

    val_dataset = problem.load_val_dataset(
        size=opts.val_graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    t0 = time.time()
    avg_cost, cost = validate(model, val_dataset, opts)
    elapsed = time.time() - t0

    cost_arr = cost.detach().cpu().numpy().astype(np.float64)
    return {
        "checkpoint": str(checkpoint_path.name),
        "val_size": int(opts.val_size),
        "val_graph_size": int(opts.val_graph_size),
        "avg_cost": float(avg_cost),
        "cost_first10": cost_arr[:10].tolist(),
        "cost_mean": float(np.mean(cost_arr)),
        "cost_std": float(np.std(cost_arr)),
        "cost_min": float(np.min(cost_arr)),
        "cost_max": float(np.max(cost_arr)),
        "wall_seconds": float(elapsed),
    }


def main():
    fixtures = {}

    print("[1/5] capture topology…", flush=True)
    fixtures["topology"] = capture_topology()
    (FIXTURES / "golden_topology.json").write_text(json.dumps(fixtures["topology"], indent=2))

    print("[2/5] capture throughput…", flush=True)
    fixtures["throughput"] = capture_throughput()
    (FIXTURES / "golden_throughput.json").write_text(json.dumps(fixtures["throughput"], indent=2))

    print("[3/5] capture dataset hashes…", flush=True)
    fixtures["dataset"] = capture_dataset()
    (FIXTURES / "golden_dataset.json").write_text(json.dumps(fixtures["dataset"], indent=2))

    print("[4/5] capture baseline orderings…", flush=True)
    fixtures["baselines"] = capture_baselines()
    (FIXTURES / "golden_baselines.json").write_text(json.dumps(fixtures["baselines"], indent=2))

    print("[5/5] capture n=8 validation (CPU)…", flush=True)
    ckpt = PROJECT_ROOT / "Variable_user_n10_epoch480.pth"
    fixtures["validation_n8"] = capture_validation_n8(ckpt)
    (FIXTURES / "golden_n8_validation.json").write_text(json.dumps(fixtures["validation_n8"], indent=2))

    summary_path = FIXTURES / "golden_summary.json"
    summary_path.write_text(json.dumps(fixtures, indent=2))
    print(f"\nWrote {summary_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
