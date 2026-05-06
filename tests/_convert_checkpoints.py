"""Convert legacy full-pickle checkpoints to state_dict-only files.

Legacy checkpoints at the project root (`Variable_user_n10_epoch{300,480}.pth`)
were saved via `torch.save(model, ...)` — full pickle, tied to the import path
`nets.attention_model.AttentionModel`. After Phase 3 renames `nets/ → attention_model/`,
those pickles fail to load.

This script:
    1. Loads each legacy checkpoint as full pickle (works pre-refactor).
    2. Extracts `model.state_dict()`.
    3. Saves to `output/checkpoints/variable_user_n10_epoch{N}.pth`
       along with the args needed to rebuild the model.
    4. Moves the original full-pickle into `output/checkpoints/legacy/` (gitignored).

Run BEFORE Phase 3:
    uv run python tests/_convert_checkpoints.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = ["convert-ckpt"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "output" / "checkpoints"
LEGACY_DIR = CHECKPOINT_DIR / "legacy"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_NAMES = ["Variable_user_n10_epoch300.pth", "Variable_user_n10_epoch480.pth"]


def model_init_args(model) -> dict:
    """Best-effort extraction of __init__ args from an AttentionModel instance.

    These mirror the kwargs passed in run.py when building the model. Saved
    alongside state_dict so post-refactor code can rebuild the same model
    architecture without guessing.
    """
    out = {}
    for attr in (
        "embedding_dim",
        "hidden_dim",
        "n_heads",
        "n_encode_layers",
        "tanh_clipping",
        "mask_inner",
        "mask_logits",
        "normalization",
        "checkpoint_encoder",
        "shrink_size",
    ):
        if hasattr(model, attr):
            out[attr] = getattr(model, attr)
    return out


def convert_one(legacy_name: str) -> dict:
    src = PROJECT_ROOT / legacy_name
    if not src.exists():
        return {"name": legacy_name, "skipped": True, "reason": "not found"}

    print(f"Loading legacy pickle: {src}")
    model = torch.load(str(src), weights_only=False, map_location="cpu")
    state = model.state_dict() if hasattr(model, "state_dict") else model

    # Pull the epoch number out of the filename for the new name.
    # "Variable_user_n10_epoch480.pth" → "variable_user_n10_epoch480.pth"
    new_name = legacy_name.lower()
    dst = CHECKPOINT_DIR / new_name

    payload = {
        "model_state_dict": state,
        "model_init_args": model_init_args(model),
        "source_legacy_filename": legacy_name,
    }
    torch.save(payload, str(dst))
    print(f"  → wrote state_dict: {dst}  ({dst.stat().st_size / 1e6:.2f} MB)")

    # Move (not copy) the original into legacy/ to avoid duplicate large files.
    legacy_dst = LEGACY_DIR / legacy_name
    if not legacy_dst.exists():
        shutil.move(str(src), str(legacy_dst))
        print(f"  → archived original: {legacy_dst}")
    else:
        # Already archived in a prior run; remove the root copy.
        src.unlink()
        print(f"  → original already archived; removed {src}")

    return {
        "name": legacy_name,
        "new_path": str(dst.relative_to(PROJECT_ROOT)),
        "init_args": payload["model_init_args"],
    }


def main():
    results = [convert_one(n) for n in LEGACY_NAMES]
    print("\nResults:")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
