# `commands/`

Copy-paste-runnable shell wrappers around the entry-point Python scripts.
Each script forwards `$@` so you can override any individual flag.

| Script              | What it does                                           |
|---------------------|--------------------------------------------------------|
| `train.sh`          | Train ASOPA from scratch (10 users, 501 epochs)        |
| `train-baseline.sh` | Run all `duibi_*` non-RL baselines on the val dataset  |
| `validate.sh`       | Validate a saved checkpoint (default epoch 480)        |
| `show.sh`           | Visualize order-influence + speed/utility distributions|
| `test.sh`           | Run the pytest suite (default skips slow + gpu marks)  |

## Conventions

- All scripts use `set -euo pipefail` — fail fast on any error.
- All scripts run **CPU-only** by default (`--no_cuda`). Strip that flag
  on a GPU host or pass `--no_cuda=False` (argparse store-true → flag presence is the signal).
- All scripts forward `$@` so you can override any individual option:
  ```bash
  ./commands/train.sh --user_num 8 --n_epochs 5
  ./commands/test.sh -m slow
  ```

## CPU vs GPU

This VM has **no GPU**, so the default flag set is CPU-only. To run on a
GPU host:

1. Install the CUDA torch extras: `uv sync --extra cu121`
2. Remove the `--no_cuda` flag (argparse store-true), e.g.:
   ```bash
   uv run python train.py --user_num 10 --n_epochs 501
   ```
