#!/usr/bin/env python
"""Top-level training launcher.

Thin alias of `train.py`. Kept under this name for muscle-memory and
documentation back-compat. Both forms work:

    uv run python run.py   --user_num 10 --n_epochs 1 --no_cuda
    uv run python train.py --user_num 10 --n_epochs 1 --no_cuda
"""

from configurations import get_options
from train import main


if __name__ == "__main__":
    main(get_options())
