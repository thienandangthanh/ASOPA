#!/usr/bin/env bash
# Run the baseline-comparison launcher: load the validation dataset and
# benchmark every non-RL ordering method (g/w sort, tabu, Qian heuristic,
# exhaustive). Output is the table printed by `show_speed_performance_dataset`.

set -euo pipefail

uv run python run_baseline.py \
    --user_num 10 \
    --val_user_num 8 \
    --val_size 100 \
    --no_cuda \
    "$@"
