#!/usr/bin/env bash
# Validate a trained ASOPA checkpoint.
#
# Loads `output/checkpoints/variable_user_n{user_num}_epoch{val_epoch}.pth`
# (state_dict format) and reports avg utility + per-sample inference time
# on the held-out `n{val_user_num}_valdataset.mat` validation set.

set -euo pipefail

uv run python ASOPA_validation.py \
    --user_num 10 \
    --val_user_num 8 \
    --val_epoch 480 \
    --val_size 1000 \
    --no_cuda \
    "$@"
