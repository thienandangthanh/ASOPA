#!/usr/bin/env bash
# Train ASOPA from scratch on the canonical 10-user topology.
#
# CPU-only by default; pass --no-no_cuda or unset --no_cuda to allow GPU.
# Output: best-epoch checkpoints in output/checkpoints/ and per-epoch
# validation history in input_data/output/.

set -euo pipefail

uv run python train.py \
    --user_num 10 \
    --val_user_num 8 \
    --batch_size 64 \
    --epoch_size 1280 \
    --val_size 1000 \
    --eval_batch_size 128 \
    --baseline rollout \
    --n_epochs 501 \
    --no_cuda \
    "$@"
