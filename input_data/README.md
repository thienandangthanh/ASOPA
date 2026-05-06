# `input_data/`

Holds all *data* the codebase reads or writes — separated from source
code so teammates can find inputs/outputs without grep'ing.

## Layout

```
input_data/
├── input/         # (placeholder) future generated training data
├── output/        # per-epoch validation .mat files written by run.py
├── variable/      # (placeholder) snapshots of stochastic env state
└── dependencies/  # external reference data, never modified by training
    ├── val/       # n*_valdataset.mat — fixed validation samples per user-count
    ├── top10/     # paper-reference Top-10 results
    └── top15/     # paper-reference Top-15 results
```

## What goes where

- **`dependencies/`** — Anything the team committed *once* and reads at runtime.
  Don't write here from training code.
- **`output/`** — `run.py` writes `n{N}_performance_value_{val_size}.mat`
  per training run. Safe to rm-rf; will be regenerated.
- **`input/`** and **`variable/`** — kept for future work; currently empty.
  Put generated training data or stochastic snapshots here when needed.
