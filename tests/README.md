# Tests

Behavior-preservation suite for the ASOPA refactor.

The tests were captured against pre-refactor code paths (`nets/`, `problems/noop/`,
`resource_allocation_optimization.py`, `my_utils.py`, `options.py`, `conf.py`,
`reinforce_baselines.py`). They must stay green through every refactor phase.
After Phase 3 imports change to the new package names (`attention_model/`,
`sic_ordering/`, `power_allocation/`, `utils/`, `configurations/`).

## Layout

```
tests/
├── conftest.py                         # shared fixtures, sys.path setup, CPU-forcing
├── _capture_golden.py                  # one-shot pre-refactor fixture capture (Phase 0)
├── _convert_checkpoints.py             # one-shot legacy .pth → state_dict (Phase 0)
├── fixtures/
│   ├── golden_topology.json
│   ├── golden_throughput.json
│   ├── golden_dataset.json
│   ├── golden_baselines.json
│   ├── golden_n8_validation.json
│   └── golden_summary.json
├── unit/
│   ├── test_seeding.py
│   ├── test_topology.py
│   ├── test_throughput.py
│   ├── test_dataset.py
│   ├── test_baselines.py
│   ├── test_state_noop.py
│   ├── test_graph_encoder.py
│   ├── test_attention_forward.py
│   └── test_reinforce_baselines.py
└── integration/
    ├── test_validation_smoke.py
    └── test_one_epoch.py
```

## Running

```bash
# Default (skips slow + GPU tests)
uv run pytest tests/ -v

# Include slow tests (exhaustive search, 1-epoch training, full validation)
uv run pytest tests/ -v -m "slow or not slow"

# Run only the slow suite
uv run pytest tests/ -v -m slow

# GPU-marked tests (must be run by user — VM has no GPU)
uv run pytest tests/ -v -m gpu
```

## Markers

- `slow` — takes >5s on CPU (exhaustive search, full validation, 1-epoch training).
- `gpu` — requires CUDA; **NOT runnable in this VM**. Hand off to the user.

## Re-capturing goldens

If a non-bug behavior change is intentional (rare!), regenerate fixtures:

```bash
uv run python tests/_capture_golden.py
```

This rewrites `tests/fixtures/golden_*.json`. Commit both code and fixtures
together with a clear note on why behavior changed.

## CPU-only constraint

The VM running these tests has no GPU. `conftest.py` sets `CUDA_VISIBLE_DEVICES=""`
before any torch import. Tests that genuinely need GPU must use `@pytest.mark.gpu`
and the user must run them separately on a GPU host.
