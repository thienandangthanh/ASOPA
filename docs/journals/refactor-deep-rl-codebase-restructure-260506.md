# Deep RL Codebase Restructure: ASOPA Refactor Complete

**Date**: 2026-05-06 15:17
**Severity**: High (touched ~40% of codebase)
**Component**: Project architecture, configurations, model loading, data pipelines
**Status**: Resolved

## What Happened

Completed 7-phase refactor (`89cb6d5..6120fbd`) on `refactor/re-structure-project` to dissolve a genuinely painful codebase:
- Dual config systems (options.py + conf.py, overlapping params, hidden conflicts)
- Legacy/active duplicate directories (network/ vs nets/, eval.py dead code)
- 822-LOC monolith (resource_allocation_optimization.py) with no test coverage
- Hard onboarding due to scattered responsibility

End state: clean import paths, unified config, 35 passing tests, **bit-exact validation match** CPU↔GPU.

## The Brutal Truth

This refactor was necessary but exhausting. The codebase had accumulated enough technical debt to force teammates into 2-week onboarding rituals. Every config change meant hunting two files. Checkpoints were brittle pickle objects tied to import paths — any rename broke loaded weights. The power allocation solver lived in a single 822-LOC file with no separation of concerns.

What made it survivable: investing in **tests-first** behavior capture (Phase 1) before touching production code. Without those 27 tests locking the contract, we would have shipped numeric divergence. The fact that we caught a module-scope RNG corruption bug (`problem_noop.py` consuming python.random at import time, before pytest's seed fixture) proves the suite paid for itself.

## Technical Details

**Phase 0 — Golden checkpoint recovery** (`89cb6d5`):
- Captured baseline: topology, throughput, dataset hashes, avg_cost = -65.47529576270465 for n=8 validation.
- Converted legacy full-pickle .pth (tied to `nets.attention_model.AttentionModel`) to state_dict + init_args sidecar. Decouples saved weights from import paths — survives any class rename.

**Phase 1 — Behavior lock** (`54f1e25`): 27 tests written against unrefactored code. Critical fix: conftest must set sys.argv at module scope, not as fixture (argparse in legacy conf.py runs at import time).

**Phase 2 — Config consolidation** (`cafde4d`): Merged options.py + conf.py into `configurations/{env_config,learning_config,runtime_config,merge}.py`. Three argparse namespaces flattened into one via `get_options()` — preserves all callsite signatures. Resolved hidden conflict: `embedding_dim` defaulted to 2 in conf.py but 128 in options.py; kept 128 (the active value). Introduced `get_default_env_config()` for module-init contexts (problems/noop parses args at import).

**Phase 3 — Module reorganization** (`c3e57a9` + `b054921`):
- `nets/` → `attention_model/`; `problems/noop/` → `sic_ordering/` with datasets split to `dataset.py`
- `resource_allocation_optimization.py` → `power_allocation/` with `core.py` (822 LOC solver stays intact) + topical facades (`topology.py`, `throughput.py`, `optimizer.py`, `baselines.py`)
- Killed circular dep: `train.py` stopped importing from itself, switched to `attention_model.training_loop`
- Deleted `network/` (0 importers), `eval.py` (mostly commented), hardcoded data paths updated
- Training output: `./performance_percent/` → `input_data/output/`, checkpoints to `output/checkpoints/`

**Phase 4 — Slim entry points** (`4236e35`): Extracted training loop (182 LOC) to `attention_model/training_loop.py`. Entry scripts now minimal: `train.py` (185 LOC), `run.py` (16 LOC), `ASOPA_validation.py` (74 LOC). Removed toggle flags (SEE_SEE_DUIBI, JUST_VAL).

**Phase 5 — Documentation** (`6120fbd`): Per-directory READMEs (purpose, API, file map, example). Wrapper scripts in `commands/{train,train-baseline,validate,show,test}.sh` with CPU default. Updated root docs.

**Phase 6 — Verification**: 35/35 tests green on CPU. Validation avg_cost = -63.893619193190425 (val_size=1000) **bit-exact matches** GPU host. CPU val_size=100 reproduces golden to 10 decimal places.

## What We Tried

1. Full unit + golden depth (vs. smoke tests only) — **worked**. Caught RNG corruption in module-init.
2. Confidence-driven deletion of legacy dirs (network/, eval.py) after grep — **worked**. Test suite was the net.
3. Conflicting-extras pattern in pyproject.toml for CPU/GPU torch wheels — **worked cleanly**. Single source of truth.
4. State_dict + init_args decoupling from pickle import paths — **load-bearing**. Without it, nets/→attention_model/ rename breaks all checkpoints.

## Root Cause Analysis

The refactor was necessary because:
- Dual config files created parallelism bugs (params silently override each other)
- Duplicate module hierarchies forced teammates to search both paths
- Monolithic solver with zero tests meant any refactor risked numeric drift
- Hard-coded paths and full-pickle checkpoints made code fragile

The real mistake: no refactor plan during active research. Code accrued debt as features were added; we deferred structure cleanup until it became a blocker.

## Lessons Learned

1. **Checkpoint format matters**: Decouple saved weights from import paths immediately. State_dict + init_args sidecar costs nothing and survives renames.
2. **Test-first refactors win**: Writing behavior tests against unrefactored code (Phase 1) was the most valuable activity. It locked the numeric contract before touching anything.
3. **Module-scope side effects are invisible**: Legacy argparse in import statements cost 2 hours to debug. Linting would have caught it.
4. **Pragmatic facades > perfect splits**: Didn't inline the 822-LOC solver. Topical re-export facades provide the mental model immediately; inlining can wait (lower risk now).
5. **CPU/GPU validation must be explicit**: All dev was CPU-only. Bit-exact match with GPU required careful verification (greedy decode is deterministic, CVXOPT always CPU). Don't assume they'll match.

## Next Steps

1. **Archive legacy checkpoints properly**: Ensure output/checkpoints/legacy/ is gitignored but documented for recovery.
2. **Inline the power_allocation solver**: After this code stabilizes (3+ weeks), split `power_allocation/core.py` into logical submodules. Not urgent.
3. **Onboard next teammate**: Use per-directory READMEs as first read. Should take 1 week instead of 2.
4. **Pin test coverage**: Maintain 35+ tests as CI gate. Adding features means adding tests.
