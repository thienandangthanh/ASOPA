# `utils/`

Cross-cutting utilities — RNG seeding, problem registry, model loading,
beam search, REINFORCE baselines, and small tensor helpers.

Nothing in here should depend on `attention_model`, `sic_ordering`, or
`power_allocation`. (Sub-modules that need each other coordinate via
late imports inside functions, e.g. `RolloutBaseline.eval`.)

## Public API

```python
from utils         import load_problem, torch_load_cpu, move_to, load_model, sample_many
from utils         import seed_everything                       # re-exported from utils.seeding
from utils.seeding import seed_everything
from utils.reinforce_baselines import (
    NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline,
)
```

## Files

| File                    | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `__init__.py`           | Re-exports `functions.*` and `seed_everything`                   |
| `seeding.py`            | `seed_everything(seed)` — resets python/numpy/torch RNG          |
| `functions.py`          | `load_problem`, `torch_load_cpu`, `move_to`, `load_model`, `load_args`, `sample_many`, `do_batch_rep`, `run_all_in_pool`, `parse_softmax_temperature` |
| `reinforce_baselines.py`| `Baseline`, `NoBaseline`, `ExponentialBaseline`, `RolloutBaseline`, `WarmupBaseline`, `BaselineDataset` |
| `beam_search.py`        | `beam_search`, `CachedLookup`, `segment_topk_idx`, `backtrack`   |
| `boolmask.py`           | `mask_long2bool`, `mask_long_scatter` — long↔bool mask conversions |
| `lexsort.py`            | Lexicographic sort helpers                                       |
| `tensor_functions.py`   | `compute_in_batches` and friends                                 |
| `log_utils.py`          | `log_values` — TensorBoard helpers                               |

## Adding a new problem

1. Create your problem class in a new package (e.g. `my_problem/`).
2. Register it in `utils/functions.py:load_problem`:
   ```python
   def load_problem(name):
       from sic_ordering import NOOP
       from my_problem  import MYPROB
       problem = {"noop": NOOP, "myprob": MYPROB}.get(name)
       assert problem is not None, "Unsupported problem: {!r}".format(name)
       return problem
   ```
3. Pass `--problem myprob` on the CLI.
