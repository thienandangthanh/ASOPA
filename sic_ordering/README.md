# `sic_ordering/`

Defines the SIC-ordering optimization problem (NOOP) plus the PyTorch
Datasets that feed it.

The attention model learns a *decode order* over users; this module
turns that order into a numeric reward (negative weighted-α throughput)
by handing the resulting power-allocation sub-problem to
`power_allocation`.

## Public API

```python
from sic_ordering import (
    NOOP,                  # problem class — has get_costs, make_state, etc.
    NOOPDataset,
    NOOPValDataset,
    NOOP_allnum_Dataset,
    StateNOOP,
    noop_users,            # canonical training topology (built at import)
    val_noop_users,        # canonical validation topology
)
```

## Files

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `problem_noop.py` | `NOOP` class with `get_costs(input, pi)`, `make_state`, etc.   |
| `state_noop.py`   | `StateNOOP` NamedTuple — visited-mask + step-counter           |
| `dataset.py`      | `NOOPDataset` (train), `NOOPValDataset` (val .mat), `NOOP_allnum_Dataset` (variable-user training) |

## Why `NOOPValDataset` reads `input_data/dependencies/val/*.mat`

The validation set is *frozen across runs* so that comparison numbers in
the paper stay reproducible. The .mat files in
`input_data/dependencies/val/n{N}_valdataset.mat` contain pre-sampled
channel gains and weights, indexed by user count `N`.

## Example: cost a fixed decode order on one batch

```python
import torch
from sic_ordering import NOOP

ds = NOOP.load_val_dataset(size=8, num_samples=2)
batch = torch.stack([ds[0], ds[1]])         # (2, 8, 3)
pi    = torch.tensor([[0,1,2,3,4,5,6,7],
                       [7,6,5,4,3,2,1,0]])    # decode orders to test
cost, _mask = NOOP.get_costs(batch, pi)      # (2,) — negative utilities
```
