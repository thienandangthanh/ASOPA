"""SIC ordering policy + problem definition for NOMA.

The `NOOP` class is the public entry point — it implements the
two-stage cost (apply ordering, then convex-optimize power) consumed by
the attention model's REINFORCE loop. Datasets that feed the model live in
`sic_ordering.dataset`.
"""

from sic_ordering.dataset import (
    NOOPDataset,
    NOOPValDataset,
    NOOP_allnum_Dataset,
)
from sic_ordering.problem_noop import NOOP, noop_users, val_noop_users
from sic_ordering.state_noop import StateNOOP

__all__ = [
    "NOOP",
    "NOOPDataset",
    "NOOPValDataset",
    "NOOP_allnum_Dataset",
    "StateNOOP",
    "noop_users",
    "val_noop_users",
]
