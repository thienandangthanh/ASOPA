"""Throughput evaluation — given a (decoded) ordering and channel state,
compute the network utility (weighted-α sum-throughput).

Re-exports the canonical implementation from `power_allocation.core`.
"""

from power_allocation.core import (  # noqa: F401
    get_max_sum_weighted_alpha_throughput,
    get_objective_throughput,
    sort_by_decode_order,
)

__all__ = [
    "sort_by_decode_order",
    "get_max_sum_weighted_alpha_throughput",
    "get_objective_throughput",
]
