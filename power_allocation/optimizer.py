"""Convex optimization for the power-allocation sub-problem.

Given a fixed decode order and channel/weight vectors, `get_optimal_p`
returns the per-user transmit powers that maximise weighted-α throughput.
For α=1 (proportional fair), the problem is convex and solved with CVXOPT;
for α∈[0,1) it uses Successive Convex Approximation (SCA); for α>1 a
trivial closed-form is used.

Re-exports the canonical implementation from `power_allocation.core`.
"""

from power_allocation.core import get_optimal_p  # noqa: F401

__all__ = ["get_optimal_p"]
