"""Non-RL ordering baselines (the `duibi_*` family).

These are the comparison methods that the attention model is benchmarked
against: simple sorts (g/w ascending/descending), Qian's heuristic
insertion, tabu search variants, exhaustive permutation, and pure random.
All return a `(decode_order, max_sum_weighted_alpha_throughput)` tuple
unless they are slow ground-truth methods.

Re-exports the canonical implementation from `power_allocation.core`.
"""

from power_allocation.core import (  # noqa: F401
    duibi_exhaustive_search,
    duibi_g_order_asc,
    duibi_g_order_desc,
    duibi_heuristic_method_qian,
    duibi_heuristic_method_qian_random,
    duibi_random,
    duibi_tabu_search_gd,
    duibi_tabu_search_wd,
    duibi_w_order_aesc,
    duibi_w_order_desc,
    get_optimal_ranking_policy,
)

__all__ = [
    "duibi_g_order_asc",
    "duibi_g_order_desc",
    "duibi_w_order_desc",
    "duibi_w_order_aesc",
    "duibi_heuristic_method_qian",
    "duibi_heuristic_method_qian_random",
    "duibi_tabu_search_gd",
    "duibi_tabu_search_wd",
    "duibi_exhaustive_search",
    "duibi_random",
    "get_optimal_ranking_policy",
]
