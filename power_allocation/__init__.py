"""Power-allocation module — convex optimization, throughput evaluation, and
non-RL baselines for NOMA networks.

Logical sub-modules:

    topology     User class + generate_topology / generate_val_topology
    optimizer    get_optimal_p (CVXOPT-based) and α-specific solvers
    throughput   sort_by_decode_order, get_max_sum_weighted_alpha_throughput,
                 get_objective_throughput
    baselines    duibi_*  — non-RL ordering heuristics (g/w sort, tabu, exhaustive)

The canonical implementation currently lives in `power_allocation.core`;
the sub-modules are thin re-export facades so call sites can import from
the matching topic without coupling to file layout. A future cleanup may
physically split `core.py` along these boundaries.
"""

from power_allocation.baselines import (  # noqa: F401
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
from power_allocation.optimizer import get_optimal_p  # noqa: F401
from power_allocation.throughput import (  # noqa: F401
    get_max_sum_weighted_alpha_throughput,
    get_objective_throughput,
    sort_by_decode_order,
)
from power_allocation.topology import (  # noqa: F401
    User,
    generate_topology,
    generate_val_topology,
)

__all__ = [
    # topology
    "User",
    "generate_topology",
    "generate_val_topology",
    # throughput
    "sort_by_decode_order",
    "get_max_sum_weighted_alpha_throughput",
    "get_objective_throughput",
    # optimizer
    "get_optimal_p",
    # baselines
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
