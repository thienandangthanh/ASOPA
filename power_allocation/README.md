# `power_allocation/`

Convex optimization, throughput evaluation, and non-RL ordering baselines.

Given a fixed *decode order* over users, this module solves the
power-allocation sub-problem and reports the resulting weighted-α
network throughput. The attention model uses these functions to compute
its REINFORCE reward; the `duibi_*` baselines provide non-learning
comparison methods.

## Public API

```python
from power_allocation import (
    User,
    generate_topology,
    generate_val_topology,
    set_users_g, set_users_w,
    get_users_g_hat, get_users_w_hat,
    sort_by_decode_order,
    get_max_sum_weighted_alpha_throughput,
    get_objective_throughput,
    get_optimal_p,
    duibi_g_order_asc,
    duibi_g_order_desc,
    duibi_w_order_desc,
    duibi_w_order_aesc,
    duibi_heuristic_method_qian,
    duibi_tabu_search_gd,
    duibi_tabu_search_wd,
    duibi_exhaustive_search,
    duibi_random,
)
```

## Files

| File              | Purpose                                                                     |
|-------------------|-----------------------------------------------------------------------------|
| `core.py`         | Canonical implementation (~820 LOC) — User, generators, all `duibi_*`, optimizers |
| `topology.py`     | User topology + helpers (`set_users_g/w`, `get_users_g_hat/w_hat`, `random_set_users_g`) |
| `throughput.py`   | `sort_by_decode_order`, `get_max_sum_weighted_alpha_throughput`, `get_objective_throughput` |
| `optimizer.py`    | `get_optimal_p` (CVXOPT for α=1, SCA for α<1, trivial for α>1)              |
| `baselines.py`    | All `duibi_*` non-RL baselines + `get_optimal_ranking_policy`               |

> **Implementation note**: `topology.py`, `throughput.py`, `optimizer.py`,
> `baselines.py` currently re-export from `core.py`. The split exists to
> give callers a topical import name (`from power_allocation.baselines
> import duibi_exhaustive_search`) without forcing an immediate inline
> code split. A future cleanup may inline.

## α — fairness exponent

| α range  | Solver                              | Meaning                          |
|----------|-------------------------------------|----------------------------------|
| α = 1    | CVXOPT interior point (closed form) | Proportional fairness (default)  |
| α ∈ [0,1)| Successive Convex Approximation     | Closer to throughput maximization|
| α > 1    | Trivial — all power to one user     | Highest fairness                 |

## Example: utility of a fixed order

```python
from power_allocation import (
    generate_topology, get_max_sum_weighted_alpha_throughput,
    sort_by_decode_order, set_users_g, set_users_w,
)
import numpy as np

users = generate_topology(5, 20, 100, 1, 32)
real  = [u for u in users if u.p_max > 0]
set_users_g(real, np.array([u.g_hat * 1.5 for u in real]))
set_users_w(real, np.array([u.w_hat for u in real]))
ordered = sort_by_decode_order(real, [0, 1, 2, 3, 4])
print(get_max_sum_weighted_alpha_throughput(users=ordered))
```
