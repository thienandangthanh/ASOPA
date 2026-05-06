"""User topology — `User` data class, topology generators, and the
small helpers that mutate or read attributes off a User list.

The `User`, `generate_topology`, `generate_val_topology` symbols are
re-exported from `power_allocation.core` (canonical implementation).
The mutation helpers (`set_users_g/w`, `get_users_g_hat/w_hat`,
`get_usrs_g`, `random_set_users_g`) live here directly since they were
previously in the now-deleted `my_utils.py`.
"""

from __future__ import annotations

import numpy as np

from power_allocation.core import (  # noqa: F401
    User,
    generate_topology,
    generate_val_topology,
)


def set_users_g(users, tg_list) -> None:
    """Set per-user instantaneous channel gain `g` from a list of values."""
    assert len(users) == len(tg_list), "channel gain count must match user count"
    for user, tg in zip(users, tg_list):
        user.g = tg


def set_users_w(users, tw_list) -> None:
    """Set per-user throughput weight `w` from a list of values."""
    assert len(users) == len(tw_list), "weight count must match user count"
    for user, tw in zip(users, tw_list):
        user.w = tw


def get_usrs_g(users) -> np.ndarray:
    """Return current `g` values as an array."""
    return np.asarray([u.g for u in users])


def get_users_g_hat(users) -> np.ndarray:
    """Return per-user mean uplink channel gain (`g_hat`) as an array."""
    return np.asarray([u.g_hat for u in users])


def get_users_w_hat(users) -> np.ndarray:
    """Return per-user weight estimate (`w_hat`) as an array."""
    return np.asarray([u.w_hat for u in users])


def random_set_users_g(users) -> None:
    """Randomize each user's instantaneous channel gain via Rayleigh × g_hat."""
    g_hat = get_users_g_hat(users)
    g = np.random.rayleigh(1, size=[len(users)]) * g_hat
    set_users_g(users, g)


__all__ = [
    "User",
    "generate_topology",
    "generate_val_topology",
    "set_users_g",
    "set_users_w",
    "get_usrs_g",
    "get_users_g_hat",
    "get_users_w_hat",
    "random_set_users_g",
]
