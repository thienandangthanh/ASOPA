"""User topology — `User` data class plus topology generators.

Re-exports the canonical implementation from `power_allocation.core`. A
future cleanup pass may inline these here once the call shape stabilises.
"""

from power_allocation.core import (  # noqa: F401
    User,
    generate_topology,
    generate_val_topology,
)

__all__ = ["User", "generate_topology", "generate_val_topology"]
