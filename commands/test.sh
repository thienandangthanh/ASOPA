#!/usr/bin/env bash
# Run the behavior-preservation test suite.
#
# Default: skip @pytest.mark.gpu and @pytest.mark.slow tests.
# Pass arguments to switch:
#   ./commands/test.sh -m slow                  # run only slow tests
#   ./commands/test.sh -m "slow or not slow"    # everything except gpu
#   ./commands/test.sh -m gpu                   # GPU-only (NOT runnable in this VM)

set -euo pipefail

uv run pytest tests/ -v "$@"
