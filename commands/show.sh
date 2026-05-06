#!/usr/bin/env bash
# Visualize SIC-ordering performance and channel-quality distributions.
# Opens matplotlib windows interactively; pipe through Xvfb if running headless.

set -euo pipefail

uv run python show.py "$@"
