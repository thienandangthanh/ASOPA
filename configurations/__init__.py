"""Single source of truth for all CLI options.

Three argparse modules — env (NOMA domain), learning (RL/NN hyperparams),
runtime (logging, paths, seed, CUDA) — are merged into a flat Namespace by
`get_options()`. Existing call shape is preserved: code can still access
`opts.user_num`, `opts.lr_model`, `opts.no_cuda` etc. without any nesting.

Usage:
    from configurations import get_options
    opts = get_options()                  # parses sys.argv
    opts = get_options(['--user_num=8'])  # parses an explicit list (used in tests)

For modules that need the env defaults at import time (e.g. legacy
problem_noop.py side effects), call `get_default_env_config()` — it returns a
Namespace populated with the defaults from `env_config.py`, no CLI parsing.
"""

from configurations.merge import get_default_env_config, get_options

__all__ = ["get_options", "get_default_env_config"]
