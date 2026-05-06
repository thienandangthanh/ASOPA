"""Build and parse the unified options namespace.

`get_options()` constructs one argparse parser, registers env / learning /
runtime arg groups onto it, parses, and applies cross-cutting derivations
(device selection, run-name timestamp, save_dir construction, baseline
warmup defaulting, sanity assertions). Returns a flat `argparse.Namespace`
matching the legacy shape — `opts.user_num`, `opts.lr_model`, etc.
"""

from __future__ import annotations

import argparse
import os
import time
from argparse import Namespace
from typing import Sequence

import torch

from configurations.env_config import ENV_DEFAULTS, register_env_args
from configurations.learning_config import LEARNING_DEFAULTS, register_learning_args  # noqa: F401
from configurations.runtime_config import RUNTIME_DEFAULTS, register_runtime_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "ASOPA — Attention-based SIC ordering and power allocation for NOMA. "
            "Args are split into env / learning / runtime groups; pass --help "
            "to see them all."
        )
    )
    register_env_args(parser)
    register_learning_args(parser)
    register_runtime_args(parser)
    return parser


def _resolve_aliases(opts: Namespace) -> None:
    """Reconcile `--graph_size` ↔ `--user_num` and `--val_graph_size` ↔ `--val_user_num`.

    `user_num` is canonical. If the user passes `--graph_size`, copy it onto
    `user_num`; otherwise mirror `user_num` onto `graph_size` so legacy code
    that reads `opts.graph_size` still works.
    """
    if opts.graph_size is not None and opts.user_num != opts.graph_size:
        opts.user_num = opts.graph_size
    opts.graph_size = opts.user_num

    if opts.val_graph_size is not None and opts.val_user_num != opts.val_graph_size:
        opts.val_user_num = opts.val_graph_size
    opts.val_graph_size = opts.val_user_num


def _apply_derivations(opts: Namespace) -> None:
    """Compute fields that depend on combinations of parsed args."""
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.user_num),
        opts.run_name,
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == "rollout" else 0


def _validate(opts: Namespace) -> None:
    """Cross-field assertions preserved from legacy options.py."""
    assert (
        opts.bl_warmup_epochs == 0 or opts.baseline == "rollout"
    ), "bl_warmup_epochs > 0 requires baseline=='rollout'"
    assert (
        opts.epoch_size % opts.batch_size == 0
    ), "epoch_size must be an integer multiple of batch_size"
    assert (
        opts.load_path is None or opts.resume is None
    ), "Pass at most one of --load_path / --resume"


def get_options(args: Sequence[str] | None = None) -> Namespace:
    """Parse CLI args (or an explicit list) and return a flat options Namespace."""
    parser = _build_parser()
    opts = parser.parse_args(args)
    _resolve_aliases(opts)
    _apply_derivations(opts)
    _validate(opts)
    return opts


def get_default_env_config() -> Namespace:
    """Return env defaults without parsing CLI.

    Used by modules that need the env values at *import time* and cannot
    rely on `get_options()` (which would error on unknown CLI args, e.g.
    when imported from pytest). Defaults are the same as the parser would
    produce for the env group, plus `seed` from runtime defaults (legacy
    module-init code reads `_env.seed`).
    """
    merged = {}
    merged.update(ENV_DEFAULTS)
    # Convenience aliases mirroring _resolve_aliases() output.
    merged["graph_size"] = merged["user_num"]
    merged["val_graph_size"] = merged["val_user_num"]
    # Legacy module-init code (problem_noop.py) reads _env.seed.
    merged["seed"] = RUNTIME_DEFAULTS["seed"]
    return Namespace(**merged)


def get_default_options() -> Namespace:
    """Return *all* defaults (env + learning + runtime) without parsing CLI.

    Useful for tests and for modules that want a pre-populated Namespace.
    The result includes derived fields (use_cuda, run_name, save_dir,
    bl_warmup_epochs) so it behaves like the output of `get_options([])`.
    """
    parser = _build_parser()
    opts = parser.parse_args([])
    _resolve_aliases(opts)
    _apply_derivations(opts)
    _validate(opts)
    return opts
