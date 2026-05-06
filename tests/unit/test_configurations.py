"""Unit tests for the new configurations package: env_config, learning_config,
runtime_config, and the merging logic in merge.py.
"""

from __future__ import annotations

import pytest


def test_get_options_returns_flat_namespace():
    """`get_options([])` produces a flat Namespace with all expected fields."""
    from configurations import get_options

    opts = get_options([])

    # env
    assert opts.user_num == 10
    assert opts.val_user_num == 8
    assert opts.d_min == 20.0
    assert opts.d_max == 100.0
    assert opts.w_min == 1.0
    assert opts.w_max == 32.0
    assert opts.noise == pytest.approx(3.981e-15, rel=1e-6)
    assert opts.alpha == 1.0

    # learning
    assert opts.lr_model == 1e-4
    assert opts.batch_size == 64
    assert opts.embedding_dim == 128
    assert opts.baseline == "rollout"

    # runtime
    assert opts.seed == 1234
    assert opts.problem == "noop"
    assert opts.no_cuda is False  # not passed, default
    assert opts.log_dir == "logs"


def test_graph_size_alias_overrides_user_num():
    from configurations import get_options

    opts = get_options(["--graph_size=8"])
    assert opts.user_num == 8
    assert opts.graph_size == 8


def test_user_num_mirrors_to_graph_size_when_no_alias():
    from configurations import get_options

    opts = get_options(["--user_num=5"])
    assert opts.user_num == 5
    assert opts.graph_size == 5


def test_epoch_size_must_be_multiple_of_batch_size():
    from configurations import get_options

    with pytest.raises(AssertionError, match="epoch_size must be"):
        get_options(["--batch_size=64", "--epoch_size=100"])


def test_bl_warmup_defaults_to_one_for_rollout():
    from configurations import get_options

    opts = get_options(["--baseline=rollout"])
    assert opts.bl_warmup_epochs == 1


def test_bl_warmup_defaults_to_zero_for_exponential():
    from configurations import get_options

    opts = get_options(["--baseline=exponential"])
    assert opts.bl_warmup_epochs == 0


def test_load_path_and_resume_are_mutually_exclusive():
    from configurations import get_options

    with pytest.raises(AssertionError, match="at most one of --load_path"):
        get_options(["--load_path=foo", "--resume=bar"])


def test_get_default_env_config_includes_seed_alias():
    """get_default_env_config exposes seed for legacy module-init code."""
    from configurations import get_default_env_config

    env = get_default_env_config()
    assert env.user_num == 10
    assert env.val_user_num == 8
    assert env.seed == 1234  # pulled from runtime defaults for back-compat
    assert env.graph_size == env.user_num
