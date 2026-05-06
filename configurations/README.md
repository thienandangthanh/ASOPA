# `configurations/`

Single source of truth for all CLI options. Replaces the legacy `options.py`
and `conf.py` (deleted). Args are grouped by *what they configure*, not by
where they happened to live historically.

## Public API

```python
from configurations import get_options, get_default_env_config

opts = get_options()                    # parse sys.argv
opts = get_options(['--user_num=8'])    # explicit list (used in tests)

env = get_default_env_config()          # env defaults without parsing CLI
```

`opts` is a flat `argparse.Namespace`. Existing call sites that read
`opts.user_num`, `opts.lr_model`, `opts.no_cuda`, `opts.seed`, etc. work
unchanged.

## Files

- `env_config.py` — NOMA domain knobs (`user_num`, `d_min/max`, `w_min/max`,
  `noise`, `alpha`, `num_min/max`, `val_user_num`, `val_epoch`).
- `learning_config.py` — RL/NN hyperparameters (`lr_model`, `batch_size`,
  `epoch_size`, `n_epochs`, `embedding_dim`, `hidden_dim`, `n_encode_layers`,
  `tanh_clipping`, `normalization`, `baseline`, `exp_beta`, `bl_*`,
  `eval_batch_size`, `checkpoint_encoder`, `shrink_size`, `model`,
  `max_grad_norm`, `lr_decay`, `lr_critic`).
- `runtime_config.py` — execution + I/O (`seed`, `no_cuda`, `log_dir`,
  `output_dir`, `run_name`, `log_step`, `checkpoint_epochs`, `load_path`,
  `resume`, `no_tensorboard`, `no_progress_bar`, `eval_only`, `val_dataset`,
  `val_size`, `problem`, `data_distribution`, `epoch_start`).
- `merge.py` — builds a single `argparse` parser, registers all three
  groups, parses, and applies derivations + assertions:
  - `opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda`
  - `opts.run_name = "{run_name}_{YYYYMMDDTHHMMSS}"`
  - `opts.save_dir = "outputs/{problem}_{user_num}/{run_name}"`
  - `opts.bl_warmup_epochs` defaults to 1 for `rollout`, 0 otherwise
  - `opts.graph_size ↔ opts.user_num` (alias mirroring)
  - asserts `epoch_size % batch_size == 0`
  - asserts at most one of `--load_path` / `--resume`

## Why three modules?

Teammates were getting confused which config controls what. Now:

- Tweaking the **physical scenario** (more users, different noise floor) →
  edit `env_config.py`.
- Tweaking the **learning algorithm** (deeper net, bigger batch) →
  edit `learning_config.py`.
- Tweaking **how the run is observed** (output paths, tensorboard) →
  edit `runtime_config.py`.

Adding a new arg goes in the matching module and is automatically picked up
by `get_options()`.

## Back-compat aliases

- `--graph_size` ↔ `--user_num` (canonical: `user_num`)
- `--val_graph_size` ↔ `--val_user_num` (canonical: `val_user_num`)

The aliases mirror in both directions: passing either CLI flag updates both
attributes on the returned Namespace.
