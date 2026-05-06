# ASOPA — Attention-Based SIC Ordering and Power Allocation for NOMA

Open-source implementation of the deep-RL framework from
**Huang et al., "Attention-Based SIC Ordering and Power Allocation for
Non-Orthogonal Multiple Access Networks", IEEE TMC 2025**
([DOI: 10.1109/TMC.2024.3470828](https://doi.org/10.1109/TMC.2024.3470828)).

## Project layout

```
ASOPA/
├── attention_model/      # Neural network: attention encoder + pointer decoder + REINFORCE loop
├── sic_ordering/         # NOOP problem class + datasets feeding the model
├── power_allocation/     # Convex optimization, throughput evaluation, non-RL baselines
├── utils/                # Generic utilities: seeding, problem registry, REINFORCE baselines
├── configurations/       # Argparse modules: env / learning / runtime → flat opts namespace
├── input_data/           # External-loaded data (Val/, Top10/, Top15/) + run outputs
├── output/               # Trained checkpoints (state_dict format)
├── logs/                 # TensorBoard event files
├── commands/             # Copy-paste-runnable shell wrappers
├── tests/                # Behavior-preservation unit + integration suite
├── docs/                 # ARCHITECTURE.md, API_REFERENCE.md, etc.
├── tensorboard_logger/   # Vendored TensorBoard logger
├── train.py              # Entry: parse opts → build → call training_loop.train_epoch loop
├── run.py                # Entry: thin alias of train.py
├── run_baseline.py       # Entry: dispatch val dataset through all duibi_* baselines
├── ASOPA_validation.py   # Entry: load checkpoint + run greedy validation
├── show.py               # Entry: visualize results
└── pyproject.toml        # uv-managed dependencies (no requirements.txt)
```

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

### Install (CPU only — VM / laptop without GPU)

```bash
uv sync --extra cpu
```

### Install (GPU — CUDA 12.1)

```bash
uv sync --extra cu121
```

### Train

```bash
./commands/train.sh                  # 10 users, 501 epochs (CPU)
uv run python train.py --user_num 8 --n_epochs 5  # short run
```

### Validate

```bash
./commands/validate.sh                                      # epoch 480 default
uv run python ASOPA_validation.py --val_epoch 300 --user_num 10
```

### Compare against non-RL baselines

```bash
./commands/train-baseline.sh
```

### Run tests

```bash
./commands/test.sh                 # default: skip slow + gpu marks
./commands/test.sh -m slow         # slow tests (full validation, 1-epoch training)
```

## Configuration

All CLI args live in `configurations/`, split by what they configure:

- **`env_config.py`** — NOMA domain (`--user_num`, `--d_min/max`, `--w_min/max`, `--noise`, `--alpha`, `--num_min/max`, `--val_user_num`)
- **`learning_config.py`** — RL/NN (`--lr_model`, `--batch_size`, `--n_epochs`, `--embedding_dim`, `--hidden_dim`, `--n_encode_layers`, `--baseline`, `--exp_beta`, …)
- **`runtime_config.py`** — Execution & I/O (`--seed`, `--no_cuda`, `--log_dir`, `--output_dir`, `--load_path`, `--resume`, `--no_tensorboard`, …)

`get_options()` merges all three into a single flat `argparse.Namespace`,
so call sites continue to read `opts.user_num`, `opts.lr_model` directly.

## Configurable user count

`--user_num` (canonical) and `--graph_size` (legacy alias) are mutually
mirrored; pass either. Likewise `--val_user_num` ↔ `--val_graph_size`.

## Cite

```bibtex
@ARTICLE{10700682,
  author={Huang, Liang and Zhu, Bincheng and Nan, Runkai and Chi, Kaikai and Wu, Yuan},
  journal={IEEE Transactions on Mobile Computing},
  title={Attention-Based SIC Ordering and Power Allocation for Non-Orthogonal Multiple Access Networks},
  year={2025},
  volume={24}, number={2}, pages={939-955},
  doi={10.1109/TMC.2024.3470828}
}
```

## Authors

- [Liang Huang](https://scholar.google.com/citations?user=NifLoZ4AAAAJ) — lianghuang@zjut.edu.cn
- [Bincheng Zhu](https://ieeexplore.ieee.org/author/37089420307) — bczhu@zjut.edu.cn
- [Runkai Nan](https://ieeexplore.ieee.org/author/37089596991) — rknan@zjut.edu.cn
- [Kaikai Chi](https://scholar.google.com/citations?user=MrdiGtMAAAAJ&hl=en&oi=ao) — kkchi@zjut.edu.cn
- [Yuan Wu](https://scholar.google.com/citations?hl=en&user=H1bxY_4AAAAJ) — yuanwu@um.edu.mo
