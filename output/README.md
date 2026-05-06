# `output/`

Persistent training outputs.

## Layout

```
output/
└── checkpoints/                                # state_dict .pth files
    ├── variable_user_n10_epoch300.pth          # converted from legacy full pickle
    ├── variable_user_n10_epoch480.pth          # converted from legacy full pickle
    └── legacy/                                 # original full-pickle checkpoints
        └── …                                   # (gitignored — local only)
```

## Checkpoint format

Each `.pth` is a `state_dict` payload created by `train.py:_save_checkpoint`:

```python
{
    "model_state_dict":  <state dict of inner model>,
    "model_init_args": {
        "embedding_dim":      128,
        "hidden_dim":         128,
        "n_heads":            8,
        "n_encode_layers":    3,
        "tanh_clipping":      10.0,
        "mask_inner":         True,
        "mask_logits":        True,
        "normalization":      "batch",
        "checkpoint_encoder": False,
        "shrink_size":        None,
    },
    "epoch": 480,
}
```

`init_args` survive the `nets/ → attention_model/` rename — the model
class is reconstructed at load time, then `load_state_dict` is called.
This is intentionally decoupled from import paths.

## Loading a checkpoint

```python
import torch
from attention_model.attention_model import AttentionModel
from utils import load_problem

ckpt = torch.load("output/checkpoints/variable_user_n10_epoch480.pth",
                  weights_only=False, map_location="cpu")
problem = load_problem("noop")
model = AttentionModel(
    ckpt["model_init_args"]["embedding_dim"],
    ckpt["model_init_args"]["hidden_dim"],
    problem,
    n_encode_layers=ckpt["model_init_args"]["n_encode_layers"],
    mask_inner=ckpt["model_init_args"]["mask_inner"],
    mask_logits=ckpt["model_init_args"]["mask_logits"],
    normalization=ckpt["model_init_args"]["normalization"],
    tanh_clipping=ckpt["model_init_args"]["tanh_clipping"],
    checkpoint_encoder=ckpt["model_init_args"].get("checkpoint_encoder", False),
    shrink_size=ckpt["model_init_args"].get("shrink_size"),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## `legacy/`

Original full-pickle checkpoints (saved via `torch.save(model, path)`)
were tied to the import path `nets.attention_model.AttentionModel`. They
are **archived locally only** (gitignored) so you can recover them if
something goes wrong with state_dict loading.
