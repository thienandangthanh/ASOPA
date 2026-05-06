# `attention_model/`

The neural network that learns SIC ordering — multi-head graph attention
encoder + sequential pointer decoder, trained with REINFORCE.

## Public API

```python
from attention_model.attention_model import AttentionModel, set_decode_type
from attention_model.graph_encoder  import GraphAttentionEncoder, MultiHeadAttention
from attention_model.training_loop  import train_epoch, validate, rollout
```

## Files

| File                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `attention_model.py`   | `AttentionModel` (nn.Module) — encoder + sequential decoder + REINFORCE forward |
| `graph_encoder.py`     | `GraphAttentionEncoder`, `MultiHeadAttention`, `SkipConnection`, `Normalization` |
| `pointer_network.py`   | `PointerNetwork`, `CriticNetworkLSTM` — alternative seq2seq baseline    |
| `critic_network.py`    | Minimal critic head used by some baseline modes                         |
| `training_loop.py`     | `train_epoch`, `validate`, `rollout`, `train_batch`, `clip_grad_norms`  |

## Cost flow

```
input (batch, n_users, 3)        # [p_max, w, g*1e9]
   │
   ▼  init_embed: Linear(3→128)
graph_encoder
   │
   ▼  multi-head attention × n_encode_layers
node embeddings + graph embedding
   │
   ▼  pointer-style sequential decoder, masked
decode order π
   │
   ▼  problem.get_costs(input, π)  →  -weighted-α throughput
cost   ←  REINFORCE loss = (cost − baseline) × log_likelihood
```

## Example: 1-batch greedy forward

```python
import torch
from attention_model.attention_model import AttentionModel, set_decode_type
from utils import load_problem

problem = load_problem("noop")
model = AttentionModel(128, 128, problem, n_encode_layers=3,
                       mask_inner=True, mask_logits=True,
                       normalization="batch", tanh_clipping=10.0)
set_decode_type(model, "greedy")
model.eval()

inp = problem.load_val_dataset(size=8, num_samples=2)[0:2]  # (batch, n, 3)
with torch.no_grad():
    cost, log_likelihood = model(inp)
```
