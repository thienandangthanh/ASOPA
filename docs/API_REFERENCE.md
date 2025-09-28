# ASOPA API Reference

## Table of Contents
- [Core Models](#core-models)
- [Problem Classes](#problem-classes)
- [Optimization Functions](#optimization-functions)
- [Utility Functions](#utility-functions)
- [Data Classes](#data-classes)
- [Configuration](#configuration)

## Core Models

### AttentionModel

The main neural network for SIC ordering optimization.

```python
class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=2, 
                 tanh_clipping=10., mask_inner=True, mask_logits=True, 
                 normalization='batch', n_heads=8, checkpoint_encoder=False, 
                 shrink_size=None):
```

**Parameters:**
- `embedding_dim` (int): Dimension of input embeddings
- `hidden_dim` (int): Dimension of hidden layers
- `problem`: Problem instance (NOOP)
- `n_encode_layers` (int): Number of encoder layers
- `tanh_clipping` (float): Clipping value for tanh activation
- `mask_inner` (bool): Whether to mask inner attention
- `mask_logits` (bool): Whether to mask output logits
- `normalization` (str): Normalization type ('batch' or 'instance')
- `n_heads` (int): Number of attention heads
- `checkpoint_encoder` (bool): Enable gradient checkpointing
- `shrink_size` (int): Batch shrinking threshold

**Methods:**

#### `forward(input, return_pi=False)`
Forward pass through the attention model.

**Parameters:**
- `input` (torch.Tensor): Input tensor of shape (batch_size, graph_size, node_dim)
- `return_pi` (bool): Whether to return the output sequences

**Returns:**
- `cost` (torch.Tensor): Negative network utility
- `log_likelihood` (torch.Tensor): Log probability of the sequence
- `pi` (torch.Tensor, optional): Output sequence if return_pi=True

**Example:**
```python
model = AttentionModel(embedding_dim=128, hidden_dim=128, problem=NOOP())
input_data = torch.randn(32, 10, 3)  # batch_size=32, users=10, features=3
cost, log_prob = model(input_data)
```

#### `set_decode_type(decode_type, temp=None)`
Set the decoding strategy.

**Parameters:**
- `decode_type` (str): 'greedy' or 'sampling'
- `temp` (float, optional): Temperature for sampling

**Example:**
```python
model.set_decode_type("greedy")  # Deterministic decoding
model.set_decode_type("sampling", temp=1.0)  # Stochastic decoding
```

### GraphAttentionEncoder

Graph attention encoder for processing user features.

```python
class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, normalization='batch'):
```

**Parameters:**
- `n_heads` (int): Number of attention heads
- `embed_dim` (int): Embedding dimension
- `n_layers` (int): Number of encoder layers
- `normalization` (str): Normalization type

**Methods:**

#### `forward(x, mask=None)`
Process input features through graph attention.

**Parameters:**
- `x` (torch.Tensor): Input features (batch_size, graph_size, embed_dim)
- `mask` (torch.Tensor, optional): Attention mask

**Returns:**
- `embeddings` (torch.Tensor): Node embeddings
- `graph_embed` (torch.Tensor): Graph-level embedding

## Problem Classes

### NOOP

Non-Orthogonal Multiple Access optimization problem.

```python
class NOOP:
    NAME = 'noop'
```

**Static Methods:**

#### `get_costs(dataset, pi)`
Calculate network utility for given SIC ordering.

**Parameters:**
- `dataset` (torch.Tensor): Input data tensor
- `pi` (torch.Tensor): SIC ordering sequence

**Returns:**
- `cost` (torch.Tensor): Negative network utility
- `mask` (torch.Tensor): Validity mask (None for NOOP)

**Example:**
```python
dataset = torch.randn(32, 10, 3)  # batch_size=32, users=10, features=3
pi = torch.randint(0, 10, (32, 10))  # Random SIC ordering
cost, mask = NOOP.get_costs(dataset, pi)
```

#### `make_dataset(size, num_samples, seed=1234, filename=None, distribution=None)`
Generate training dataset.

**Parameters:**
- `size` (int): Number of users
- `num_samples` (int): Number of samples
- `seed` (int): Random seed
- `filename` (str, optional): Dataset file path
- `distribution` (str, optional): Data distribution

**Returns:**
- `dataset`: Dataset instance

#### `load_val_dataset(size, num_samples, filename=None, distribution=None)`
Load validation dataset.

**Parameters:**
- `size` (int): Number of users
- `num_samples` (int): Number of samples
- `filename` (str, optional): Dataset file path
- `distribution` (str, optional): Data distribution

**Returns:**
- `dataset`: Validation dataset

#### `make_state(input)`
Create initial state for the problem.

**Parameters:**
- `input` (torch.Tensor): Input data

**Returns:**
- `state`: StateNOOP instance

### StateNOOP

State representation for the NOOP problem.

```python
class StateNOOP(NamedTuple):
    g: torch.Tensor          # Input features
    ids: torch.Tensor        # Batch indices
    first_a: torch.Tensor    # First action
    prev_a: torch.Tensor     # Previous action
    visited_: torch.Tensor    # Visited nodes mask
    i: torch.Tensor          # Current step
```

**Properties:**
- `visited` (torch.Tensor): Visited nodes mask

**Static Methods:**

#### `initialize(g, visited_dtype=torch.uint8)`
Initialize state from input data.

**Parameters:**
- `g` (torch.Tensor): Input features
- `visited_dtype` (torch.dtype): Data type for visited mask

**Returns:**
- `state`: Initialized StateNOOP

**Methods:**

#### `update(selected)`
Update state with selected action.

**Parameters:**
- `selected` (torch.Tensor): Selected node indices

**Returns:**
- `state`: Updated StateNOOP

#### `all_finished()`
Check if all nodes have been visited.

**Returns:**
- `bool`: True if finished

#### `get_current_node()`
Get current node index.

**Returns:**
- `torch.Tensor`: Current node index

#### `get_mask()`
Get mask for valid actions.

**Returns:**
- `torch.Tensor`: Action validity mask

## Optimization Functions

### Power Allocation

#### `get_max_sum_weighted_alpha_throughput(users=[], alpha=1, noise=3.981e-15, use_nlopt=False)`
Calculate maximum weighted proportional fairness.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter (1 for proportional fairness)
- `noise` (float): Gaussian white noise power
- `use_nlopt` (bool): Whether to use NLOPT solver

**Returns:**
- `float`: Maximum weighted throughput

**Example:**
```python
users = [User(i, tp_max=1.0, tg=0.5, tw=1.0) for i in range(5)]
max_throughput = get_max_sum_weighted_alpha_throughput(users, alpha=1.0)
```

#### `get_optimal_p(users=[], alpha=1, noise=3.981e-15, use_nlopt=False)`
Solve optimal power allocation.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power
- `use_nlopt` (bool): Whether to use NLOPT solver

**Returns:**
- `list`: Optimal power allocation

**Example:**
```python
users = [User(i, tp_max=1.0, tg=0.5, tw=1.0) for i in range(5)]
optimal_power = get_optimal_p(users, alpha=1.0)
```

#### `get_objective_throughput(users=[], p=None, alpha=1, noise=3.981e-15, need_user_throughput_list=False)`
Calculate objective throughput for given power allocation.

**Parameters:**
- `users` (list): List of User objects
- `p` (list): Power allocation
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power
- `need_user_throughput_list` (bool): Whether to return per-user throughput

**Returns:**
- `float` or `list`: Objective throughput or per-user throughput list

### Baseline Methods

#### `duibi_exhaustive_search(users=[], alpha=1, noise=3.981e-15, need_throughput_his=False)`
Exhaustive search for optimal SIC ordering.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power
- `need_throughput_his` (bool): Whether to return throughput history

**Returns:**
- `tuple`: (optimal_order, max_throughput, top15_throughput, top15_orders)

#### `duibi_heuristic_method_qian(users=[], alpha=1, noise=3.981e-15, need_random=False)`
Qian's heuristic method for SIC ordering.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power
- `need_random` (bool): Whether to randomize user order

**Returns:**
- `tuple`: (decode_order, max_throughput)

#### `duibi_tabu_search_gd(users=[], alpha=1, noise=3.981e-15, need_random=False)`
Tabu search with greedy descent initialization.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power
- `need_random` (bool): Whether to randomize user order

**Returns:**
- `tuple`: (decode_order, max_throughput)

#### `duibi_g_order_asc(users=[], alpha=1, noise=3.981e-15)`
Channel gain ascending order baseline.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power

**Returns:**
- `tuple`: (decode_order, max_throughput)

#### `duibi_g_order_desc(users=[], alpha=1, noise=3.981e-15)`
Channel gain descending order baseline.

**Parameters:**
- `users` (list): List of User objects
- `alpha` (float): Fairness parameter
- `noise` (float): Gaussian white noise power

**Returns:**
- `tuple`: (decode_order, max_throughput)

## Data Classes

### User

Represents a network user in the NOMA system.

```python
class User:
    def __init__(self, tid, tdecode_order=-1, tp_max=1., tg=0.5, tw=1, td=1):
```

**Parameters:**
- `tid` (int): User ID
- `tdecode_order` (int): SIC decoding order
- `tp_max` (float): Maximum transmit power
- `tg` (float): Channel gain
- `tw` (float): Throughput weight
- `td` (float): Distance to base station

**Attributes:**
- `id` (int): User ID
- `decode_order` (int): SIC decoding order
- `p_max` (float): Maximum transmit power
- `g_hat` (float): Average channel gain
- `w_hat` (float): Average throughput weight
- `g` (float): Current channel gain
- `w` (float): Current throughput weight
- `d` (float): Distance to base station

**Example:**
```python
user = User(tid=0, tp_max=1.0, tg=0.5, tw=2.0, td=50.0)
print(f"User {user.id}: power_limit={user.p_max}, weight={user.w}")
```

### Dataset Classes

#### NOOPDataset

Training dataset for NOOP problem.

```python
class NOOPDataset(Dataset):
    def __init__(self, num_samples=1000, seed=1234, size=10, filename=None, distribution=None):
```

**Parameters:**
- `num_samples` (int): Number of samples
- `seed` (int): Random seed
- `size` (int): Number of users
- `filename` (str, optional): Dataset file path
- `distribution` (str, optional): Data distribution

**Methods:**
- `__len__()`: Return dataset size
- `__getitem__(idx)`: Return sample at index

#### NOOPValDataset

Validation dataset for NOOP problem.

```python
class NOOPValDataset(Dataset):
    def __init__(self, num_samples=1000, seed=1234, size=10, filename=None, distribution=None):
```

**Parameters:**
- `num_samples` (int): Number of samples
- `seed` (int): Random seed
- `size` (int): Number of users
- `filename` (str, optional): Dataset file path
- `distribution` (str, optional): Data distribution

## Utility Functions

### Topology Generation

#### `generate_topology(user_number=10, d_min=20, d_max=100, w_min=1, w_max=32, max_num=10)`
Generate user topology for simulation.

**Parameters:**
- `user_number` (int): Number of users
- `d_min` (float): Minimum distance to base station
- `d_max` (float): Maximum distance to base station
- `w_min` (float): Minimum throughput weight
- `w_max` (float): Maximum throughput weight
- `max_num` (int): Maximum number of users (for padding)

**Returns:**
- `list`: List of User objects

#### `generate_val_topology(user_number=8, d_min=20, d_max=100, w_min=1, w_max=32, max_num=10)`
Generate validation topology.

**Parameters:**
- `user_number` (int): Number of users
- `d_min` (float): Minimum distance to base station
- `d_max` (float): Maximum distance to base station
- `w_min` (float): Minimum throughput weight
- `w_max` (float): Maximum throughput weight
- `max_num` (int): Maximum number of users (for padding)

**Returns:**
- `list`: List of User objects

### User Management

#### `sort_by_decode_order(users=[], decode_order=None, need_order=False)`
Sort users by SIC decoding order.

**Parameters:**
- `users` (list): List of User objects
- `decode_order` (list, optional): Decoding order
- `need_order` (bool): Whether to return the order

**Returns:**
- `list` or `tuple`: Sorted users or (sorted_users, decode_order)

#### `get_users_g_hat(users)`
Extract average channel gains from users.

**Parameters:**
- `users` (list): List of User objects

**Returns:**
- `numpy.ndarray`: Average channel gains

#### `get_users_w_hat(users)`
Extract average throughput weights from users.

**Parameters:**
- `users` (list): List of User objects

**Returns:**
- `numpy.ndarray`: Average throughput weights

#### `set_users_g(users, g_values)`
Set channel gains for users.

**Parameters:**
- `users` (list): List of User objects
- `g_values` (list): Channel gain values

#### `set_users_w(users, w_values)`
Set throughput weights for users.

**Parameters:**
- `users` (list): List of User objects
- `w_values` (list): Throughput weight values

## Configuration

### Training Configuration (`conf.py`)

```python
# Key training parameters
parser.add_argument('--user_num', default=10)           # Number of users
parser.add_argument('--val_user_num', default=8)          # Validation users
parser.add_argument('--epoch_num', default=2)            # Training epochs
parser.add_argument('--lr', default=1e-3)                 # Learning rate
parser.add_argument('--alpha', default=1)                # Fairness parameter
parser.add_argument('--noise', default=3.981e-15)        # Gaussian noise
parser.add_argument('--seed', default=1234)              # Random seed
```

### Model Configuration (`options.py`)

```python
# Model architecture parameters
parser.add_argument('--embedding_dim', default=128)      # Embedding dimension
parser.add_argument('--hidden_dim', default=128)         # Hidden layer size
parser.add_argument('--n_encode_layers', default=3)       # Encoder layers
parser.add_argument('--n_heads', default=8)               # Attention heads
parser.add_argument('--batch_size', default=64)           # Batch size
parser.add_argument('--epoch_size', default=1280)         # Epoch size
```

### Environment Variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# PyTorch settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CVXOPT settings
export CVXOPT_BUILD_FFTW=1
```

---

## Error Handling

### Common Exceptions

#### `ValueError`
Raised when invalid parameters are provided.

```python
try:
    optimal_p = get_optimal_p(users, alpha=-1)  # Invalid alpha
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

#### `RuntimeError`
Raised when optimization fails.

```python
try:
    throughput = get_max_sum_weighted_alpha_throughput(users)
except RuntimeError as e:
    print(f"Optimization failed: {e}")
```

#### `torch.cuda.OutOfMemoryError`
Raised when GPU memory is insufficient.

```python
try:
    model = AttentionModel(...).cuda()
except torch.cuda.OutOfMemoryError:
    print("GPU memory insufficient, using CPU")
    model = AttentionModel(...).cpu()
```

### Debugging Tips

1. **Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check tensor shapes:**
```python
print(f"Input shape: {input.shape}")
print(f"Model output shape: {output.shape}")
```

3. **Validate data ranges:**
```python
assert input.min() >= 0, "Negative input values"
assert input.max() <= 1, "Input values too large"
```

4. **Monitor memory usage:**
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated()}")
print(f"GPU memory cached: {torch.cuda.memory_reserved()}")
```