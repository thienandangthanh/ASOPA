# AGENTS.md - AI Coding Agent Instructions

This document provides specific instructions and context for AI coding agents working on the ASOPA (Attention-Based SIC Ordering and Power Allocation) project.

## Documentation References

For comprehensive information, refer to:
- **[Documentation Index](docs/INDEX.md)** - Complete documentation overview
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and technical details
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Development Guide](docs/DEVELOPMENT_GUIDE.md)** - Development environment and workflow
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## Project Overview

**ASOPA** is a deep reinforcement learning framework for optimizing Non-Orthogonal Multiple Access (NOMA) networks. The system uses attention-based neural networks to determine optimal Successive Interference Cancellation (SIC) ordering and convex optimization for power allocation.

### Core Problem
Maximize weighted proportional fairness in uplink NOMA networks by jointly optimizing SIC ordering and power allocation.

### Key Technologies
- **PyTorch 2.3.1** for neural networks (pinned version for compatibility)
- **CVXOPT** for convex optimization
- **CUDA 12.1+** for GPU acceleration
- **Python 3.8+** as the primary language

## DevPod Environment Commands

This project uses **DevPod** for consistent development environments. AI agents should use the following commands:

### Starting the Development Environment
```bash
# Start DevPod environment with VS Code
devpod up --ide vscode .
```

### Remote Command Execution
AI agents should use this format for remote command execution:

```bash
# Basic command execution
ssh asopa.devpod 'source .venv/bin/activate && command_here'

# Check Python version and environment
ssh asopa.devpod 'source .venv/bin/activate && python --version && which python'

# Install new dependencies
ssh asopa.devpod 'source .venv/bin/activate && pip install package_name'

# Run training with specific parameters
ssh asopa.devpod 'source .venv/bin/activate && python run.py --n_epochs 10 --graph_size 5'

# Execute validation scripts
ssh asopa.devpod 'source .venv/bin/activate && python ASOPA_validation.py'

# Run tests
ssh asopa.devpod 'source .venv/bin/activate && python -m pytest tests/'

# Check GPU availability (if CUDA is configured)
ssh asopa.devpod 'source .venv/bin/activate && python -c "import torch; print(torch.cuda.is_available())"'
```

### Important Notes for AI Agents
1. **Virtual Environment**: Every new shell session requires activating the Python virtual environment with `source .venv/bin/activate`
2. **Remote Execution**: All Python commands must be executed within the DevPod environment
3. **Environment Isolation**: The DevPod environment ensures consistent dependencies and configurations across different development sessions

## Project Structure

```
ASOPA/
├── nets/                          # Neural network implementations
│   ├── attention_model.py         # Core attention model (MAIN MODEL)
│   ├── graph_encoder.py           # Graph attention encoder
│   ├── critic_network.py          # Baseline network
│   └── pointer_network.py          # Alternative model
├── problems/                      # Problem definitions
│   └── noop/                      # NOMA optimization problem
│       ├── problem_noop.py        # Problem definition & cost calculation
│       └── state_noop.py          # RL state management
├── resource_allocation_optimization.py  # Power allocation optimization
├── run.py                         # Main training entry point
├── ASOPA_validation.py            # Model validation
├── train.py                       # Training loop implementation
├── conf.py                        # Configuration parameters
├── options.py                     # Training options
└── docs/                          # Comprehensive documentation
```

## Key Files for AI Agents

### Critical Files (High Priority)
1. **`nets/attention_model.py`** - Core neural network architecture
2. **`resource_allocation_optimization.py`** - Power optimization solver
3. **`problems/noop/problem_noop.py`** - Problem definition and cost calculation
4. **`train.py`** - Training loop and policy gradient implementation
5. **`run.py`** - Main orchestration script

### Important Files (Medium Priority)
1. **`nets/graph_encoder.py`** - Graph attention mechanism
2. **`problems/noop/state_noop.py`** - RL state management
3. **`conf.py`** - Configuration parameters
4. **`options.py`** - Training options
5. **`ASOPA_validation.py`** - Validation and evaluation

### Supporting Files (Lower Priority)
1. **`utils/`** - Utility functions
2. **`nets/critic_network.py`** - Baseline network
3. **`nets/pointer_network.py`** - Alternative model
4. **`my_utils.py`** - Custom utilities

## Architecture Context

### Two-Stage Optimization Approach
```
Input Data → Attention Model → SIC Ordering → Power Allocation → Network Utility
     ↓              ↓                ↓              ↓                    ↓
[Channel gains,  Graph Attention   Decoding      Convex              Weighted
 User weights]   Encoder + Decoder  Order π      Optimization        Proportional
                                                                    Fairness
```

### Key Components
1. **AttentionModel**: Determines SIC decoding order using graph attention
2. **Power Optimizer**: Solves convex optimization for power allocation
3. **Problem Definition**: Defines NOMA optimization problem and cost calculation
4. **Training Pipeline**: Policy gradient training with baseline network

## Common Tasks for AI Agents

### 1. Adding New Attention Mechanisms
**Location**: `nets/attention_model.py` or create new file in `nets/`
**Pattern**:
```python
class MyAttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, problem, **kwargs):
        super().__init__()
        # Your implementation
    def forward(self, input, return_pi=False):
        # Your forward pass
    def set_decode_type(self, decode_type, temp=None):
        # Set decoding strategy
```

### 2. Adding New Optimization Methods
**Location**: `resource_allocation_optimization.py`
**Pattern**:
```python
def my_power_optimization(users, alpha=1.0, noise=3.981e-15):
    """Custom power optimization method."""
    # Your implementation
    return optimal_power

# Integrate with existing framework
def get_optimal_p(users, alpha=1.0, noise=3.981e-15, solver='cvxopt'):
    if solver == 'my_method':
        return my_power_optimization(users, alpha, noise)
    # ... existing solvers
```

### 3. Adding New Baseline Methods
**Location**: `resource_allocation_optimization.py`
**Pattern**:
```python
def duibi_my_baseline(users, alpha=1.0, noise=3.981e-15):
    """Custom baseline method."""
    # Your implementation
    decode_order = [0, 1, 2, ...]  # Your ordering
    users_order = sort_by_decode_order(users, decode_order)
    max_throughput = get_max_sum_weighted_alpha_throughput(users_order, alpha, noise)
    return decode_order, max_throughput
```

### 4. Modifying Training Loop
**Location**: `train.py`
**Key functions**:
- `train_epoch()`: Main training loop
- `train_batch()`: Single batch training
- `validate()`: Model validation

### 5. Adding New Problem Variants
**Location**: `problems/my_problem/`
**Pattern**:
```python
class MyProblem:
    NAME = 'my_problem'
    @staticmethod
    def get_costs(dataset, pi):
        # Your cost calculation
    @staticmethod
    def make_dataset(size, num_samples, **kwargs):
        # Your dataset generation
```

## Configuration Management

### Key Parameters
```python
# conf.py - Problem parameters
parser.add_argument('--user_num', default=10)           # Number of users
parser.add_argument('--val_user_num', default=8)       # Validation users
parser.add_argument('--alpha', default=1)             # Fairness parameter
parser.add_argument('--noise', default=3.981e-15)     # Gaussian noise

# options.py - Model parameters
parser.add_argument('--embedding_dim', default=128)    # Embedding dimension
parser.add_argument('--hidden_dim', default=128)       # Hidden layer size
parser.add_argument('--n_heads', default=8)            # Attention heads
parser.add_argument('--batch_size', default=64)        # Batch size
```

### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0          # GPU selection
OMP_NUM_THREADS=4               # CPU threads
ASOPA_DEBUG=1                   # Debug mode
```

## Testing Guidelines

### Unit Tests
```python
def test_attention_model():
    model = AttentionModel(embedding_dim=64, hidden_dim=64, problem=NOOP())
    input_data = torch.randn(4, 5, 3)  # batch_size=4, users=5, features=3
    cost, log_prob = model(input_data)
    assert cost.shape == (4,)
    assert log_prob.shape == (4,)
    assert torch.isfinite(cost).all()
```

### Integration Tests
```python
def test_training_pipeline():
    opts = get_options(['--n_epochs', '2', '--graph_size', '5', '--no_cuda'])
    run(opts)  # Should not raise exceptions
```

## Debugging Tips

### Common Issues
1. **NaN/Inf values**: Check input data and model outputs
2. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
3. **Optimization failures**: Check problem formulation and constraints
4. **Training instability**: Reduce learning rate or add gradient clipping

### Debugging Tools
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check tensor values
def check_tensor_values(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# Monitor gradients
def monitor_gradients(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    print(f"Total gradient norm: {total_norm ** 0.5:.4f}")
```

## Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large models
- Enable batch shrinking for variable user numbers
- Use mixed precision training when possible

### Computational Optimization
- Vectorize operations where possible
- Cache expensive computations
- Use parallel processing for independent operations

## Extension Points

### Research Directions
1. **New attention mechanisms** for better SIC ordering
2. **Advanced optimization** methods for power allocation
3. **Multi-objective** optimization frameworks
4. **Distributed training** for large-scale problems
5. **Real-time adaptation** for dynamic networks

### Implementation Areas
- **Neural architectures** in `nets/`
- **Optimization methods** in `resource_allocation_optimization.py`
- **Problem variants** in `problems/`
- **Training strategies** in `train.py`
- **Baseline methods** for comparison

## Common Commands

### Training
```bash
# Basic training
ssh asopa.devpod 'source .venv/bin/activate && python run.py --n_epochs 300 --graph_size 10'

# Training with specific configuration
ssh asopa.devpod 'source .venv/bin/activate && python run.py --model attention --embedding_dim 128 --hidden_dim 128'

# CPU-only training
ssh asopa.devpod 'source .venv/bin/activate && python run.py --no_cuda'
```

### Validation
```bash
# Validate pre-trained model
ssh asopa.devpod 'source .venv/bin/activate && python ASOPA_validation.py'

# Run baseline comparisons
ssh asopa.devpod 'source .venv/bin/activate && python run_baseline.py'
```

### Development
```bash
# Run tests
ssh asopa.devpod 'source .venv/bin/activate && python -m pytest tests/'

# Check code style
ssh asopa.devpod 'source .venv/bin/activate && black --check .'
ssh asopa.devpod 'source .venv/bin/activate && flake8 .'

# Type checking
ssh asopa.devpod 'source .venv/bin/activate && mypy nets/ problems/'
```

## File Dependencies

### Critical Dependencies
- `attention_model.py` depends on `graph_encoder.py`
- `problem_noop.py` depends on `resource_allocation_optimization.py`
- `train.py` depends on `attention_model.py` and `problem_noop.py`
- `run.py` orchestrates all components

### Import Patterns
```python
# Standard imports
import torch
import torch.nn as nn
import numpy as np

# Project imports
from nets.attention_model import AttentionModel
from problems.noop import NOOP
from resource_allocation_optimization import get_optimal_p
```

## Error Patterns to Avoid

### Common Mistakes
1. **Hardcoding values** instead of using configuration
2. **Ignoring device placement** (CPU vs GPU)
3. **Not validating inputs** before processing
4. **Missing error handling** for optimization failures
5. **Inconsistent tensor shapes** across batches

### Best Practices
1. **Always validate inputs** and outputs
2. **Use configuration parameters** instead of hardcoded values
3. **Handle edge cases** gracefully
4. **Add comprehensive tests** for new functionality
5. **Update documentation** when adding features

## Agent-Specific Instructions

### For Code Generation Agents
- **Always include type hints** and docstrings
- **Follow the established patterns** in existing code
- **Add error handling** for edge cases
- **Include tests** for new functionality
- **Update configuration** when adding parameters

### For Debugging Agents
- **Check tensor shapes and devices** first
- **Validate input data** for NaN/Inf values
- **Monitor gradient flow** during training
- **Use debugging tools** provided in the codebase
- **Check configuration parameters** for correctness

### For Optimization Agents
- **Profile performance** before and after changes
- **Consider memory usage** when optimizing
- **Maintain backward compatibility** when possible
- **Test on different hardware** configurations
- **Document performance improvements**

### For Testing Agents
- **Write comprehensive tests** for all new code
- **Test edge cases** and error conditions
- **Verify integration** with existing components
- **Check performance** doesn't regress
- **Update test documentation** as needed

---

## Quick Reference

### Key Classes
- `AttentionModel`: Main neural network
- `NOOP`: Problem definition
- `StateNOOP`: RL state management
- `User`: Network user representation

### Key Functions
- `get_optimal_p()`: Power allocation optimization
- `get_max_sum_weighted_alpha_throughput()`: Calculate network utility
- `duibi_*()`: Baseline comparison methods

### Key Files
- `nets/attention_model.py`: Core model
- `resource_allocation_optimization.py`: Optimization
- `problems/noop/problem_noop.py`: Problem definition
- `train.py`: Training implementation

---

This document should provide AI coding agents with sufficient context to work effectively on the ASOPA project. For additional details, refer to the comprehensive documentation in the `docs/` directory.