# ASOPA Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Architecture Guide](#architecture-guide)
- [API Reference](#api-reference)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Project Overview

**ASOPA (Attention-Based SIC Ordering and Power Allocation)** is a deep reinforcement learning framework for optimizing Non-Orthogonal Multiple Access (NOMA) networks. The system addresses the joint optimization of Successive Interference Cancellation (SIC) ordering and power allocation to maximize weighted proportional fairness.

### Key Features
- **Attention-based neural network** for SIC ordering optimization
- **Convex optimization** for power allocation
- **Reinforcement learning** training with policy gradient methods
- **Comprehensive baseline comparisons** (exhaustive search, heuristic methods)
- **Modular architecture** for easy extension and modification

### Paper Reference
```
@ARTICLE{10700682,
  author={Huang, Liang and Zhu, Bincheng and Nan, Runkai and Chi, Kaikai and Wu, Yuan},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Attention-Based SIC Ordering and Power Allocation for Non-Orthogonal Multiple Access Networks}, 
  year={2025},
  volume={24},
  number={2},
  pages={939-955}
}
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.8.0+
- CUDA 12.9+ (recommended)
- CVXOPT for optimization

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ASOPA

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Running Validation
```bash
# Validate ASOPA performance
python ASOPA_validation.py

# Run baseline comparisons
python run_baseline.py
```

### Training from Scratch
```bash
# Train ASOPA model
python run.py --n_epochs 300 --graph_size 10
```

## Architecture Guide

### System Overview
```
Input Data → Attention Network → SIC Ordering → Power Optimization → Network Utility
     ↓              ↓                ↓              ↓                    ↓
[Channel gains,  Graph Attention   Decoding      Convex              Weighted
 User weights]   Encoder + Decoder  Order π      Optimization        Proportional
                                                                    Fairness
```

### Core Components

#### 1. Attention Model (`nets/attention_model.py`)
- **Purpose**: Determines optimal SIC decoding order
- **Architecture**: Graph Attention Network with sequential decoding
- **Input**: User features [power_limit, weight, channel_gain]
- **Output**: SIC ordering sequence

#### 2. Power Allocation (`resource_allocation_optimization.py`)
- **Purpose**: Solves convex optimization for power allocation
- **Methods**: CVXOPT solver with Successive Convex Approximation
- **Objective**: Maximize weighted proportional fairness

#### 3. Problem Definition (`problems/noop/`)
- **Purpose**: Defines NOMA optimization problem
- **Components**: Cost calculation, state management, dataset generation
- **Metrics**: Network utility, execution time

#### 4. Training Pipeline (`train.py`, `run.py`)
- **Method**: Policy gradient with baseline network
- **Baseline**: Rollout baseline for variance reduction
- **Curriculum**: Variable user numbers (5-10 users)

## API Reference

### Core Classes

#### `AttentionModel`
```python
class AttentionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=2, ...)
    def forward(self, input, return_pi=False)
    def set_decode_type(self, decode_type, temp=None)
```

#### `NOOP` (Problem Class)
```python
class NOOP:
    @staticmethod
    def get_costs(dataset, pi)  # Calculate network utility
    @staticmethod
    def make_dataset(*args, **kwargs)  # Generate training data
    @staticmethod
    def load_val_dataset(*args, **kwargs)  # Load validation data
```

#### `User` (Network User)
```python
class User:
    def __init__(self, tid, tdecode_order=-1, tp_max=1., tg=0.5, tw=1, td=1)
    # Properties: id, decode_order, p_max, g_hat, w_hat, g, w, d
```

### Key Functions

#### Power Allocation
```python
def get_max_sum_weighted_alpha_throughput(users=[], alpha=1, noise=3.981e-15)
def get_optimal_p(users=[], alpha=1, noise=3.981e-15)
```

#### Baseline Methods
```python
def duibi_exhaustive_search(users=[], alpha=1, noise=3.981e-15)
def duibi_heuristic_method_qian(users=[], alpha=1, noise=3.981e-15)
def duibi_tabu_search_gd(users=[], alpha=1, noise=3.981e-15)
```

## Development Guide

### Project Structure
```
ASOPA/
├── nets/                    # Neural network implementations
│   ├── attention_model.py   # Core attention model
│   ├── graph_encoder.py     # Graph attention encoder
│   └── critic_network.py    # Baseline network
├── problems/                # Problem definitions
│   └── noop/               # NOMA optimization problem
├── utils/                   # Utility functions
├── Val/                     # Validation datasets
├── Top10/, Top15/          # Baseline results
├── run.py                   # Main training script
├── ASOPA_validation.py      # Validation script
└── conf.py                  # Configuration
```

### Adding New Features

#### 1. New Attention Mechanism
```python
# Create new file: nets/my_attention_model.py
class MyAttentionModel(nn.Module):
    def __init__(self, ...):
        # Your implementation
    def forward(self, input, return_pi=False):
        # Your forward pass
```

#### 2. New Baseline Method
```python
# Add to resource_allocation_optimization.py
def duibi_my_method(users=[], alpha=1, noise=3.981e-15):
    # Your baseline implementation
    return decode_order, max_throughput
```

#### 3. New Problem Variant
```python
# Create new file: problems/my_problem/problem_my.py
class MyProblem:
    NAME = 'my_problem'
    @staticmethod
    def get_costs(dataset, pi):
        # Your cost calculation
```

### Configuration Management

#### Training Parameters (`conf.py`)
```python
# Key parameters
parser.add_argument('--user_num', default=10)           # Number of users
parser.add_argument('--val_user_num', default=8)       # Validation users
parser.add_argument('--epoch_num', default=2)          # Training epochs
parser.add_argument('--lr', default=1e-3)             # Learning rate
parser.add_argument('--alpha', default=1)             # Fairness parameter
```

#### Model Parameters (`options.py`)
```python
# Model architecture
parser.add_argument('--embedding_dim', default=128)    # Embedding dimension
parser.add_argument('--hidden_dim', default=128)      # Hidden layer size
parser.add_argument('--n_encode_layers', default=3)   # Encoder layers
parser.add_argument('--n_heads', default=8)            # Attention heads
```

### Testing and Validation

#### Unit Tests
```python
# Example test structure
def test_attention_model():
    model = AttentionModel(...)
    input_data = generate_test_data()
    output = model(input_data)
    assert output.shape == expected_shape

def test_power_allocation():
    users = generate_test_users()
    optimal_p = get_optimal_p(users)
    assert all(p >= 0 for p in optimal_p)
```

#### Integration Tests
```python
# Test complete pipeline
def test_training_pipeline():
    # Test data generation
    # Test model training
    # Test validation
    # Test baseline comparison
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
python run.py --no_cuda
```

#### 2. Dependency Conflicts
```bash
# Create clean environment
conda create -n asopa python=3.8
conda activate asopa
pip install -r requirements.txt
```

#### 3. Memory Issues
```python
# Reduce batch size in options.py
parser.add_argument('--batch_size', default=32)  # Reduce from 64

# Enable gradient checkpointing
parser.add_argument('--checkpoint_encoder', action='store_true')
```

#### 4. Optimization Solver Issues
```python
# CVXOPT solver settings in resource_allocation_optimization.py
solvers.options['show_progress'] = False
solvers.options['refinement'] = 2
solvers.options['abstol'] = 1e-6
```

### Performance Optimization

#### Training Speed
- Use GPU acceleration
- Increase batch size (if memory allows)
- Enable gradient checkpointing
- Use mixed precision training

#### Memory Usage
- Reduce embedding dimensions
- Use smaller batch sizes
- Enable gradient accumulation
- Use CPU for validation

### Debugging Tips

#### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. Check Intermediate Results
```python
# In attention_model.py
print(f'Input shape: {input.shape}')
print(f'Embeddings shape: {embeddings.shape}')
print(f'Output order: {pi}')
```

#### 3. Validate Data Flow
```python
# Check data consistency
assert input.shape[1] == len(users)
assert all(w > 0 for w in weights)
assert all(g > 0 for g in channel_gains)
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** following the coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit pull request**

### Coding Standards

#### Python Style
- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and small

#### Code Organization
- Separate concerns (model, data, training)
- Use meaningful variable names
- Add comments for complex logic
- Maintain backward compatibility

### Documentation Standards

#### Function Documentation
```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
```

#### Class Documentation
```python
class MyClass:
    """
    Brief description of the class.
    
    Attributes:
        attr1: Description of attribute 1
        attr2: Description of attribute 2
    """
```

### Testing Requirements

- **Unit tests** for all new functions
- **Integration tests** for new features
- **Performance benchmarks** for optimizations
- **Documentation tests** for examples

### Release Process

1. **Update version numbers**
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Publish to PyPI** (if applicable)

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CVXOPT Documentation](https://cvxopt.org/)
- [Graph Attention Networks Paper](https://arxiv.org/abs/1710.10903)
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

For questions or support, please open an issue or contact the development team.