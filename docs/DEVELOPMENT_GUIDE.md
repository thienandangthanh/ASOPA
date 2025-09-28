# ASOPA Development Guide

## Table of Contents
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Structure](#code-structure)
- [Adding New Features](#adding-new-features)
- [Testing](#testing)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.8.0 or higher
- CUDA 12.9+ (recommended for training)
- Git for version control

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ASOPA
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv asopa_env
   source asopa_env/bin/activate  # On Windows: asopa_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import cvxopt; print('CVXOPT: OK')"
   ```

### Development Tools

Install additional development tools:

```bash
pip install black flake8 pytest pytest-cov mypy
```

## Development Environment

### DevPod Environment Setup

This project uses **DevPod** for consistent development environments. Follow these steps to set up and use the DevPod environment:

#### Starting DevPod Environment

```bash
# Start DevPod environment with VS Code
devpod up --ide vscode .
```

#### Executing Commands in DevPod

After starting the DevPod environment, you can access it in two ways:

##### For Human Developers (Interactive Shell)
Connect to an interactive bash shell in the DevPod environment:

```bash
# Connect to interactive shell
ssh asopa.devpod

# Once connected, activate virtual environment
source .venv/bin/activate

# Now you can run commands directly
python --version
python run.py --help
pip list
python run.py --n_epochs 10 --graph_size 5
```

##### For AI Agents (Remote Command Execution)
AI agents should use the command template for remote execution:

```bash
# AI agents use this format for remote command execution
ssh asopa.devpod 'source .venv/bin/activate && python --version'
ssh asopa.devpod 'source .venv/bin/activate && python run.py --help'
ssh asopa.devpod 'source .venv/bin/activate && pip list'
```

#### Important Notes

1. **Virtual Environment**: Every new shell session requires activating the Python virtual environment with `source .venv/bin/activate`
2. **Remote Execution**: All Python commands must be executed within the DevPod environment
3. **Environment Isolation**: The DevPod environment ensures consistent dependencies and configurations across different development sessions
4. **Interactive vs Remote**: Human developers can use interactive SSH sessions, while AI agents use remote command execution

#### Common DevPod Commands

##### For Human Developers (Interactive Shell)
```bash
# Connect to DevPod
ssh asopa.devpod

# Activate virtual environment
source .venv/bin/activate

# Check Python version and environment
python --version && which python

# Install new dependencies
pip install package_name

# Run training with specific parameters
python run.py --n_epochs 10 --graph_size 5

# Execute validation scripts
python ASOPA_validation.py

# Run tests
python -m pytest tests/

# Check GPU availability (if CUDA is configured)
python -c "import torch; print(torch.cuda.is_available())"
```

##### For AI Agents (Remote Command Execution)
```bash
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

### IDE Configuration

#### VS Code Setup

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./asopa_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true
    }
}
```

#### PyCharm Setup

1. Open project in PyCharm
2. Configure Python interpreter to use virtual environment
3. Enable code inspection and formatting
4. Set up run configurations for training and validation

### Environment Variables

Create `.env` file:

```bash
# CUDA settings
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1

# PyTorch settings
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Development settings
ASOPA_DEBUG=1
ASOPA_LOG_LEVEL=DEBUG
```

### Git Configuration

Set up Git hooks for code quality:

```bash
# Pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
black --check .
flake8 .
pytest tests/ -v
EOF

chmod +x .git/hooks/pre-commit
```

## Code Structure

### Project Organization

```
ASOPA/
├── docs/                          # Documentation
│   ├── README.md
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── TROUBLESHOOTING.md
├── nets/                          # Neural network implementations
│   ├── __init__.py
│   ├── attention_model.py         # Core attention model
│   ├── graph_encoder.py           # Graph attention encoder
│   ├── critic_network.py          # Baseline network
│   └── pointer_network.py         # Alternative model
├── problems/                      # Problem definitions
│   ├── __init__.py
│   └── noop/                      # NOMA optimization problem
│       ├── __init__.py
│       ├── problem_noop.py        # Problem definition
│       └── state_noop.py          # State management
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── functions.py               # General utilities
│   ├── tensor_functions.py        # Tensor operations
│   ├── data_utils.py              # Data processing
│   └── log_utils.py               # Logging utilities
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_models.py             # Model tests
│   ├── test_optimization.py       # Optimization tests
│   └── test_data.py               # Data tests
├── scripts/                       # Utility scripts
│   ├── train.py                   # Training script
│   ├── validate.py                # Validation script
│   └── benchmark.py               # Benchmarking script
├── configs/                       # Configuration files
│   ├── default.yaml               # Default configuration
│   ├── training.yaml              # Training configuration
│   └── validation.yaml            # Validation configuration
├── run.py                         # Main entry point
├── conf.py                        # Configuration parser
├── options.py                     # Training options
└── requirements.txt               # Dependencies
```

### Coding Standards

#### Python Style Guide

Follow PEP 8 with these additional rules:

```python
# Use type hints
def calculate_throughput(users: List[User], power: List[float]) -> float:
    """Calculate network throughput for given power allocation.
    
    Args:
        users: List of User objects
        power: Power allocation vector
        
    Returns:
        Total network throughput
    """
    pass

# Use meaningful variable names
user_count = len(users)
max_power = max(user.p_max for user in users)

# Use constants for magic numbers
MAX_USERS = 20
DEFAULT_NOISE = 3.981e-15
```

#### Documentation Standards

```python
class AttentionModel(nn.Module):
    """Attention-based model for SIC ordering optimization.
    
    This model uses graph attention networks to determine optimal
    Successive Interference Cancellation (SIC) ordering for NOMA networks.
    
    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layers
        problem: Problem instance (NOOP)
        n_encode_layers: Number of encoder layers
        n_heads: Number of attention heads
        
    Example:
        >>> model = AttentionModel(embedding_dim=128, hidden_dim=128, problem=NOOP())
        >>> input_data = torch.randn(32, 10, 3)
        >>> cost, log_prob = model(input_data)
    """
    
    def forward(self, input: torch.Tensor, return_pi: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the attention model.
        
        Args:
            input: Input tensor of shape (batch_size, graph_size, node_dim)
            return_pi: Whether to return the output sequences
            
        Returns:
            Tuple containing:
                - cost: Negative network utility
                - log_likelihood: Log probability of the sequence
                - pi: Output sequence (if return_pi=True)
        """
        pass
```

## Adding New Features

### 1. New Attention Mechanism

#### Step 1: Create Model Class

```python
# nets/my_attention_model.py
import torch
import torch.nn as nn
from typing import Tuple

class MyAttentionModel(nn.Module):
    """Custom attention model implementation."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, problem, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.problem = problem
        
        # Your custom architecture
        self.encoder = MyCustomEncoder(embedding_dim, hidden_dim)
        self.decoder = MyCustomDecoder(embedding_dim, hidden_dim)
        
    def forward(self, input: torch.Tensor, return_pi: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass implementation."""
        # Your implementation
        pass
        
    def set_decode_type(self, decode_type: str, temp: float = None):
        """Set decoding strategy."""
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp
```

#### Step 2: Register Model

```python
# nets/__init__.py
from .attention_model import AttentionModel
from .my_attention_model import MyAttentionModel

MODEL_REGISTRY = {
    'attention': AttentionModel,
    'my_attention': MyAttentionModel,
    'pointer': PointerNetwork
}

def get_model(model_name: str, **kwargs):
    """Get model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)
```

#### Step 3: Add Configuration

```python
# options.py
parser.add_argument('--model', default='attention', 
                   choices=['attention', 'my_attention', 'pointer'],
                   help="Model architecture to use")
```

#### Step 4: Add Tests

```python
# tests/test_my_attention.py
import pytest
import torch
from nets.my_attention_model import MyAttentionModel
from problems.noop import NOOP

def test_my_attention_model():
    """Test custom attention model."""
    model = MyAttentionModel(embedding_dim=64, hidden_dim=64, problem=NOOP())
    input_data = torch.randn(4, 5, 3)  # batch_size=4, users=5, features=3
    
    cost, log_prob = model(input_data)
    
    assert cost.shape == (4,)
    assert log_prob.shape == (4,)
    assert torch.isfinite(cost).all()
    assert torch.isfinite(log_prob).all()
```

### 2. New Optimization Method

#### Step 1: Implement Solver

```python
# resource_allocation_optimization.py

def my_power_optimization(users: List[User], alpha: float = 1.0, 
                         noise: float = 3.981e-15) -> List[float]:
    """Custom power optimization method.
    
    Args:
        users: List of User objects
        alpha: Fairness parameter
        noise: Gaussian white noise power
        
    Returns:
        Optimal power allocation
    """
    # Your optimization implementation
    user_count = len(users)
    
    # Example: Simple equal power allocation
    if alpha == 1.0:
        # Proportional fairness
        optimal_power = [user.p_max for user in users]
    else:
        # Other fairness criteria
        optimal_power = [user.p_max / user_count for user in users]
    
    return optimal_power
```

#### Step 2: Integrate with Framework

```python
# resource_allocation_optimization.py

def get_optimal_p(users: List[User], alpha: float = 1.0, 
                 noise: float = 3.981e-15, solver: str = 'cvxopt') -> List[float]:
    """Get optimal power allocation using specified solver.
    
    Args:
        users: List of User objects
        alpha: Fairness parameter
        noise: Gaussian white noise power
        solver: Optimization solver ('cvxopt', 'my_method', 'nlopt')
        
    Returns:
        Optimal power allocation
    """
    if solver == 'my_method':
        return my_power_optimization(users, alpha, noise)
    elif solver == 'cvxopt':
        return _get_optimal_p_cvxopt(users, alpha, noise)
    else:
        raise ValueError(f"Unknown solver: {solver}")
```

#### Step 3: Add Configuration

```python
# conf.py
parser.add_argument('--power_solver', default='cvxopt',
                   choices=['cvxopt', 'my_method', 'nlopt'],
                   help="Power allocation solver")
```

### 3. New Baseline Method

#### Step 1: Implement Baseline

```python
# resource_allocation_optimization.py

def duibi_my_baseline(users: List[User], alpha: float = 1.0, 
                     noise: float = 3.981e-15) -> Tuple[List[int], float]:
    """Custom baseline method for SIC ordering.
    
    Args:
        users: List of User objects
        alpha: Fairness parameter
        noise: Gaussian white noise power
        
    Returns:
        Tuple of (decode_order, max_throughput)
    """
    # Your baseline implementation
    # Example: Random ordering
    import random
    
    decode_order = list(range(len(users)))
    random.shuffle(decode_order)
    
    users_order = sort_by_decode_order(users, decode_order)
    max_throughput = get_max_sum_weighted_alpha_throughput(users_order, alpha, noise)
    
    return decode_order, max_throughput
```

#### Step 2: Add to Comparison Suite

```python
# run_baseline.py

BASELINE_METHODS = {
    'exhaustive': duibi_exhaustive_search,
    'heuristic': duibi_heuristic_method_qian,
    'tabu_gd': duibi_tabu_search_gd,
    'tabu_wd': duibi_tabu_search_wd,
    'g_asc': duibi_g_order_asc,
    'g_desc': duibi_g_order_desc,
    'w_asc': duibi_w_order_aesc,
    'w_desc': duibi_w_order_desc,
    'random': duibi_random,
    'my_baseline': duibi_my_baseline
}

def run_baseline_comparison(method_name: str, users: List[User]):
    """Run baseline method comparison."""
    if method_name not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method_name}")
    
    method = BASELINE_METHODS[method_name]
    decode_order, max_throughput = method(users)
    
    return {
        'method': method_name,
        'decode_order': decode_order,
        'max_throughput': max_throughput
    }
```

### 4. New Problem Variant

#### Step 1: Create Problem Class

```python
# problems/my_problem/problem_my.py
import torch
from typing import NamedTuple, Tuple

class MyProblem:
    """Custom problem variant."""
    
    NAME = 'my_problem'
    
    @staticmethod
    def get_costs(dataset: torch.Tensor, pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate costs for custom problem."""
        # Your cost calculation
        batch_size = dataset.size(0)
        costs = torch.zeros(batch_size)
        
        for i in range(batch_size):
            # Custom cost calculation
            costs[i] = custom_cost_function(dataset[i], pi[i])
        
        return costs, None
    
    @staticmethod
    def make_dataset(size: int, num_samples: int, **kwargs):
        """Generate training dataset."""
        return MyDataset(size=size, num_samples=num_samples, **kwargs)
    
    @staticmethod
    def load_val_dataset(size: int, num_samples: int, **kwargs):
        """Load validation dataset."""
        return MyValDataset(size=size, num_samples=num_samples, **kwargs)
    
    @staticmethod
    def make_state(input: torch.Tensor):
        """Create initial state."""
        return MyState.initialize(input)
```

#### Step 2: Create State Class

```python
# problems/my_problem/state_my.py
import torch
from typing import NamedTuple

class MyState(NamedTuple):
    """State representation for custom problem."""
    
    # Problem-specific state variables
    data: torch.Tensor
    ids: torch.Tensor
    current_step: torch.Tensor
    # Add more state variables as needed
    
    @staticmethod
    def initialize(data: torch.Tensor):
        """Initialize state from input data."""
        batch_size = data.size(0)
        return MyState(
            data=data,
            ids=torch.arange(batch_size, device=data.device),
            current_step=torch.zeros(1, device=data.device)
        )
    
    def update(self, selected: torch.Tensor):
        """Update state with selected action."""
        return self._replace(current_step=self.current_step + 1)
    
    def all_finished(self) -> bool:
        """Check if all steps are finished."""
        return self.current_step.item() >= self.data.size(1)
    
    def get_mask(self) -> torch.Tensor:
        """Get action validity mask."""
        # Return mask for valid actions
        return torch.zeros(self.data.size(0), self.data.size(1), dtype=torch.bool)
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration
├── test_models.py              # Model tests
├── test_optimization.py        # Optimization tests
├── test_data.py                # Data tests
├── test_training.py             # Training tests
├── test_validation.py           # Validation tests
└── fixtures/                    # Test fixtures
    ├── sample_data.py
    └── sample_models.py
```

### Writing Tests

#### Model Tests

```python
# tests/test_models.py
import pytest
import torch
from nets.attention_model import AttentionModel
from problems.noop import NOOP

class TestAttentionModel:
    """Test suite for AttentionModel."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return AttentionModel(
            embedding_dim=64,
            hidden_dim=64,
            problem=NOOP(),
            n_encode_layers=2,
            n_heads=4
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        return torch.randn(4, 5, 3)  # batch_size=4, users=5, features=3
    
    def test_forward_pass(self, model, sample_input):
        """Test forward pass."""
        cost, log_prob = model(sample_input)
        
        assert cost.shape == (4,)
        assert log_prob.shape == (4,)
        assert torch.isfinite(cost).all()
        assert torch.isfinite(log_prob).all()
    
    def test_decode_types(self, model, sample_input):
        """Test different decode types."""
        model.set_decode_type("greedy")
        cost_greedy, _ = model(sample_input)
        
        model.set_decode_type("sampling")
        cost_sampling, _ = model(sample_input)
        
        assert cost_greedy.shape == cost_sampling.shape
    
    def test_gradient_flow(self, model, sample_input):
        """Test gradient flow."""
        model.train()
        cost, log_prob = model(sample_input)
        loss = cost.mean() + log_prob.mean()
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
```

#### Optimization Tests

```python
# tests/test_optimization.py
import pytest
import numpy as np
from resource_allocation_optimization import (
    User, get_optimal_p, get_max_sum_weighted_alpha_throughput
)

class TestPowerOptimization:
    """Test suite for power optimization."""
    
    @pytest.fixture
    def sample_users(self):
        """Create sample users."""
        return [
            User(tid=0, tp_max=1.0, tg=0.5, tw=1.0),
            User(tid=1, tp_max=1.0, tg=0.8, tw=2.0),
            User(tid=2, tp_max=1.0, tg=0.3, tw=1.5)
        ]
    
    def test_optimal_power_allocation(self, sample_users):
        """Test optimal power allocation."""
        optimal_p = get_optimal_p(sample_users, alpha=1.0)
        
        assert len(optimal_p) == len(sample_users)
        assert all(p >= 0 for p in optimal_p)
        assert all(p <= user.p_max for p, user in zip(optimal_p, sample_users))
    
    def test_throughput_calculation(self, sample_users):
        """Test throughput calculation."""
        max_throughput = get_max_sum_weighted_alpha_throughput(sample_users, alpha=1.0)
        
        assert isinstance(max_throughput, float)
        assert max_throughput >= 0
        assert np.isfinite(max_throughput)
    
    def test_different_alpha_values(self, sample_users):
        """Test different alpha values."""
        alphas = [0.5, 1.0, 1.5]
        
        for alpha in alphas:
            optimal_p = get_optimal_p(sample_users, alpha=alpha)
            throughput = get_max_sum_weighted_alpha_throughput(sample_users, alpha=alpha)
            
            assert len(optimal_p) == len(sample_users)
            assert isinstance(throughput, float)
            assert np.isfinite(throughput)
```

#### Integration Tests

```python
# tests/test_integration.py
import pytest
import torch
from run import run
from options import get_options

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_training_pipeline(self):
        """Test complete training pipeline."""
        opts = get_options([
            '--n_epochs', '2',
            '--graph_size', '5',
            '--batch_size', '4',
            '--epoch_size', '8',
            '--no_cuda'
        ])
        
        # This should not raise any exceptions
        run(opts)
    
    def test_validation_pipeline(self):
        """Test validation pipeline."""
        # Load pre-trained model and validate
        # Implementation depends on available models
        pass
    
    def test_baseline_comparison(self):
        """Test baseline comparison."""
        # Test that all baseline methods work
        from resource_allocation_optimization import (
            duibi_exhaustive_search,
            duibi_heuristic_method_qian,
            duibi_tabu_search_gd
        )
        
        users = [User(tid=i, tp_max=1.0, tg=0.5, tw=1.0) for i in range(3)]
        
        for method in [duibi_exhaustive_search, duibi_heuristic_method_qian, duibi_tabu_search_gd]:
            decode_order, max_throughput = method(users)
            assert len(decode_order) == len(users)
            assert isinstance(max_throughput, float)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=nets --cov=problems --cov-report=html

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_models.py::TestAttentionModel::test_forward_pass
```

## Debugging

### Debugging Tools

#### 1. PyTorch Debugging

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check for NaN/Inf values
def check_tensor_values(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# Monitor gradient norms
def monitor_gradients(model):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
```

#### 2. Memory Debugging

```python
# Monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear GPU cache
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### 3. Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.debug(f"Input shape: {input.shape}")
logger.info(f"Training epoch {epoch} completed")
logger.warning(f"Low performance: {performance}")
logger.error(f"Training failed: {error}")
```

### Common Debugging Scenarios

#### 1. Training Issues

```python
# Check data loading
def debug_data_loading(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch.shape}")
        if i >= 2:  # Check first few batches
            break

# Check model output
def debug_model_output(model, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data)
        print(f"Model output: {output}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
```

#### 2. Optimization Issues

```python
# Check optimization convergence
def debug_optimization(users, alpha, noise):
    optimal_p = get_optimal_p(users, alpha, noise)
    throughput = get_max_sum_weighted_alpha_throughput(users, alpha, noise)
    
    print(f"Optimal power: {optimal_p}")
    print(f"Max throughput: {throughput}")
    
    # Check constraints
    for i, (p, user) in enumerate(zip(optimal_p, users)):
        assert 0 <= p <= user.p_max, f"Power constraint violated for user {i}"
```

#### 3. Performance Issues

```python
# Profile training time
import time

def profile_training_step(model, batch):
    start_time = time.time()
    
    # Forward pass
    forward_start = time.time()
    cost, log_prob = model(batch)
    forward_time = time.time() - forward_start
    
    # Backward pass
    backward_start = time.time()
    loss = cost.mean()
    loss.backward()
    backward_time = time.time() - backward_start
    
    total_time = time.time() - start_time
    
    print(f"Forward time: {forward_time:.4f}s")
    print(f"Backward time: {backward_time:.4f}s")
    print(f"Total time: {total_time:.4f}s")
```

## Performance Optimization

### 1. Model Optimization

#### Gradient Checkpointing

```python
# Enable gradient checkpointing
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    problem=NOOP(),
    checkpoint_encoder=True  # Enable checkpointing
)
```

#### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def training_step(model, batch, optimizer):
    optimizer.zero_grad()
    
    with autocast():
        cost, log_prob = model(batch)
        loss = cost.mean()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### Model Parallelism

```python
# Multi-GPU training
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Model sharding (for very large models)
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

def setup_distributed():
    init_process_group(backend='nccl')
    model = DistributedDataParallel(model)
```

### 2. Data Optimization

#### Efficient Data Loading

```python
# Optimize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

#### Data Preprocessing

```python
# Preprocess data once
def preprocess_dataset(dataset):
    processed_data = []
    for item in dataset:
        # Expensive preprocessing
        processed_item = expensive_preprocessing(item)
        processed_data.append(processed_item)
    return processed_data

# Cache processed data
import pickle

def load_cached_data(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        data = preprocess_dataset(raw_dataset)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return data
```

### 3. Training Optimization

#### Learning Rate Scheduling

```python
# Cosine annealing
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Warmup + cosine
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

#### Gradient Accumulation

```python
# Accumulate gradients for larger effective batch size
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    cost, log_prob = model(batch)
    loss = cost.mean() / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
```

## Contributing Guidelines

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Update documentation**
6. **Run test suite**: `pytest`
7. **Submit pull request**

### Code Review Process

#### Before Submitting

- [ ] Code follows PEP 8 style guide
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No hardcoded values
- [ ] Error handling is implemented
- [ ] Performance is acceptable

#### Review Checklist

- [ ] Code is readable and well-commented
- [ ] Architecture is sound
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Performance impact is considered
- [ ] Backward compatibility is maintained

### Release Process

1. **Update version numbers**
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Publish release notes**

### Issue Reporting

When reporting issues, include:

- **Environment**: OS, Python version, PyTorch version
- **Steps to reproduce**: Clear, minimal steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback
- **Code**: Minimal code to reproduce the issue

### Feature Requests

When requesting features, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: What other approaches were considered?
- **Implementation**: Any implementation ideas?
- **Testing**: How should it be tested?

---

## Additional Resources

- [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)

For questions or support, please open an issue or contact the development team.