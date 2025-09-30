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

For complete, up-to-date setup and environment instructions, please start with the Development Guide:

- [Getting Started](DEVELOPMENT_GUIDE.md#getting-started) — prerequisites, environment setup, installation, verification
- [Development Environment](DEVELOPMENT_GUIDE.md#development-environment) — using DevPod, remote execution, and IDE setup

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

For comprehensive development instructions, environment setup, and contribution guidelines, please refer to the dedicated [Development Guide](DEVELOPMENT_GUIDE.md)

## Troubleshooting

For detailed troubleshooting information, common issues, and solutions, please refer to the dedicated [Troubleshooting Guide](TROUBLESHOOTING.md)

## Contributing

For comprehensive contribution guidelines, including development workflow, code review process, and release process please refer to the dedicated [Contributing Guide](CONTRIBUTING.md).

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CVXOPT Documentation](https://cvxopt.org/)
- [Graph Attention Networks Paper](https://arxiv.org/abs/1710.10903)
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

For questions or support, please open an issue or contact the development team.
