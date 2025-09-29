# ASOPA Documentation Index

Welcome to the comprehensive documentation for the ASOPA (Attention-Based SIC Ordering and Power Allocation) project. This documentation suite provides everything you need to understand, use, and extend the ASOPA framework.

## Documentation Overview

### [README.md](README.md) - Getting Started
**Start here for new users**
- Project overview and key features
- Quick start guide
- Installation instructions
- Basic usage examples
- Paper reference and citations

### [ARCHITECTURE.md](ARCHITECTURE.md) - System Design
**Deep dive into the system architecture**
- High-level system overview
- Core components and their interactions
- Neural network architecture details
- Optimization framework
- Data flow diagrams
- Performance considerations

### [API_REFERENCE.md](API_REFERENCE.md) - Complete API Documentation
**Comprehensive reference for all functions and classes**
- Core model classes (AttentionModel, GraphAttentionEncoder)
- Problem definition classes (NOOP, StateNOOP)
- Optimization functions
- Utility functions
- Configuration parameters
- Error handling and exceptions

### [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) - Development Workflow
**Guide for developers and contributors**
- Development environment setup
- Code structure and standards
- Adding new features (models, optimizers, baselines)
- Testing strategies
- Debugging techniques
- Performance optimization
- Contributing guidelines

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem Solving
**Solutions to common issues**
- Common installation problems
- Training and performance issues
- Memory and optimization problems
- Debugging tools and techniques
- Frequently asked questions
- Getting help and support

### [AGENTS.md](../AGENTS.md) - AI Agent Instructions
**Specific instructions for AI coding agents**
- Agent-specific commands and patterns
- Common tasks for agents
- Quick reference for agents
- Links to detailed documentation

## Quick Navigation

### For New Users
1. **Start with** [README.md](README.md) for project overview
2. **Follow** the Quick Start section for installation
3. **Run** the validation examples to verify setup
4. **Read** [ARCHITECTURE.md](ARCHITECTURE.md) for system understanding

### For Researchers
1. **Study** [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
2. **Reference** [API_REFERENCE.md](API_REFERENCE.md) for implementation details
3. **Use** [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for extending the framework
4. **Check** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

### For Developers
1. **Setup** development environment using [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
2. **Understand** code structure and standards
3. **Implement** new features following the guidelines
4. **Test** thoroughly using provided test frameworks
5. **Contribute** following the contribution guidelines

### For AI Agents
1. **Read** [AGENTS.md](../AGENTS.md) for agent-specific instructions
2. **Reference** [API_REFERENCE.md](API_REFERENCE.md) for implementation details
3. **Follow** [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for development workflow
4. **Check** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

## Key Concepts

### ASOPA Framework
- **Attention-based neural network** for SIC ordering optimization
- **Convex optimization** for power allocation
- **Reinforcement learning** training with policy gradient methods
- **Modular architecture** for easy extension

### Core Components
- **AttentionModel**: Main neural network for SIC ordering
- **GraphAttentionEncoder**: Graph attention mechanism
- **Power Allocation Optimizer**: Convex optimization solver
- **Problem Definition**: NOMA optimization problem setup
- **Training Pipeline**: Policy gradient training framework

### Key Features
- **Variable user numbers** (5-20+ users)
- **Multiple baseline comparisons** (exhaustive search, heuristics)
- **Comprehensive validation** suite
- **Extensible architecture** for new research

## Usage Examples

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
python ASOPA_validation.py

# Train from scratch
python run.py --n_epochs 300 --graph_size 10

# Run baseline comparisons
python run_baseline.py
```

### Advanced Usage
```python
# Custom attention model
from nets.my_attention_model import MyAttentionModel
model = MyAttentionModel(embedding_dim=128, hidden_dim=128, problem=NOOP())

# Custom optimization
from resource_allocation_optimization import my_power_optimization
optimal_power = my_power_optimization(users, alpha=1.0)

# Custom baseline
from resource_allocation_optimization import my_baseline_method
decode_order, throughput = my_baseline_method(users)
```

## Development Workflow

### Adding New Features
1. **Create** new module following code standards
2. **Implement** functionality with proper documentation
3. **Add** comprehensive tests
4. **Update** documentation
5. **Submit** pull request

### Testing Strategy
- **Unit tests** for individual components
- **Integration tests** for complete pipeline
- **Performance tests** for optimization
- **Regression tests** for bug fixes

### Code Quality
- **PEP 8** style compliance
- **Type hints** for all functions
- **Comprehensive documentation**
- **Error handling** for edge cases

## Performance Considerations

### Training Optimization
- **Mixed precision** training for speed
- **Gradient checkpointing** for memory
- **Multi-GPU** support for scale
- **Curriculum learning** for efficiency

### Inference Optimization
- **Greedy decoding** for speed
- **Model quantization** for size
- **Batch optimization** for throughput
- **Caching** for repeated computations

## Configuration

### Key Parameters
- **Model architecture**: embedding_dim, hidden_dim, n_heads
- **Training**: learning_rate, batch_size, n_epochs
- **Problem**: user_num, alpha, noise
- **Hardware**: use_cuda, num_workers

### Environment Setup
- **Python 3.8+** required
- **PyTorch 2.3.1** (pinned version for compatibility)
- **CUDA 12.1+** for GPU acceleration
- **CVXOPT** for optimization

## Extending ASOPA

### Research Directions
- **New attention mechanisms** for better SIC ordering
- **Advanced optimization** methods for power allocation
- **Multi-objective** optimization frameworks
- **Distributed training** for large-scale problems
- **Real-time adaptation** for dynamic networks

### Implementation Areas
- **Neural architectures** in `nets/`
- **Optimization methods** in `resource_allocation_optimization.py`
- **Problem variants** in `problems/`
- **Training strategies** in `train.py`
- **Baseline methods** for comparison

## Contributing

### How to Contribute
1. **Fork** the repository
2. **Create** feature branch
3. **Implement** changes with tests
4. **Update** documentation
5. **Submit** pull request

### Contribution Areas
- **Bug fixes** and improvements
- **New features** and capabilities
- **Documentation** enhancements
- **Performance** optimizations
- **Test coverage** improvements

## Support

### Getting Help
- **Check** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- **Search** existing GitHub issues
- **Create** new issue with detailed information
- **Contact** development team

### Community
- **GitHub** repository for code and issues
- **Documentation** for comprehensive guides
- **Examples** for usage patterns
- **Tests** for validation

---

## Documentation Maintenance

This documentation is maintained by the ASOPA development team. For updates, corrections, or suggestions:

1. **Submit** issue for documentation problems
2. **Create** pull request for improvements
3. **Contact** team for major changes
4. **Follow** contribution guidelines

---

**Last Updated**: 2024
**Version**: 1.0
**Maintainers**: ASOPA Development Team

For the most up-to-date information, please refer to the individual documentation files and the main repository.