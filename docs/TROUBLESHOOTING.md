# ASOPA Troubleshooting Guide

## Table of Contents
- [Common Issues](#common-issues)
- [Training Issues](#training-issues)
- [Performance Problems](#performance-problems)
- [Memory Issues](#memory-issues)
- [Optimization Problems](#optimization-problems)
- [Debugging Tools](#debugging-tools)
- [FAQ](#faq)

## Common Issues

### Issue: ImportError when running scripts

**Symptoms:**
```
ImportError: No module named 'torch'
ImportError: No module named 'cvxopt'
ModuleNotFoundError: No module named 'nets'
```

**Solutions:**

1. **Check virtual environment**:
   ```bash
   # Activate virtual environment
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   
   # Verify Python path
   which python
   ```

2. **Reinstall dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Check Python path**:
   ```bash
   # Add project root to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **Verify installation**:
   ```python
   import torch
   import cvxopt
   from nets.attention_model import AttentionModel
   print("All imports successful!")
   ```

### Issue: CUDA/GPU not detected

**Symptoms:**
```
CUDA not available
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Check CUDA installation**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Force CPU usage**:
   ```bash
   python run.py --no_cuda
   ```

3. **Check CUDA version compatibility**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version
   ```

4. **Reduce batch size**:
   ```python
   # In options.py
   parser.add_argument('--batch_size', default=32)  # Reduce from 64
   ```

### Issue: Model loading fails

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
FileNotFoundError: Variable_user_n10_epoch300.pth
```

**Solutions:**

1. **Check model file exists**:
   ```bash
   ls -la *.pth
   ```

2. **Verify model compatibility**:
   ```python
   import torch
   checkpoint = torch.load('model.pth', map_location='cpu')
   print(checkpoint.keys())
   ```

3. **Load with proper device**:
   ```python
   # Load on CPU first
   model = torch.load('model.pth', map_location='cpu')
   # Then move to GPU if needed
   if torch.cuda.is_available():
       model = model.cuda()
   ```

## Training Issues

### Issue: Training loss is NaN

**Symptoms:**
```
loss: nan
RuntimeError: Function 'LogSoftmaxBackward' returned nan values
```

**Solutions:**

1. **Check input data**:
   ```python
   # Add data validation
   def validate_input(input_data):
       if torch.isnan(input_data).any():
           print("NaN detected in input data")
       if torch.isinf(input_data).any():
           print("Inf detected in input data")
       print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
   ```

2. **Reduce learning rate**:
   ```python
   # In conf.py
   parser.add_argument('--lr', default=1e-4)  # Reduce from 1e-3
   ```

3. **Add gradient clipping**:
   ```python
   # In train.py
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Check model initialization**:
   ```python
   # Initialize weights properly
   def init_weights(m):
       if isinstance(m, nn.Linear):
           nn.init.xavier_uniform_(m.weight)
           nn.init.constant_(m.bias, 0)
   
   model.apply(init_weights)
   ```

### Issue: Training is too slow

**Symptoms:**
```
Training takes hours per epoch
GPU utilization is low
```

**Solutions:**

1. **Increase batch size** (if memory allows):
   ```python
   # In options.py
   parser.add_argument('--batch_size', default=128)  # Increase from 64
   ```

2. **Enable mixed precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       cost, log_prob = model(batch)
       loss = cost.mean()
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Use multiple GPUs**:
   ```python
   if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
   ```

4. **Optimize data loading**:
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=4,  # Parallel data loading
       pin_memory=True,
       persistent_workers=True
   )
   ```

### Issue: Model doesn't converge

**Symptoms:**
```
Loss doesn't decrease
Performance doesn't improve
```

**Solutions:**

1. **Check learning rate schedule**:
   ```python
   # Add learning rate scheduling
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=10
   )
   ```

2. **Increase training epochs**:
   ```python
   # In options.py
   parser.add_argument('--n_epochs', default=500)  # Increase from 300
   ```

3. **Check baseline network**:
   ```python
   # Verify baseline is working
   baseline_value = baseline.eval(batch, cost)
   print(f"Baseline value: {baseline_value}")
   ```

4. **Add curriculum learning**:
   ```python
   # Start with easier problems
   if epoch < 100:
       user_range = (5, 7)
   elif epoch < 200:
       user_range = (7, 9)
   else:
       user_range = (9, 10)
   ```

## Performance Problems

### Issue: Low GPU utilization

**Symptoms:**
```
GPU utilization < 50%
Training is slow
```

**Solutions:**

1. **Increase batch size**:
   ```python
   parser.add_argument('--batch_size', default=128)
   ```

2. **Enable gradient accumulation**:
   ```python
   accumulation_steps = 4
   
   for i, batch in enumerate(dataloader):
       cost, log_prob = model(batch)
       loss = cost.mean() / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Optimize model architecture**:
   ```python
   # Reduce model size if possible
   model = AttentionModel(
       embedding_dim=64,  # Reduce from 128
       hidden_dim=64,     # Reduce from 128
       n_encode_layers=2   # Reduce from 3
   )
   ```

### Issue: Inference is slow

**Symptoms:**
```
Validation takes too long
Real-time inference not possible
```

**Solutions:**

1. **Use greedy decoding**:
   ```python
   model.set_decode_type("greedy")  # Faster than sampling
   ```

2. **Reduce batch size for inference**:
   ```python
   # In ASOPA_validation.py
   opts.eval_batch_size = 1  # Process one at a time
   ```

3. **Enable model optimization**:
   ```python
   # TorchScript optimization
   model = torch.jit.script(model)
   
   # Or TensorRT optimization (if available)
   import torch_tensorrt
   model = torch_tensorrt.compile(model)
   ```

## Memory Issues

### Issue: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions:**

1. **Reduce batch size**:
   ```python
   parser.add_argument('--batch_size', default=16)  # Reduce from 64
   ```

2. **Enable gradient checkpointing**:
   ```python
   model = AttentionModel(
       embedding_dim=128,
       hidden_dim=128,
       problem=NOOP(),
       checkpoint_encoder=True  # Enable checkpointing
   )
   ```

3. **Use CPU for some operations**:
   ```python
   # Move large tensors to CPU
   large_tensor = large_tensor.cpu()
   # Process on CPU
   result = process_on_cpu(large_tensor)
   # Move back to GPU if needed
   result = result.cuda()
   ```

4. **Clear GPU cache**:
   ```python
   torch.cuda.empty_cache()
   ```

### Issue: System memory issues

**Symptoms:**
```
MemoryError
System runs out of RAM
```

**Solutions:**

1. **Reduce dataset size**:
   ```python
   # In conf.py
   parser.add_argument('--train_data_num', default=10000)  # Reduce from 120000
   parser.add_argument('--val_data_num', default=100)      # Reduce from 1000
   ```

2. **Use data streaming**:
   ```python
   # Load data on demand
   class StreamingDataset(Dataset):
       def __init__(self, data_path):
           self.data_path = data_path
           self.data_files = os.listdir(data_path)
       
       def __getitem__(self, idx):
           # Load data file on demand
           return torch.load(os.path.join(self.data_path, self.data_files[idx]))
   ```

3. **Process in smaller chunks**:
   ```python
   # Process data in chunks
   chunk_size = 1000
   for i in range(0, len(dataset), chunk_size):
       chunk = dataset[i:i+chunk_size]
       process_chunk(chunk)
   ```

## Optimization Problems

### Issue: CVXOPT solver fails

**Symptoms:**
```
cvxopt.base.CVXOPTError: Solver failed
RuntimeError: Optimization failed
```

**Solutions:**

1. **Check problem formulation**:
   ```python
   # Verify constraints are feasible
   def check_feasibility(users):
       for user in users:
           assert user.p_max > 0, f"Invalid power limit for user {user.id}"
           assert user.g > 0, f"Invalid channel gain for user {user.id}"
           assert user.w > 0, f"Invalid weight for user {user.id}"
   ```

2. **Adjust solver parameters**:
   ```python
   # In resource_allocation_optimization.py
   solvers.options['show_progress'] = False
   solvers.options['refinement'] = 2
   solvers.options['abstol'] = 1e-6
   solvers.options['reltol'] = 1e-5
   solvers.options['feastol'] = 1e-6
   ```

3. **Use alternative solver**:
   ```python
   # Try different solver
   def get_optimal_p(users, alpha=1, noise=3.981e-15, solver='cvxopt'):
       if solver == 'cvxopt':
           return _get_optimal_p_cvxopt(users, alpha, noise)
       elif solver == 'scipy':
           return _get_optimal_p_scipy(users, alpha, noise)
       else:
           raise ValueError(f"Unknown solver: {solver}")
   ```

### Issue: Optimization is slow

**Symptoms:**
```
Power allocation takes too long
Training is bottlenecked by optimization
```

**Solutions:**

1. **Cache optimization results**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_power_optimization(user_features_hash, alpha, noise):
       return solve_power_allocation(user_features, alpha, noise)
   ```

2. **Use approximate methods**:
   ```python
   # Use faster approximation for training
   def approximate_power_allocation(users, alpha, noise):
       if alpha == 1.0:
           # Simple proportional allocation
           total_weight = sum(user.w for user in users)
           return [user.w / total_weight * user.p_max for user in users]
       else:
           # Other approximations
           return [user.p_max / len(users) for user in users]
   ```

3. **Parallel optimization**:
   ```python
   from multiprocessing import Pool
   
   def parallel_power_optimization(user_batches):
       with Pool() as pool:
           results = pool.map(solve_power_allocation, user_batches)
       return results
   ```

## Debugging Tools

### 1. PyTorch Debugging

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check for NaN/Inf
def check_tensor_values(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
    print(f"{name} range: [{tensor.min():.4f}, {tensor.max():.4f}]")

# Monitor gradients
def monitor_gradients(model):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name} gradient norm: {param_norm:.4f}")
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.4f}")
```

### 2. Memory Monitoring

```python
# Monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")

# Clear GPU cache
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. Performance Profiling

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

### 4. Data Validation

```python
# Validate input data
def validate_input_data(input_data):
    print(f"Input shape: {input_data.shape}")
    print(f"Input dtype: {input_data.dtype}")
    print(f"Input device: {input_data.device}")
    
    # Check for invalid values
    if torch.isnan(input_data).any():
        print("Warning: NaN values in input")
    if torch.isinf(input_data).any():
        print("Warning: Inf values in input")
    
    # Check value ranges
    for i in range(input_data.shape[-1]):
        feature = input_data[:, :, i]
        print(f"Feature {i} range: [{feature.min():.4f}, {feature.max():.4f}]")
```

## FAQ

### Q: How do I run ASOPA on a different number of users?

**A:** Modify the configuration parameters:

```python
# In conf.py
parser.add_argument('--user_num', default=15)        # Training users
parser.add_argument('--val_user_num', default=12)    # Validation users

# In options.py
parser.add_argument('--graph_size', default=15)      # Model graph size
parser.add_argument('--val_graph_size', default=12)  # Validation graph size
```

### Q: How do I change the fairness parameter α?

**A:** Modify the alpha parameter:

```python
# In conf.py
parser.add_argument('--alpha', default=0.5)  # Change from 1.0 to 0.5

# This affects the objective function:
# α = 1: Proportional fairness (log utility)
# α < 1: More fair (closer to max-min)
# α > 1: Less fair (closer to max-sum)
```

### Q: How do I add my own baseline method?

**A:** Follow these steps:

1. **Implement the baseline**:
   ```python
   def my_baseline_method(users, alpha=1, noise=3.981e-15):
       # Your implementation
       decode_order = [0, 1, 2, ...]  # Your ordering
       users_order = sort_by_decode_order(users, decode_order)
       max_throughput = get_max_sum_weighted_alpha_throughput(users_order, alpha, noise)
       return decode_order, max_throughput
   ```

2. **Add to comparison suite**:
   ```python
   # In run_baseline.py
   BASELINE_METHODS['my_method'] = my_baseline_method
   ```

3. **Run comparison**:
   ```bash
   python run_baseline.py --method my_method
   ```

### Q: How do I modify the attention mechanism?

**A:** Create a new attention model:

1. **Create new model file**:
   ```python
   # nets/my_attention.py
   class MyAttentionModel(nn.Module):
       def __init__(self, ...):
           # Your custom architecture
       def forward(self, input, return_pi=False):
           # Your forward pass
   ```

2. **Register the model**:
   ```python
   # In run.py
   model_class = {
       'attention': AttentionModel,
       'my_attention': MyAttentionModel,
   }.get(opts.model, AttentionModel)
   ```

3. **Run with new model**:
   ```bash
   python run.py --model my_attention
   ```

### Q: How do I debug training issues?

**A:** Use these debugging techniques:

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check intermediate values**:
   ```python
   def debug_model_output(model, input_data):
       model.eval()
       with torch.no_grad():
           output = model(input_data)
           print(f"Model output: {output}")
           print(f"Output shape: {output.shape}")
   ```

3. **Monitor training metrics**:
   ```python
   # Add to training loop
   print(f"Epoch {epoch}: loss={loss:.4f}, reward={reward:.4f}")
   ```

### Q: How do I optimize for my specific hardware?

**A:** Adjust these parameters:

1. **For GPU with limited memory**:
   ```python
   parser.add_argument('--batch_size', default=16)
   parser.add_argument('--checkpoint_encoder', action='store_true')
   ```

2. **For CPU-only systems**:
   ```bash
   python run.py --no_cuda
   ```

3. **For multiple GPUs**:
   ```python
   if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
   ```

### Q: How do I reproduce the paper results?

**A:** Follow these steps:

1. **Use exact configuration**:
   ```python
   # In conf.py - use paper settings
   parser.add_argument('--user_num', default=10)
   parser.add_argument('--val_user_num', default=8)
   parser.add_argument('--alpha', default=1)
   parser.add_argument('--noise', default=3.981e-15)
   ```

2. **Load pre-trained model**:
   ```bash
   python ASOPA_validation.py
   ```

3. **Run baseline comparisons**:
   ```bash
   python run_baseline.py
   ```

4. **Compare results** with paper figures

### Q: How do I extend ASOPA for new research?

**A:** Consider these extension points:

1. **New attention mechanisms** in `nets/`
2. **New optimization methods** in `resource_allocation_optimization.py`
3. **New baseline methods** for comparison
4. **New problem variants** in `problems/`
5. **New training strategies** in `train.py`

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs** for error messages
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Environment details (OS, Python, PyTorch versions)
   - Steps to reproduce
   - Error messages
   - Relevant code snippets
4. **Contact the development team**

## Contributing Fixes

If you find a solution to a problem:

1. **Test your fix** thoroughly
2. **Update this guide** with the solution
3. **Submit a pull request** with your fix
4. **Add tests** to prevent regression

---

For additional support, please refer to the main documentation or contact the development team.
