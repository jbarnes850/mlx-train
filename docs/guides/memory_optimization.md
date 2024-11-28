# Memory Optimization Guide

## Overview

Efficient memory management is crucial for training large models on Apple Silicon devices. This guide covers MLX's memory optimization strategies and best practices for distributed training.

## Memory Management Features

### 1. Unified Memory Model

MLX uses a unified memory model([1](https://ml-explore.github.io/mlx/build/html/index.html)) where arrays live in shared memory. This means:

- No explicit data transfers between CPU and GPU
- Operations can run on any supported device
- Efficient memory utilization across devices

### 2. Automatic Memory Optimization

The MemoryOptimizer class provides automatic memory management:

```python
from mlx_train.core import MemoryOptimizer

# Get memory configuration suggestions
config = MemoryOptimizer.suggest_config(
    model_size=1e9, # 1B parameters
    num_devices=2
)
print(f"Suggested batch size: {config['suggested_batch_size']}")
print(f"Gradient accumulation steps: {config['gradient_accumulation']}")
print(f"Activation checkpointing: {config['activation_checkpointing']}")
```

### 3. Memory Recovery Strategies

When encountering memory issues, the optimizer can suggest recovery configurations:

```python
# Get recovery suggestions when OOM occurs
new_config = MemoryOptimizer.suggest_recovery_config(current_config)

# Apply suggested changes:
# - Reduced batch size
# - Enabled gradient checkpointing
# - Optimized optimizer states
```

## Best Practices

### 1. Model Initialization

```python
from mlx_train.models import ModelBuilder

# Build model with memory optimizations
model = ModelBuilder.build(
    config=model_config,
    quantize=True, # Enable weight quantization
    model_type="transformer"
)

# Estimate model memory requirements
model_size = ModelBuilder.estimate_model_size(model)
```

### 2. Training Configuration

Key memory optimization parameters:

- **Batch Size**: Automatically scaled based on available memory
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Activation Checkpointing**: Trades computation for memory
- **Mixed Precision**: Reduces memory footprint

### 3. Distributed Training

Memory considerations for multi-device setups:

```python
# Memory-optimized distributed configuration
config = MemoryOptimizer.suggest_config(
model_size=model_size,
num_devices=num_devices
)

# Monitor per-device memory usage
memory_used = mx.metal.get_active_memory() / 1e9 # GB
memory_total = config["hardware"]["memory_per_device"]
utilization = (memory_used / memory_total) 100
```

## Memory Monitoring

The training visualization system provides real-time memory metrics:

```python
from mlx_train.training import TrainingVisualizer
visualizer = TrainingVisualizer(
num_devices=num_devices,
config=training_config
)

# Monitor memory usage during training
metrics = TrainingMetrics(
memory_used=current_memory,
memory_total=total_memory,
device_utilization=utilization
)
```

## Troubleshooting

Common memory issues and solutions:

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Enable gradient accumulation
   - Activate memory checkpointing
   - Consider model sharding

2. **Memory Fragmentation**
   - Use contiguous memory allocations
   - Monitor memory patterns
   - Clear unused caches

3. **Device Synchronization**
   - Optimize data transfer patterns
   - Balance load across devices
   - Monitor device utilization

## See Also

- [Model Development Guide](model_development.md)
- [Training API Documentation](../api/training.md)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
