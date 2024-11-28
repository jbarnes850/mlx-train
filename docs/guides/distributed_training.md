# Distributed Training Guide

## Overview

MLX Distributed Training enables seamless scaling across multiple Apple Silicon devices. This guide covers setup, best practices, and troubleshooting for distributed training.

## Setup Requirements

1. **Hardware Requirements**
   - Multiple Apple Silicon devices (M1/M2/M3)
   - All devices on same network
   - Sufficient memory per device

2. **Network Configuration**

   ```bash
   # Set optimal TCP links for MLX
   export OMPI_MCA_btl_tcp_links=4
   
   # Optional: Set specific network interface
   export OMPI_MCA_btl_tcp_if_include=en0
   ```

## Quick Start

1. **Configure Devices**

   ```bash
   # On each device:
   mlx-train configure --distributed
   ```

2. **Launch Training**

   ```bash
   # Start training across all devices
   mlx-train train --distributed
   ```

## Memory Management

The framework automatically handles memory optimization:

```python
# Memory is automatically managed per device
memory_config = MemoryOptimizer.suggest_config(
    model_size=1e9,  # 1B parameters
    num_devices=2
)
```

## Best Practices

### 1. Memory Optimization

- Enable gradient checkpointing for large models
- Use mixed precision training
- Monitor memory usage per device

### 2. Network Configuration

- Ensure stable network connection
- Use recommended TCP link settings
- Keep devices on same subnet

### 3. Batch Size Scaling

- Scale batch size with number of devices
- Monitor memory usage
- Use gradient accumulation if needed

## Monitoring Training

The framework provides real-time monitoring:

```bash
# View training progress
mlx-train monitor

# Check device status
mlx-train status
```

## Error Recovery

The framework includes automatic error recovery:

1. **Memory Errors**
   - Automatic batch size reduction
   - Gradient checkpointing enablement
   - Memory usage optimization

2. **Network Issues**
   - Automatic reconnection
   - Gradient synchronization validation
   - Training state recovery

## Common Issues

### 1. Memory Errors

```bash
# Reduce batch size
mlx-train train --batch-size 16

# Enable memory optimization
mlx-train train --optimize-memory
```

### 2. Network Issues

```bash
# Check network connectivity
mlx-train check-network

# Reset distributed setup
mlx-train reset-distributed
```

### 3. Performance Issues

```bash
# Monitor performance
mlx-train monitor --performance

# Optimize settings
mlx-train optimize
```

## Advanced Configuration

For fine-grained control:

```python
from mlx_train import DistributedConfig

config = DistributedConfig(
    num_devices=4,
    batch_size_per_device=32,
    gradient_accumulation=4
)
```

## Next Steps

- [Memory Optimization Guide](memory_optimization.md)
- [Model Development Guide](model_development.md)
- [Advanced Training Guide](advanced_training.md)
