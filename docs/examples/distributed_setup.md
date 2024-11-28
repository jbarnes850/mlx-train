# Distributed Training Example

This guide demonstrates how to set up and run distributed training across multiple Apple Silicon devices.

## Environment Setup

```bash
# First, ensure all devices are on the same network
mlx-train check-network

# Configure each device
export OMPI_MCA_btl_tcp_links=4  # Optimal for MLX
```

## Project Configuration

```python
from mlx_train import ProjectConfig, HardwareConfig
from mlx_train.core import EnvironmentManager, DistributedController

# Initialize environment
env = EnvironmentManager()
env.setup_dependencies()

# Create distributed configuration
config = ProjectConfig(
    project_dir="./distributed-training",
    model_type="transformer",
    hardware=HardwareConfig(
        num_devices=2,
        btl_tcp_links=4
    ),
    model_config={
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "vocab_size": 50257,
        "max_position_embeddings": 2048
    },
    training_config={
        "batch_size": 32,  # Per device
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_steps": 50000,
        "gradient_accumulation": 4,
        "mixed_precision": True
    }
)
```

## Memory Optimization

```python
from mlx_train.utils import MemoryOptimizer

# Get memory-optimized configuration
memory_config = MemoryOptimizer.suggest_config(
    model_size=2e9,  # 2B parameters
    num_devices=2
)

print(f"""
Memory Configuration:
- Estimated memory per device: {memory_config['estimated_memory_gb']:.1f}GB
- Suggested batch size: {memory_config['suggested_batch_size']}
- Gradient accumulation steps: {memory_config['gradient_accumulation']}
- Activation checkpointing: {memory_config['activation_checkpointing']}
""")
```

## Distributed Training Setup

```python
from mlx_train.training import TrainingOrchestrator
from mlx_train.models import ModelBuilder
from mlx_train.utils import DistributedMetricsTracker

# Initialize distributed controller
controller = DistributedController()

# Build model
model = ModelBuilder.build(
    config=config.model_config,
    model_type="transformer",
    quantize=True
)

# Synchronize model across devices
model = controller.synchronize_model(model)

# Create distributed orchestrator
orchestrator = TrainingOrchestrator(
    model=model,
    config=config,
    distributed=controller
)

# Initialize metrics tracker
metrics = DistributedMetricsTracker(num_devices=2)
```

## Training Loop

```python
from mlx_train.utils import ExperimentTracker

# Setup experiment tracking
tracker = ExperimentTracker(config.project_dir)
tracker.start_run("distributed_training", config)

try:
    # Start training
    orchestrator.train()
    
    # Monitor metrics per device
    metrics.update(
        device_id=controller.rank,
        {
            "loss": current_loss,
            "learning_rate": current_lr,
            "throughput": samples_per_second
        }
    )
    
    # Check synchronization
    if not metrics.check_sync():
        print("Warning: Devices may be out of sync")
        
finally:
    tracker.end_run()
```

## Checkpoint Management

```python
# Save distributed checkpoint
controller.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=current_epoch,
    metrics=metrics.get_all_metrics()
)

# Load checkpoint and resume training
model, start_epoch = controller.load_checkpoint(model, optimizer)
```

## Complete Training Script

```python
import mlx.core as mx
from mlx_train import (
    ProjectConfig,
    HardwareConfig,
    ModelBuilder,
    TrainingOrchestrator,
    DistributedController,
    ExperimentTracker,
    MemoryOptimizer
)

def main():
    # Setup environment
    env = EnvironmentManager()
    env.setup_dependencies()
    
    # Configuration
    config = ProjectConfig(
        project_dir="./distributed-training",
        model_type="transformer",
        hardware=HardwareConfig(num_devices=2),
        model_config={
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 32,
            "vocab_size": 50257
        },
        training_config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "mixed_precision": True
        }
    )
    
    # Initialize distributed training
    controller = DistributedController()
    
    # Memory optimization
    memory_config = MemoryOptimizer.suggest_config(
        model_size=2e9,
        num_devices=2
    )
    
    # Build and synchronize model
    model = ModelBuilder.build(
        config=config.model_config,
        model_type="transformer",
        quantize=True
    )
    model = controller.synchronize_model(model)
    
    # Setup training
    orchestrator = TrainingOrchestrator(
        model=model,
        config=config,
        distributed=controller
    )
    
    # Setup tracking
    tracker = ExperimentTracker(config.project_dir)
    tracker.start_run("distributed_training", config)
    
    try:
        orchestrator.train()
    finally:
        tracker.end_run()
        
        # Save final checkpoint
        if controller.rank == 0:  # Primary device only
            controller.save_checkpoint(
                model=model,
                optimizer=orchestrator.optimizer,
                epoch=orchestrator.current_epoch,
                metrics=tracker.get_metrics()
            )

if __name__ == "__main__":
    main()
```

## Monitoring Training

```bash
# View training progress across devices
mlx-train monitor --distributed

# Check device synchronization
mlx-train check-sync

# View resource usage
mlx-train status --resources
```

## Common Issues and Solutions

1. **Memory Errors**

   ```bash
   # Reduce per-device batch size
   mlx-train train --batch-size 16 --distributed
   
   # Enable gradient checkpointing
   mlx-train train --checkpoint-activation --distributed
   ```

2. **Network Issues**

   ```bash
   # Verify network connectivity
   mlx-train check-network
   
   # Reset distributed setup
   mlx-train reset-distributed
   ```

3. **Performance Issues**

   ```bash
   # Monitor performance metrics
   mlx-train monitor --performance --distributed
   
   # Optimize network settings
   export OMPI_MCA_btl_tcp_links=4
   ```

## Next Steps

- [Memory Optimization Guide](../guides/memory_optimization.md)
- [Advanced Training Guide](../guides/advanced_training.md)
- [Model Export Guide](../guides/model_export.md)
