# Basic Training Example

This example demonstrates how to train a simple model using MLX-Train.

## Setup Project

```bash
mlx-train init basic-training
cd basic-training
```

## Configure Training

```python
from mlx_train import ProjectConfig, HardwareConfig

config = ProjectConfig(
    project_dir="./basic-training",
    model_type="transformer",
    hardware=HardwareConfig(num_devices=1),
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    mixed_precision=True
)

config.save()
```

## Define Model

```python
from mlx_train.models import BaseModel, ModelRegistry
import mlx.nn as nn

@ModelRegistry.register("custom")
class SimpleTransformer(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            hidden_size=config["hidden_size"]
        )
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"])
        
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
        
    def loss_fn(self, x, y):
        return nn.losses.cross_entropy(x, y)
        
    @classmethod
    def from_config(cls, config):
        return cls(config)
```

## Load Data

```python
from mlx_train.data import DatasetManager

# Initialize dataset manager
manager = DatasetManager(config)

# Load dataset (HuggingFace or local)
dataset = manager.setup_dataset()

# Create optimized dataloader
train_loader = manager.get_dataloader(dataset["train"])
```

## Memory Optimization

```python
from mlx_train.utils import MemoryOptimizer

# Get memory-optimized configuration
memory_config = MemoryOptimizer.suggest_config(
    model_size=1e9,  # 1B parameters
    num_devices=1
)

print(f"Suggested batch size: {memory_config['batch_size']}")
print(f"Gradient accumulation steps: {memory_config['gradient_accumulation']}")
```

## Train Model

```python
from mlx_train.training import TrainingOrchestrator
from mlx_train.models import ModelBuilder

# Build model with memory optimization
model = ModelBuilder.build(
    config={
        "num_layers": 6,
        "num_heads": 8,
        "hidden_size": 512,
        "vocab_size": 50257
    },
    model_type="custom",
    quantize=True  # Enable quantization for memory efficiency
)

# Create training orchestrator
orchestrator = TrainingOrchestrator(
    model=model,
    config=config
)

# Start training with progress tracking
orchestrator.train(train_loader)
```

## Monitor Training

```python
from mlx_train.utils import ExperimentTracker

# Initialize experiment tracker
tracker = ExperimentTracker(config.project_dir)

# Start tracking run
tracker.start_run(
    name="basic_training",
    config=config
)

# Monitor metrics
tracker.log_metrics({
    "loss": current_loss,
    "learning_rate": current_lr,
    "throughput": samples_per_second
})

# Compare runs
tracker.compare_runs(metric="loss")
```

## Export Model

```python
from mlx_train.core import export_mlx

# Export trained model
export_mlx(
    checkpoint="checkpoints/final.pt",
    output_dir="exports",
    quantize=True
)
```

## Serve Model

```python
from mlx_train.serving import ModelServer

# Start model server
server = ModelServer(
    model_path="exports/model",
    port=8000
)

server.serve()
```

## Complete Training Script

Here's a complete script combining all the above:

```python
import mlx.core as mx
from mlx_train import (
    ProjectConfig, 
    HardwareConfig,
    ModelBuilder,
    DatasetManager,
    TrainingOrchestrator,
    ExperimentTracker
)

def main():
    # Configuration
    config = ProjectConfig(
        project_dir="./basic-training",
        model_type="transformer",
        hardware=HardwareConfig(num_devices=1),
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10,
        mixed_precision=True
    )
    
    # Setup data
    manager = DatasetManager(config)
    dataset = manager.setup_dataset()
    train_loader = manager.get_dataloader(dataset["train"])
    
    # Build model
    model = ModelBuilder.build(
        config={
            "num_layers": 6,
            "num_heads": 8,
            "hidden_size": 512,
            "vocab_size": 50257
        },
        model_type="custom",
        quantize=True
    )
    
    # Setup training
    orchestrator = TrainingOrchestrator(
        model=model,
        config=config
    )
    
    # Setup tracking
    tracker = ExperimentTracker(config.project_dir)
    tracker.start_run("basic_training", config)
    
    # Train
    try:
        orchestrator.train(train_loader)
    finally:
        tracker.end_run()
        
    # Export model
    export_mlx(
        checkpoint="checkpoints/final.pt",
        output_dir="exports",
        quantize=True
    )

if __name__ == "__main__":
    main()
```

## Next Steps

- [Distributed Training Example](distributed_setup.md)
- [Custom Model Example](custom_model.md)
- [Memory Optimization Guide](../guides/memory_optimization.md)
