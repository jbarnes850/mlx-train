# Custom Model Example

This guide demonstrates how to create and train a custom model with advanced features like LoRA fine-tuning and memory optimization.

## Project Setup

```bash
mlx-train init custom-model
cd custom-model
```

## Define Custom Model

```python
from mlx_train.models import BaseModel, ModelRegistry
import mlx.nn as nn

@ModelRegistry.register("custom_transformer")
class CustomTransformer(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Model configuration
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.vocab_size = config["vocab_size"]
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(config["max_position_embeddings"], self.hidden_size)
        
        # Transformer layers with LoRA adaptation
        self.layers = [
            nn.TransformerEncoder(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                mlp_dim=self.hidden_size * 4,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ]
        
        # Output head
        self.head = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add position embeddings
        positions = mx.arange(input_ids.shape[1])
        x = x + self.position_embedding(positions)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            
        # Project to vocabulary
        return self.head(x)
        
    def loss_fn(self, logits, labels):
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1)
        )
        
    @classmethod
    def from_config(cls, config):
        return cls(config)
```

## Add LoRA Fine-tuning

```python
from mlx_train.models.architectures import LoRALayer

class LoRATransformer(CustomTransformer):
    def __init__(self, config):
        super().__init__(config)
        
        # Add LoRA layers to attention
        self.lora_layers = []
        for layer in self.layers:
            attention = layer.self_attention
            self.lora_layers.append(
                LoRALayer(
                    in_features=self.hidden_size,
                    out_features=self.hidden_size,
                    rank=config.get("lora_rank", 8),
                    alpha=config.get("lora_alpha", 16)
                )
            )
            # Wrap attention with LoRA
            layer.self_attention = self.lora_layers[-1].wrap_layer(attention)
```

## Configure Training

```python
from mlx_train import ProjectConfig, HardwareConfig

config = ProjectConfig(
    project_dir="./custom-model",
    model_type="custom_transformer",
    hardware=HardwareConfig(num_devices=2),  # Multi-device training
    model_config={
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "vocab_size": 50257,
        "max_position_embeddings": 2048,
        "dropout": 0.1,
        "lora_rank": 8,  # For LoRA fine-tuning
        "lora_alpha": 16
    },
    training_config={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_steps": 10000,
        "warmup_steps": 1000,
        "mixed_precision": True
    }
)

config.save()
```

## Memory-Optimized Training

```python
from mlx_train.training import TrainingOrchestrator
from mlx_train.models import ModelBuilder
from mlx_train.utils import MemoryOptimizer

# Get memory-optimized configuration
memory_config = MemoryOptimizer.suggest_config(
    model_size=1e9,  # 1B parameters
    num_devices=2
)

print(f"Suggested batch size: {memory_config['batch_size']}")
print(f"Gradient accumulation steps: {memory_config['gradient_accumulation']}")

# Build model with memory optimization
model = ModelBuilder.build(
    config=config.model_config,
    model_type="custom_transformer",
    quantize=True  # Enable quantization
)

# Setup distributed training
orchestrator = TrainingOrchestrator(
    model=model,
    config=config
)

# Train with progress tracking
orchestrator.train()
```

## Monitor Training

```python
from mlx_train.utils import ExperimentTracker

tracker = ExperimentTracker(config.project_dir)
tracker.start_run("custom_model_training", config)

# Log custom metrics
tracker.log_metrics({
    "loss": current_loss,
    "learning_rate": current_lr,
    "lora_gradients": lora_grad_norm
})

# Compare with baseline
tracker.compare_runs(["baseline", "custom_model_training"])
```

## Export and Serve

```python
from mlx_train.core import export_mlx
from mlx_train.serving import ModelServer

# Export model
export_mlx(
    checkpoint="checkpoints/final.pt",
    output_dir="exports",
    quantize=True
)

# Serve model
server = ModelServer(
    model_path="exports/model",
    port=8000
)
server.serve()
```

## Complete Training Script

```python
import mlx.core as mx
from mlx_train import (
    ProjectConfig,
    HardwareConfig,
    ModelBuilder,
    TrainingOrchestrator,
    ExperimentTracker,
    MemoryOptimizer
)

def main():
    # Configuration
    config = ProjectConfig(
        project_dir="./custom-model",
        model_type="custom_transformer",
        hardware=HardwareConfig(num_devices=2),
        model_config={
            "hidden_size": 1024,
            "num_layers": 12,
            "num_heads": 16,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "dropout": 0.1,
            "lora_rank": 8
        },
        training_config={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "mixed_precision": True
        }
    )
    
    # Memory optimization
    memory_config = MemoryOptimizer.suggest_config(
        model_size=1e9,
        num_devices=2
    )
    
    # Build model
    model = ModelBuilder.build(
        config=config.model_config,
        model_type="custom_transformer",
        quantize=True
    )
    
    # Setup training
    orchestrator = TrainingOrchestrator(
        model=model,
        config=config
    )
    
    # Setup tracking
    tracker = ExperimentTracker(config.project_dir)
    tracker.start_run("custom_model", config)
    
    # Train
    try:
        orchestrator.train()
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
- [Memory Optimization Guide](../guides/memory_optimization.md)
- [Model Architecture Guide](../guides/model_architectures.md)
