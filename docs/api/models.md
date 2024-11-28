# Model API Reference

## BaseModel

Abstract base class for all models in the framework.

```python
from mlx_train.models import BaseModel
import mlx.nn as nn

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.encoder = nn.TransformerEncoder(
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            hidden_size=config["hidden_size"]
        )
        
    def forward(self, x):
        return self.encoder(x)
        
    def loss_fn(self, x, y):
        return nn.losses.cross_entropy(x, y)
        
    @classmethod
    def from_config(cls, config):
        return cls(config)
```

## ModelBuilder

Factory class for model creation and initialization with memory optimizations.

```python
from mlx_train.models import ModelBuilder

# Build model from config
model = ModelBuilder.build(
    config={
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "model_type": "transformer"
    },
    pretrained=True,
    quantize=True
)

# Estimate model memory requirements
size_bytes = ModelBuilder.estimate_model_size(model)
print(f"Model size: {size_bytes / 1e9:.2f}GB")
```

## ModelRegistry

Registry for model architectures with type safety.

```python
from mlx_train.models import ModelRegistry, BaseModel

@ModelRegistry.register("custom_transformer")
class CustomTransformer(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Model implementation
        
    @classmethod
    def from_config(cls, config):
        return cls(config)

# Get registered model
model_cls = ModelRegistry.get_model("custom_transformer")
```

## Pre-built Architectures

### TransformerModel

```python
from mlx_train.models.architectures import TransformerModel

model = TransformerModel(
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    vocab_size=50257,
    max_position_embeddings=1024
)
```

### LoRALayer

Low-Rank Adaptation layer for efficient fine-tuning.

```python
from mlx_train.models.architectures import LoRALayer

lora = LoRALayer(
    in_features=768,
    out_features=768,
    rank=8,  # Lower rank = more compression
    alpha=16  # Scaling factor
)

# Apply LoRA to existing model
original_layer = model.self_attention
model.self_attention = lora.wrap_layer(original_layer)
```

## Memory Management

### Efficient Loading

```python
# Load large models efficiently
weights = ModelBuilder.load_sharded(
    path="path/to/model",
    num_shards=2
)

# Quantize weights for memory efficiency
quantized = ModelBuilder.quantize_weights(
    weights,
    bits=8  # 8-bit quantization
)
```

### Gradient Checkpointing

```python
from mlx_train.models import enable_checkpointing

# Enable gradient checkpointing for memory efficiency
model = enable_checkpointing(model)
```

## Best Practices

1. **Memory Optimization**
   - Use `ModelBuilder` for automatic memory optimization
   - Enable quantization for large models
   - Use gradient checkpointing for deep models
   - Monitor memory usage during training

2. **Custom Models**
   - Inherit from `BaseModel`
   - Register with `ModelRegistry`
   - Implement required abstract methods
   - Use type hints for better IDE support

3. **Performance**
   - Use mixed precision training
   - Enable memory optimizations
   - Profile model before training
   - Monitor device utilization

## Error Handling

```python
try:
    model = ModelBuilder.build(config)
except ValueError as e:
    print(f"Invalid configuration: {e}")
except MemoryError as e:
    print(f"Memory optimization failed: {e}")
```

## See Also

- [Training API](training.md)
- [Core API](core.md)
- [Model Development Guide](../guides/model_development.md)
