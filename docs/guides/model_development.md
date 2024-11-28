# Model Development Guide

## Overview

This guide covers best practices for developing models with MLX, including architecture design, memory optimization, and distributed training considerations.

## Model Architecture

### Base Model Class

All models should inherit from the BaseModel class:

```python
from mlx_train.models import BaseModel
import mlx.nn as nn
class CustomModel(BaseModel):
    def init(self, config):
        super().init()
        self.layers = [
nn.Linear(config.input_dim, config.hidden_dim),
nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss_fn(self, x, y):
        pred = self(x)
        return mx.mean((pred - y)2)
```

### Memory-Efficient Design

1. **Gradient Checkpointing**
   - Use for large models
   - Trade computation for memory
   - Enable selectively on expensive layers

```python
from mlx_train.core import checkpoint
class TransformerLayer(nn.Module):
    def call(self, x):
        return checkpoint(self.forward, x)
```

### Model Parallelism

- Split large models across devices
- Balance computation and communication
- Use automated sharding when possible

## Model Configuration

Use structured configs for reproducibility:

```python
from dataclasses import dataclass
@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    vocab_size: int = 32000
```

## Best Practices

1. **Type Hints**
   - Use type annotations
   - Improves code readability
   - Enables better IDE support

2. **Documentation**
   - Document architecture decisions
   - Include shape assertions
   - Add docstrings for complex methods

3. **Testing**
   - Unit test core components
   - Verify shape handling
   - Test memory efficiency

4. **Performance Optimization**
   - Profile critical paths
   - Optimize data loading
   - Monitor memory usage

## Model Registry

Register models for easy instantiation:

```python
from mlx_train.models import ModelRegistry
@ModelRegistry.register("custom_transformer")
class CustomTransformer(BaseModel):
```

### Usage

```python
model = ModelRegistry.get_model("custom_transformer")(config)
```
