# Training API Reference

## Overview

The Training API provides core functionality for model training, data loading, and optimization in MLX. This API is designed for efficient distributed training on Apple Silicon devices.

## Trainer

Core training loop and optimization management.

```python
from mlx_train.training import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device_config=hw_config,
    batch_size=32,
    gradient_accumulation=4
)

# Train for multiple epochs
trainer.train(
    train_dataset=train_data,
    val_dataset=val_data,
    epochs=10,
    checkpoint_dir="checkpoints"
)

# Evaluate model
metrics = trainer.evaluate(test_dataset)
```

## DataLoader

Efficient data loading for distributed training.

```python
from mlx_train.training import DataLoader

loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    # Process batch
    loss = model(batch)
```

## Callbacks

Training callbacks for monitoring and control.

```python
from mlx_train.training import ModelCheckpoint, EarlyStopping

callbacks = [
    ModelCheckpoint(
        filepath="best_model.pt",
        monitor="val_loss",
        save_best_only=True
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3
    )
]

trainer.train(callbacks=callbacks)
```

## Distributed Training

Support for multi-device training configurations.

```python
from mlx_train.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    optimizer=optimizer,
    device_config=hw_config,
    strategy="data_parallel"
)

# Train across multiple devices
trainer.train(train_dataset, val_dataset)
```

## Best Practices

1. **Gradient Accumulation**
   - Use for larger effective batch sizes
   - Helps with memory constraints
   - Monitor memory usage

2. **Checkpointing**
   - Save checkpoints regularly
   - Use validation metrics for best model selection
   - Implement backup strategies

3. **Performance Optimization**
   - Profile training loops
   - Optimize data loading
   - Monitor device utilization

4. **Distributed Training**
   - Configure device communication properly
   - Balance load across devices
   - Monitor synchronization overhead

## Error Handling

```python
try:
    trainer.train(train_dataset, val_dataset)
except OutOfMemoryError:
    print("Reduce batch size or enable gradient accumulation")
except DeviceError as e:
    print(f"Device error occurred: {e}")
```

## See Also

- [Core API](core.md)
- [Model API](models.md)
- [Memory Optimization Guide](../guides/memory_optimization.md)
