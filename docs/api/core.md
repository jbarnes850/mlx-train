# Core API Reference

## ProjectConfig

Configuration management for training projects.

```python
from mlx_train.core import ProjectConfig

config = ProjectConfig(
    project_dir="./my_project",
    model_type="transformer",
    hardware=HardwareConfig(num_devices=2),
    batch_size=32,
    learning_rate=1e-4
)

# Save configuration
config.save()

# Load existing configuration
loaded_config = ProjectConfig.load("path/to/config.json")
```

## HardwareManager

Hardware detection and optimization for MLX.

```python
from mlx_train.core import HardwareManager

# Initialize hardware manager
hw_manager = HardwareManager()

# Detect available devices
hardware_config = hw_manager.detect_hardware()
print(f"Found {hardware_config.num_devices} devices")

# Create MPI hostfile for distributed training
hostfile = hw_manager.create_hostfile(devices)
```

## MemoryOptimizer

Memory optimization utilities for large models.

```python
from mlx_train.core import MemoryOptimizer

# Calculate optimal batch size
batch_size = MemoryOptimizer.optimize_batch_size(
    model_size=1e9,  # 1B parameters
    memory_limit=8e9  # 8GB memory
)

# Get memory configuration suggestions
config = MemoryOptimizer.suggest_config(
    model_size=1e9,
    num_devices=2
)

# Load large models efficiently
weights = MemoryOptimizer.load_sharded("path/to/weights")

# Quantize weights for memory efficiency
quantized = MemoryOptimizer.quantize_weights(weights, bits=8)
```

## DeviceDiscovery

Device detection and information gathering.

```python
from mlx_train.core import DeviceDiscovery, DeviceInfo

# Initialize discovery
discovery = DeviceDiscovery()

# Get available devices
devices: List[DeviceInfo] = discovery.discover_devices()

# Device information
for device in devices:
    print(f"""
    Hostname: {device.hostname}
    Device ID: {device.device_id}
    Memory: {device.memory:.1f}GB
    Type: {device.device_type}
    """)
```

## EnvironmentManager

Environment setup and validation.

```python
from mlx_train.core import EnvironmentManager

# Initialize environment manager
env = EnvironmentManager()

# Setup distributed environment
world = env.setup_distributed(config)

# Verify dependencies
env.setup_dependencies()
```

## Export Utilities

Model export and conversion utilities.

```python
from mlx_train.core import export_mlx, export_gguf

# Export to MLX format
export_mlx(
    checkpoint="path/to/checkpoint",
    output_dir="exports",
    quantize=True
)

# Export to GGUF format (for Ollama)
export_gguf(
    checkpoint="path/to/checkpoint",
    output_dir="exports"
)
```

## Configuration Types

### HardwareConfig

```python
@dataclass
class HardwareConfig:
    num_devices: int
    btl_tcp_links: int = 4  # MLX recommended value
```

### DeviceInfo

```python
@dataclass
class DeviceInfo:
    hostname: str
    device_id: int
    memory: float  # GB
    device_type: str
```

## Error Handling

The core modules include comprehensive error handling:

```python
try:
    config = ProjectConfig.load("config.json")
except FileNotFoundError:
    print("Configuration file not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Best Practices

1. **Memory Management**
   - Always use MemoryOptimizer for large models
   - Enable quantization when possible
   - Monitor memory usage during training

2. **Distributed Setup**
   - Verify network configuration
   - Use recommended TCP link settings
   - Monitor device synchronization

3. **Configuration**
   - Save configurations for reproducibility
   - Validate before training
   - Use type hints for better IDE support

## See Also

- [Training API](training.md)
- [Model API](models.md)
- [Distributed Training Guide](../guides/distributed_training.md)
