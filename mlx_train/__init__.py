"""MLX Training Framework"""

__version__ = "0.1.0"

from mlx_train.core.hardware import HardwareConfig, HardwareManager
from mlx_train.models.base import BaseModel
from mlx_train.models.registry import ModelRegistry, ModelType
from mlx_train.models.architectures.test_model import TestModel

__all__ = [
    "HardwareConfig",
    "HardwareManager",
    "BaseModel",
    "ModelRegistry",
    "ModelType",
    "TestModel",
    "ModelBuilder",
    "DatasetManager"
] 