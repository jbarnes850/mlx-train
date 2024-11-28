from typing import Optional, Union, Dict
import mlx.core as mx
from mlx_train.models.registry import ModelRegistry
from mlx_train.models.base import BaseModel
from mlx_train.utils.memory import MemoryOptimizer

class ModelBuilder:
    """Enhanced model builder with memory optimizations"""
    
    @staticmethod
    def build(
        config: Dict,
        model_type: str = "custom",
        pretrained: bool = False,
        quantize: bool = False
    ) -> BaseModel:
        """Build a model with memory optimizations"""
        # Get model class from registry
        model_cls = ModelRegistry.get_model(model_type)
        
        # Create model instance
        model = model_cls.from_config(config)
        
        if pretrained:
            weights = MemoryOptimizer.load_sharded(config["pretrained_path"])
            if quantize:
                weights = MemoryOptimizer.quantize_weights(weights)
            model.update(weights)
        
        mx.eval(model.parameters())
        return model

    @staticmethod
    def estimate_model_size(model: BaseModel) -> int:
        """Estimate model memory requirements"""
        total_params = 0
        for p in model.parameters().values():
            total_params += p.size
        
        # Estimate size in bytes (assuming float32)
        return total_params * 4