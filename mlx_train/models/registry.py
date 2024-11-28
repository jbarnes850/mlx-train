from enum import Enum
from typing import Type, Dict
from mlx_train.models.base import BaseModel

class ModelType(Enum):
    CUSTOM = "custom"
    LLM = "llm"
    VISION = "vision"

class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str):
        def wrapper(model_class: Type[BaseModel]):
            cls._models[model_type] = model_class
            return model_class
        return wrapper
    
    @classmethod
    def get_model(cls, model_type: str) -> Type[BaseModel]:
        if model_type not in cls._models:
            raise ValueError(f"Model type {model_type} not found in registry")
        return cls._models[model_type] 