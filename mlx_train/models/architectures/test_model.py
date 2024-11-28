import mlx.core as mx
import mlx.nn as nn
from mlx_train.models.base import BaseModel
from mlx_train.models.registry import ModelRegistry

@ModelRegistry.register("custom")
class TestModel(BaseModel):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.linear(x)
        
    def __call__(self, x):
        return self.forward(x)
        
    def loss_fn(self, x, y):
        output = self(x)
        return mx.mean((output - y) ** 2)
    
    @classmethod
    def from_config(cls, config):
        return cls(config["hidden_size"]) 