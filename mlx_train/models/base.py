from abc import ABC, abstractmethod
import mlx.core as mx
import mlx.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        """Forward pass of the model"""
        raise NotImplementedError
    
    def loss_fn(self, x, y):
        """Loss function for training"""
        raise NotImplementedError
    
    @classmethod
    def from_config(cls, config):
        """Create model instance from config"""
        raise NotImplementedError