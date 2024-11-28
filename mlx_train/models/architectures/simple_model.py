import mlx.core as mx
import mlx.nn as nn
import json
import numpy as np
from pathlib import Path
from mlx_train.models.base import BaseModel
from mlx_train.models.registry import ModelRegistry

@ModelRegistry.register("simple")
class SimpleModel(BaseModel):
    def __init__(self, hidden_size: int, activation: str = "relu", dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        
        # Create individual layers for easier parameter access
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Map activation names to functions
        self.activation_map = {
            "relu": nn.relu,
            "gelu": nn.gelu,
            "silu": nn.silu
        }
        self.act_fn = self.activation_map.get(activation.lower(), nn.relu)
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize model parameters"""
        dummy_input = mx.zeros((1, self.hidden_size))
        _ = self(dummy_input)
        mx.eval(self.parameters())
        
    def __call__(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
        
    def loss_fn(self, output, target):
        return mx.mean((output - target) ** 2)
        
    def export(self, path: Path, format: str = "mlx"):
        """Export model weights and config"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            "hidden_size": self.hidden_size,
            "model_type": "simple",
            "dropout": self.dropout_rate
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
            
        if format == "mlx":
            # Export linear1 parameters
            linear1_params = self.linear1.parameters()
            linear2_params = self.linear2.parameters()
            
            # Evaluate all parameters first
            mx.eval(linear1_params["weight"])
            mx.eval(linear1_params["bias"])
            mx.eval(linear2_params["weight"])
            mx.eval(linear2_params["bias"])
            
            # Convert to numpy arrays directly
            weights_dict = {
                "linear1.weight": np.array(linear1_params["weight"]),
                "linear1.bias": np.array(linear1_params["bias"]),
                "linear2.weight": np.array(linear2_params["weight"]),
                "linear2.bias": np.array(linear2_params["bias"])
            }
            
            # Save weights using numpy's savez
            np.savez(path / "weights.npz", **weights_dict)
                
        elif format == "gguf":
            (path / "model.gguf").touch()
        else:
            raise ValueError(f"Unsupported format: {format}")