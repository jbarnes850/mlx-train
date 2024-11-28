import pytest
import mlx.core as mx
from pathlib import Path
import shutil
import json
import numpy as np
from mlx_train.models import SimpleModel

def test_model_export(simple_model, tmp_path):
    """Test model export compatibility"""
    mlx_path = tmp_path / "mlx"
    simple_model.export(mlx_path, format="mlx")
    
    # Verify files
    assert mlx_path.exists()
    assert (mlx_path / "config.json").exists()
    assert (mlx_path / "weights.npz").exists()
    
    # Verify config contents
    with open(mlx_path / "config.json") as f:
        config = json.load(f)
        assert config["hidden_size"] == simple_model.hidden_size
        assert config["model_type"] == "simple"
    
    # Verify weights with new structure
    weights = np.load(mlx_path / "weights.npz")
    expected_keys = {"linear1.weight", "linear1.bias", 
                    "linear2.weight", "linear2.bias"}
    assert set(weights.files) == expected_keys
    
    # Verify weight shapes
    assert weights["linear1.weight"].shape == (simple_model.hidden_size, simple_model.hidden_size)
    assert weights["linear1.bias"].shape == (simple_model.hidden_size,)
    assert weights["linear2.weight"].shape == (simple_model.hidden_size, simple_model.hidden_size)
    assert weights["linear2.bias"].shape == (simple_model.hidden_size,)
    
    # Test GGUF format export
    gguf_path = tmp_path / "gguf"
    simple_model.export(gguf_path, format="gguf")
    assert gguf_path.exists()
    assert (gguf_path / "model.gguf").exists()
    
    # Cleanup using tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True) 