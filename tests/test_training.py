import pytest
import mlx.core as mx
from mlx_train.models import SimpleModel

def test_model_creation(simple_model):
    """Test basic model creation"""
    assert simple_model is not None
    assert isinstance(simple_model, SimpleModel)
    
    # Test parameter initialization
    params = simple_model.parameters()
    assert "linear1" in params
    assert "linear2" in params
    for layer in ["linear1", "linear2"]:
        assert "weight" in params[layer]
        assert "bias" in params[layer]
        # Verify parameters are initialized
        mx.eval(params[layer]["weight"])
        mx.eval(params[layer]["bias"])