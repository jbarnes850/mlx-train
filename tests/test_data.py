import pytest
import mlx.core as mx
import numpy as np
from mlx_train.data import DatasetManager, DataPreprocessor
from datasets import Dataset

def test_dataset_basic(basic_config):
    """Test basic dataset functionality"""
    manager = DatasetManager(basic_config)
    preprocessor = DataPreprocessor(basic_config)
    
    # Create minimal test data - convert MLX arrays to numpy
    test_data = Dataset.from_dict({
        "input_ids": mx.array(np.random.uniform(size=(10, basic_config["hidden_size"]))).tolist(),
        "labels": mx.array(np.random.uniform(size=(10, basic_config["hidden_size"]))).tolist()
    })
    
    # Test basic dataloader
    dataloader = manager.get_dataloader(test_data)
    batch = next(iter(dataloader))
    
    # Verify basic batch properties
    x, y = batch
    assert isinstance(x, mx.array)
    assert isinstance(y, mx.array)
    assert x.shape[0] == basic_config["batch_size"]
    
    # Test preprocessor
    sample_text = ["hello world", "testing mlx"]
    processed = preprocessor.tokenize(sample_text)
    assert "input_ids" in processed
    assert "labels" in processed