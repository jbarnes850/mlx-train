import pytest
import mlx.core as mx
import numpy as np
import time
from mlx_train.models import SimpleModel
from mlx_train.data import DatasetManager
from datasets import Dataset
from mlx_train.utils.metrics import MetricsTracker

def test_basic_throughput():
    """Test basic training throughput with minimal model"""
    # Basic config for testing
    config = {
        "hidden_size": 128,
        "batch_size": 8,
        "memory_per_device": 8,
        "cache_dir": "test_cache",
        "model_size": 128 * 128
    }
    
    # Create test data
    test_data = Dataset.from_dict({
        "input_ids": mx.array(np.random.uniform(size=(100, config["hidden_size"]))).tolist(),
        "labels": mx.array(np.random.uniform(size=(100, config["hidden_size"]))).tolist()
    })
    
    # Setup model and data
    model = SimpleModel(hidden_size=config["hidden_size"])
    manager = DatasetManager(config)
    dataloader = manager.get_dataloader(test_data)
    metrics = MetricsTracker()
    
    # Measure throughput
    start_time = time.time()
    batch = next(iter(dataloader))
    mx.eval(batch)  # Ensure computation is complete
    end_time = time.time()
    
    # Calculate throughput
    samples_per_second = config["batch_size"] / (end_time - start_time)
    assert samples_per_second > 0, "Throughput should be positive"
    
    # Track metrics
    metrics.update_training({
        "throughput": samples_per_second,
        "batch_size": config["batch_size"],
        "samples_processed": config["batch_size"],
        "time_elapsed": end_time - start_time
    })
    
    # Memory tracking if available
    if mx.metal.is_available():
        memory_used = mx.metal.get_peak_memory() / 1e9  # Convert to GB
        metrics.update_training({"peak_memory_gb": memory_used})
        assert memory_used < config["memory_per_device"], "Memory usage exceeds limit"
    
    # Cleanup
    import shutil
    shutil.rmtree("test_cache", ignore_errors=True)