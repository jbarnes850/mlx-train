import os
import sys
from pathlib import Path
import pytest
import mlx.core as mx
from datasets import Dataset
from mlx_train.models import SimpleModel

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def pytest_configure(config):
    """Setup pytest configuration"""
    # Register custom markers properly
    config.addinivalue_line(
        "markers", 
        "distributed: mark test as requiring multiple devices"
    )
    config.addinivalue_line(
        "markers", "export: mark test as export functionality test"
    )
    config.addinivalue_line(
        "markers", "critical: mark test as critical path test"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Setup test environment"""
    os.environ["MLX_DISTRIBUTED"] = "1"
    os.environ["PYTHONPATH"] = str(project_root)
    yield

@pytest.fixture
def basic_config():
    """Basic test configuration"""
    return {
        "hidden_size": 128,
        "batch_size": 8,
        "memory_per_device": 8,
        "cache_dir": "test_cache",
        "model_size": 128 * 128
    }

@pytest.fixture
def test_data(basic_config):
    """Generate test data"""
    return Dataset.from_dict({
        "input_ids": mx.random.uniform(shape=(10, basic_config["hidden_size"])),
        "labels": mx.random.uniform(shape=(10, basic_config["hidden_size"]))
    })

@pytest.fixture
def mock_hardware_config():
    """Mock hardware configuration for testing"""
    return {
        "num_devices": 2,
        "total_memory_gb": 32.0,
        "total_tflops": 14.2
    }

@pytest.fixture
def mock_distributed_setup(mock_hardware_config):
    """Mock distributed training setup"""
    return {
        "world_size": mock_hardware_config["num_devices"],
        "rank": 0,
        "local_rank": 0,
        "device_ids": [0, 1]
    }

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    return SimpleModel(hidden_size=128)

@pytest.fixture
def test_config():
    """Basic test configuration"""
    return {
        "hidden_size": 128,
        "batch_size": 8,
        "memory_per_device": 8,
        "model_size": 128 * 128
    }

@pytest.fixture
def model_config():
    """Model configuration for testing"""
    return {
        "hidden_size": 128,
        "activation": "relu",
        "dropout": 0.1
    }

@pytest.fixture
def simple_model(model_config):
    """Create SimpleModel instance for testing"""
    return SimpleModel(**model_config)
    