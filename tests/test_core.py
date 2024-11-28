import pytest
from mlx_train.core.hardware import HardwareConfig

def test_hardware_config():
    """Test basic hardware configuration"""
    config = HardwareConfig(num_devices=1)
    assert config.num_devices == 1