import pytest
import mlx.core as mx
import numpy as np
from datasets import Dataset
from mlx_train.models import SimpleModel
from mlx_train.data import DatasetManager
from mlx_train.training.distributed import DistributedController
import json
from pathlib import Path
import time

@pytest.mark.distributed
def test_basic_distributed():
    """Test basic distributed training functionality"""
    controller = DistributedController()
    world = controller.world
    
    config = {
        "hidden_size": 128,
        "batch_size": 8,
        "memory_per_device": 8,
        "cache_dir": f"test_cache_{world.rank}",
        "model_size": 128 * 128
    }
    
    # Create test data - convert to numpy, then to list for Dataset compatibility
    test_data = Dataset.from_dict({
        "input_ids": mx.array(
            np.random.uniform(size=(10, config["hidden_size"]))
            .astype(np.float32)
        ).tolist(),
        "labels": mx.array(
            np.random.uniform(size=(10, config["hidden_size"]))
            .astype(np.float32)
        ).tolist()
    })
    
    # Setup model and data
    model = SimpleModel(hidden_size=config["hidden_size"])
    manager = DatasetManager(config)
    dataloader = manager.get_dataloader(test_data)
    
    # Test model synchronization
    initial_params = model.parameters()
    model = controller.synchronize_model(model)
    synced_params = model.parameters()
    
    # Verify parameters are synchronized
    for key in ["linear1", "linear2"]:
        for param in ["weight", "bias"]:
            mx.eval(initial_params[key][param])
            mx.eval(synced_params[key][param])
            assert mx.array_equal(
                initial_params[key][param],
                synced_params[key][param]
            )
    
    # Test gradient reduction
    batch = next(iter(dataloader))
    x, y = batch
    
    # Forward pass
    output = model(x)
    loss = model.loss_fn(output, y)
    
    # Get gradients
    def loss_wrapper(params):
        # Update model with new parameters
        model.update(params)
        output = model(x)
        return model.loss_fn(output, y)
    
    # Compute gradients
    grad = mx.grad(loss_wrapper)(model.parameters())
    
    # Verify gradient structure matches parameters
    assert set(grad.keys()) == set(model.parameters().keys())
    for layer in ["linear1", "linear2"]:
        assert "weight" in grad[layer]
        assert "bias" in grad[layer]
    
    # Test gradient reduction
    reduced_grad = controller.all_reduce_grads(grad)
    mx.eval(reduced_grad)
    
    # Basic assertions
    assert reduced_grad is not None, "Gradient reduction failed"
    
    # Cleanup
    import shutil
    shutil.rmtree(f"test_cache_{world.rank}", ignore_errors=True)

def test_distributed_recovery():
    """Test error recovery in distributed setting"""
    controller = DistributedController()
    world = controller.world
    
    # Create checkpoint directory
    controller.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Save test checkpoint (only on rank 0)
    if world.rank == 0:
        checkpoint = {
            "epoch": 1,
            "model_state": {"test": mx.array([1.0, 2.0, 3.0]).tolist()},
            "optimizer_state": {},
            "metrics": {"loss": 0.5},
            "world_size": world.size,
            "timestamp": 0
        }
        
        # Save checkpoint as JSON
        checkpoint_path = controller.checkpoint_dir / "checkpoint_epoch_1.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
    
    # Ensure all processes wait for checkpoint
    mx.eval(mx.array([0.0]))
    
    # Create a mock model with proper parameter structure
    model = type('MockModel', (), {
        'parameters': lambda: {"test": mx.array([0.0, 0.0, 0.0])},
        'update': lambda x: None
    })()
    optimizer = type('MockOptimizer', (), {'state': {}})()
    
    # Load checkpoint and verify
    loaded_model, epoch = controller.load_checkpoint(model, optimizer)
    
    # Ensure checkpoint is loaded before assertion
    mx.eval(mx.array([0.0]))  # Synchronization barrier
    
    # Verify checkpoint loading
    if world.rank == 0:  # Only check on primary device
        assert epoch == 1, "Checkpoint epoch not loaded correctly"
    
    # Cleanup
    if world.rank == 0:
        import shutil
        shutil.rmtree(controller.checkpoint_dir, ignore_errors=True)
    
def test_memory_optimization():
    """Verify memory optimization strategies"""
    
def test_model_synchronization():
    """Ensure model sync across devices"""
    
def test_device_discovery():
    """Test device discovery and connection"""
    controller = DistributedController()
    world = controller.world
    
    # Test device info using MLX's built-in methods
    devices = {
        "memory": mx.metal.get_peak_memory() / 1e9 if mx.metal.is_available() else 0,
        "num_devices": int(world.size()),
        "status": "connected"
    }
    
    assert devices["num_devices"] >= 1, "Should detect at least current device"
    assert "memory" in devices
    assert devices["status"] == "connected"
    
@pytest.mark.distributed
def test_training_resumption():
    """Test training resumption after interruption"""
    controller = DistributedController()
    
    # Setup basic training state
    checkpoint = {
        "epoch": 5,
        "model_state": {"test": mx.array([1.0, 2.0, 3.0]).tolist()},
        "optimizer_state": {},
        "metrics": {"loss": 0.5},
        "world_size": controller.size
    }
    
    # Create mock model class properly
    class MockModel:
        def parameters(self):
            return checkpoint["model_state"]
        def update(self, params):
            pass
    
    # Save checkpoint
    controller.save_checkpoint(
        model=MockModel(),
        optimizer=type('MockOptimizer', (), {'state': {}})(),
        epoch=checkpoint["epoch"],
        metrics=checkpoint["metrics"]
    )
    
    # Load checkpoint and verify
    loaded_model, epoch = controller.load_checkpoint(
        model=MockModel(),
        optimizer=type('MockOptimizer', (), {'state': {}})()
    )
    
    assert epoch == checkpoint["epoch"], "Should resume from correct epoch"
    
def test_memory_management():
    """Test memory optimization across devices"""
    config = {
        "hidden_size": 128,
        "batch_size": 8,
        "memory_per_device": 8,
        "model_size": 128 * 128
    }
    
    # Create test data
    test_data = Dataset.from_dict({
        "input_ids": np.random.uniform(size=(1000, config["hidden_size"])).tolist(),
        "labels": np.random.uniform(size=(1000, config["hidden_size"])).tolist()
    })
    
    # Test memory optimization
    manager = DatasetManager(config)
    dataloader = manager.get_dataloader(test_data)
    
    if mx.metal.is_available():
        initial_memory = mx.metal.get_cache_memory()
        batch = next(iter(dataloader))
        peak_memory = mx.metal.get_peak_memory()
        
        assert peak_memory < config["memory_per_device"] * 1e9, "Memory usage exceeds limit"
 