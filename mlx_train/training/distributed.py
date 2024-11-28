import mlx.core as mx
from mlx.utils import tree_map
import os
from pathlib import Path
import json
import time

class DistributedController:
    def __init__(self):
        """Initialize distributed controller"""
        self.world = mx.distributed.init()
        # Call the size and rank methods to get values
        self.size = int(self.world.size())  # Add parentheses to call the method
        self.rank = int(self.world.rank())  # Add parentheses to call the method
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save training checkpoint"""
        if self.rank == 0:  # Only primary device saves
            # Convert model state to serializable format
            model_state = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in model.parameters().items()
            }
            
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state,
                "metrics": metrics,
                "world_size": self.size,
                "timestamp": time.time()
            }
            
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
            with open(path, "w") as f:
                json.dump(checkpoint, f)
                
    def load_checkpoint(self, model, optimizer):
        """Load latest checkpoint if exists"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.json"))
        if not checkpoints:
            return model, 0
            
        latest = checkpoints[-1]
        try:
            with open(latest) as f:
                checkpoint = json.load(f)
                
            # Verify world size matches
            if checkpoint["world_size"] != self.size:
                raise ValueError("Checkpoint world size doesn't match current setup")
                
            # Convert loaded state back to MLX arrays
            model_state = {
                k: mx.array(v) if isinstance(v, (list, float)) else v
                for k, v in checkpoint["model_state"].items()
            }
            
            # Update model and optimizer
            model.update(model_state)
            optimizer.state.update(checkpoint.get("optimizer_state", {}))
            
            return model, checkpoint["epoch"]
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return model, 0
    
    def all_reduce_grads(self, grads):
        """Average gradients across all devices"""
        if self.size == 1:
            return grads
            
        try:
            size_float = float(self.size)  # Convert to float for division
            return tree_map(
                lambda x: mx.distributed.all_sum(x) / size_float,
                grads
            )
        except Exception as e:
            print(f"Error in gradient reduction: {e}")
            return grads

    def synchronize_model(self, model):
        """Ensure model weights are synchronized across devices"""
        if self.size == 1:
            return model
            
        try:
            size_float = float(self.size)  # Convert to float for division
            
            def reduce_fn(x):
                return mx.distributed.all_sum(x) / size_float
                
            params = tree_map(reduce_fn, model.parameters())
            model.update(params)
            return model
        except Exception as e:
            print(f"Error synchronizing model: {e}")
            return model