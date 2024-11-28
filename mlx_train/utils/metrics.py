from typing import Dict
from collections import defaultdict
import numpy as np
import json
import time
from pathlib import Path

class MetricsTracker:
    """Tracks training and validation metrics"""
    
    def __init__(self):
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
    def update_training(self, metrics: Dict):
        """Update training metrics"""
        for k, v in metrics.items():
            self.train_metrics[k].append(v)
            
    def update_validation(self, metrics: Dict):
        """Update validation metrics"""
        for k, v in metrics.items():
            self.val_metrics[k].append(v)
            
    def get_epoch_metrics(self) -> Dict:
        """Get average metrics for epoch"""
        return {
            k: np.mean(v) for k, v in self.train_metrics.items()
        }
        
    def get_validation_metrics(self) -> Dict:
        """Get validation metrics"""
        return {
            k: np.mean(v) for k, v in self.val_metrics.items()
        }
        
    def get_all_metrics(self) -> Dict:
        """Get all metrics"""
        return {
            "train": dict(self.train_metrics),
            "validation": dict(self.val_metrics)
        }
        
    def reset(self):
        """Reset metrics for new epoch"""
        self.train_metrics.clear()
        self.val_metrics.clear() 
        
    def save_progress(self, path: Path):
        """Save training progress for analysis"""
        metrics = {
            "train": {k: np.array(v).tolist() for k, v in self.train_metrics.items()},
            "validation": {k: np.array(v).tolist() for k, v in self.val_metrics.items()},
            "timestamp": time.time(),
            "total_samples": self.total_samples
        }
        
        with open(path, "w") as f:
            json.dump(metrics, f) 

class DistributedMetricsTracker:
    """Track and validate distributed training metrics"""
    
    def __init__(self, num_devices: int):
        self.num_devices = num_devices
        self.device_metrics = {i: {} for i in range(num_devices)}
        
    def update(self, device_id: int, metrics: Dict):
        """Update metrics for a specific device"""
        self.device_metrics[device_id] = metrics
        
    def check_sync(self) -> bool:
        """Check if devices are in sync"""
        if not self.device_metrics:
            return True
            
        # Compare loss values across devices
        losses = [m.get('loss', 0) for m in self.device_metrics.values()]
        max_diff = max(losses) - min(losses)
        
        return max_diff < 1e-6  # Tolerance threshold