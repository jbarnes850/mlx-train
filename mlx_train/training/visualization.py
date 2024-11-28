from typing import Dict, List, Optional
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import mlx.core as mx
import time
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    loss: float
    learning_rate: float
    samples_per_second: float
    memory_used: float
    memory_total: float
    network_bandwidth: Optional[float] = None  # MB/s for distributed
    device_utilization: Optional[float] = None  # Percentage

class TrainingVisualizer:
    """Real-time training visualization with distributed support"""
    
    def __init__(self, num_devices: int, config: Dict):
        self.num_devices = num_devices
        self.config = config
        self.console = Console()
        self.start_time = time.time()
        self.history: List[TrainingMetrics] = []
        
    def generate_view(self, metrics: TrainingMetrics) -> Layout:
        """Generate comprehensive training view"""
        layout = Layout()
        
        # Main layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=8),
            Layout(name="resources", size=6)
        )
        
        # Header with training status
        self._update_header(layout)
        
        # Training metrics section
        self._update_metrics_section(layout, metrics)
        
        # Resource utilization section
        self._update_resource_section(layout, metrics)
        
        return layout
    
    def _update_header(self, layout: Layout):
        """Update header with training status"""
        elapsed = time.time() - self.start_time
        header = Panel(
            f"🚀 Training on {self.num_devices} device{'s' if self.num_devices > 1 else ''} | "
            f"Time: {elapsed/3600:.1f}h | "
            f"Epoch: {self.config.get('current_epoch', 0) + 1}/{self.config['num_epochs']}",
            style="bold blue"
        )
        layout["header"].update(header)
    
    def _update_metrics_section(self, layout: Layout, metrics: TrainingMetrics):
        """Update training metrics visualization"""
        # Create metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta", box=None)
        metrics_table.add_column("Metric")
        metrics_table.add_column("Current")
        metrics_table.add_column("Best")
        
        # Add core metrics
        best_loss = min([m.loss for m in self.history + [metrics]], default=metrics.loss)
        metrics_table.add_row(
            "Loss",
            f"{metrics.loss:.4f}",
            f"{best_loss:.4f}",
            style="green" if metrics.loss == best_loss else "white"
        )
        
        metrics_table.add_row(
            "Learning Rate",
            f"{metrics.learning_rate:.6f}",
            ""
        )
        
        metrics_table.add_row(
            "Throughput",
            f"{metrics.samples_per_second:.1f} samples/s",
            f"{max([m.samples_per_second for m in self.history + [metrics]], default=0):.1f} samples/s"
        )
        
        layout["metrics"].update(Panel(metrics_table, title="Training Progress"))
    
    def _update_resource_section(self, layout: Layout, metrics: TrainingMetrics):
        """Update resource utilization visualization"""
        resources = Table(show_header=True, header_style="bold cyan", box=None)
        resources.add_column("Device")
        resources.add_column("Memory")
        resources.add_column("Utilization")
        if self.num_devices > 1:
            resources.add_column("Network")
        
        # Memory usage bar
        memory_percent = (metrics.memory_used / metrics.memory_total) * 100
        memory_bar = self._create_progress_bar(memory_percent)
        
        # Device utilization bar
        if metrics.device_utilization:
            util_bar = self._create_progress_bar(metrics.device_utilization)
        else:
            util_bar = "N/A"
        
        # Add row for each device
        for device in range(self.num_devices):
            row = [
                f"Device {device}",
                f"{memory_bar} {memory_percent:.1f}%",
                f"{util_bar}"
            ]
            
            # Add network bandwidth for distributed training
            if self.num_devices > 1 and metrics.network_bandwidth:
                row.append(f"{metrics.network_bandwidth:.1f} MB/s")
            
            resources.add_row(*row)
        
        layout["resources"].update(Panel(resources, title="Resource Utilization"))
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a simple progress bar"""
        filled = int(width * percentage / 100)
        return f"[{'=' * filled}{' ' * (width - filled)}]"
    
    def save_history(self, path: Path):
        """Save training history to file"""
        history = [vars(m) for m in self.history]
        with open(path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2) 