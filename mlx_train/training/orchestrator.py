import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict
from pathlib import Path
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import numpy as np
from rich.prompt import Confirm
from mlx_train.training.visualization import TrainingMetrics, TrainingVisualizer
from core.distributed import DistributedController
from data.manager import DatasetManager

console = Console()

class TrainingOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.distributed = DistributedController()
        
        # Initialize components
        self.model = self._build_model()
        self.optimizer = self._setup_optimizer()
        self.dataset = DatasetManager(config)
        
        # Training state
        self.start_time = None
        self.samples_processed = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def train(self):
        """Run training with live monitoring"""
        visualizer = TrainingVisualizer(
            num_devices=self.distributed.size,
            config=self.config
        )
        
        with Live(refresh_per_second=1) as live:
            for epoch in range(self.current_epoch, self.config["num_epochs"]):
                try:
                    # Training loop
                    metrics = TrainingMetrics(
                        loss=self.current_loss,
                        learning_rate=self.optimizer.learning_rate,
                        samples_per_second=self.samples_processed / (time.time() - self.start_time),
                        memory_used=mx.metal.get_active_memory() / 1e9,  # GB
                        memory_total=self.config["hardware"]["memory_per_device"],
                        device_utilization=self._get_device_utilization(),
                        network_bandwidth=self._get_network_bandwidth() if self.distributed.size > 1 else None
                    )
                    
                    # Update visualization
                    live.update(visualizer.generate_view(metrics))
                    visualizer.history.append(metrics)
                    
                except Exception as e:
                    self._handle_training_error(e)
    
    def _train_epoch(self) -> Dict:
        """Train single epoch with progress tracking"""
        total_loss = 0
        num_batches = 0
        
        for batch in self.dataset.get_batches():
            # Forward pass and loss
            loss, grads = self._compute_loss_and_grads(batch)
            
            # All-reduce gradients
            grads = self.distributed.all_reduce_grads(grads)
            
            # Update model
            self.optimizer.apply_gradients(self.model, grads)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.samples_processed += len(batch[0])
            
            # Evaluate gradients
            mx.eval(self.model.parameters())
            
        return {
            "loss": total_loss / num_batches,
            "samples_per_second": self.samples_processed / (time.time() - self.start_time)
        }
    
    def _generate_training_view(self) -> Layout:
        """Generate comprehensive training view"""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats"),
            Layout(name="memory", size=3)
        )
        
        # Header with training status
        layout["header"].update(Panel(
            f"🚀 Training on {self.distributed.size} device(s) | Epoch {self.current_epoch + 1}/{self.config['num_epochs']}",
            style="bold blue"
        ))
        
        # Training statistics
        stats_table = Table(show_header=True, header_style="bold")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        elapsed = time.time() - self.start_time
        samples_per_sec = self.samples_processed / elapsed if elapsed > 0 else 0
        
        stats_table.add_row("Best Loss", f"{self.best_loss:.4f}")
        stats_table.add_row("Samples/Second", f"{samples_per_sec:.1f}")
        stats_table.add_row("Total Samples", str(self.samples_processed))
        stats_table.add_row("Time Elapsed", f"{elapsed/3600:.1f}h")
        
        layout["stats"].update(stats_table)
        
        # Memory usage (per device)
        if mx.metal.is_available():
            memory_table = Table(title="Memory Usage")
            memory_table.add_column("Device")
            memory_table.add_column("Used")
            memory_table.add_column("Total")
            
            memory_used = mx.metal.get_active_memory() / 1e9  # Convert to GB
            memory_total = self.config["hardware"]["memory_per_device"]
            memory_percent = (memory_used / memory_total) * 100
            
            memory_table.add_row(
                f"Device {self.distributed.rank}",
                f"{memory_used:.1f}GB",
                f"{memory_total}GB ({memory_percent:.1f}%)"
            )
            
            layout["memory"].update(memory_table)
        
        return layout
    
    def _check_training_health(self, metrics: Dict) -> bool:
        """Basic training health checks"""
        if not metrics:
            return False
        
        # Check for NaN loss
        if np.isnan(metrics["loss"]):
            console.print("[red]Warning: NaN loss detected[/red]")
            return False
        
        # Check for zero gradients
        if metrics.get("grad_norm", 1.0) < 1e-7:
            console.print("[yellow]Warning: Very small gradients detected[/yellow]")
        
        return True