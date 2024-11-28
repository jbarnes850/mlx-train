from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
import mlx.core as mx
import plotext as plt
from datetime import datetime

console = Console()

@dataclass
class ExperimentRun:
    id: str
    name: str
    config: Dict
    metrics: Dict[str, List[float]]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    hardware_info: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)

class ExperimentTracker:
    """Track and compare training runs"""
    
    def __init__(self, project_dir: Path):
        self.experiments_dir = project_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        self.current_run: Optional[ExperimentRun] = None
        
    def start_run(self, name: str, config: Dict):
        """Start new experiment run"""
        run_id = f"{name}_{int(time.time())}"
        
        self.current_run = ExperimentRun(
            id=run_id,
            name=name,
            config=config,
            metrics={},
            start_time=time.time(),
            hardware_info=self._get_hardware_info()
        )
        
        self._save_run()
        console.print(f"[green]Started experiment run: {run_id}[/green]")
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for current run"""
        if not self.current_run:
            raise RuntimeError("No active experiment run")
            
        for name, value in metrics.items():
            if name not in self.current_run.metrics:
                self.current_run.metrics[name] = []
            self.current_run.metrics[name].append(value)
            
        self._save_run()
        
    def end_run(self, status: str = "completed"):
        """End current experiment run"""
        if not self.current_run:
            return
            
        self.current_run.end_time = time.time()
        self.current_run.status = status
        self._save_run()
        
        duration = self.current_run.end_time - self.current_run.start_time
        console.print(f"[green]Experiment {self.current_run.id} {status} in {duration:.2f}s[/green]")
        
    def compare_runs(self, metric: str = "loss"):
        """Compare different experiment runs"""
        runs = self._load_all_runs()
        
        table = Table(title="Experiment Comparison")
        table.add_column("Run")
        table.add_column("Status")
        table.add_column(f"Best {metric}")
        table.add_column("Duration")
        
        for run in runs:
            if metric in run.metrics:
                best_metric = min(run.metrics[metric])
                duration = (run.end_time or time.time()) - run.start_time
                
                table.add_row(
                    run.name,
                    run.status,
                    f"{best_metric:.4f}",
                    f"{duration:.1f}s"
                )
                
        console.print(table)
        
        # Plot metrics
        self._plot_metrics(runs, metric)
        
    def _save_run(self):
        """Save current run to disk"""
        if not self.current_run:
            return
            
        run_file = self.experiments_dir / f"{self.current_run.id}.json"
        with open(run_file, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)
            
    def _load_all_runs(self) -> List[ExperimentRun]:
        """Load all experiment runs"""
        runs = []
        for run_file in self.experiments_dir.glob("*.json"):
            with open(run_file) as f:
                data = json.load(f)
                runs.append(ExperimentRun(**data))
        return sorted(runs, key=lambda x: x.start_time)
        
    def _get_hardware_info(self) -> Dict:
        """Get current hardware information"""
        return {
            "num_devices": mx.distributed.get_world_size(),
            "device_type": "metal" if mx.metal.is_available() else "cpu",
            "timestamp": datetime.now().isoformat()
        }
        
    def _plot_metrics(self, runs: List[ExperimentRun], metric: str):
        """Plot metrics comparison"""
        plt.clear_figure()
        
        for run in runs:
            if metric in run.metrics:
                plt.plot(run.metrics[metric], label=run.name)
                
        plt.title(f"{metric} Comparison")
        plt.show() 