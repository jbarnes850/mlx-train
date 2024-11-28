from typing import Iterator, Tuple, Dict, Union, Optional, List
import mlx.core as mx
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
import json
from pathlib import Path
from mlx_train.utils.memory import MemoryOptimizer

console = Console()

class DatasetManager:
    """Enhanced dataset management with local data support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.memory_limit = config.get("memory_per_device", 8) * 1e9  # Convert GB to bytes
        self.cache_dir = Path(config.get("cache_dir", "cache"))
        self.cache_dir.mkdir(exist_ok=True)
        
    def setup_dataset(self) -> Dataset:
        """Interactive dataset setup with enhanced local support"""
        source = self._prompt_data_source()
        
        if source == "huggingface":
            return self._setup_hf_dataset()
        elif source == "local":
            return self._setup_local_dataset()
        else:  # synthetic
            return self._setup_synthetic_dataset()
    
    def _setup_local_dataset(self) -> Dataset:
        """Enhanced local dataset setup with format detection"""
        supported_formats = {
            ".csv": self._load_csv,
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
            ".txt": self._load_text,
            ".parquet": self._load_parquet
        }
        
        # Get data path
        path = Path(console.input("[bold blue]Enter path to your dataset: [/bold blue]"))
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
            
        # Detect format and load
        format_loader = supported_formats.get(path.suffix.lower())
        if not format_loader:
            raise ValueError(f"Unsupported file format. Supported: {list(supported_formats.keys())}")
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(f"Loading {path.suffix} dataset...", total=None)
            dataset = format_loader(path)
            
        # Validate and optimize
        dataset = self._validate_and_optimize_dataset(dataset)
        return dataset
    
    def _validate_and_optimize_dataset(self, dataset: Dataset) -> Dataset:
        """Validate and optimize dataset for MLX"""
        # Check memory requirements
        estimated_size = sum(
            x.nbytes if hasattr(x, 'nbytes') else len(str(x)) 
            for x in dataset.values()
        )
        
        if estimated_size > self.memory_limit:
            console.print("[yellow]Warning: Dataset might exceed memory limits. Enabling streaming...[/yellow]")
            dataset = dataset.to_streaming()
        
        # Validate format
        required_columns = {"input_ids", "labels"}
        if not all(col in dataset.features for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        return dataset
    
    def _load_csv(self, path: Path) -> Dataset:
        """Load and process CSV data"""
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)
    
    def _load_json(self, path: Path) -> Dataset:
        """Load and process JSON data"""
        with open(path) as f:
            data = json.load(f)
        return Dataset.from_dict(data)
    
    def _load_jsonl(self, path: Path) -> Dataset:
        """Load and process JSONL data"""
        return Dataset.from_json(str(path))
    
    def _load_text(self, path: Path) -> Dataset:
        """Load and process text data"""
        with open(path) as f:
            texts = [line.strip() for line in f]
        return Dataset.from_dict({"text": texts})
    
    def _load_parquet(self, path: Path) -> Dataset:
        """Load and process Parquet data"""
        return Dataset.from_parquet(str(path))
    
    def get_dataloader(self, dataset: Dataset) -> Iterator[Tuple[mx.array, mx.array]]:
        """Create MLX-optimized dataloader"""
        def prepare_batch(examples: Dict) -> Tuple[mx.array, mx.array]:
            x = mx.array(examples["input_ids"])
            y = mx.array(examples["labels"])
            return x, y
        
        # Calculate optimal batch size with default model size if not provided
        model_size = self.config.get("model_size", self.config["hidden_size"] * self.config["hidden_size"])
        optimal_batch = MemoryOptimizer.optimize_batch_size(
            model_size,
            self.memory_limit
        )
        actual_batch = min(self.batch_size, optimal_batch)
        
        # Create batches
        for i in range(0, len(dataset), actual_batch):
            batch = dataset[i:i + actual_batch]
            yield prepare_batch(batch)