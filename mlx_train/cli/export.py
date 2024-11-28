import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import json
import shutil
from typing import Optional
import torch
from core.export import export_mlx, export_gguf

console = Console()
app = typer.Typer()

@app.command()
def export_model(
    checkpoint_path: Path,
    output_dir: Path,
    format: str = "mlx",
    quantize: bool = True
):
    """Export trained model for inference"""
    try:
        # Create export directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        console.print("[bold blue]Loading checkpoint...[/bold blue]")
        checkpoint = torch.load(checkpoint_path)
        
        # Export based on format
        if format == "mlx":
            _export_mlx(checkpoint, output_dir, quantize)
        elif format == "gguf":  # For Ollama compatibility
            _export_gguf(checkpoint, output_dir)
        
        # Export model config
        config = {
            "model_type": checkpoint["config"]["model_type"],
            "architecture": checkpoint["config"]["architecture"],
            "tokenizer": checkpoint["config"]["tokenizer"],
            "quantized": quantize
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        console.print(Panel(
            f"""
            [green]✓ Model exported successfully![/green]
            
            Location: {output_dir}
            Format: {format}
            Quantized: {quantize}
            
            To serve the model:
            [yellow]mlx-train serve {output_dir}[/yellow]
            """,
            title="Export Complete",
            expand=False
        ))
        
    except Exception as e:
        console.print(f"[bold red]Export failed: {e}[/bold red]")
        raise

@app.command()
def serve(
    model_path: Path,
    port: int = 8000,
    quantize: bool = True
):
    """Serve exported model locally"""
    from serving.server import ModelServer
    
    server = ModelServer(model_path, port)
    server.load_model(quantize=quantize)
    server.serve() 

def _export_mlx(checkpoint, output_dir, quantize):
    """Export model to MLX format"""
    return export_mlx(checkpoint, output_dir, quantize)

def _export_gguf(checkpoint, output_dir):
    """Export model to GGUF format"""
    return export_gguf(checkpoint, output_dir) 