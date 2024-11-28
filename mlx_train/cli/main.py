import typer
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
from typing import Optional
from pathlib import Path
import os

from core.environment import EnvironmentManager
from core.hardware import HardwareManager
from core.config import ProjectConfig
from core.memory_optimizer import MemoryOptimizer

app = typer.Typer(
    help="🚀 MLX Training Framework",
    no_args_is_help=True,
    add_completion=False
)
console = Console()

def show_welcome():
    """Display beautiful welcome screen"""
    layout = Layout()
    layout.split_column(
        Layout(Panel.fit(
            """
            🔮 MLX Training Framework
            Train AI models across Apple Silicon devices with ease
            """,
            style="bold blue",
            box=box.ROUNDED
        )),
        Layout(Panel(
            """
            • Automatic hardware optimization
            • Distributed training support
            • Simple model development
            • Local model serving
            """,
            title="Features",
            box=box.ROUNDED
        ))
    )
    console.print(layout)

def handle_error(func):
    """Error handling decorator with user-friendly messages"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            console.print(Panel(
                f"[red]Error: {str(e)}[/red]\n\n"
                f"[yellow]Need help? Visit: https://github.com/ml-explore/mlx[/yellow]",
                title="❌ Error Occurred",
                style="red"
            ))
            if os.getenv("MLX_DEBUG"):
                raise
            raise typer.Exit(1)
    return wrapper

@app.command()
def launch():
    """Interactive project setup and training launcher"""
    show_welcome()
    
    # Environment setup
    with console.status("[bold blue]Checking environment..."):
        env_manager = EnvironmentManager()
        if not env_manager.verify_installation():
            if Confirm.ask("MLX environment not found. Set up now?"):
                env_manager.setup_dependencies()

def configure_project():
    """Interactive project configuration with presets"""
    console = Console()
    
    # Get project directory
    project_dir = Path(Prompt.ask(
        "Project directory",
        default="mlx-project"
    ))
    
    # Model Configuration
    model_preset = Prompt.ask(
        "Select model configuration",
        choices=[
            "small (good for testing)",
            "medium (balanced)",
            "large (requires multiple devices)",
            "custom"
        ]
    )
    
    # Map presets to configurations
    preset_configs = {
        "small": {
            "hidden_size": 768,
            "num_layers": 6,
            "num_heads": 8,
            "batch_size": 16,
            "model_type": "transformer"
        },
        "medium": {
            "hidden_size": 1024,
            "num_layers": 12,
            "num_heads": 16,
            "batch_size": 32,
            "model_type": "transformer"
        },
        "large": {
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 32,
            "batch_size": 64,
            "model_type": "transformer"
        }
    }
    
    config = preset_configs.get(model_preset.split()[0], {})
    
    # Custom configuration if selected
    if model_preset == "custom":
        config = {
            "hidden_size": int(Prompt.ask("Hidden size", default="1024")),
            "num_layers": int(Prompt.ask("Number of layers", default="12")),
            "num_heads": int(Prompt.ask("Number of attention heads", default="16")),
            "batch_size": int(Prompt.ask("Batch size", default="32")),
            "model_type": Prompt.ask(
                "Model type",
                choices=["transformer", "mlp", "custom"],
                default="transformer"
            )
        }
    
    # Training Configuration
    training_config = {
        "optimizer": Prompt.ask(
            "Select optimizer",
            choices=["adam", "adamw", "sgd"],
            default="adamw"
        ),
        "learning_rate": float(Prompt.ask(
            "Learning rate",
            default="0.0001"
        )),
        "mixed_precision": Confirm.ask(
            "Enable mixed precision training?",
            default=True
        )
    }
    
    # Hardware Optimization
    hw_manager = HardwareManager()
    hardware_config = hw_manager.detect_hardware()
    
    if hardware_config.num_devices > 1:
        training_config["distributed"] = Confirm.ask(
            f"Enable distributed training across {hardware_config.num_devices} devices?",
            default=True
        )
    
    # Memory Optimization
    memory_config = MemoryOptimizer.suggest_config(
        model_size=config["hidden_size"] * config["num_layers"],
        num_devices=hardware_config.num_devices
    )
    
    # Show Configuration Summary
    console.print("\n[bold blue]Configuration Summary:[/bold blue]")
    
    summary = Table.grid(padding=1)
    summary.add_row("Model Size:", f"{config['hidden_size'] * config['num_layers'] / 1e6:.1f}M parameters")
    summary.add_row("Batch Size:", f"{config['batch_size']} per device")
    summary.add_row("Hardware:", f"{hardware_config.num_devices} device(s)")
    summary.add_row("Memory Usage:", f"{memory_config['estimated_memory_gb']:.1f}GB per device")
    
    console.print(Panel(summary))
    
    if Confirm.ask("Save this configuration?", default=True):
        # Create ProjectConfig with all necessary parameters
        project_config = ProjectConfig(
            project_dir=project_dir,
            model_type=config["model_type"],
            hardware=hardware_config,
            batch_size=config["batch_size"],
            learning_rate=training_config["learning_rate"],
            **config,  # Include all model configuration
            **training_config  # Include all training configuration
        )
        project_config.save()
        console.print("[green]Configuration saved successfully![/green]")
    
    return project_config, training_config, hardware_config, memory_config

if __name__ == "__main__":
    app() 