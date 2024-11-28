from pathlib import Path
from typing import Dict, Optional
import shutil
import typer
from rich.console import Console
from rich.prompt import Prompt

console = Console()

class ProjectTemplate:
    """Project template manager with example configurations"""
    
    TEMPLATES = {
        "minimal": {
            "description": "Basic model training setup",
            "config": {
                "model_type": "custom",
                "hidden_size": 768,
                "num_layers": 6,
                "num_heads": 8
            }
        },
        "transformer": {
            "description": "Transformer model with attention",
            "config": {
                "model_type": "transformer",
                "hidden_size": 1024,
                "num_layers": 12,
                "num_heads": 16
            }
        },
        "distributed": {
            "description": "Multi-device training setup",
            "config": {
                "model_type": "distributed",
                "hidden_size": 2048,
                "num_layers": 24,
                "num_heads": 32
            }
        }
    }
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.template_dir = Path(__file__).parent.parent / "templates"
    
    def create_project(self, template_name: str = "minimal"):
        """Create new project from template"""
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Template {template_name} not found")
            
        template = self.TEMPLATES[template_name]
        
        # Create project structure
        self._create_directories()
        self._copy_template_files(template_name)
        self._create_config(template["config"])
        
        console.print(f"[green]✓ Created new project from {template_name} template[/green]")
        console.print(f"[blue]Description: {template['description']}[/blue]")
        
    def _create_directories(self):
        """Create project directory structure"""
        dirs = [
            "data",
            "models",
            "configs",
            "experiments",
            "checkpoints"
        ]
        for dir_name in dirs:
            (self.project_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def _copy_template_files(self, template_name: str):
        """Copy template files to project directory"""
        template_path = self.template_dir / template_name
        if template_path.exists():
            for file in template_path.glob("**/*"):
                if file.is_file():
                    dest = self.project_dir / file.relative_to(template_path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest) 