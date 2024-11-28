from pydantic import BaseModel
from pathlib import Path
import json
from typing import Optional
from mlx_train.core.hardware import HardwareConfig

class ProjectConfig(BaseModel):
    project_dir: Path
    model_type: str
    hardware: HardwareConfig
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    
    def save(self, path: Optional[Path] = None):
        """Save config to JSON"""
        if path is None:
            path = self.project_dir / "config.json"
        
        config_dict = self.model_dump()
        config_dict["project_dir"] = str(config_dict["project_dir"])
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load config from JSON"""
        with open(path) as f:
            config_dict = json.load(f)
        
        config_dict["project_dir"] = Path(config_dict["project_dir"])
        return cls(**config_dict) 