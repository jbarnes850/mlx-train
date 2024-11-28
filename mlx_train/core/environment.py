import os
import mlx.core as mx
from pathlib import Path

class EnvironmentManager:
    def setup_distributed(self, config):
        """Setup MLX distributed environment"""
        os.environ["MLX_DISTRIBUTED"] = "1"
        os.environ["OMPI_MCA_btl_tcp_links"] = "4"
        return mx.distributed.init() 

    def setup_dependencies(self):
        try:
            # Add version checks
            import mlx
            import mpi4py
            return True
        except ImportError as e:
            raise EnvironmentError(f"Missing required dependency: {e}")