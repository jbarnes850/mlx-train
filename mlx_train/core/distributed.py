import mlx.core as mx
from mpi4py import MPI
from typing import Dict, Any

class DistributedController:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
    def all_reduce_grads(self, grads: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """All-reduce gradients across devices"""
        for k, v in grads.items():
            # Convert to numpy for MPI communication
            np_array = v.numpy()
            self.comm.Allreduce(MPI.IN_PLACE, np_array, op=MPI.SUM)
            # Convert back to MLX array and normalize
            grads[k] = mx.array(np_array) / self.size
        return grads
    
    def broadcast(self, data: Any, root: int = 0) -> Any:
        """Broadcast data from root to all processes"""
        return self.comm.bcast(data, root=root) 