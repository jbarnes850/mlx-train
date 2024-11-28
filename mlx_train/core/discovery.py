import mlx.core as mx
from dataclasses import dataclass
from typing import List, Optional
import socket

@dataclass
class DeviceInfo:
    hostname: str
    device_id: int
    memory: float  # GB
    device_type: str

class DeviceDiscovery:
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover available MLX devices"""
        devices = []
        hostname = socket.gethostname()
        
        # For now, we assume one device per process
        # In future, we can expand this to handle multiple devices
        devices.append(DeviceInfo(
            hostname=hostname,
            device_id=0,
            memory=mx.metal.get_peak_memory() / 1e9 if mx.metal.is_available() else 0,
            device_type="metal" if mx.metal.is_available() else "cpu"
        ))
        
        return devices