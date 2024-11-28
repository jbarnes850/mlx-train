import mlx.core as mx
from dataclasses import dataclass
from typing import List
from .discovery import DeviceInfo, DeviceDiscovery

@dataclass
class HardwareConfig:
    num_devices: int
    btl_tcp_links: int = 4  # MLX recommended value

class HardwareManager:
    def __init__(self):
        self.discovery = DeviceDiscovery()
        
    def detect_hardware(self) -> HardwareConfig:
        """Detect available devices for distributed training"""
        devices = self.discovery.discover_devices()
        
        return HardwareConfig(
            num_devices=len(devices),
            btl_tcp_links=4  # MLX recommended value
        )
    
    def create_hostfile(self, devices: List[DeviceInfo]) -> str:
        """Create MPI hostfile content"""
        return "\n".join(f"{device.hostname} slots=1" for device in devices)