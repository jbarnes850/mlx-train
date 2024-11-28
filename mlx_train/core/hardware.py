import mlx.core as mx
from dataclasses import dataclass
from typing import List
from .discovery import DeviceInfo, DeviceDiscovery

@dataclass
class HardwareConfig:
    def __init__(self):
        """Initialize hardware config"""
        self.num_devices = 1  # Default to single device for now
        self.total_memory_gb = self._get_total_memory()
        self.total_tflops = self._calculate_tflops()
    
    def _get_total_memory(self):
        """Get total available memory in GB"""
        if mx.metal.is_available():
            return mx.metal.get_peak_memory() / 1e9
        return 8.0  # Default to 8GB
    
    def _calculate_tflops(self):
        """Calculate approximate TFLOPS"""
        return 14.2  # Default for M1 Max/Pro

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