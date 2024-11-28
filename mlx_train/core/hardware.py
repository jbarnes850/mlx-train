import mlx.core as mx
from dataclasses import dataclass
from typing import List
from .discovery import DeviceInfo, DeviceDiscovery

@dataclass
class HardwareConfig:
    def __init__(self):
        """Initialize hardware config"""
        self.num_devices = 1  # Default to single device
        self.total_memory_gb = self._get_total_memory()
        self.total_tflops = self._calculate_tflops()
        self.device_type = self._detect_device_type()
        
    def _get_total_memory(self):
        """Get total available memory in GB"""
        try:
            if mx.metal.is_available():
                # Convert bytes to GB with proper rounding
                memory = float(mx.metal.get_peak_memory()) / (1024 * 1024 * 1024)
                return round(memory, 1)
        except Exception:
            pass
        # Fallback: detect system memory
        import psutil
        return round(psutil.virtual_memory().total / (1024 * 1024 * 1024), 1)
    
    def _detect_device_type(self):
        """Detect specific Apple Silicon chip and variant"""
        import subprocess
        try:
            cmd = ["system_profiler", "SPHardwareDataType"]
            output = subprocess.check_output(cmd).decode()
            
           
            if "M4" in output:  
                if "Max" in output:
                    return "M4 Max"
                elif "Pro" in output:
                    return "M4 Pro"
                return "M4"
            elif "M3" in output:
                if "Max" in output:
                    return "M3 Max"
                elif "Pro" in output:
                    return "M3 Pro"
                return "M3"
            elif "M2" in output:
                if "Max" in output:
                    return "M2 Max"
                elif "Pro" in output:
                    return "M2 Pro"
                elif "Ultra" in output:
                    return "M2 Ultra"
                return "M2"
            elif "M1" in output:
                if "Max" in output:
                    return "M1 Max"
                elif "Pro" in output:
                    return "M1 Pro"
                elif "Ultra" in output:
                    return "M1 Ultra"
                return "M1"
        except:
            pass
        return "Apple Silicon"
    
    def _calculate_tflops(self):
        """Calculate approximate TFLOPS based on device type"""
        
        device_tflops = {
            # M4 Family 
            "M4": 11.0,  
            "M4 Pro": 19.0,  
            "M4 Max": 40.0,  
            
            # M3 Family
            "M3": 16.5,
            "M3 Pro": 20.0,
            "M3 Max": 40.0,
            
            # M2 Family
            "M2": 15.8,
            "M2 Pro": 19.0,
            "M2 Max": 23.0,
            "M2 Ultra": 46.0,
            
            # M1 Family
            "M1": 11.0,
            "M1 Pro": 11.0,
            "M1 Max": 16.0,
            "M1 Ultra": 32.0,
            
            "Apple Silicon": 14.2  # Default fallback
        }
        return device_tflops.get(self._detect_device_type(), 14.2)
    
    def get_memory_bandwidth(self):
        """Get memory bandwidth in GB/s"""
        device_bandwidth = {
            # M4 Family
            "M4": 120,      
            "M4 Pro": 273,  
            "M4 Max": 546,  
            
            # M3 Family
            "M3": 100,
            "M3 Pro": 200,
            "M3 Max": 400,
            
            # M2 Family
            "M2": 100,
            "M2 Pro": 200,
            "M2 Max": 400,
            "M2 Ultra": 800,
            
            # M1 Family
            "M1": 68.25,
            "M1 Pro": 200,
            "M1 Max": 400,
            "M1 Ultra": 800,
            
            "Apple Silicon": 100  # Default fallback
        }
        return device_bandwidth.get(self._detect_device_type(), 100)

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