from dataclasses import dataclass
import subprocess
import socket
import platform
from typing import List, Optional
import netifaces
import zeroconf

@dataclass
class DeviceInfo:
    hostname: str
    ip_address: str
    device_type: str
    memory_gb: float
    status: str = "available"
    
class DeviceDiscovery:
    SERVICE_TYPE = "_mlx-train._tcp.local."
    
    def __init__(self):
        self.zc = zeroconf.Zeroconf()
        
    def discover_devices(self) -> List[DeviceInfo]:
        """Discover available Apple Silicon devices on the network"""
        devices = []
        
        # Add local device first
        local_device = self._get_local_device()
        if local_device:
            devices.append(local_device)
        
        # Discover network devices
        network_devices = self._discover_network_devices()
        devices.extend(network_devices)
        
        return devices
    
    def _get_local_device(self) -> Optional[DeviceInfo]:
        """Get information about the local device"""
        try:
            # Get system info
            cmd = ["system_profiler", "SPHardwareDataType"]
            output = subprocess.check_output(cmd).decode()
            
            # Parse device type and memory
            device_type = self._parse_device_type(output)
            memory_gb = self._parse_memory(output)
            
            # Get hostname and IP
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            return DeviceInfo(
                hostname=hostname,
                ip_address=ip_address,
                device_type=device_type,
                memory_gb=memory_gb,
                status="active"
            )
        except Exception as e:
            print(f"Warning: Could not get local device info: {e}")
            return None
    
    def _discover_network_devices(self) -> List[DeviceInfo]:
        """Discover Apple Silicon devices on the network"""
        devices = []
        
        # Get all network interfaces
        for interface in netifaces.interfaces():
            try:
                # Skip loopback
                if interface == "lo" or interface == "lo0":
                    continue
                    
                # Get interface addresses
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET not in addrs:
                    continue
                
                # Scan network
                network = addrs[netifaces.AF_INET][0]
                ip = network['addr']
                netmask = network['netmask']
                
                # Use zeroconf for service discovery
                browser = zeroconf.ServiceBrowser(
                    self.zc,
                    self.SERVICE_TYPE,
                    handlers=[self._on_service_state_change]
                )
                
            except Exception as e:
                print(f"Warning: Could not scan interface {interface}: {e}")
                continue
        
        return devices
    
    def _on_service_state_change(self, zc: zeroconf.Zeroconf, 
                               service_type: str,
                               name: str, 
                               state_change: zeroconf.ServiceStateChange) -> None:
        """Handle discovered MLX training services"""
        if state_change is zeroconf.ServiceStateChange.Added:
            info = zc.get_service_info(service_type, name)
            if info:
                # Parse device info from service properties
                properties = info.properties
                device = DeviceInfo(
                    hostname=properties.get(b'hostname', b'').decode(),
                    ip_address=socket.inet_ntoa(info.addresses[0]),
                    device_type=properties.get(b'device_type', b'').decode(),
                    memory_gb=float(properties.get(b'memory_gb', 0)),
                    status="available"
                )
                self.devices.append(device)