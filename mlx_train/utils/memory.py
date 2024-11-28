from typing import Dict, Any
import mlx.core as mx

class MemoryOptimizer:
    """Memory optimization utilities for large models"""
    
    @staticmethod
    def optimize_batch_size(model_size: int, memory_limit: int) -> int:
        """Calculate optimal batch size based on model and memory"""
        # Reserve 20% for gradients and optimizer states
        available_memory = memory_limit * 0.8
        # Estimate memory per sample (model size + activations)
        mem_per_sample = model_size * 1.5
        return max(1, int(available_memory / mem_per_sample))
    
    @staticmethod
    def load_sharded(path: str) -> Dict[str, Any]:
        """Load large models in sharded format"""
        try:
            # First try safetensors format
            return mx.load_safetensors(path)
        except:
            # Fallback to regular loading with memory checks
            if mx.metal.is_available():
                current_mem = mx.metal.get_active_memory()
                mx.metal.clear_cache()  # Clear cache before loading
            
            weights = mx.load(path)
            return weights
    
    @staticmethod
    def quantize_weights(weights: Dict[str, mx.array], bits: int = 8):
        """Quantize model weights for memory efficiency"""
        if bits not in [4, 8]:
            raise ValueError("Only 4 and 8 bit quantization supported")
            
        quantized = {}
        for name, param in weights.items():
            # Skip non-weight tensors
            if any(x in name for x in ['bias', 'norm']):
                quantized[name] = param
                continue
                
            # Quantize weights
            quantized[name] = mx.quantize(param, bits)
        
        return quantized 
    
    @staticmethod
    def suggest_config(model_size: int, num_devices: int) -> dict:
        """Suggest optimal memory configuration"""
        # Calculate memory requirements
        param_memory = model_size * 4  # 4 bytes per parameter
        
        # Account for optimizer states (e.g., Adam has 2 states per parameter)
        optimizer_memory = param_memory * 2
        
        # Estimate activation memory (rough approximation)
        activation_memory = param_memory * 0.5
        
        # Total memory per device
        total_memory = (param_memory + optimizer_memory + activation_memory) / num_devices
        
        # Convert to GB
        memory_gb = total_memory / (1024 ** 3)
        
        return {
            "estimated_memory_gb": memory_gb,
            "suggested_batch_size": max(1, int(32 / memory_gb)),  # Scale batch size with memory
            "activation_checkpointing": memory_gb > 8,  # Enable for large models
            "gradient_accumulation": max(1, int(memory_gb / 4))  # Scale with memory
        }
    
    @staticmethod
    def suggest_recovery_config(current_config: Dict) -> Dict:
        """Suggest configuration changes for recovery"""
        new_config = current_config.copy()
        
        # Reduce batch size
        new_config["batch_size"] = max(1, current_config["batch_size"] // 2)
        
        # Enable gradient checkpointing
        new_config["gradient_checkpointing"] = True
        
        # Adjust optimizer memory
        if current_config["optimizer"] == "adamw":
            new_config["optimizer_no_state"] = True
            
        return new_config