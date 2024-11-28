import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Union
from datasets import Dataset

class DataPreprocessor:
    """Basic data preprocessing utilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def prepare_batch(self, batch: Dict) -> tuple:
        """Convert batch to MLX arrays"""
        x = mx.array(batch["input_ids"])
        y = mx.array(batch["labels"])
        return x, y
        
    def tokenize(self, texts: Union[str, List[str]]) -> Dict:
        """Basic tokenization - placeholder for now"""
        if isinstance(texts, str):
            texts = [texts]
            
        # Simple space tokenization for testing
        tokens = [text.split() for text in texts]
        return {
            "input_ids": tokens,
            "labels": tokens[1:] + [tokens[0]]  # Shift for testing
        } 