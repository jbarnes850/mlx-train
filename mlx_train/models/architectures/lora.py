import mlx.core as mx
import mlx.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.scale = 1.0

    def __call__(self, x):
        return self.lora_b(self.lora_a(x)) * self.scale 