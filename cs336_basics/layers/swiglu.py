'''
# SwiGLU
SwiGLU = SiLU (Swish) Activation + GLU
* SiLU (Swish): sigmoid(x)*x, 
* GLU: Gated Linear Unit
    * element-wise product of 'linear transform' & 'linear transform + sigmoid
'''
    
import torch
import torch.nn as nn

from cs336_basics.layers.linear import Linear

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)*x
    
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        self.silu = SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.silu(self.w1(x))
        out2 = self.w3(x)
        out = self.w2(out1*out2)
        return out
        
    