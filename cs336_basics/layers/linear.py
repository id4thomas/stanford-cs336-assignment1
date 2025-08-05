import torch
import torch.nn as nn

import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        # Initialize
        ## calc stddev
        stddev = math.sqrt(2/(in_features + out_features))
        weight = torch.empty(out_features, in_features, dtype=dtype, device=device)
        weight = nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=stddev,
            a=-3.0*stddev,
            b=3.0*stddev
        )
        self.weight = nn.Parameter(weight)
        
        self.device=device
        self.dtype=dtype
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)