import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        # Make gain parameter
        gain = torch.empty(d_model, dtype=dtype, device=device)
        self.gain = nn.Parameter(gain, requires_grad=True)
        
        self.d_model=d_model
        self.eps = eps
        
        self.device = device
        self.dtype=dtype
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process (batch, seq, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Calculate RMS
        ms = torch.sum(
            torch.square(x),
            dim=-1,
            keepdim=True
        ) + self.eps
        ms = ms/self.d_model
        rms = torch.sqrt(ms)
        
        result = x*self.gain
        result = result/rms
        return result.to(in_dtype)