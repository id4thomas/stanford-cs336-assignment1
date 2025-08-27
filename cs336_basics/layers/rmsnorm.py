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
        # for stability -> set to float32
        # weight = torch.empty(d_model, dtype=dtype, device=device)
        # initialize with ones for stability
        weight = torch.ones(d_model, dtype=torch.float32, device=device)
        self.weight = nn.Parameter(weight, requires_grad=True)
        
        self.d_model=d_model
        self.eps = eps
        
        self.device = device
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process (batch, seq, d_model)
        in_dtype = x.dtype

        # cast to float32 for stability
        x = x.to(torch.float32)
        
        # Calculate RMS
        ms = torch.sum(
            torch.square(x),
            dim=-1,
            keepdim=True
        )
        ms = ms/self.d_model
        # careful: add eps after squared mean calculation
        # rms = torch.sqrt(ms + self.eps)
        # rms = torch.sqrt(ms + self.eps)
        # use rsqrt to avoid nans
        inv_rms = torch.rsqrt(torch.clamp(ms + self.eps, min=1e-12))
        
        result = x*self.weight
        # result = result/rms
        result = result*inv_rms
        return result.to(in_dtype)