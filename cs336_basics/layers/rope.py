import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ):
        super().__init__()
        
        # initialize inv_freq
        dim_half_range = torch.arange(0, d_k, 2, dtype=torch.int64).to(dtype=torch.float)
        dim_half_range = dim_half_range/d_k
        
        inv_freq = 1.0 / (theta**dim_half_range)
        
        # register buffer
        self.register_buffer(
            'inv_freq',
            inv_freq,
            persistent=False # setting False excludes this from state_dict
        )
        
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        '''
        x: shape (..., seq_len, d_k)
        token_positions: shape (..., seq_len)
        '''
        # Calculate Frequency
        # (batch, seq_len, d_k//2)
        freq = torch.einsum("... i, ... j->... ij", token_positions, self.inv_freq)
        
        # rotation
        cos = freq.cos()
        sin = freq.sin()
        
        # Split x (..., seq_len, d_k//2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Rotate 
        x_new_even = x_even*cos - x_odd*sin
        x_new_odd = x_even*sin + x_odd*cos
        
        # Interweave
        x_new_even = x_new_even.unsqueeze(-1)
        x_new_odd = x_new_odd.unsqueeze(-1)
        
        # stack into (..., seq_len, d_k//2, 2) -> flatten
        x_out = torch.cat([x_new_even, x_new_odd], dim=-1)
        return x_out.flatten(-2)