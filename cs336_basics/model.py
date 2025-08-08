from typing import Optional

import torch
import torch.nn as nn

from cs336_basics.layers import *

class TransformerBlock(nn.Module):
    '''Pre-norm Transformer block'''
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float
    ):
        super().__init__()
        d_head = d_model//num_heads
        
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        
        self.rope = RoPE(
            theta=theta,
            d_k=d_head,
            max_seq_len=max_seq_len
        )
        
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.rope
        )
        
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff
        )
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor]=None
    ):
        out1 = self.ln1(x)
        # attn on normalized
        out1 = self.attn(out1, token_positions=token_positions)
        # residual
        out1 = x + out1
        
        out2 = self.ln2(out1)
        out2 = self.ffn(out2)
        return out1 + out2