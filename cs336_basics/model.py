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
        theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        d_head = d_model//num_heads
        
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        
        self.rope = RoPE(
            theta=theta,
            d_k=d_head,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.rope,
            device=device,
            dtype=dtype
        )
        
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor]=None
    ):
        # pre-norm
        out1 = self.ln1(x)
        # attn on normalized
        out1 = self.attn(out1, token_positions=token_positions)
        # residual
        out1 = x + out1
        
        out2 = self.ln2(out1)
        out2 = self.ffn(out2)
        return out1 + out2
    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding=d_model
        )
        
        layers = []
        for _ in range(num_layers):
            layer = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, in_indices: torch.Tensor):
        # in_indices: (batch_size, seq_len)
        batch_size, seq_len = in_indices.shape
        
        # (batch, seq_len, d_model)
        input_ids = self.token_embeddings(in_indices)
        position_ids = torch.arange(0, seq_len).view(1, seq_len)
        
        out = input_ids
        for layer_i, layer in enumerate(self.layers):
            out = layer(out, token_positions=position_ids)

        out = self.ln_final(out)
        out = self.lm_head(out)
        return out
        
        