import math
from typing import Optional

import torch
import torch.nn as nn

from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rope import RoPE

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    '''
    x_max = x.max(dim=dim, keepdim=True).values
    x_ = x - x_max
    x_exp = torch.exp(x_)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    x_exp = x_exp/x_exp_sum
    return x_exp

def make_causal_attention_mask(seq_len: int) -> torch.Tensor:
    '''
    ex.
    tensor([[ True, False, False],
        [ True,  True, False],
        [ True,  True,  True]])
    '''
    causal_mask = torch.ones(seq_len, seq_len)
    causal_mask = torch.triu(causal_mask, diagonal=1)==0
    return causal_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor
    ):
        '''
        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        '''
        # QK^T (..., seq_len, seq_len)
        qk = Q@K.transpose(-2, -1)
        
        # normalize
        qk/=math.sqrt(Q.shape[-1])
        
        # mask
        mask_values = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))
        qk += mask_values
        
        # softmax
        qk = softmax(qk, dim=-1)
        
        return qk@V
        
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Optional[RoPE] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Init Weights
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
                
        self.sdpa = ScaledDotProductAttention()
        self.rope = rope
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # make causal attention mask
        mask = make_causal_attention_mask(seq_len)
        
        # calc linear first 'then' view
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k)
        ## permute
        # (batch, seq, n_h, d_k) -> (batch, n_h, seq, d_k)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
            
        if not self.rope is None:
            # rope should be applied to Query, Key
            # each head should apply rope separately
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k)
        V = V.permute(0, 2, 1, 3)
        
        # (batch, seq, n_h, d_k)
        out = self.sdpa(Q, K, V, mask=mask)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch, seq_len, -1)
        
        out = self.output_proj(out)
        return out
        
        
        
        
        