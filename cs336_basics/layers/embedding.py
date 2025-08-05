import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        # Initialize
        weight = torch.empty(num_embeddings, embedding)
        weight = nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)
        
        self.device=device
        self.dtype=dtype
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        return self.weight[token_ids, :]