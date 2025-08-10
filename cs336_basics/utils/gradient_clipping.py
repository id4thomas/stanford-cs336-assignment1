from collections.abc import Iterable

import torch 

@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    # warning; 'norm' is across 'all' parameters
    gradients = [
        p.grad for p in parameters if not p.grad is None
    ]
    norm = torch.linalg.vector_norm(torch.stack(gradients), ord=2)
    
    for p in parameters:
        if p.grad is None:
            continue
        coef = max_l2_norm / (norm + eps)
        coef = torch.clamp(coef, max=1.0) # set max to 1.0
        p.grad.mul_(coef)  # in-place scale
    return