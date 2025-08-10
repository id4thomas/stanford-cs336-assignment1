import torch

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