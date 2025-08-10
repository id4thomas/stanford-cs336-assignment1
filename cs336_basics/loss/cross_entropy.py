import torch

from cs336_basics.layers import softmax

def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce: str='mean'
):
    '''take predicted logits and compute cross entropy
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.
    
    Apply logsumexp trick
    * correct_logit - logsumexp(logits)
    '''
    input_ = input - input.max(dim=-1, keepdim=True).values
    logsumexp = torch.exp(input_).sum(dim=-1).log()
    
    # use torch.gather
    correct_logits = torch.gather(
        input_,
        dim=-1,
        index=target.unsqueeze(dim=1)
    ).squeeze(dim=1)
    loss = logsumexp - correct_logits
    if reduce=='mean':
        loss = loss.mean()
    elif reduce=='sum':
        loss = loss.sum()
    return loss
    