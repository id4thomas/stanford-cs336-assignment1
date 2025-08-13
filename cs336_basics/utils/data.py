import random

import numpy as np
import numpy.typing as npt

import torch

def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    sampled input sequences
    corresponding next-token targets"""
    seq_len = x.shape[0]
    start_idxs = list(range(seq_len-context_length))
    
    # Sample start_idxs
    start_idxs = random.sample(start_idxs, k=batch_size)
    
    # Make input, target ids
    batch_input_ids = []
    batch_target_ids = []
    for idx in start_idxs:
        input_ids = x[idx:idx+context_length]
        target_ids = x[idx+1:idx+1+context_length]
        # print(input_ids, len(input_ids), type(input_ids))
        # print(target_ids, len(target_ids), type(target_ids))
        
        batch_input_ids.append(input_ids)
        batch_target_ids.append(target_ids)

    batch_input_ids = np.stack(batch_input_ids, axis=0)
    batch_target_ids = np.stack(batch_target_ids, axis=0)
    
    input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    target_ids = torch.tensor(batch_target_ids, dtype=torch.long, device=device)
    return input_ids, target_ids
        