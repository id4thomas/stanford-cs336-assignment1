import random

import numpy as np
import numpy.typing as npt

import torch

# def get_batch(
#     x: npt.NDArray,
#     batch_size: int,
#     context_length: int,
#     device: str
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns
#     sampled input sequences
#     corresponding next-token targets"""
#     seq_len = x.shape[0]
#     start_idxs = list(range(seq_len-context_length))
    
#     # Sample start_idxs
#     start_idxs = random.sample(start_idxs, k=batch_size)
    
#     # Make input, target ids
#     batch_input_ids = []
#     batch_target_ids = []
#     for idx in start_idxs:
#         input_ids = x[idx:idx+context_length]
#         target_ids = x[idx+1:idx+1+context_length]
#         # print(input_ids, len(input_ids), type(input_ids))
#         # print(target_ids, len(target_ids), type(target_ids))
        
#         batch_input_ids.append(input_ids)
#         batch_target_ids.append(target_ids)

#     batch_input_ids = np.stack(batch_input_ids, axis=0)
#     batch_target_ids = np.stack(batch_target_ids, axis=0)
    
#     input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
#     target_ids = torch.tensor(batch_target_ids, dtype=torch.long, device=device)
#     return input_ids, target_ids

import numpy as np
import torch
import numpy.typing as npt

def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    sampled input sequences
    corresponding next-token targets
    """
    seq_len = x.shape[0]
    
    # 1. Efficiently sample random starting indices directly into a NumPy array
    start_idxs = np.random.randint(0, seq_len - context_length, size=(batch_size,))

    # 2. Use broadcasting to create a 2D matrix of indices for the entire batch
    # This replaces the entire for-loop
    # Shape: (batch_size, context_length)
    indices = start_idxs[:, np.newaxis] + np.arange(context_length)
    
    # 3. Use the index matrix to grab all sequences at once. This is extremely fast.
    batch_input_ids = x[indices]
    batch_target_ids = x[indices + 1] # Targets are simply the next token

    # 4. Convert to PyTorch tensors and move to the target device
    # Use torch.from_numpy for efficiency and .to() to move and change type
    input_ids = torch.from_numpy(batch_input_ids).to(device=device, dtype=torch.long)
    target_ids = torch.from_numpy(batch_target_ids).to(device=device, dtype=torch.long)
    
    return input_ids, target_ids