import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.model import Transformer
from cs336_basics.loss import cross_entropy
from cs336_basics.utils import (
    get_batch,
    load_checkpoint
)

def initialize_model(
    config,
    device=None,
):
    dtype = config.get("dtype", "float32")
    if dtype=='bfloat16':
        dtype=torch.bfloat16
    elif dtype=='float32':
        dtype=torch.float32
    elif dtype=='float16':
        dtype=torch.float16
    else:
        raise ValueError(f"dtype {dtype} not recognized")
    
    model = Transformer(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        device=device,
        dtype=dtype
    )
    model.to(device)
    return model

def load_model(model_dir, weight_name, device="cpu"):
    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        model_config = json.load(f)
    model = initialize_model(model_config, device=device)
    model = torch.compile(model, backend="aot_eager")
    
    # Load Checkpoint
    weight_dir = os.path.join(model_dir, f"{weight_name}.pt")
    load_checkpoint(weight_dir, model)
    return model

def load_data(data_dir):
    return np.memmap(
        data_dir,
        dtype=np.uint16,
        mode="r"
    )

def get_eval_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
)-> tuple[torch.Tensor, torch.Tensor]:
    seq_len = dataset.shape[0]
    
    for i in tqdm(range(0, seq_len-context_length, batch_size)):
        start = i
        end = min(i+batch_size, seq_len-context_length-1)
        
        start_idxs = np.array(range(start, end))
        indices = start_idxs[:, np.newaxis] + np.arange(context_length)
        
        batch_input_ids = dataset[indices]
        batch_target_ids = dataset[indices + 1]
        
        input_ids = torch.from_numpy(batch_input_ids).to(device=device, dtype=torch.long)
        target_ids = torch.from_numpy(batch_target_ids).to(device=device, dtype=torch.long)
        yield input_ids, target_ids

@torch.no_grad()    
def calculate_loss(model, input_ids, target_ids):
    logits = model(input_ids)                 # [B, T, V]
    
    B, T, V = logits.shape
    loss = cross_entropy(
        logits.view(B * T, V),
        target_ids.view(B * T),
        reduce='mean'
    )
    return loss.item()

def evaluate(model, dataset, batch_size, num_batches, device="cpu"):
    context_length = model.context_length
    
    model.eval()
    
    loss = 0.0
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            input_ids, target_ids = get_batch(
                dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            batch_loss = calculate_loss(model, input_ids, target_ids)
            loss += batch_loss

    loss /= num_batches
    return loss

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train LM")
    parser.add_argument("--data_dir", type=str, help="Path to evaluation dataset")
    parser.add_argument("--model_dir", type=str, help="Path to model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint Name")    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of Eval Batches to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    
    # Load Model
    model = load_model(
        args.model_dir,
        args.checkpoint,
        device=args.device
    )
    print(f"Loaded Model weight {args.checkpoint}")
    
    # Load Dataset
    dataset = load_data(args.data_dir)
    print(f'Loaded dataset {args.data_dir}')
    
    # Evaluate
    loss = evaluate(
        model,
        dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=args.device
    )
    print('Eval Loss: {:.5f}'.format(loss))