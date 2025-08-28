import argparse
import json
import os
import pickle
import time

import torch

from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.model import Transformer
from cs336_basics.layers import softmax

from cs336_basics.utils import load_checkpoint

def load_tokenizer(tokenizer_dir):
    vocab_path = os.path.join(tokenizer_dir, "vocab.pkl")
    merges_path = os.path.join(tokenizer_dir, "merges.pkl")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
        
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )
    return tokenizer

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

@torch.no_grad()
def generate(model, tokenizer, device="cpu"):
    prompt = " "
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).to(device)
    eos_token = tokenizer.encode("<|endoftext|>")[0]
    
    # Simple Greedy Generation
    max_tokens = 256
    generated_ids = []
    
    sampled_id = -1
    n = 0
    start = time.time()
    while n < max_tokens:
        last_logits = model(input_ids.unsqueeze(0)).squeeze(0)[-1, :]
        
        probs = softmax(last_logits, dim = -1)
        sampled_id = torch.argmax(probs, keepdims = True, dim = -1)
        input_ids = torch.cat((input_ids, sampled_id))
        
        sampled_id_val = sampled_id[0].item()
        if sampled_id_val == eos_token:
            break
        
        generated_ids.append(sampled_id_val)
        n+=1
    end = time.time()
    print("Generated IDS:", generated_ids)
    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated Text of len {len(generated_text)} in {end-start:.3f}")
    print(generated_text)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train LM")
    parser.add_argument("--tokenizer_dir", type=str, help="Path to load tokenizer from")
    parser.add_argument("--model_dir", type=str, help="Path to model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint Name")    
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    
    # Load Tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir)
    print('Loaded Tokenizer')
    
    # Load Model
    model = load_model(
        args.model_dir,
        args.checkpoint,
        device=args.device
    )
    print(f"Loaded Model weight {args.checkpoint}")
    
    model.eval()
    generate(model, tokenizer, device=args.device)