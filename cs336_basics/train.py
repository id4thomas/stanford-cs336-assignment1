from typing import IO, Any, BinaryIO

import numpy as np
import torch

# Tokenizer
from cs336_basics.model import TransformerBlock, Transformer
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.layers import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    ScaledDotProductAttention,
    SiLU,
    SwiGLU,
    softmax
)
from cs336_basics.loss import (
    cross_entropy
)
from cs336_basics.optim import(
    AdamW
)
from cs336_basics.optim.lr_scheduler import CosineAnnealingLRScheduler
from cs336_basics.utils import (
    get_batch,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint
)

def initialize_run(config):
    run_config = config.get('run', {})
    

def load_data(fpath):
    return np.memmap(
        fpath,
        dtype=np.uint16,
        mode="r"
    )

def initialize_mdoel(config):
    model = Transformer(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
    )
    return model

def initialize_optimizer(config, params, max_steps: int = -1):
    # Init Optim
    optimizer_type = config.get('optim', "adamw")
    optimizer_args = config["optim_args"]
    if optimizer_type=='adamw':
        optimizer = AdamW(
            params=params,
            lr=optimizer_args.get("lr", 1e-3),
            betas=optimizer_args.get("betas", (0.9, 0.999)),
            eps=optimizer_args.get("eps", 1e-8),
            weight_decay=optimizer_args.get("weight_decay", 1e-2),
        )
    else:
        raise ValueError(f"optimizer {optimizer_type} not defined")
    
    # Init Scheduler
    scheduler_type = config["scheduler"]
    scheduler_args = config["scheduler_args"]

    if scheduler_type=="cosine":
        warmup_ratio = scheduler_args.get("warmup_ratio", 0.01)
        warmup_steps = int(warmup_ratio*max_steps)
        scheduler = CosineAnnealingLRScheduler(
            optimizer,
            max_learning_rate=optimizer_args.get("lr", 1e-3),
            min_learning_rate=scheduler_args.get('min_learning_rate', 1e-3),
            warmup_iters=warmup_steps,
            cosine_cycle_iters=max_steps,
        )
    else:
        raise ValueError(f"scheduler {scheduler_type} not defined")
        
    return optimizer, scheduler


def train(config):
    pass

if __name__=="__main__":
    # Example config
    # mostly follow HF TrainingArgs
    config = {
        "run": {
            "name": "test",
            "out_dir": ""
        },
        "data":{
            "train_data_path": "",
            "val_data_path": "",
        },
        "model": {
            "vocab_size": 10_000,
            "context_length": 256,
            "d_model": 512,
            "num_layers": 4,
            "num_heads": 16,
            "d_ff": 1344,
            "rope_theta": 10_000
        },
        "optimizer": {
            "optim": "adamw",
            "optim_args": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2
            },
            "scheduler": "cosine",
            "scheduler_args": {
                "min_learning_rate": 0.1,
                "warmup_ratio": 0.01,
                "cosine_cycle_iters": 0.01,
            }
        },
        "training": {
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_steps": 1,
            "save_steps": 1
        }
        
    }
    
    train(config)