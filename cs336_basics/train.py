import argparse
from datetime import datetime
import json
import os
from math import isnan
from typing import IO, Any, BinaryIO
from tqdm import tqdm

import numpy as np
import torch

# Set wandb credentials with env variable `WANDB_API_KEY`
import wandb

# Tokenizer
from cs336_basics.model import TransformerBlock, Transformer

from cs336_basics.tokenizer.bpe import BPETokenizer
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
    """Initialize wandb run loggin"""
    project = config.get('project', 'cs336-assignment1')
    run_name = config.get('name', datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    wandb.init(
        project=project,
        name=run_name
    )
    
def load_data(fpath):
    return np.memmap(
        fpath,
        dtype=np.uint16,
        mode="r"
    )

def initialize_model(
    config,
    device=None,
    dtype=None
):
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

def evaluate(model, val_dataset, context_length, batch_size, device, num_batches: int = 50):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids, target_ids = get_batch(
                val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            # Ensure correct dtypes
            # input_ids = input_ids.to(torch.int32)
            # target_ids = target_ids.to(torch.int32)

            logits = model(input_ids)                 # [B, T, V]
            B, T, V = logits.shape
            loss = cross_entropy(
                logits.view(B * T, V),
                target_ids.view(B * T),
                reduce='mean'
            )
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if len(losses) else float("nan")

def train(config):
    # Initialize
    run_config = config["run"]
    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    
    ## Initialize Data
    ### Load Data File
    train_dataset = load_data(data_config['train_data_path'])
    val_dataset = load_data(data_config['val_data_path'])
    
    ### Initialize DataLoader

    ## Initialize Model
    dtype = training_config["dtype"]
    if dtype=='bfloat16':
        dtype=torch.bfloat16
    elif dtype=='float32':
        dtype=torch.float32
    elif dtype=='float16':
        dtype=torch.float16
    else:
        raise ValueError(f"dtype {dtype} not recognized")
        
    model = initialize_model(model_config)
    
    ## Initialize Optimizer
    max_steps = training_config.get('max_steps', 1)
    optimizer, scheduler = initialize_optimizer(
        optimizer_config,
        model.parameters(),
        max_steps=max_steps
    )
    
    # Start Training
    device = training_config.get('device', 'mps')
    batch_size = training_config.get("batch_size", 8)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    clip_grad_norm = float(training_config.get("clip_grad_norm", 1.0))
    
    eval_steps = training_config.get("eval_steps", 1)
    save_steps = training_config.get("save_steps", 1)
    out_dir = training_config["out_dir"]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    eval_batches = val_dataset.shape[0] // batch_size
    
    ## Load Model to Device
    model.to(device)
    # model.to(torch.bfloat16)
    # model = torch.compile(model, backend="aot_eager")
    model.train()
    
    ## Initialize Run
    initialize_run(run_config)
    
    ## Start Training Loop
    global_step=0
    accum_step=0
    best_val = float("inf")
    for step in tqdm(range(max_steps)):
        global_step+=1
        
        # Get Batch
        input_ids, target_ids = get_batch(
            train_dataset,
            batch_size=batch_size,
            context_length=model_config['context_length'],
            device=device
        )
        # use int32 with mps (TypeError: Trying to convert UInt16 to the MPS backend but it does not have support for that dtype)
        # input_ids = input_ids.to(torch.int32)
        # target_ids = target_ids.to(torch.int32)

        
        # ===== Forward =====
        logits = model(input_ids)   # expected [B, T, V]
        print(logits.shape)
        print(logits[0][0])
        B, T, V = logits.shape
        loss = cross_entropy(
            logits.view(B * T, V),
            target_ids.view(B * T),
            reduce='mean'
        )
        
        # Scale loss for grad accumulation
        loss_accum = loss / gradient_accumulation_steps
        loss_accum.backward()
        accum_step += 1
        
        # ===== Step / clip / schedule =====
        if accum_step % gradient_accumulation_steps == 0:
            # Optional gradient clipping
            if clip_grad_norm is not None and clip_grad_norm > 0:
                gradient_clipping(model.parameters(), clip_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0

        # ===== Logging =====
        # current LR (first param group)
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        train_log = {
            "train/loss": loss.item(),
            "train/lr": current_lr,
            "train/step": global_step
        }
        print(train_log)
        wandb.log(train_log, step=global_step)


        # ===== Eval =====
        if (global_step % eval_steps) == 0:
            val_loss = evaluate(
                model=model,
                val_dataset=val_dataset,
                context_length=model_config['context_length'],
                batch_size=batch_size,
                device=device,
                num_batches=eval_batches
            )
            wandb.log({"eval/loss": val_loss, "eval/step": global_step}, step=global_step)

            # Save best checkpoint
            if val_loss < best_val and not isnan(val_loss):
                best_val = val_loss
                if out_dir:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        iteration=global_step,
                        out=os.path.join(out_dir, f"best_step_{global_step}.pt")
                    )
       
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_step"] = global_step

    

if __name__=="__main__":
    # Example config
    # mostly follow HF TrainingArgs
    # config = {
    #     "run": {
    #         "project": "test",
    #         "name": "test"
    #     },
    #     "data":{
    #         "train_data_path": "",
    #         "val_data_path": "",
    #     },
    #     "model": {
    #         "vocab_size": 10_000,
    #         "context_length": 256,
    #         "d_model": 512,
    #         "num_layers": 4,
    #         "num_heads": 16,
    #         "d_ff": 1344,
    #         "rope_theta": 10_000
    #     },
    #     "optimizer": {
    #         "optim": "adamw",
    #         "optim_args": {
    #             "lr": 1e-5,
    #             "betas": [0.9, 0.999],
    #             "eps": 1e-8,
    #             "weight_decay": 1e-2
    #         },
    #         "scheduler": "cosine",
    #         "scheduler_args": {
    #             "min_learning_rate": 0.1,
    #             "warmup_ratio": 0.01,
    #             "cosine_cycle_iters": 0.01,
    #         }
    #     },
    #     "training": {
    #         "max_steps": 1,
    #         "batch_size": 1,
    #         "gradient_accumulation_steps": 1,
    #         "eval_steps": 1,
    #         "save_steps": 1,
    #         "device": "mps",
    #         "out_dir": ""
    #     }
        
    # }
    parser = argparse.ArgumentParser(description="Train LM")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    train(config)