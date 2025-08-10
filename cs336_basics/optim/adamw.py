from collections.abc import Callable, Iterable
import math
from typing import List, Optional, Union

import torch
from torch.optim import (
    Optimizer
)
from torch.optim.optimizer import ParamsT


class AdamW(Optimizer):
    """AdamW Optimizer
    Stateful: for each param, keep track of running estimate of 1st, 2nd moments
    """
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}".format(lr))
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(
        self,
        closure: Optional[Callable]=None
    ):
        # closure: function that re-evalutes model and returns loss
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p] # get state associated with p
                grad = p.grad.data # get gradient of loss wrt p
                
                # Update 1st moment estimate
                m = state.get("m", 0.0)
                m = beta1*m + (1-beta1)*grad
                state["m"] = m
                
                # Update 2nd moment estimate
                v = state.get("v", 0.0)
                v = beta2*v + (1-beta2)*torch.square(grad)
                state["v"] = v
                
                # Update Params
                t = state.get("t", 0) # get iteration number
                state["t"] = t+1
                
                ## compute adjusted lr
                adjusted_lr = lr * (math.sqrt(1 - beta2**(t+1)))
                adjusted_lr = adjusted_lr / (1 - beta1**(t+1))
                p.data -= adjusted_lr*m*(1/(torch.sqrt(v) + eps))
                
                # Apply weight decay
                p.data -= lr*weight_decay*p.data

        return loss