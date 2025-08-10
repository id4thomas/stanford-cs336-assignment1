from collections.abc import Callable, Iterable
import math
from typing import List, Optional, Union

import torch
from torch.optim import (
    Optimizer
)
from torch.optim.optimizer import ParamsT

class SGD(Optimizer):
    def __init__(
        self,
        params: Optimizer.ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
    ):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}".format(lr))
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(
        self,
        closure: Optional[Callable]=None
    ):
        # closure: function that re-evalutes model and returns loss
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p] # get state associated with p
                t = state.get("t", 0) # get iteration number
                grad = p.grad.data # get gradient of loss wrt p
                
                # Perform Update
                p.data -= (lr/math.sqrt(t+1)) * grad # lr/sqrt(t+1) is lr decay (not default in sgd)
                state["t"] = t+1
                
        return loss