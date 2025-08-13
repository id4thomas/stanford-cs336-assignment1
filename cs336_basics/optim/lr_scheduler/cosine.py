import math
from typing_extensions import override

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

def cosine_annealing_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # Warmup
    if it < warmup_iters:
        return (it/warmup_iters) * max_learning_rate
    # Cosine Annealing
    elif warmup_iters<=it and it<=cosine_cycle_iters:
        lr = min_learning_rate
        lr += 0.5 * (1 + math.cos(((it-warmup_iters)*math.pi)/(cosine_cycle_iters-warmup_iters))) * (max_learning_rate-min_learning_rate)
        return lr
    # Post-annealing
    else:
        return min_learning_rate
    
class CosineAnnealingLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.max_learning_rate=max_learning_rate
        self.min_learning_rate=min_learning_rate
        self.warmup_iters=warmup_iters
        self.cosine_cycle_iters=cosine_cycle_iters
    
        super().__init__(optimizer)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rate."""
        # Initial Steps
        # if self._isinitial:
        #     return [group["lr"] for group in self.optimizer.param_groups]
        
        step_lr = cosine_annealing_lr(
            it=self._step_count,
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_iters=self.warmup_iters,
            cosine_cycle_iters=self.cosine_cycle_iters,
        )
        return [
            step_lr for group in self.optimizer.param_groups
        ]