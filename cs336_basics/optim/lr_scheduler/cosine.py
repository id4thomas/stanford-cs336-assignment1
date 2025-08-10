import math

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
    else:
        return min_learning_rate