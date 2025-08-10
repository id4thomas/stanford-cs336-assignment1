# 4. Training a Transformer LM

## 4-1. Cross-entropy Loss
Notes
* Take predicted and target logits -> compute -plogq (p is real, q is predicted)
* subtract largest element for numerical stability
    * `torch.max` with `keepdim=True`
* cancel out log and exp (**Log-Sum-Exp Trick**)
    * when p=1.0, -logq = -log(exp(x)/sum(exp(x)))
    * -> -(log(exp(x))-log(sum(exp(x))))
    * -> -x + log(sum(exp(x))) -> log(sum(exp(x))) - x
* use `torch.gather` to collect predicted logits of correct class

Using torch.gather to collect correct class logits
* `dim`: which axis to look along, all other axes are matched exactly between input, index
```
# (batch) -> (batch, 1)
# need to unsqueeze to match shape as input
target_ = target.unsqueeze(dim=1)

correct_logits = torch.gather(
    input_,
    dim=-1,
    index=target.unsqueeze(dim=1)
).squeeze(dim=1)
```

Testcases
```
tests/test_nn_utils.py::test_cross_entropy PASSED
```

## 4-2. Optimizer
Notes
* 

AdamW Optimizer

Testcases
```
tests/test_optimizer.py::test_adamw PASSED
```


## 4-3. Learning rate scheduling
Cosine Annealing Scheduling


Testcases
```
tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED
```

## 4-4. Gradient Clipping
Notes
* calculate l2 norm across 'all' parameters that require grad

Q. doesn't this require too much computation to calculate norm across everything?
* All gradients are already computed during backprop and stored in memory.
* Computing the L2 norm just means summing up (grad**2).sum() for each parameter and taking a square root —
this is tiny compared to the cost of backprop itself.

Testcases
```
tests/test_nn_utils.py::test_gradient_clipping PASSED
```