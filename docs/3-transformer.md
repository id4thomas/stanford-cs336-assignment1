# 3. Transformer Language Model Architecture

## 3-1. Building Blocks (Linear & Embedding)
Notes
* linear weights are stored as (d_out, d_in) and used like x@W.T
* weight init is done with `torch.init.trunc_normal_`

Testcases:
```
tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
```

## 3-2. Pre-Norm Transformer Block
### 3-2-1. RMSNorm
Testcases:
```
tests/test_model.py::test_rmsnorm PASSED
```

### 3-2-2. SwiGLU Feed-forward Network
SwiGLU = SiLU (Swish) Activation + GLU
* SiLU (Swish): sigmoid(x)*x, 
* GLU: Gated Linear Unit
    * element-wise product of 'linear transform' & 'linear transform + sigmoid
* SwiGLU: w2(silu(w1(x)) * w3(x))
    * 3 linear layers
    * w1, w3: d_model -> d_ff
    * w2: d_ff -> d_model

Testcases:
```
tests/test_model.py::test_silu_matches_pytorch PASSED
tests/test_model.py::test_swiglu PASSED
```

### 3-2-3. Relative Positional Embeddings (RoPE)
RoPE (Rotary Position Embeddings)
* Pairwise Rotation Matrix: rotates embedding elements (in blocks of 2) by angle
    * angle = token positon / (constant^(2k/d))
    * k: dimension index of embedding

**Implementation Details**
Inverse Frequency Init
* rope implementation in transformers calculates 'inv_freq' at init time
* dynamically calculates freqs -> rotation emb during forward
    * pre-calculating for all possible token position allocates too much memory

Rotation Calculation
* Split input x into 2 parts (odd idxs, even idxs)
    * x -> x_even (x[..., 0::2]), x_odd (x[..., 1::2])
* Apply Rotation
    * x_rot_even = x_even * cos - x_odd * sin
    * x_rot_odd  = x_even * sin + x_odd * cos
* Stack -> Flatten
    * stack([x_rot_even, x_rot_odd], dim=-1): innermost index 0 picks from x_rot_even, index 1 from x_rot_odd
    * flatten(-2): 

flatten walks the last two dims in this order -> interleaves even / odd
```
x_out[..., 0,0],
x_out[..., 0,1],
x_out[..., 1,0],
x_out[..., 1,1],
…,
x_out[..., (H/2-1),0],
x_out[..., (H/2-1),1]
```

Testcases:
```
tests/test_model.py::test_rope PASSED
```

### 3-2-4.Attention
**Scaled Dot-Product Attention**

Softmax:
* Use the trick of subtracting the maximum value in the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    * softmax operation is invariant to adding any constant c to all inputs

Scaled Dot-Product Attention
* mask pre-softmax values by adding -inf to masking positions (True to 0.0, False to -inf)
    * `torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))`

**Causal Multi-Head Self-Attention**

Handling Multi-head:
* calc linear first -> view -> permute
    * linear: (batch, seq, d_model) -> (batch, seq, d_k)
    * view: (batch, seq, d_k) -> (batch, seq, n_heads, d_head)
    * permute: (batch, seq, n_heads, d_head) -> (batch, n_heads, seq, d_head)
        * view as 'n_heads' number of (seq, d_head)

Applying RoPE:
* rope should be applied to Query, Key
* each head should apply rope separately

Testcases:
```
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED
tests/test_model.py::test_scaled_dot_product_attention PASSED

tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
```

### 3-2-5. Transformer Block
* attend: out1 = x + causalmha(rmsnorm(x))
* feed-forward: out2 = out1 + ffnn(rmsnorm(out1))

Testcases:
```
tests/test_model.py::test_transformer_block PASSED
```

## 3-3. Transformer LM
### 3-3-1. Transformer Model
Notes:
* apply rmsnorm before lm head
* layers (List[TransformerBlock]) should be enclosed like `nn.ModuleList(layers)`

Testcases:
```
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED
```