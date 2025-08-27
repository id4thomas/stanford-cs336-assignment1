# Training OWT
## 1. Data Preparation
### 1-1. Train BPE Tokenizer
```
VOCAB SIZE 32000 NUM PROCESSES 8
OUTPUT TO results/tokenizer/owt-v32000
Available CPU Count: 10

Start Traininig
Calculating Pre-token Frequencies done in 480.815s with 8 processes
Building pair_freqs / pair_to_keys done in 58.840s
Made heap of size 19592
Merging done in 977.923s
Training Complete in 1533.620s!
Vocab, Merges saved to results/tokenizer/owt-v32000
```

### 1-2. Tokenize Corpus
**Train (50.42 minutes)**
- each chunk npy is 650M
```
OUTPUT TO results/dataset/owt_train
	BOUNDARIES: [0, 1490070394, 2980128270, 4470223005, 5960269363, 7450321387, 8940385101, 10430448049, 11920511059]
Chunked corpus in 28.395s

[chunk 0] tokenized in 1933.960s
[chunk 1] tokenized in 1917.328s
[chunk 2] tokenized in 3016.706s
[chunk 3] tokenized in 1933.557s
[chunk 4] tokenized in 1936.513s
[chunk 5] tokenized in 1941.453s
[chunk 6] tokenized in 1934.819s
[chunk 7] tokenized in 2076.881s

[chunk 0] saved in 14.182s shape (341047999,)
[chunk 1] saved in 18.111s shape (340965037,)
[chunk 2] saved in 7.404s shape (340902829,)
[chunk 3] saved in 14.776s shape (341139217,)
[chunk 4] saved in 14.408s shape (341258640,)
[chunk 5] saved in 13.198s shape (340801061,)
[chunk 6] saved in 14.506s shape (340580356,)
[chunk 7] saved in 7.811s shape (340425368,)

dataset tokenized in 3025.020s
```

**Valid (46.241s)**
- each chunk npy is 35M
```
OUTPUT TO results/dataset/owt_valid
	BOUNDARIES: [0, 36335216, 72505172, 108752143, 145027268, 181256470, 217499287, 253752435, 289998753]
Chunked corpus in 0.508s

[chunk 0] tokenized in 45.797s
[chunk 1] tokenized in 45.718s
[chunk 2] tokenized in 45.876s
[chunk 3] tokenized in 45.758s
[chunk 4] tokenized in 45.695s
[chunk 5] tokenized in 45.716s
[chunk 6] tokenized in 45.725s
[chunk 7] tokenized in 45.631s

[chunk 0] saved in 0.207s shape (8347897,)
[chunk 1] saved in 0.222s shape (8254651,)
[chunk 2] saved in 0.191s shape (8265543,)
[chunk 3] saved in 0.208s shape (8328409,)
[chunk 4] saved in 0.206s shape (8322533,)
[chunk 5] saved in 0.224s shape (8264607,)
[chunk 6] saved in 0.225s shape (8356564,)
[chunk 7] saved in 0.209s shape (8260894,)

dataset tokenized in 46.251s
```

## 2. Training Model
### 2-1. Hyperparameters
...

### 2-2. Dataloader