# CS336 Spring 2025 Assignment 1: Basics
## 2. Byte-Pair Encoding (BPE) Tokenizer
[[Docs (testcase results)](./docs/2-tokenizer.md)]

### Training BPE
Methodology:
```
1. Initialize Vocab
2. Pretokenize
    2-1. Calculate Pretoken frequencies (Pretoken using GPT-2 PAT pattern)
        * This part is done in multiprocessing
    2-2. Make Double Linked List of Bytes (split pretoken)
3. Count byte pair frequencies, record locations
    * pair_counts, pair_positions
4. Merge (loop)
    4-1. Find byte pair with max count
    4-2. Add to merges, vocab
    4-3. Iterate through pair_positions[pair] -> Update left/right of pair
```

Training Speed by num_processes (M1 Max):

| input_file | 1 Process | 4 Processes | 8 Processes | 16 Processes |
| --- | --- | --- | --- | --- |
| corpus.en | 3.363 | 3.140 | 3.141 | 3.085 |
| tinystories_sample_5M.txt | 18.560 | 21.056 | 22.420 | 22.646 |


## 3. Transformer Language Model Architecture
TBD

## 4. Training a Transformer LM
TBD

----
For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

