# CS336 Spring 2025 Assignment 1: Basics
## Docs
| section | notes |
| --- | --- |
| [2-tokenizer](./docs/2-tokenizer.md) | |
| [3-transformer](./docs/3-transformer.md) | |
| [4-training](./docs/4-training.md) | |
| [5-training-loop](./docs/5-training-loop.md) |  |

## Notes
### [2-tokenizer] Training BPE
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


### [5-training-loop]
**Tokenizer Training**

Example Logs:



**Model Training (Updated 250813)**

Example Logs:
```
(base) ➜  stanford-cs336-assignment1 git:(main) ✗ ./train.sh
wandb: Currently logged in as: id4thomas to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.0
wandb: Run data is saved locally in /Users/id4thomas/github/stanford-cs336/stanford-cs336-assignment1/wandb/run-20250813_212836-56nybw0u
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run test-run1
wandb: ⭐️ View project at https://wandb.ai/id4thomas/cs336-assignment1
wandb: 🚀 View run at https://wandb.ai/id4thomas/cs336-assignment1/runs/56nybw0u
0%|                                                                                                                                                                                                                                   | 0/20 [00:00<?, ?it/s]{'train/loss': 6.907755374908447, 'train/lr': 0.0009757729755661011, 'train/step': 1}
5%|██████████▉                                                                                                                                                                                                                | 1/20 [00:00<00:07,  2.52it/s]{'train/loss': 6.907796382904053, 'train/lr': 0.0009460482294732421, 'train/step': 2}
{'train/loss': 6.907742023468018, 'train/lr': 0.000905463412215599, 'train/step': 3}
...
{'train/loss': 6.907790184020996, 'train/lr': 1e-05, 'train/step': 20}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:16<00:00,  1.24it/s]
wandb: 
wandb: 🚀 View run test-run1 at: https://wandb.ai/id4thomas/cs336-assignment1/runs/56nybw0u
wandb: Find logs at: wandb/run-20250813_212836-56nybw0u/logs
```

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

