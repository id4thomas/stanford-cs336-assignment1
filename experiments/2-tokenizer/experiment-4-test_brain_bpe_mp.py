'''
## Speed Comparison (corpus.en)
# v1
NUM PROCESSES 1 9.970
NUM PROCESSES 4 3.005
NUM PROCESSES 8 2.789
NUM PROCESSES 16 2.806

# v2
NUM PROCESSES 1 3.363
NUM PROCESSES 4 3.140
NUM PROCESSES 8 3.141
NUM PROCESSES 16 3.085

## Speed Comparison (tinystories_sample_5M.txt)
# v1
NUM PROCESSES 1 43.399
NUM PROCESSES 4 62.345
NUM PROCESSES 8 59.174
NUM PROCESSES 16 48.879

# v2
NUM PROCESSES 1 18.560
NUM PROCESSES 4 21.056
NUM PROCESSES 8 22.420
NUM PROCESSES 16 22.646
'''


import json
import multiprocessing
import os
import sys
import time
from typing import BinaryIO, Dict, List, Set, Tuple

import pandas as pd
from transformers import AutoTokenizer
sys.path.append('assignment1-basics')
# from cs336_basics.tokenizer.train_mp import train_bpe
from cs336_basics.tokenizer.train_mp_v2 import train_bpe

from tests.common import gpt2_bytes_to_unicode

## Adapter
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs
    )

FIXTURES_PATH="assignment1-basics/tests/fixtures"

def main():
    # 1. test_train_bpe
    print("## test_train_bpe")
    input_path =  os.path.join(FIXTURES_PATH, "corpus.en")
    reference_vocab_path = os.path.join(FIXTURES_PATH, "train-bpe-reference-vocab.json")
    reference_merges_path = os.path.join(FIXTURES_PATH, "train-bpe-reference-merges.txt")

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
        
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        num_processes=8
    )

    # Tests
    print(set(vocab.keys()) == set(reference_vocab.keys()))
    print(set(vocab.values()) == set(reference_vocab.values()))


    # 2. test_train_bpe_special_tokens
    print("## test_train_bpe_special_tokens")
    input_path =  os.path.join(FIXTURES_PATH, "tinystories_sample_5M.txt")
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        num_processes=8
    )
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    print("CHECKING SPECIAL TOKENS")
    for word_bytes in vocabs_without_specials:
        if b"<|" in word_bytes:
            print(word_bytes)

    import pickle

    with open('assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl', 'rb') as f:
        expected_data = pickle.load(f)

    print('values set diff')
    print(set(vocab.values()) - set(expected_data['vocab_values']))
    print(set(expected_data['vocab_values']) - set(vocab.values()))
        

    ## Compare Training Time
    fname = 'corpus.en'
    # fname = 'tinystories_sample_5M.txt'
    print(f"## Speed Comparison ({fname})")
    for num_processes in [1,4,8,16]:
        start = time.time()
        input_path =  os.path.join(FIXTURES_PATH, fname)
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=1000,
            special_tokens=["<|endoftext|>"],
            num_processes=8
        )
        print("NUM PROCESSES {} {:.3f}".format(num_processes, time.time()-start))

if __name__=='__main__':
    main()