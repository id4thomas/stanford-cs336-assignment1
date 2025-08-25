import argparse
import multiprocessing as mp
import os
import pickle
import time
from typing import BinaryIO

import numpy as np

from cs336_basics.tokenizer.train_bpe import find_chunk_boundaries
from cs336_basics.tokenizer.bpe import BPETokenizer

def chunk_corpus(
    dataset_path: str,
    result_path: str,
    num_processes: int = 1
):
    # Chunk dataset
    with open(dataset_path, 'rb') as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token = b"<|endoftext|>"
        )
        print("\tBOUNDARIES:", boundaries)
        
        for chunk_i in range(len(boundaries)-1):
            start = boundaries[chunk_i]
            end = boundaries[chunk_i+1]
            
            f.seek(start)
            
            text = f.read(end - start).decode("utf-8", errors="ignore")  
            with open(os.path.join(result_path, f"chunk{chunk_i}.txt"), 'w') as wf:
                wf.write(text)

def load_tokenizer(tokenizer_path):
    vocab_path = os.path.join(tokenizer_path, "vocab.pkl")
    merges_path = os.path.join(tokenizer_path, "merges.pkl")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
        
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )
    return tokenizer

class FileIterator:
    def __init__(self, file: BinaryIO, mini_chunk_size: int = 4096):
        self.file = file
        self.mini_chunk_size = mini_chunk_size

    def __iter__(self):
        return self

    def __next__(self):
        block = self.file.read(self.mini_chunk_size)
        if not block:
            raise StopIteration
        return block

def _tokenize_fn(
    chunk_i: int,
    chunk_path: str,
    tokenizer_path: str,
    result_path: str,
    chunk_size: int = 81_920
):
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Tokenize
    print("chunk {} start tokenization".format(chunk_i))
    start = time.time()
    tokens = []
    with open(os.path.join(chunk_path, f"chunk{chunk_i}.txt"), 'r') as f:
        it = FileIterator(
            f,
            mini_chunk_size=chunk_size
        )
        for id in tokenizer.encode_iterable(it):
            tokens.append(id)
                
        # for id in tokenizer.encode_iterable(f):
        #     tokens.append(id)
    
    end = time.time()
    print("chunk {} tokenized in {:.3f}s".format(chunk_i, end-start))
    
    start = time.time()
    tokenized_text = np.array(tokens, dtype=np.uint16)
    with open(os.path.join(result_path, f"chunk{chunk_i}.npy"), 'wb') as f:
        np.save(f, tokenized_text)
    end = time.time()
    print("chunk {} saved in {:.3f}s shape {}".format(chunk_i, end-start, str(tokenized_text.shape)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train LM")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset file")
    parser.add_argument("--output_path", type=str, help="Path to save")
    parser.add_argument("--tokenizer_path", type=str, help="Path to load tokenizer from")
    parser.add_argument("--num_processes", type=int, help="Number of processes to use")
    args = parser.parse_args()
    
    # Chunk Corpus
    # Uses about 10GB Each
    chunk_output_path = os.path.join(args.output_path, "chunks")
    if not os.path.exists(chunk_output_path):
        os.makedirs(chunk_output_path)
    
    start = time.time()    
    chunk_corpus(
        args.dataset_path,
        chunk_output_path,
        num_processes=args.num_processes
    )
    end = time.time()
    print("Chunked corpus in {:.3f}s".format(end-start))

    # Tokenize
    tokenized_output_path = os.path.join(args.output_path, "tokens")
    if not os.path.exists(tokenized_output_path):
        os.makedirs(tokenized_output_path)
    
    
    start = time.time()
    with mp.Pool(args.num_processes) as pool:
        pool.starmap(
            _tokenize_fn,
            [
                (
                    chunk_i,
                    chunk_output_path,
                    args.tokenizer_path,
                    tokenized_output_path,
                )
                for chunk_i in range(args.num_processes)
            ]
        )
    end = time.time()
    print("dataset tokenized in {:.3f}s".format(end-start))

    # Aggregate