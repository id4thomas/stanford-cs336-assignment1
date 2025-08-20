import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.bpe import BPETokenizer


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train LM")
    parser.add_argument("--input_path", type=str, help="Path to train file")
    parser.add_argument("--output_path", type=str, help="Path to save")
    parser.add_argument("--vocab_size", type=int, default = 10000, help="Target vocab size")
    parser.add_argument("--num_processes", type=int, default=1, help="Num Processes")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        raise ValueError(f"Path {args.output_path} doesn't exist")
    
    # M1 Max - 8 Perf Cores, 2 Eff Cores
    cpu_count = mp.cpu_count()
    print(f"Available CPU Count: {cpu_count}")
    
    # Fix to eos
    special_tokens = ["<|endoftext|>"]
    print("\n\nStart Traininig")
    start = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
        verbose=True
    )
    end = time.time()
    print("Training Complete in {:.3f}s!".format(end-start))
    
    # Save Vocab / Merges
    vocab_path = os.path.join(args.output_path, "vocab.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    merges_path = os.path.join(args.output_path, "merges.pkl")
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
        
    print(f"Vocab, Merges saved to {args.output_path}")
    
    # Test Tokenizer
    print("Running Tests:")
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens
    )
    
    # test_roundtrip_unicode_string
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    print("ENCODED:", encoded_ids)
    
    decoded_string = tokenizer.decode(encoded_ids)
    print("DECODED:", decoded_string)
    