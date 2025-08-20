# train_bpe
## v2
```
1. Initialize Vocab
2. Pre-tokenize (multi-process)
    * Calculate Pretoken frequencies (Pretoken using GPT-2 PAT pattern)
3. Build byte pair freq, reverse index pretoken
    * byte pair freq (pair_freqs): count of tuple(bytes, bytes) pair 
    * reverse index pretoken (pair_to_keys): index source (pre-token) of pair
4. Merge (loop)
    * use max-heap (-pair_count, reverse_order(pair)), use custom pair class
    4-1. Find byte pair with max count
    4-2. Add to merges, vocab
    4-3. Iterate through indexed 'key' from pair_to_keys
        * decrement 'all' pair frequencies of the old key
        * merge the target pair -> create new key
        * increment 'all' pair frequencies of the new key

freqs = {
    (b'A', b'B', b'C'): 5,
    ...
}

pair_freqs = {
    (b'A', b'B'): 5,
    (b'B', b'C'): 8,   # (5 from ABC + 3 from BC)
    (b'C', b'A'): 2
}

pairs_to_keys = {
    (b'A', b'B'): { (b'A', b'B', b'C') },
    (b'B', b'C'): { (b'A', b'B', b'C'), (b'B', b'C') },
    (b'C', b'A'): { (b'C', b'A') }
}
```

## v1
* hangs when creating linked list if the dataset is large

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