'''Methodology
1. Initialize Vocab
2. Read Files, Pretokenize
    2-1. Find chunk bounaries
    2-2. Process each chunk, gather byte pairs per pre-token
'''
from collections import defaultdict, Counter
from functools import reduce
import heapq

import os
import regex as re
from typing import List, Dict, Tuple
 
import multiprocessing as mp
from typing import BinaryIO

# GPT-2 Pretokenization pattern
PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class TokenNode:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
        # For determining pre-tokenization boundary
        self.is_next_connected = True



def initialize_vocab(special_tokens):
    """Initiialize Vocab"""
    vocab: Dict[int, bytes] = {}
    cur_vocab_size = 0
    
    encoded_special_tokens = [x.encode('utf-8') for x in special_tokens]
    for tok in encoded_special_tokens:
        vocab[cur_vocab_size]=tok
        cur_vocab_size+=1
        
    ## 256 utf-8 bytes (8bit -> 0 ~ 255)
    # byte can represent 256 values (unicode string is sequence of bytes)
    # start with single-byte -> merge
    for i in range(256):
        vocab[cur_vocab_size]=bytes([i])
        cur_vocab_size+=1
    
    return vocab, cur_vocab_size

# Provided Function
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize_text(
    input_path: str,
    start: int,
    end: int,
    special_tokens: List[str]
) -> Dict[str, int]:
    '''Return counter of pretokens'''
    split_pat = re.compile(
        "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    )
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")  
    
    if special_tokens:
        parts = split_pat.split(text)
    else:
        parts = [text]
        
    pretok_freq: Dict[str, int] = {}
    
    for part in parts:
        if part in special_tokens:
            continue
    
        ## Iterate through pretokens
        for pretok_match in re.finditer(PAT, part):
            pretok = pretok_match.group()
            pretok_freq[pretok] = pretok_freq.get(pretok, 0) + 1
        
    return pretok_freq
        
def _merge_freqs(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    """Adds frequencies from dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result

def pretokenize(
    input_path: str,
    special_tokens: List[str],
    num_processes: int = 1
) -> Dict[str, int]:
    with open(input_path, 'rb') as f:
        # 1. Find Boundaries
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=b"<|endoftext|>"
        )
        
        # 2. Calculate pretoken frequencies
        with mp.Pool(num_processes) as pool:
            pretok_freqs = pool.starmap(
                _pretokenize_text,
                [
                    (
                        input_path,
                        boundaries[b_i-1],
                        boundaries[b_i],
                        special_tokens
                    )
                    for b_i in range(1, len(boundaries))
                ]
            )
    pretok_freqs = reduce(_merge_freqs, pretok_freqs, {})
    return pretok_freqs

def add_node(byte_val, prev):
    """Helper to create and link a new TokenNode."""
    node = TokenNode(byte_val)
    if prev:
        prev.next = node
        node.prev = prev
    return node

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    '''
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
    '''
    # 1. Initialize Vocab
    vocab, cur_vocab_size = initialize_vocab(special_tokens)
    
    # 2. Pretokenize
    ## 2-1. Calculate Pretoken frequencies
    pretok_freqs = pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    
    ## 2-2. Make Double Linked Lists
    head = None
    prev = None
    
    for pretok, freq in pretok_freqs.items():
        pretok_bytes = pretok.encode('utf-8')
        
        for _ in range(freq):
            for byte in pretok_bytes:
                prev = add_node(bytes([byte]), prev)
                if head is None:
                    head=prev
            prev.is_next_connected=False
  
    # 3. Count Pair Frequencies, Record Location
    pair_positions = defaultdict(set)
    node = head
    while node and node.next:
        # print(node.val, node.is_next_connected)
        if not node.is_next_connected:
            node=node.next
            continue
        
        pair_positions[
            (node.val, node.next.val)
        ].add(node)
        node = node.next
    pair_counts = {pair: len(nodes) for pair, nodes in pair_positions.items()}
    
    # 4. Merge
    remaining_merges = vocab_size-cur_vocab_size
    merges: List[Tuple[bytes, bytes]] = []
    
    for i in range(remaining_merges):
        # Break Ties - preferring the lexicographically greater pair.
        max_count_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                pair[0],#.decode('utf-8', errors='ignore'),
                pair[1]#.decode('utf-8', errors='ignore')
            )
        )
        
        # Add to merges
        merges.append(max_count_pair)
        remaining_merges-=1
        
        # Add new vocab
        merged_val = b''.join(max_count_pair)
        vocab[cur_vocab_size]=merged_val
        cur_vocab_size+=1
        
        # Iterate through merge
        max_count_pair_positions = list(pair_positions[max_count_pair])
        for node_a in max_count_pair_positions:
            # Re-validate if still merge-able
            if (
                node_a.next is None
                or node_a.val!=max_count_pair[0]
                or not node_a.is_next_connected 
                or node_a.next.val!=max_count_pair[1]
            ):
                continue
            if not node_a in pair_positions[max_count_pair]:
                # print("HI")
                continue
            
            node_b = node_a.next
            
            # 1. Merge Node
            new_node = TokenNode(merged_val)
            new_node.prev=node_a.prev
            new_node.next=node_b.next
            new_node.is_next_connected=node_b.is_next_connected
 
            # 2. Update Left
            if node_a.prev:
                if node_a.prev.is_next_connected:
                    # Remove previous
                    prev_pair = (node_a.prev.val, node_a.val)
                    pair_counts[prev_pair]-=1
                    pair_positions[prev_pair].discard(node_a.prev)
                    
                    # Add new merged version
                    new_pair = (node_a.prev.val, merged_val)
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                    pair_positions[new_pair].add(node_a.prev)

                node_a.prev.next=new_node
            
            # 3. Update Right
            if node_b.next:
                if node_b.is_next_connected:
                    # Remove previous
                    prev_pair = (node_b.val, node_b.next.val)
                    pair_counts[prev_pair]-=1
                    pair_positions[prev_pair].discard(node_b)

                    # Add new merged version
                    new_pair = (merged_val, node_b.next.val)
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + 1
                    pair_positions[new_pair].add(new_node)

                node_b.next.prev=new_node
            
            node_a.val=None
            node_b.val=None
        del pair_counts[max_count_pair]
        del pair_positions[max_count_pair]
    return vocab, merges

if __name__=='__main__':
    from time import time
    start = time.time()
    (vocab, merges) = train_bpe(
        input_path="../../tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        num_processes=1
    )
    print("corpus.en done in {:.3f} with 1 process".format(time.time()-start))
