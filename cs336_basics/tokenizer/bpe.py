import json
from typing import Any, List, Dict, Tuple, Iterable, Iterator

import regex as re

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab=vocab
        self.inv_vocab = {v:k for k,v in vocab.items()}

        self.merges=merges
        if isinstance(special_tokens, list) and special_tokens:
            # Sort to ensure case ['<|eot|><|eot|>', '<|eot|>']
            self.special_tokens=sorted(special_tokens, key=lambda x: (-len(x), x))
            self.split_pat = re.compile(
                "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            )
        else:
            self.special_tokens=special_tokens
            self.split_pat = None
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        
        with open(merges_filepath) as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
        
        tokenizer = cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens
        )
        return tokenizer
    
    def id_to_token(self, token_id: int) -> bytes:
        return self.vocab[token_id]
    
    def token_to_id(self, token: bytes) -> int:
        if token not in self.inv_vocab:
            raise ValueError(
                "token {} not in vocab".format(token.decode('utf-8', errors='ignore'))
            )
        return self.inv_vocab[token]
        
    
    def merge(self, indices: List[bytes], merge_pair: Tuple[bytes, bytes]) -> List[bytes]:
        merged_index = b''.join(merge_pair)
        new_indices = []
        
        i=0
        while i < len(indices):
            if i+1 < len(indices):
                pair = (indices[i], indices[i+1])
                # pair = (bytes([indices[i]]), bytes([indices[i+1]]))
                if pair==merge_pair:
                    new_indices.append(merged_index)
                    i+=2
                else:
                    new_indices.append(indices[i])
                    i+=1
            else:
                new_indices.append(indices[i])
                i+=1
        return new_indices
    
    def tokenize(self, text: str) -> list[bytes]:
        if self.special_tokens:
            parts = self.split_pat.split(text)
        else:
            parts = [text]
        
        if self.special_tokens:
            parts = self.split_pat.split(text)
        else:
            parts = [text]
        
        indices = []
        for part in parts:
            # Handle special tokens
            if self.special_tokens and part in self.special_tokens:
                indices.append(part.encode('utf-8'))
                continue
            
            for pretok_match in re.finditer(PAT, part):
                pretok = pretok_match.group()
                # Tokenize
                part_bytes = pretok.encode('utf-8')
                part_indices = list(map(lambda x: bytes([x]), part_bytes))
                
                # Merge
                for merge_pair in self.merges:
                    part_indices = self.merge(part_indices, merge_pair)
            
                indices.extend(part_indices)
        return indices
    
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        indices = [self.token_to_id(x) for x in tokens]
        return indices
             
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for x in iterable:
            for token in self.encode(x):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token(x) for x in ids]
        return b''.join(tokens).decode('utf-8', errors='ignore')