# 2. Byte-Pair Encoding (BPE) Tokenizer

## 2-1. test_train_bpe (Updated 250731)
### Testcases
```
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

## 2-2. test_tokenizer (Updated 250801)
### Testcases
```
tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage SKIPPED (rlimit support for non-linux systems is spotty.)
tests/test_tokenizer.py::test_encode_memory_usage SKIPPED (rlimit support for non-linux systems is spotty.)
```

Fix History
* `test_encode_special_token_double_newline_non_whitespace`, `test_address_matches_tiktoken`: add PAT based pretokenization during tokenize
* `test_encode_iterable_tinystories_sample_roundtrip`: yield per token not yieling list of tokens
    * `At index 498 diff: 628 != 198, Right contains one more item: 198`
* `test_overlapping_special_tokens`: sort special tokens first to catch the overlapping first


## Training
Tinystores:
```
VOCAB SIZE 10000 NUM PROCESSES 8
OUTPUT TO results/tokenizer/tinystories-v10000
Available CPU Count: 10


Start Traininig
Calculating Pre-token Frequencies done in 74.518s with 8 processes
Building pair_freqs / pair_to_keys done in 0.099s
Made heap of size 2108
Merging done in 3.086s
Training Complete in 77.807s!
Vocab, Merges saved to results/tokenizer/tinystories-v10000
Running Tests:
ENCODED: [73, 196, 170, 294, 196, 179, 268, 196, 181, 120, 483, 33, 196, 189, 64, 33, 241, 160, 154, 132]
DECODED: Héllò hôw are ü? 🙃
```

OWT
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