#!/bin/bash

# train_bpe
echo "TESTING: train_bpe"
uv run pytest tests/test_train_bpe.py

# tokenizer
echo "TESTING: tokenizer"
uv run pytest tests/test_tokenizer.py