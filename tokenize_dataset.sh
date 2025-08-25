#!/bin/bash
NUM_PROCESSES=8

# Tinystories
TOKENIZER_PATH="results/tokenizer/tinystories-v10000"
## train
# DATASET_PATH="data/TinyStoriesV2-GPT4-train.txt"
# OUTPUT_PATH="results/dataset/TinyStoriesV2-GPT4-train"
## valid
DATASET_PATH="data/TinyStoriesV2-GPT4-valid.txt"
OUTPUT_PATH="results/dataset/TinyStoriesV2-GPT4-valid"

# OWT-Train
# DATASET_PATH="data/owt_train.txt"
# OUTPUT_PATH="results/dataset/owt_train"

echo "OUTPUT TO ${OUTPUT_PATH}"

uv run cs336_basics/tokenize_dataset.py \
    --dataset_path ${DATASET_PATH} \
    --output_path ${OUTPUT_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --num_processes ${NUM_PROCESSES}

