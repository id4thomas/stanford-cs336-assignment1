#!/bin/bash
NUM_PROCESSES=8
VOCAB_SIZE=32000

# Testcases
# OUTPUT_PATH="results/tokenizer/corpus-en-v${VOCAB_SIZE}"
# OUTPUT_PATH="results/tokenizer/tinystories-sample-${VOCAB_SIZE}"
# OUTPUT_PATH="results/tokenizer/tinystories-v${VOCAB_SIZE}"
OUTPUT_PATH="results/tokenizer/owt-v${VOCAB_SIZE}"


echo "VOCAB SIZE ${VOCAB_SIZE} NUM PROCESSES ${NUM_PROCESSES}"
mkdir -p $OUTPUT_PATH
echo "OUTPUT TO ${OUTPUT_PATH}"


# DATA_FILE="tests/fixtures/corpus.en"
# DATA_FILE="tests/fixtures/tinystories_sample_5M.txt"
# DATA_FILE="data/TinyStoriesV2-GPT4-train.txt"
DATA_FILE="data/owt_train.txt"

uv run cs336_basics/train_tokenizer.py \
    --input_path ${DATA_FILE} \
    --output_path ${OUTPUT_PATH} \
    --vocab_size ${VOCAB_SIZE} \
    --num_processes ${NUM_PROCESSES}