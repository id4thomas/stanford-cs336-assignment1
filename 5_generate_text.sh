#!/bin/bash

TOKENIZER_DIR="results/tokenizer/tinystories-v10000"
RUN_NAME="tinystories-run2"
CHECKPOINT_NAME="best_step_8800"

BATCH_SIZE=32
NUM_BATCHES=32

echo "EVALUATING ${RUN_NAME} ${CHECKPOINT_NAME}"
uv run cs336_basics/generate.py \
    --tokenizer_dir ${TOKENIZER_DIR} \
    --model_dir results/model/${RUN_NAME} \
    --checkpoint ${CHECKPOINT_NAME} \
    --device mps