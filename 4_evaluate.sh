#!/bin/bash

DATA_NAME="TinyStoriesV2-GPT4-valid"
RUN_NAME="tinystories-run2"

CHECKPOINT_NAME="best_step_8800"

BATCH_SIZE=32
NUM_BATCHES=32

echo "EVALUATING ${RUN_NAME} ${CHECKPOINT_NAME}"
uv run cs336_basics/evaluate.py \
    --data_dir results/dataset/${DATA_NAME}/data.npy \
    --model_dir results/model/${RUN_NAME} \
    --checkpoint ${CHECKPOINT_NAME} \
    --batch_size ${BATCH_SIZE} \
    --num_batches ${NUM_BATCHES} \
    --device mps