#!/bin/bash
source .env

# CONFIG_NAME="test-1"
# CONFIG_NAME="tinystories-test"
CONFIG_NAME="tinystories-run1"
CONFIG_NAME="tinystories-run2"

echo "RUN ${CONFIG_NAME}"
uv run cs336_basics/train.py --config configs/${CONFIG_NAME}.json