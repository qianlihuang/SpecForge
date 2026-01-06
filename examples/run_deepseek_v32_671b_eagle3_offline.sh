#!/bin/bash
# Train Eagle3 for DeepSeek-V3.2 - Offline mode (pre-generate hidden states)
# Usage: ./run_deepseek_v32_671b_eagle3_offline.sh [NUM_GPUS] [TP_SIZE]

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}

# Use a small sample dataset for initial testing
DATA_PATH=${DATA_PATH:-"$ROOT_DIR/cache/dataset/deepseek-v32-sample.jsonl"}
HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH:-"$ROOT_DIR/cache/hidden_states/deepseek-v32-sample"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-671B-eagle3-sample-offline"}

echo "============================================"
echo "DeepSeek-V3.2 Eagle3 Offline Training"
echo "============================================"
echo "NUM_GPUS: $NUM_GPUS"
echo "TP_SIZE: $TP_SIZE"
echo "DATA_PATH: $DATA_PATH"
echo "HIDDEN_STATES_PATH: $HIDDEN_STATES_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "============================================"

# Step 1: Generate hidden states
echo ""
echo "[Step 1/2] Generating hidden states..."
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path /data/models/DeepSeek-V3.2 \
    --enable-aux-hidden-states \
    --data-path $DATA_PATH \
    --output-path $HIDDEN_STATES_PATH \
    --chat-template deepseek-v32 \
    --max-length 4096 \
    --tp-size $TP_SIZE \
    --batch-size 2 \
    --sglang-page-size 64 \
    --sglang-mem-fraction-static 0.70 \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC

echo ""
echo "[Step 1/2] Hidden states generation completed!"
echo "============================================"

# Step 2: Train Eagle3 offline
echo ""
echo "[Step 2/2] Training Eagle3 model..."
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data/models/DeepSeek-V3.2 \
    --draft-model-config $ROOT_DIR/configs/deepseek-v32-671b-eagle3.json \
    --train-data-path $DATA_PATH \
    --train-hidden-states-path $HIDDEN_STATES_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 5 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --chat-template deepseek-v32 \
    --cache-dir $ROOT_DIR/cache \
    --log-interval 1 \
    --save-interval 5

echo ""
echo "============================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"
