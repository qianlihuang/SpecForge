#!/bin/bash
# Train Eagle3 for DeepSeek-V3.2 - Online mode (generate hidden states on-the-fly)
# Usage: ./run_deepseek_v32_671b_eagle3_online.sh [NUM_GPUS] [TP_SIZE]

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}

# Use a small sample dataset for initial testing
DATA_PATH=${DATA_PATH:-"$ROOT_DIR/cache/dataset/deepseek-v32-sample.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-671B-eagle3-sample-online"}

echo "============================================"
echo "DeepSeek-V3.2 Eagle3 Online Training"
echo "============================================"
echo "NUM_GPUS: $NUM_GPUS"
echo "TP_SIZE: $TP_SIZE"
echo "DATA_PATH: $DATA_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "============================================"

# Train Eagle3 online (hidden states generated on-the-fly)
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path deepseek-ai/DeepSeek-V3.2 \
    --draft-model-config $ROOT_DIR/configs/deepseek-v32-671b-eagle3.json \
    --train-data-path $DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --sglang-page-size 64 \
    --sglang-mem-fraction-static 0.70 \
    --chat-template deepseek-v32 \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60 \
    --log-interval 10 \
    --save-interval 100

echo ""
echo "============================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"
