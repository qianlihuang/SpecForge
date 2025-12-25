#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for deepseek-v3.2
NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# generate hidden states
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/prepare_hidden_states.py \
    --target-model-path deepseek-ai/DeepSeek-V3.2 \
    --enable-aux-hidden-states \
    --data-path $ROOT_DIR/cache/dataset/perfect-blend.jsonl \
    --output-path $ROOT_DIR/cache/hidden_states/perfect-blend-deepseek-v32 \
    --chat-template deepseek-v32 \
    --max-length 131072 \
    --tp-size 8 \
    --batch-size 4 \
    --page-size 64 \
    --sglang-mem-fraction-static 0.75

# train eagle3 offline
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path deepseek-ai/DeepSeek-V3.2 \
    --draft-model-config $ROOT_DIR/configs/deepseek-v32-671b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfect-blend.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/perfect-blend-deepseek-v32 \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/deepseek-v32-671B-eagle3-perfect-blend-offline \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --learning-rate 5e-5 \
    --max-length 131072 \
    --chat-template deepseek-v32 \
    --cache-dir $ROOT_DIR/cache
