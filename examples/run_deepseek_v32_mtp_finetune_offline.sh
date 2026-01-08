#!/bin/bash
# Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding (Offline Mode)
#
# This script:
# 1. Generates hidden states from the target model using SGLang
# 2. Fine-tunes the MTP layer on the generated data
# 3. Exports the model for vLLM-magik/SGLang deployment
#
# Usage: ./run_deepseek_v32_mtp_finetune_offline.sh [NUM_GPUS] [TP_SIZE]

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}

# Paths
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-"/data/models/DeepSeek-V3.2"}
DATA_PATH=${DATA_PATH:-"$ROOT_DIR/cache/dataset/deepseek-v32-sample.jsonl"}
HIDDEN_STATES_PATH=${HIDDEN_STATES_PATH:-"$ROOT_DIR/cache/hidden_states/deepseek-v32-mtp"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-mtp-finetuned"}
EXPORT_DIR=${EXPORT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-mtp-eagle"}

# Training parameters
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_LENGTH=${MAX_LENGTH:-2048}

echo "============================================"
echo "DeepSeek-V3.2 MTP Layer Fine-tuning"
echo "============================================"
echo "NUM_GPUS: $NUM_GPUS"
echo "TP_SIZE: $TP_SIZE"
echo "TARGET_MODEL_PATH: $TARGET_MODEL_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "HIDDEN_STATES_PATH: $HIDDEN_STATES_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXPORT_DIR: $EXPORT_DIR"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "============================================"

# Step 1: Generate hidden states
echo ""
echo "[Step 1/3] Generating hidden states from target model..."
echo "============================================"

if [ -d "$HIDDEN_STATES_PATH" ] && [ "$(ls -A $HIDDEN_STATES_PATH 2>/dev/null)" ]; then
    echo "Hidden states directory already exists and is not empty."
    echo "Skipping generation. Delete $HIDDEN_STATES_PATH to regenerate."
else
    torchrun \
        --standalone \
        --nproc_per_node $NUM_GPUS \
        $ROOT_DIR/scripts/prepare_mtp_hidden_states.py \
        --target-model-path $TARGET_MODEL_PATH \
        --data-path $DATA_PATH \
        --output-path $HIDDEN_STATES_PATH \
        --chat-template deepseek-v32 \
        --max-length $MAX_LENGTH \
        --tp-size $TP_SIZE \
        --batch-size $BATCH_SIZE \
        --sglang-page-size 64 \
        --sglang-mem-fraction-static 0.70 \
        --build-dataset-num-proc $BUILD_DATASET_NUM_PROC
fi

echo ""
echo "[Step 1/3] Hidden states generation completed!"
echo "============================================"

# Step 2: Fine-tune MTP layer
echo ""
echo "[Step 2/3] Fine-tuning MTP layer..."
echo "============================================"

python $ROOT_DIR/scripts/train_mtp_layer.py \
    --target-model-path $TARGET_MODEL_PATH \
    --train-data-path $DATA_PATH \
    --train-hidden-states-path $HIDDEN_STATES_PATH \
    --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --log-interval 1 \
    --save-interval 1

echo ""
echo "[Step 2/3] Fine-tuning completed!"
echo "============================================"

# Step 3: Export model
echo ""
echo "[Step 3/3] Exporting model for EAGLE deployment..."
echo "============================================"

python $ROOT_DIR/scripts/export_mtp_model.py \
    --input-dir $OUTPUT_DIR/checkpoint-epoch-$NUM_EPOCHS \
    --target-model-path $TARGET_MODEL_PATH \
    --output-dir $EXPORT_DIR

echo ""
echo "============================================"
echo "Pipeline completed!"
echo "============================================"
echo ""
echo "Exported model: $EXPORT_DIR"
echo ""
echo "Usage with vLLM-magik:"
echo "  vllm serve $TARGET_MODEL_PATH \\"
echo "    --speculative_config '{\"method\":\"eagle\",\"model\":\"$EXPORT_DIR\", \"num_speculative_tokens\": 3}'"
echo ""
echo "Usage with SGLang:"
echo "  python -m sglang.launch_server --model $TARGET_MODEL_PATH \\"
echo "    --speculative-algorithm EAGLE --speculative-draft-model-path $EXPORT_DIR"
echo ""
