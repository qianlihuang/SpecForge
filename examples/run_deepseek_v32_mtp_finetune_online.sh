#!/bin/bash
# Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding (Online Mode)
#
# This script runs online training where hidden states are generated on-the-fly
# during training, similar to SpecForge's online EAGLE3 training.
#
# Usage: ./run_deepseek_v32_mtp_finetune_online.sh [NUM_GPUS] [TP_SIZE]

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}

# Paths
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-"/data/models/DeepSeek-V3.2"}
DATA_PATH=${DATA_PATH:-"$ROOT_DIR/cache/dataset/deepseek-v32-sample.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-mtp-online"}
EXPORT_DIR=${EXPORT_DIR:-"$ROOT_DIR/outputs/deepseek-v32-mtp-eagle-online"}

# Training parameters
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_LENGTH=${MAX_LENGTH:-2048}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}

# SGLang parameters
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
SGLANG_PAGE_SIZE=${SGLANG_PAGE_SIZE:-64}

echo "============================================"
echo "DeepSeek-V3.2 MTP Layer Fine-tuning (Online)"
echo "============================================"
echo "NUM_GPUS: $NUM_GPUS"
echo "TP_SIZE: $TP_SIZE"
echo "TARGET_MODEL_PATH: $TARGET_MODEL_PATH"
echo "DATA_PATH: $DATA_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXPORT_DIR: $EXPORT_DIR"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "============================================"

# Step 1: Online training (hidden states generated on-the-fly)
echo ""
echo "[Step 1/2] Online training with on-the-fly hidden state generation..."
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_mtp_layer_online.py \
    --target-model-path $TARGET_MODEL_PATH \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --chat-template deepseek-v32 \
    --max-length $MAX_LENGTH \
    --tp-size $TP_SIZE \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --sglang-mem-fraction-static $SGLANG_MEM_FRACTION \
    --sglang-page-size $SGLANG_PAGE_SIZE

echo ""
echo "[Step 1/2] Online training completed!"
echo "============================================"

# Step 2: Export model
echo ""
echo "[Step 2/2] Exporting model for EAGLE deployment..."
echo "============================================"

python $ROOT_DIR/scripts/export_mtp_model.py \
    --input-dir $OUTPUT_DIR/checkpoint-epoch-$NUM_EPOCHS \
    --target-model-path $TARGET_MODEL_PATH \
    --output-dir $EXPORT_DIR

echo ""
echo "============================================"
echo "Online training pipeline completed!"
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
