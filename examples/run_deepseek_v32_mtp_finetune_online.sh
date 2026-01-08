#!/bin/bash
# Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding (Online Mode)
#
# This script runs online training where hidden states are generated on-the-fly
# during training. Includes FULL MTP architecture (decoder block with attention + MoE)
# to match the inference architecture in vLLM-magik and SGLang.
#
# Architecture (training = inference):
#   hidden_states (from layer 60) + input_ids
#   -> enorm(embed), hnorm(hidden)
#   -> eh_proj(concat)
#   -> DecoderBlock (attention + MoE)  <-- Included!
#   -> norm -> lm_head -> logits
#
# Training modes (set TRAINING_MODE env var):
#   full        - Train full MTP layer (default, best quality)
#   freeze_moe  - Train projection + attention, freeze MoE experts
#   projection  - Train projection only (WARNING: architecture mismatch!)
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
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

# Training mode: full, freeze_moe, freeze_attention, or projection
TRAINING_MODE=${TRAINING_MODE:-"full"}

# SGLang parameters
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.70}
SGLANG_PAGE_SIZE=${SGLANG_PAGE_SIZE:-64}

echo "============================================"
echo "DeepSeek-V3.2 MTP Layer Fine-tuning (Online, Full Architecture)"
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
echo "TRAINING_MODE: $TRAINING_MODE"
echo "============================================"

# Step 1: Online training (hidden states generated on-the-fly)
echo ""
echo "[Step 1/2] Online training with on-the-fly hidden state generation (mode: $TRAINING_MODE)..."
echo "============================================"

# Set training mode flags
TRAIN_FLAGS=""
case $TRAINING_MODE in
    "full")
        echo "Training FULL MTP layer (projection + attention + MoE)"
        ;;
    "freeze_moe")
        TRAIN_FLAGS="--freeze-moe"
        echo "Training with FROZEN MoE (projection + attention only)"
        ;;
    "freeze_attention")
        TRAIN_FLAGS="--freeze-attention"
        echo "Training with FROZEN attention (projection + MoE only)"
        ;;
    "projection")
        TRAIN_FLAGS="--projection-only"
        echo "WARNING: Training PROJECTION only - architecture mismatch with inference!"
        ;;
    *)
        echo "Unknown training mode: $TRAINING_MODE"
        echo "Valid modes: full, freeze_moe, freeze_attention, projection"
        exit 1
        ;;
esac

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_mtp_full_online.py \
    --target-model-path $TARGET_MODEL_PATH \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --chat-template deepseek-v32 \
    --max-length $MAX_LENGTH \
    --tp-size $TP_SIZE \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --sglang-mem-fraction-static $SGLANG_MEM_FRACTION \
    --sglang-page-size $SGLANG_PAGE_SIZE \
    $TRAIN_FLAGS

echo ""
echo "[Step 1/2] Online training completed!"
echo "============================================"

# Step 2: Export model
echo ""
echo "[Step 2/2] Exporting model for EAGLE deployment..."
echo "============================================"

# Export with FP8 quantization by default (matches original model format)
QUANTIZE_FLAG=${QUANTIZE_TO_FP8:-"--quantize-to-fp8"}

python $ROOT_DIR/scripts/export_mtp_model_full.py \
    --checkpoint-dir $OUTPUT_DIR/checkpoint-epoch-$NUM_EPOCHS \
    --target-model-path $TARGET_MODEL_PATH \
    --output-dir $EXPORT_DIR \
    $QUANTIZE_FLAG

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
