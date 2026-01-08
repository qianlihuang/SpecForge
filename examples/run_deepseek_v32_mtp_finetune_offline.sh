#!/bin/bash
# Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding (Offline Mode)
#
# This script fine-tunes the FULL MTP layer (layer 61) including the decoder block,
# ensuring architecture compatibility with vLLM-magik and SGLang inference.
#
# Architecture (training = inference):
#   hidden_states (from layer 60) + input_ids
#   -> enorm(embed), hnorm(hidden)
#   -> eh_proj(concat)
#   -> DecoderBlock (attention + MoE)  <-- Included for architecture match!
#   -> norm -> lm_head -> logits
#
# Pipeline:
# 1. Generates hidden states from the target model using SGLang
# 2. Fine-tunes the MTP layer (full or with frozen components)
# 3. Exports the model for vLLM-magik/SGLang EAGLE deployment
#
# Training modes (set TRAINING_MODE env var):
#   full        - Train full MTP layer (default, best quality)
#   freeze_moe  - Train projection + attention, freeze MoE experts
#   projection  - Train projection only (WARNING: architecture mismatch!)
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
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

# Training mode: full, freeze_moe, freeze_attention, or projection
TRAINING_MODE=${TRAINING_MODE:-"full"}

echo "============================================"
echo "DeepSeek-V3.2 MTP Layer Fine-tuning (Full Architecture)"
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
echo "TRAINING_MODE: $TRAINING_MODE"
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
echo "[Step 2/3] Fine-tuning MTP layer (mode: $TRAINING_MODE)..."
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

python $ROOT_DIR/scripts/train_mtp_full.py \
    --target-model-path $TARGET_MODEL_PATH \
    --hidden-states-path $HIDDEN_STATES_PATH \
    --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --log-interval 1 \
    --save-interval 1 \
    $TRAIN_FLAGS

echo ""
echo "[Step 2/3] Fine-tuning completed!"
echo "============================================"

# Step 3: Export model
echo ""
echo "[Step 3/3] Exporting model for EAGLE deployment..."
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
