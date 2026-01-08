#!/bin/bash
# Run the complete DeepSeek-V3.2 MTP extraction and (optional) fine-tuning pipeline
#
# This script provides two options:
# 1. Extract MTP layer directly (no fine-tuning) - RECOMMENDED for most use cases
# 2. Fine-tune projection components and then export
#
# DeepSeek-V3.2 already has a trained MTP layer, so fine-tuning is usually NOT needed.
# Only use fine-tuning if you want to adapt the model to a specific domain.

set -e

# Configuration
TARGET_MODEL="/data/models/DeepSeek-V3.2"
OUTPUT_BASE="/sgl-workspace/SpecForge/outputs"
HIDDEN_STATES_PATH="/sgl-workspace/SpecForge/cache/hidden_states/deepseek-v32-mtp"

# Option 1: Extract MTP layer directly (RECOMMENDED)
echo "============================================================"
echo "Option 1: Extract MTP Layer (No Fine-tuning)"
echo "============================================================"
echo "This extracts the pre-trained MTP layer from DeepSeek-V3.2"
echo "No fine-tuning is performed - the model is ready to use."
echo ""

python /sgl-workspace/SpecForge/scripts/extract_deepseek_v32_mtp.py \
    --target-model-path "$TARGET_MODEL" \
    --output-dir "$OUTPUT_BASE/deepseek-v32-mtp-extracted"

echo ""
echo "Extracted MTP model saved to: $OUTPUT_BASE/deepseek-v32-mtp-extracted"
echo ""

# Option 2: Fine-tune projection components (OPTIONAL)
# Uncomment the following section if you want to fine-tune for domain adaptation

# echo "============================================================"
# echo "Option 2: Fine-tune Projection Components"
# echo "============================================================"
# echo "This fine-tunes only the lightweight projection components:"
# echo "  - enorm, hnorm (RMSNorm)"
# echo "  - eh_proj (projection layer)"
# echo "  - norm, lm_head (output)"
# echo "The decoder block (attention + MoE) remains frozen."
# echo ""
# 
# # Step 2a: Generate hidden states (if not already done)
# if [ ! -d "$HIDDEN_STATES_PATH" ]; then
#     echo "Generating hidden states..."
#     torchrun --nproc_per_node=8 /sgl-workspace/SpecForge/scripts/prepare_mtp_hidden_states.py \
#         --target-model-path "$TARGET_MODEL" \
#         --dataset-path /sgl-workspace/SpecForge/cache/dataset/deepseek-v32-sample.jsonl \
#         --output-dir "$HIDDEN_STATES_PATH" \
#         --max-length 2048
# fi
# 
# # Step 2b: Train projection components
# echo "Training projection components..."
# torchrun --nproc_per_node=8 /sgl-workspace/SpecForge/scripts/train_mtp_layer_v2.py \
#     --target-model-path "$TARGET_MODEL" \
#     --hidden-states-path "$HIDDEN_STATES_PATH" \
#     --output-dir "$OUTPUT_BASE/deepseek-v32-mtp-finetuned" \
#     --num-epochs 3 \
#     --batch-size 1 \
#     --learning-rate 1e-5
# 
# # Step 2c: Export fine-tuned model
# echo "Exporting fine-tuned model..."
# python /sgl-workspace/SpecForge/scripts/export_mtp_model_v2.py \
#     --checkpoint-dir "$OUTPUT_BASE/deepseek-v32-mtp-finetuned/checkpoint-epoch-3" \
#     --target-model-path "$TARGET_MODEL" \
#     --output-dir "$OUTPUT_BASE/deepseek-v32-mtp-finetuned-exported"

echo ""
echo "============================================================"
echo "Usage Instructions"
echo "============================================================"
echo ""
echo "For vLLM-magik:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $TARGET_MODEL \\"
echo "    --tensor-parallel-size 8 \\"
echo "    --speculative_config '{\"method\":\"eagle\",\"model\":\"$OUTPUT_BASE/deepseek-v32-mtp-extracted\", \"num_speculative_tokens\": 3}'"
echo ""
echo "For SGLang:"
echo "  python -m sglang.launch_server \\"
echo "    --model $TARGET_MODEL \\"
echo "    --tp 8 \\"
echo "    --speculative-algorithm EAGLE \\"
echo "    --speculative-draft-model-path $OUTPUT_BASE/deepseek-v32-mtp-extracted"
echo ""
