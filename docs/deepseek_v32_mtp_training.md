# DeepSeek-V3.2 MTP Layer Training for EAGLE Speculative Decoding

This guide explains how to fine-tune DeepSeek-V3.2's MTP (Multi-Token Prediction) layer 61 as an EAGLE-style draft model for speculative decoding with sglang and vllm-magik.

## Overview

DeepSeek-V3.2 uses a unique architecture with 61 layers:
- **Layers 0-59**: Standard transformer layers with MLA (Multi-Head Latent Attention) and MoE (Mixture of Experts)
- **Layer 60**: The MTP (Multi-Token Prediction) layer, which has a specialized structure

The MTP layer architecture:
```
embed_tokens (input) ─────────────────────────────────────┐
                                                          │
enorm(embed) ─────┬──────────────────────────────────────►│
                  │                                       │
hnorm(hidden) ───►└► eh_proj(concat) ──► decoder(MoE) ──►│
                                                          │
shared_head.norm ──────► lm_head ─────────────────────► output
```

This architecture makes it ideal for EAGLE-style speculative decoding, where the draft model needs both embedding tokens and hidden states from the target model.

## Prerequisites

1. **SpecForge Installation**
   ```bash
   cd /sgl-workspace/SpecForge
   pip install -e .
   ```

2. **Required Dependencies**
   ```bash
   pip install torch transformers safetensors accelerate
   ```

3. **DeepSeek-V3.2 Model**
   - Full model path (for weight extraction)
   - Config available at: `/sgl-workspace/DeepSeek-V3.2/`

## Training Workflow

### 1. Prepare Training Data

Training data should be in JSONL format with conversations:

```json
{"conversations": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "<think>reasoning</think>Answer"}]}
```

Sample data location: `cache/dataset/deepseek-v32-sample.jsonl`

### 2. Test Model Architecture

Before training, verify the model can be created:

```bash
python scripts/train_deepseek_v32_mtp.py --mode test
```

This will:
- Create a `DeepSeekV32MTPForCausalLM` model (~2.4B parameters)
- Test forward and backward passes
- Verify gradient flow

### 3. Train the Model

Full training command:

```bash
python scripts/train_deepseek_v32_mtp.py \
    --mode train \
    --config-path configs/deepseek-v32-mtp-eagle.json \
    --data-path cache/dataset/deepseek-v32-sample.jsonl \
    --output-dir outputs/deepseek-v32-mtp-eagle \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 512 \
    --device cuda
```

Key parameters:
- `--config-path`: Model configuration (MLA + MoE settings)
- `--data-path`: Training data JSONL file  
- `--output-dir`: Where to save checkpoints
- `--num-epochs`: Number of training epochs
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--learning-rate`: Learning rate for optimizer
- `--max-length`: Maximum sequence length
- `--device`: `cpu` or `cuda`

### 4. Export to NextN Format

After training, export to sglang/vllm-magik compatible format:

```bash
python scripts/export_deepseek_v32_mtp.py \
    --input-dir outputs/deepseek-v32-mtp-eagle/epoch_1 \
    --output-dir outputs/deepseek-v32-mtp-nextn \
    --target-model-path /path/to/DeepSeek-V3.2
```

This creates a `DeepseekV3ForCausalLMNextN` model that includes:
- Trained MTP layer weights
- Full MoE weights from target model
- Proper config for inference

## Inference with sglang

```bash
python -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V3.2 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path outputs/deepseek-v32-mtp-nextn \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp 8
```

Key speculative decoding parameters:
- `--speculative-algorithm EAGLE3`: Use EAGLE-3 algorithm
- `--speculative-draft-model-path`: Path to exported NextN model
- `--speculative-num-steps`: Number of draft generation steps
- `--speculative-eagle-topk`: Top-k for draft token selection
- `--speculative-num-draft-tokens`: Total draft tokens per step

## Inference with vllm-magik

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/DeepSeek-V3.2 \
    --speculative-config '{
        "method": "eagle3",
        "model": "outputs/deepseek-v32-mtp-nextn",
        "num_speculative_tokens": 4,
        "eagle_aux_hidden_state_layer_ids": [1, 29, 57]
    }' \
    --tensor-parallel-size 8
```

## Model Architecture Details

### DeepSeekV32MTPConfig

```python
{
    "hidden_size": 7168,
    "num_hidden_layers": 1,  # Single MTP layer
    "vocab_size": 129280,
    "rms_norm_eps": 1e-6,
    
    # MLA (Multi-Head Latent Attention)
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "num_attention_heads": 128,
    "v_head_dim": 128,
    
    # MoE (Mixture of Experts)
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "routed_scaling_factor": 2.5,
    "intermediate_size": 2048,  # Per expert
    "moe_intermediate_size": 2048,
    
    # RoPE
    "rope_theta": 10000.0,
    "max_position_embeddings": 163840,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0
    },
    
    # EAGLE3-specific
    "eagle_aux_hidden_state_layer_ids": [1, 29, 57]
}
```

### Training Considerations

1. **Memory Efficiency**: The training implementation uses a simplified MoE that averages expert behavior to reduce memory usage. Full MoE weights are loaded during export.

2. **Gradient Flow**: Only trainable parameters include:
   - `enorm.weight`, `hnorm.weight` (normalization)
   - `eh_proj.weight` (fusion projection)
   - MLA attention weights (q/kv projections, layer norms)
   - `decoder.mlp` (simplified MoE for training)
   - `norm.weight` (output normalization)

3. **Hidden State Layers**: EAGLE-3 uses hidden states from layers [1, 29, 57] for TTT (Test-Time Training) training objective.

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch-size`
- Reduce `--max-length`
- Use gradient accumulation
- Enable gradient checkpointing

### Export Errors

Ensure the target model path contains:
- `config.json`
- `model.safetensors.index.json`
- Weight files for layer 61 (MTP layer)

### Inference Issues

1. Verify config has `"architectures": ["DeepseekV3ForCausalLMNextN"]`
2. Check all required weight files are present
3. Ensure tokenizer files are copied

## File Structure

```
SpecForge/
├── configs/
│   └── deepseek-v32-mtp-eagle.json    # Model configuration
├── scripts/
│   ├── train_deepseek_v32_mtp.py      # Training script
│   └── export_deepseek_v32_mtp.py     # Export script
├── specforge/
│   └── modeling/
│       └── draft/
│           └── deepseek_v32_mtp.py    # Model implementation
├── outputs/
│   └── deepseek-v32-mtp-eagle/        # Training outputs
│       └── epoch_N/
│           ├── config.json
│           ├── model.safetensors
│           └── training_state.pt
└── docs/
    └── deepseek_v32_mtp_training.md   # This document
```

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- [sglang Documentation](https://docs.sglang.ai/)
- [vllm Documentation](https://docs.vllm.ai/)
