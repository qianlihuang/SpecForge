#!/usr/bin/env python3
"""
Export fine-tuned DeepSeek-V3.2 MTP model for EAGLE speculative decoding.

This script creates a model directory that can be used with vLLM-magik or SGLang
for EAGLE speculative decoding.

Usage:
    python scripts/export_mtp_model.py \
        --input-dir outputs/deepseek-v32-mtp-finetuned/checkpoint-epoch-3 \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --output-dir outputs/deepseek-v32-mtp-eagle
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned MTP model for EAGLE speculative decoding"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to original DeepSeek-V3.2 model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for exported model",
    )
    return parser.parse_args()


def create_mtp_config(target_config: dict) -> dict:
    """Create config for MTP model compatible with vLLM-magik."""
    return {
        "architectures": ["DeepSeekMTPModel"],
        "model_type": "deepseek_mtp",
        "vocab_size": target_config.get("vocab_size", 129280),
        "hidden_size": target_config.get("hidden_size", 7168),
        "intermediate_size": target_config.get("intermediate_size", 18432),
        "moe_intermediate_size": target_config.get("moe_intermediate_size", 2048),
        "num_hidden_layers": 1,  # MTP has 1 layer
        "num_nextn_predict_layers": 1,
        "num_attention_heads": target_config.get("num_attention_heads", 128),
        "num_key_value_heads": target_config.get("num_key_value_heads", 128),
        "q_lora_rank": target_config.get("q_lora_rank", 1536),
        "kv_lora_rank": target_config.get("kv_lora_rank", 512),
        "qk_nope_head_dim": target_config.get("qk_nope_head_dim", 128),
        "qk_rope_head_dim": target_config.get("qk_rope_head_dim", 64),
        "v_head_dim": target_config.get("v_head_dim", 128),
        "hidden_act": target_config.get("hidden_act", "silu"),
        "max_position_embeddings": target_config.get("max_position_embeddings", 163840),
        "initializer_range": target_config.get("initializer_range", 0.02),
        "rms_norm_eps": target_config.get("rms_norm_eps", 1e-6),
        "rope_theta": target_config.get("rope_theta", 10000),
        "rope_scaling": target_config.get("rope_scaling", {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        }),
        "attention_bias": target_config.get("attention_bias", False),
        "attention_dropout": target_config.get("attention_dropout", 0.0),
        "n_routed_experts": target_config.get("n_routed_experts", 256),
        "n_shared_experts": target_config.get("n_shared_experts", 1),
        "num_experts_per_tok": target_config.get("num_experts_per_tok", 8),
        "moe_layer_freq": target_config.get("moe_layer_freq", 1),
        "first_k_dense_replace": target_config.get("first_k_dense_replace", 3),
        "norm_topk_prob": target_config.get("norm_topk_prob", True),
        "scoring_func": target_config.get("scoring_func", "sigmoid"),
        "routed_scaling_factor": target_config.get("routed_scaling_factor", 2.5),
        "topk_group": target_config.get("topk_group", 4),
        "topk_method": target_config.get("topk_method", "noaux_tc"),
        "n_group": target_config.get("n_group", 8),
        "bos_token_id": target_config.get("bos_token_id", 0),
        "eos_token_id": target_config.get("eos_token_id", 1),
        "pad_token_id": target_config.get("pad_token_id", 0),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        # V3.2 specific
        "index_head_dim": target_config.get("index_head_dim", 128),
        "index_n_heads": target_config.get("index_n_heads", 64),
        "index_topk": target_config.get("index_topk", 2048),
    }


def load_target_mtp_weights(target_model_path: str) -> dict:
    """Load MTP layer weights from target model.
    
    In DeepSeek-V3.2:
    - Layers 0-60 are regular decoder layers  
    - Layer 61 is the MTP prediction layer (with enorm, hnorm, eh_proj, shared_head)
    """
    print(f"Loading MTP weights from target model: {target_model_path}")
    
    # Load weight index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Find all weights for layer 61 (MTP layer) and embedding
    mtp_weights = {}
    files_to_load = {}
    
    for weight_name, filename in weight_map.items():
        # MTP layer is layer 61 (not layer 60)
        if "layers.61" in weight_name or weight_name == "model.embed_tokens.weight":
            if filename not in files_to_load:
                files_to_load[filename] = []
            files_to_load[filename].append(weight_name)
    
    # Load weights
    for filename, keys in files_to_load.items():
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in keys or key == "model.embed_tokens.weight":
                    tensor = f.get_tensor(key)
                    # Skip FP8 quantized weights for now
                    if tensor.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        mtp_weights[key] = tensor
                        print(f"    Loaded: {key} {list(tensor.shape)} {tensor.dtype}")
    
    return mtp_weights


def export_model(args):
    """Export the fine-tuned MTP model."""
    print("=" * 60)
    print("Exporting DeepSeek-V3.2 MTP Model for EAGLE")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load target model config
    print("\n[Step 1/4] Loading target model config...")
    target_config_path = os.path.join(args.target_model_path, "config.json")
    with open(target_config_path, "r") as f:
        target_config = json.load(f)
    
    # Create MTP config
    print("\n[Step 2/4] Creating MTP config...")
    mtp_config = create_mtp_config(target_config)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(mtp_config, f, indent=2)
    print(f"  Saved config to: {config_path}")
    
    # Load MTP weights from target model
    print("\n[Step 3/4] Loading MTP weights from target model...")
    mtp_weights = load_target_mtp_weights(args.target_model_path)
    
    # Load fine-tuned weights if available
    finetuned_weights_path = os.path.join(args.input_dir, "model.safetensors")
    if os.path.exists(finetuned_weights_path):
        print(f"\n  Loading fine-tuned weights from: {finetuned_weights_path}")
        with safe_open(finetuned_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Map simplified model keys to vLLM-magik format
                mapped_key = map_finetuned_key(key)
                if mapped_key:
                    mtp_weights[mapped_key] = tensor
                    print(f"    Updated: {key} -> {mapped_key}")
    
    # Save weights in vLLM-magik format
    print("\n[Step 4/4] Saving model weights...")
    
    # Rename keys to match vLLM-magik expected format
    output_weights = {}
    for key, tensor in mtp_weights.items():
        # Convert to bfloat16 if not already
        if tensor.dtype not in [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = tensor.to(torch.bfloat16)
        output_weights[key] = tensor
    
    weights_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(output_weights, weights_path)
    print(f"  Saved weights to: {weights_path}")
    
    # Create weight index
    weight_index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in output_weights.values())},
        "weight_map": {k: "model.safetensors" for k in output_weights.keys()}
    }
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(weight_index, f, indent=2)
    
    # Copy tokenizer files
    print("\n  Copying tokenizer files...")
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    for f in tokenizer_files:
        src = os.path.join(args.target_model_path, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, f))
            print(f"    Copied: {f}")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModel exported to: {args.output_dir}")
    print("\nUsage with vLLM-magik:")
    print(f'  --speculative_config \'{{"method":"eagle","model":"{args.output_dir}", "num_speculative_tokens": 3}}\'')
    print("\nUsage with SGLang:")
    print(f"  --speculative-algorithm EAGLE --speculative-draft-model-path {args.output_dir}")


def map_finetuned_key(key: str) -> str:
    """Map fine-tuned model keys to vLLM-magik format.
    
    Note: In DeepSeek-V3.2, the MTP layer is layer 61 (not 60).
    """
    # Simple model -> vLLM-magik format mapping
    # The MTP layer is exported as layer 61
    mapping = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "enorm.weight": "model.layers.61.enorm.weight",
        "hnorm.weight": "model.layers.61.hnorm.weight",
        "eh_proj.weight": "model.layers.61.eh_proj.weight",
        "norm.weight": "model.layers.61.shared_head.norm.weight",
        "lm_head.weight": "model.layers.61.shared_head.head.weight",
        "input_layernorm.weight": "model.layers.61.input_layernorm.weight",
    }
    return mapping.get(key, None)


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
