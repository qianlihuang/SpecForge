#!/usr/bin/env python3
"""
Export fine-tuned DeepSeek-V3.2 MTP projection model for EAGLE speculative decoding.

This script takes fine-tuned projection weights (from train_mtp_layer_v2.py) and
combines them with the frozen decoder block from the original model to create
a complete MTP model for inference.

Fine-tuned components (from checkpoint):
- enorm, hnorm, eh_proj, norm, lm_head

Frozen components (from target model):
- All decoder block weights (attention + MoE)
- input_layernorm, post_attention_layernorm
- self_attn.*, mlp.*

Usage:
    python scripts/export_mtp_model_v2.py \
        --checkpoint-dir outputs/deepseek-v32-mtp-finetuned/checkpoint-epoch-3 \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --output-dir outputs/deepseek-v32-mtp-eagle-finetuned
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned MTP model for EAGLE speculative decoding"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory",
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
    """Create config for MTP model compatible with vLLM-magik/SGLang."""
    return {
        "architectures": ["DeepSeekMTPModel"],
        "model_type": "deepseek_mtp",
        "vocab_size": target_config.get("vocab_size", 129280),
        "hidden_size": target_config.get("hidden_size", 7168),
        "intermediate_size": target_config.get("intermediate_size", 18432),
        "moe_intermediate_size": target_config.get("moe_intermediate_size", 2048),
        "num_hidden_layers": 1,
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
        "rope_scaling": target_config.get("rope_scaling"),
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


def export_model(args):
    """Export the fine-tuned MTP model."""
    print("=" * 60)
    print("Exporting Fine-tuned MTP Model for EAGLE")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load target model config
    print("\n[Step 1/5] Loading target model config...")
    target_config_path = os.path.join(args.target_model_path, "config.json")
    with open(target_config_path, "r") as f:
        target_config = json.load(f)
    
    mtp_layer_idx = target_config.get("num_hidden_layers", 61)
    print(f"  MTP layer index: {mtp_layer_idx}")
    
    # Create MTP config
    print("\n[Step 2/5] Creating MTP config...")
    mtp_config = create_mtp_config(target_config)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(mtp_config, f, indent=2)
    print(f"  Saved config to: {config_path}")
    
    # Load fine-tuned weights
    print("\n[Step 3/5] Loading fine-tuned weights...")
    finetuned_weights = {}
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    
    # Mapping from checkpoint keys to target model keys
    finetuned_key_mapping = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "enorm.weight": f"model.layers.{mtp_layer_idx}.enorm.weight",
        "hnorm.weight": f"model.layers.{mtp_layer_idx}.hnorm.weight",
        "eh_proj.weight": f"model.layers.{mtp_layer_idx}.eh_proj.weight",
        "norm.weight": f"model.layers.{mtp_layer_idx}.shared_head.norm.weight",
        "lm_head.weight": f"model.layers.{mtp_layer_idx}.shared_head.head.weight",
    }
    
    if os.path.exists(checkpoint_path):
        print(f"  Loading from: {checkpoint_path}")
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in finetuned_key_mapping:
                    target_key = finetuned_key_mapping[key]
                    tensor = f.get_tensor(key)
                    finetuned_weights[target_key] = tensor
                    print(f"    Loaded: {key} -> {target_key} {list(tensor.shape)}")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  Will export original MTP weights only")
    
    # Load all MTP weights from target model
    print("\n[Step 4/5] Loading decoder block weights from target model...")
    index_path = os.path.join(args.target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Find all weights for MTP layer
    files_to_load = defaultdict(list)
    for weight_name, filename in weight_map.items():
        if f"layers.{mtp_layer_idx}" in weight_name or weight_name == "model.embed_tokens.weight":
            files_to_load[filename].append(weight_name)
    
    output_weights = {}
    
    for filename, keys in sorted(files_to_load.items()):
        filepath = os.path.join(args.target_model_path, filename)
        print(f"  Loading: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in keys:
                    # Check if this key should use fine-tuned weight
                    if key in finetuned_weights:
                        output_weights[key] = finetuned_weights[key]
                        print(f"    Using fine-tuned: {key}")
                    else:
                        output_weights[key] = f.get_tensor(key)
    
    print(f"  Total weights: {len(output_weights)}")
    
    # Save weights
    print("\n[Step 5/5] Saving model...")
    
    # Split into multiple files if needed
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    
    current_file_weights = {}
    current_file_size = 0
    file_idx = 1
    saved_files = []
    final_weight_map = {}
    
    for key, tensor in sorted(output_weights.items()):
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_file_size + tensor_size > MAX_FILE_SIZE and current_file_weights:
            filename = f"model-{file_idx:05d}-of-XXXXX.safetensors"
            save_file(current_file_weights, os.path.join(args.output_dir, filename))
            saved_files.append(filename)
            for k in current_file_weights:
                final_weight_map[k] = filename
            print(f"    Saved: {filename}")
            
            current_file_weights = {}
            current_file_size = 0
            file_idx += 1
        
        current_file_weights[key] = tensor
        current_file_size += tensor_size
    
    # Save remaining
    if current_file_weights:
        if len(saved_files) == 0:
            filename = "model.safetensors"
        else:
            filename = f"model-{file_idx:05d}-of-XXXXX.safetensors"
        
        save_file(current_file_weights, os.path.join(args.output_dir, filename))
        saved_files.append(filename)
        for k in current_file_weights:
            final_weight_map[k] = filename
        print(f"    Saved: {filename}")
    
    # Update filenames
    if len(saved_files) > 1:
        total_files = len(saved_files)
        for i, old_name in enumerate(saved_files):
            new_name = old_name.replace("XXXXX", f"{total_files:05d}")
            os.rename(
                os.path.join(args.output_dir, old_name),
                os.path.join(args.output_dir, new_name)
            )
            for k, v in final_weight_map.items():
                if v == old_name:
                    final_weight_map[k] = new_name
            saved_files[i] = new_name
    
    # Create weight index
    total_size = sum(t.numel() * t.element_size() for t in output_weights.values())
    weight_index = {
        "metadata": {"total_size": total_size},
        "weight_map": final_weight_map
    }
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(weight_index, f, indent=2)
    
    # Copy tokenizer
    print("\n  Copying tokenizer files...")
    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(args.target_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, fname))
            print(f"    Copied: {fname}")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModel exported to: {args.output_dir}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"\nFine-tuned weights: {list(finetuned_weights.keys())}")
    print("\nUsage with vLLM-magik:")
    print(f'  --speculative_config \'{{"method":"eagle","model":"{args.output_dir}", "num_speculative_tokens": 3}}\'')
    print("\nUsage with SGLang:")
    print(f"  --speculative-algorithm EAGLE --speculative-draft-model-path {args.output_dir}")


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
