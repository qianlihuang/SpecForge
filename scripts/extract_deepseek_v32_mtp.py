#!/usr/bin/env python3
"""
Extract DeepSeek-V3.2 MTP layer for EAGLE speculative decoding.

This script extracts the MTP layer (layer 61) from DeepSeek-V3.2 and creates
a standalone model directory that can be used with vLLM-magik or SGLang
for EAGLE speculative decoding.

NO FINE-TUNING IS NEEDED - DeepSeek-V3.2 already has a trained MTP layer.

Usage:
    python scripts/extract_deepseek_v32_mtp.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --output-dir outputs/deepseek-v32-mtp-eagle

For vLLM-magik:
    --speculative_config '{"method":"eagle","model":"outputs/deepseek-v32-mtp-eagle", "num_speculative_tokens": 3}'

For SGLang:
    --speculative-algorithm EAGLE --speculative-draft-model-path outputs/deepseek-v32-mtp-eagle
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DeepSeek-V3.2 MTP layer for EAGLE speculative decoding"
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to DeepSeek-V3.2 model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extracted MTP model",
    )
    parser.add_argument(
        "--skip-fp8",
        action="store_true",
        default=False,
        help="Skip FP8 quantized weights (use dequantized versions)",
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
        "num_hidden_layers": 1,  # MTP has 1 decoder layer
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
        # V3.2 specific - for NSA (Native Sparse Attention)
        "index_head_dim": target_config.get("index_head_dim", 128),
        "index_n_heads": target_config.get("index_n_heads", 64),
        "index_topk": target_config.get("index_topk", 2048),
    }


def extract_mtp_model(args):
    """Extract MTP layer from DeepSeek-V3.2."""
    print("=" * 60)
    print("Extracting DeepSeek-V3.2 MTP Layer for EAGLE")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load target model config
    print("\n[Step 1/4] Loading target model config...")
    target_config_path = os.path.join(args.target_model_path, "config.json")
    with open(target_config_path, "r") as f:
        target_config = json.load(f)
    
    # Verify it's DeepSeek-V3.2 (has num_nextn_predict_layers)
    if target_config.get("num_nextn_predict_layers", 0) == 0:
        print("WARNING: Model does not appear to have MTP layers (num_nextn_predict_layers=0)")
    
    # Create MTP config
    print("\n[Step 2/4] Creating MTP config...")
    mtp_config = create_mtp_config(target_config)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(mtp_config, f, indent=2)
    print(f"  Saved config to: {config_path}")
    
    # Load weight index
    print("\n[Step 3/4] Extracting MTP weights...")
    index_path = os.path.join(args.target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Find all weights for layer 61 (MTP layer) and embedding
    # Layer 61 is the MTP layer in DeepSeek-V3.2 (layers 0-60 are regular decoder layers)
    mtp_layer_idx = target_config.get("num_hidden_layers", 61)  # Should be 61
    print(f"  MTP layer index: {mtp_layer_idx}")
    
    files_to_load = defaultdict(list)
    for weight_name, filename in weight_map.items():
        # MTP layer weights
        if f"layers.{mtp_layer_idx}" in weight_name:
            files_to_load[filename].append(weight_name)
        # Main embedding (shared with MTP)
        elif weight_name == "model.embed_tokens.weight":
            files_to_load[filename].append(weight_name)
    
    print(f"  Found {sum(len(v) for v in files_to_load.values())} weight tensors in {len(files_to_load)} files")
    
    # Load and save weights
    output_weights = {}
    total_loaded = 0
    skipped_fp8 = 0
    
    for filename, keys in sorted(files_to_load.items()):
        filepath = os.path.join(args.target_model_path, filename)
        print(f"  Loading: {filename} ({len(keys)} tensors)")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in keys:
                    tensor = f.get_tensor(key)
                    
                    # Skip FP8 weights if requested (use scale_inv for dequantization later)
                    if args.skip_fp8 and tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        skipped_fp8 += 1
                        continue
                    
                    output_weights[key] = tensor
                    total_loaded += 1
    
    print(f"  Loaded {total_loaded} tensors")
    if skipped_fp8 > 0:
        print(f"  Skipped {skipped_fp8} FP8 tensors")
    
    # Save weights
    print("\n[Step 4/4] Saving MTP model...")
    
    # Split into multiple files if too large (safetensors limit is ~5GB per file)
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    
    current_file_weights = {}
    current_file_size = 0
    file_idx = 1
    saved_files = []
    final_weight_map = {}
    
    for key, tensor in sorted(output_weights.items()):
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_file_size + tensor_size > MAX_FILE_SIZE and current_file_weights:
            # Save current file
            filename = f"model-{file_idx:05d}-of-XXXXX.safetensors"
            save_file(current_file_weights, os.path.join(args.output_dir, filename))
            saved_files.append(filename)
            for k in current_file_weights:
                final_weight_map[k] = filename
            print(f"    Saved: {filename} ({len(current_file_weights)} tensors)")
            
            current_file_weights = {}
            current_file_size = 0
            file_idx += 1
        
        current_file_weights[key] = tensor
        current_file_size += tensor_size
    
    # Save remaining weights
    if current_file_weights:
        if len(saved_files) == 0:
            # Single file, use simple name
            filename = "model.safetensors"
        else:
            filename = f"model-{file_idx:05d}-of-XXXXX.safetensors"
        
        save_file(current_file_weights, os.path.join(args.output_dir, filename))
        saved_files.append(filename)
        for k in current_file_weights:
            final_weight_map[k] = filename
        print(f"    Saved: {filename} ({len(current_file_weights)} tensors)")
    
    # Update filenames with correct total
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
    
    # Copy tokenizer files
    print("\n  Copying tokenizer files...")
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    for fname in tokenizer_files:
        src = os.path.join(args.target_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, fname))
            print(f"    Copied: {fname}")
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print(f"\nMTP model extracted to: {args.output_dir}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Files: {saved_files}")
    print("\nUsage with vLLM-magik:")
    print(f'  --speculative_config \'{{"method":"eagle","model":"{args.output_dir}", "num_speculative_tokens": 3}}\'')
    print("\nUsage with SGLang:")
    print(f"  --speculative-algorithm EAGLE --speculative-draft-model-path {args.output_dir}")


def main():
    args = parse_args()
    extract_mtp_model(args)


if __name__ == "__main__":
    main()
