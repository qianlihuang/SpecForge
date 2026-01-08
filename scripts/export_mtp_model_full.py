#!/usr/bin/env python3
"""
Export fine-tuned Full MTP model for EAGLE speculative decoding.

This script exports a fine-tuned MTP layer that includes:
- Projection components: enorm, hnorm, eh_proj  
- Full decoder block: attention + MoE
- Output head: shared_head.norm, shared_head.head

The output format is compatible with vLLM-magik and SGLang for EAGLE decoding.

Usage:
    python scripts/export_mtp_model_full.py \
        --checkpoint-dir outputs/deepseek-v32-mtp-full/checkpoint-epoch-3 \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --output-dir outputs/deepseek-v32-mtp-eagle \
        --quantize-to-fp8  # Optional: quantize MoE weights to FP8
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def quantize_weight_to_fp8(
    weight_bf16: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 weight to FP8 using block-wise scaling."""
    out_features, in_features = weight_bf16.shape
    out_blocks = (out_features + block_size - 1) // block_size
    in_blocks = (in_features + block_size - 1) // block_size
    
    pad_out = out_blocks * block_size - out_features
    pad_in = in_blocks * block_size - in_features
    if pad_out > 0 or pad_in > 0:
        weight_bf16 = torch.nn.functional.pad(weight_bf16, (0, pad_in, 0, pad_out))
    
    weight_f32 = weight_bf16.to(torch.float32)
    weight_blocked = weight_f32.view(out_blocks, block_size, in_blocks, block_size)
    
    fp8_max = 448.0
    block_max = weight_blocked.abs().amax(dim=(1, 3), keepdim=True)
    scale = block_max / fp8_max
    scale = scale.clamp(min=1e-12)
    scale_inv = 1.0 / scale
    
    weight_scaled = weight_blocked / scale
    weight_scaled = weight_scaled.clamp(-fp8_max, fp8_max)
    weight_scaled = weight_scaled.view(out_blocks * block_size, in_blocks * block_size)
    scale_inv = scale_inv.squeeze(1).squeeze(-1)
    
    if pad_out > 0 or pad_in > 0:
        weight_scaled = weight_scaled[:out_features, :in_features]
    
    return weight_scaled.to(torch.float8_e4m3fn), scale_inv.to(torch.float32)


def should_quantize_to_fp8(key: str) -> bool:
    """Check if a weight should be quantized to FP8."""
    if "mlp.experts." in key and any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
        if "weight_scale_inv" not in key:
            return True
    if "mlp.shared_experts" in key and any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
        if "weight_scale_inv" not in key:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Export full MTP model for EAGLE")
    
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to fine-tuned checkpoint")
    parser.add_argument("--target-model-path", type=str, required=True,
                       help="Path to original DeepSeek-V3.2 model")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--quantize-to-fp8", action="store_true",
                       help="Quantize MoE weights to FP8 (reduces size by ~50%%)")
    parser.add_argument("--block-size", type=int, default=128,
                       help="Block size for FP8 quantization")
    
    return parser.parse_args()


def create_mtp_config(target_config: dict) -> dict:
    """Create MTP model config compatible with vLLM-magik.
    
    Note: We use model_type="deepseek_v32" instead of "deepseek_mtp" because
    vLLM-magik's SpeculativeConfig.hf_config_override() expects deepseek_v32/v3
    and transforms it to deepseek_mtp internally. Using deepseek_mtp directly
    causes Transformers AutoConfig to fail since it doesn't recognize that type.
    
    Important: num_hidden_layers must match the target model's layer count (61)
    because vLLM's EagleDeepseekV3ForCausalLM uses it as the start_layer_id when
    num_nextn_predict_layers is present. The weights are only for the MTP layer
    but the config tells vLLM which layer index to load them into.
    """
    # Get target model's layer count - needed for vLLM to compute start_layer_id
    target_num_layers = target_config.get("num_hidden_layers", 61)
    
    return {
        "architectures": ["DeepSeekMTPModel"],
        "model_type": "deepseek_v32",  # vLLM will convert this to deepseek_mtp
        "vocab_size": target_config.get("vocab_size", 129280),
        "hidden_size": target_config.get("hidden_size", 7168),
        "intermediate_size": target_config.get("intermediate_size", 18432),
        "moe_intermediate_size": target_config.get("moe_intermediate_size", 2048),
        "num_hidden_layers": target_num_layers,  # Must match target model!
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
        # Quantization config (if present)
        "quantization_config": target_config.get("quantization_config"),
    }


def map_checkpoint_key_to_target(key: str, mtp_layer_idx: int = 61) -> Optional[str]:
    """Map checkpoint key to target model key format.
    
    Checkpoint format: embed_tokens.weight, enorm.weight, decoder_block.self_attn.*, etc.
    Target format: model.layers.61.enorm.weight, model.layers.61.self_attn.*, etc.
    """
    if key == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    
    prefix_mappings = {
        "enorm.weight": f"model.layers.{mtp_layer_idx}.enorm.weight",
        "hnorm.weight": f"model.layers.{mtp_layer_idx}.hnorm.weight",
        "eh_proj.weight": f"model.layers.{mtp_layer_idx}.eh_proj.weight",
        "norm.weight": f"model.layers.{mtp_layer_idx}.shared_head.norm.weight",
        "lm_head.weight": f"model.layers.{mtp_layer_idx}.shared_head.head.weight",
    }
    
    if key in prefix_mappings:
        return prefix_mappings[key]
    
    # Map decoder_block.* to model.layers.61.*
    if key.startswith("decoder_block."):
        rest = key[len("decoder_block."):]
        return f"model.layers.{mtp_layer_idx}.{rest}"
    
    return None


def export_model(args):
    """Export the fine-tuned MTP model."""
    print("=" * 60)
    print("Exporting Full MTP Model for EAGLE")
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
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    finetuned_weights = {}
    
    if os.path.exists(checkpoint_path):
        print(f"  Loading from: {checkpoint_path}")
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                target_key = map_checkpoint_key_to_target(key, mtp_layer_idx)
                if target_key:
                    tensor = f.get_tensor(key)
                    finetuned_weights[target_key] = tensor
                    print(f"    {key} -> {target_key}")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
    
    print(f"  Loaded {len(finetuned_weights)} fine-tuned weights")
    
    # Load original MTP layer weights from target model
    print("\n[Step 4/5] Loading remaining weights from target model...")
    index_path = os.path.join(args.target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    # Find files containing MTP layer weights
    files_to_load = defaultdict(list)
    for weight_name, filename in weight_map.items():
        if f"layers.{mtp_layer_idx}" in weight_name or weight_name == "model.embed_tokens.weight":
            files_to_load[filename].append(weight_name)
    
    output_weights = {}
    stats = {"finetuned": 0, "original": 0, "quantized": 0}
    
    for filename, keys in tqdm(sorted(files_to_load.items()), desc="Loading weights"):
        filepath = os.path.join(args.target_model_path, filename)
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in keys:
                    continue
                
                # Skip scale_inv if we're using finetuned weights that need requantization
                if "weight_scale_inv" in key:
                    base_key = key.replace("_scale_inv", "")
                    if base_key in finetuned_weights:
                        # Will regenerate scale_inv after quantization
                        continue
                    else:
                        # Keep original scale_inv
                        output_weights[key] = f.get_tensor(key)
                    continue
                
                # Use fine-tuned weight if available
                if key in finetuned_weights:
                    tensor = finetuned_weights[key]
                    stats["finetuned"] += 1
                    
                    # Quantize to FP8 if requested
                    if args.quantize_to_fp8 and should_quantize_to_fp8(key):
                        if tensor.dtype in [torch.bfloat16, torch.float32, torch.float16]:
                            weight_fp8, scale_inv = quantize_weight_to_fp8(tensor, args.block_size)
                            output_weights[key] = weight_fp8
                            scale_key = key.replace(".weight", ".weight_scale_inv")
                            output_weights[scale_key] = scale_inv
                            stats["quantized"] += 1
                        else:
                            output_weights[key] = tensor
                    else:
                        output_weights[key] = tensor
                else:
                    # Use original weight
                    output_weights[key] = f.get_tensor(key)
                    stats["original"] += 1
    
    print(f"  Stats: finetuned={stats['finetuned']}, original={stats['original']}, quantized={stats['quantized']}")
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
    
    for key, tensor in tqdm(sorted(output_weights.items()), desc="Saving"):
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_file_size + tensor_size > MAX_FILE_SIZE and current_file_weights:
            filename = f"model-{file_idx:05d}-of-XXXXX.safetensors"
            save_file(current_file_weights, os.path.join(args.output_dir, filename))
            saved_files.append(filename)
            for k in current_file_weights:
                final_weight_map[k] = filename
            
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
    
    # Copy tokenizer files
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
    print(f"Fine-tuned weights: {len(finetuned_weights)}")
    print("\nUsage with vLLM-magik:")
    print(f'  --speculative_config \'{{"method":"eagle","model":"{args.output_dir}", "num_speculative_tokens": 3}}\'')
    print("\nUsage with SGLang:")
    print(f"  --speculative-algorithm EAGLE --speculative-draft-model-path {args.output_dir}")


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
