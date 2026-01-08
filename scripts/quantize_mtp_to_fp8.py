#!/usr/bin/env python3
"""
Quantize fine-tuned MTP model weights to FP8 for efficient inference.

This script takes a fine-tuned MTP checkpoint (BF16) and quantizes the MoE expert
weights to FP8 format, matching the original DeepSeek-V3.2 quantization scheme.

Usage:
    python scripts/quantize_mtp_to_fp8.py \
        --input-dir outputs/deepseek-v32-mtp-eagle \
        --output-dir outputs/deepseek-v32-mtp-eagle-fp8
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def quantize_weight_to_fp8(
    weight_bf16: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize BF16 weight to FP8 using block-wise scaling.
    
    Args:
        weight_bf16: [out_features, in_features] in BF16
        block_size: Block size for quantization (default 128 for DeepSeek)
    
    Returns:
        weight_fp8: [out_features, in_features] in FP8
        scale_inv: [out_features/block_size, in_features/block_size] in FP32
    """
    out_features, in_features = weight_bf16.shape
    
    # Calculate number of blocks
    out_blocks = (out_features + block_size - 1) // block_size
    in_blocks = (in_features + block_size - 1) // block_size
    
    # Pad if necessary
    pad_out = out_blocks * block_size - out_features
    pad_in = in_blocks * block_size - in_features
    if pad_out > 0 or pad_in > 0:
        weight_bf16 = torch.nn.functional.pad(weight_bf16, (0, pad_in, 0, pad_out))
    
    # Convert to float32 for computation
    weight_f32 = weight_bf16.to(torch.float32)
    
    # Reshape to blocks
    weight_blocked = weight_f32.view(out_blocks, block_size, in_blocks, block_size)
    
    # Compute max absolute value per block for scaling
    # FP8 E4M3 range: [-448, 448]
    fp8_max = 448.0
    block_max = weight_blocked.abs().amax(dim=(1, 3), keepdim=True)
    
    # Compute scale
    scale = block_max / fp8_max
    scale = scale.clamp(min=1e-12)
    scale_inv = 1.0 / scale
    
    # Quantize
    weight_scaled = weight_blocked / scale
    weight_scaled = weight_scaled.clamp(-fp8_max, fp8_max)
    
    # Reshape back
    weight_scaled = weight_scaled.view(out_blocks * block_size, in_blocks * block_size)
    scale_inv = scale_inv.squeeze(1).squeeze(-1)
    
    # Remove padding
    if pad_out > 0 or pad_in > 0:
        weight_scaled = weight_scaled[:out_features, :in_features]
    
    # Convert to FP8
    weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)
    
    return weight_fp8, scale_inv.to(torch.float32)


def should_quantize(key: str) -> bool:
    """Check if a weight should be quantized to FP8."""
    # Quantize MoE expert weights (gate_proj, up_proj, down_proj)
    if "mlp.experts." in key and any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
        if "weight_scale_inv" not in key:
            return True
    # Also quantize shared expert weights
    if "mlp.shared_experts" in key and any(x in key for x in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
        if "weight_scale_inv" not in key:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize MTP model to FP8")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory with BF16 model")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for FP8 model")
    parser.add_argument("--block-size", type=int, default=128,
                       help="Block size for quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Quantizing MTP Model to FP8")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Block size: {args.block_size}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(args.input_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, fname))
            print(f"Copied: {fname}")
    
    # Load weight index
    index_path = os.path.join(args.input_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files_to_process = set(weight_map.values())
    else:
        # Single file model
        files_to_process = ["model.safetensors"]
        weight_map = None
    
    # Process each file
    output_weights = {}
    stats = {"quantized": 0, "kept": 0, "total_size_before": 0, "total_size_after": 0}
    
    for filename in tqdm(sorted(files_to_process), desc="Processing files"):
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Skip existing scale_inv tensors (we'll regenerate them)
                if "weight_scale_inv" in key:
                    continue
                
                tensor = f.get_tensor(key)
                stats["total_size_before"] += tensor.numel() * tensor.element_size()
                
                if should_quantize(key):
                    # Quantize to FP8
                    if tensor.dtype == torch.bfloat16:
                        weight_fp8, scale_inv = quantize_weight_to_fp8(tensor, args.block_size)
                        output_weights[key] = weight_fp8
                        scale_key = key.replace(".weight", ".weight_scale_inv")
                        output_weights[scale_key] = scale_inv
                        stats["quantized"] += 1
                        stats["total_size_after"] += weight_fp8.numel() * weight_fp8.element_size()
                        stats["total_size_after"] += scale_inv.numel() * scale_inv.element_size()
                    elif tensor.dtype == torch.float8_e4m3fn:
                        # Already FP8, keep as is and copy scale_inv
                        output_weights[key] = tensor
                        stats["kept"] += 1
                        stats["total_size_after"] += tensor.numel() * tensor.element_size()
                else:
                    # Keep other weights as is
                    output_weights[key] = tensor
                    stats["kept"] += 1
                    stats["total_size_after"] += tensor.numel() * tensor.element_size()
    
    # Also get scale_inv from original files for weights that were already FP8
    for filename in sorted(files_to_process):
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            continue
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "weight_scale_inv" in key and key not in output_weights:
                    output_weights[key] = f.get_tensor(key)
    
    print(f"\nQuantization stats:")
    print(f"  Quantized: {stats['quantized']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Size before: {stats['total_size_before'] / (1024**3):.2f} GB")
    print(f"  Size after: {stats['total_size_after'] / (1024**3):.2f} GB")
    
    # Save weights (split into multiple files if needed)
    print("\nSaving quantized model...")
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
    if len(saved_files) > 1:
        total_size = sum(t.numel() * t.element_size() for t in output_weights.values())
        weight_index = {
            "metadata": {"total_size": total_size},
            "weight_map": final_weight_map
        }
        index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(weight_index, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Files: {len(saved_files)}")
    
    # Verify
    print("\nVerifying output...")
    total_fp8 = 0
    total_other = 0
    for fname in saved_files:
        with safe_open(os.path.join(args.output_dir, fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dtype == torch.float8_e4m3fn:
                    total_fp8 += 1
                else:
                    total_other += 1
    print(f"  FP8 tensors: {total_fp8}")
    print(f"  Other tensors: {total_other}")


if __name__ == "__main__":
    main()
