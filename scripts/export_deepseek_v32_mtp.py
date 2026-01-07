#!/usr/bin/env python3
"""
Export trained DeepSeek-V3.2 MTP draft model for sglang/vllm-magik inference.

This script converts a trained SpecForge DeepSeekV32MTPForCausalLM checkpoint
to the format expected by sglang (DeepseekV3ForCausalLMNextN).

Usage:
    python scripts/export_deepseek_v32_mtp.py \
        --input-dir outputs/deepseek-v32-mtp-eagle/epoch_0_step_1000 \
        --output-dir outputs/deepseek-v32-mtp-nextn \
        --target-model-path /path/to/DeepSeek-V3.2
"""

import argparse
import json
import os
import shutil
from typing import Dict, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def create_nextn_config(
    target_config: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Create config.json for DeepseekV3ForCausalLMNextN.
    
    The NextN config is essentially the target model config with:
    - num_hidden_layers = 1
    - architectures = ["DeepseekV3ForCausalLMNextN"]
    """
    new_config = dict(target_config)
    new_config.update({
        "num_hidden_layers": 1,
        "architectures": ["DeepseekV3ForCausalLMNextN"],
    })
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Created config: {config_path}")


def convert_weights(
    input_dir: str,
    output_dir: str,
    target_model_path: str,
) -> None:
    """
    Convert SpecForge MTP weights to sglang NextN format.
    
    Weight mapping:
    - SpecForge: embed_tokens, enorm, hnorm, eh_proj, decoder, norm, lm_head
    - sglang: model.embed_tokens, model.enorm, model.hnorm, model.eh_proj, 
              model.layers.0.*, model.shared_head.norm, model.shared_head.head
              
    Key differences:
    - SpecForge uses simplified MoE, sglang uses full MoE
    - For training, we only train certain parameters
    - For export, we need to merge with target model's MoE weights
    """
    # Load trained weights from SpecForge checkpoint
    trained_weights = {}
    
    # Look for safetensors files
    safetensors_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    
    if safetensors_files:
        for sf_file in safetensors_files:
            sf_path = os.path.join(input_dir, sf_file)
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    trained_weights[key] = f.get_tensor(key)
    else:
        # Try loading from pytorch checkpoint
        pt_files = [f for f in os.listdir(input_dir) if f.endswith(".pt") or f.endswith(".bin")]
        for pt_file in pt_files:
            pt_path = os.path.join(input_dir, pt_file)
            state_dict = torch.load(pt_path, map_location="cpu")
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    trained_weights.update(state_dict["model_state_dict"])
                else:
                    trained_weights.update(state_dict)
    
    print(f"Loaded {len(trained_weights)} weights from SpecForge checkpoint")
    
    # Load target model's MTP layer weights (layer 61)
    target_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    target_num_layers = target_config.num_hidden_layers  # Should be 61
    mtp_layer_id = target_num_layers  # Layer 61 (0-indexed: 60 is last transformer, 61 is MTP)
    
    # Find the safetensors index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
    else:
        weight_map = {}
    
    # Prepare output weights with NextN naming convention
    output_weights = {}
    
    # Map trained weights to NextN format
    weight_mapping = {
        # Embedding (shared with target model)
        "embed_tokens.weight": "model.embed_tokens.weight",
        # MTP-specific layers
        "enorm.weight": "model.enorm.weight",
        "hnorm.weight": "model.hnorm.weight",
        "eh_proj.weight": "model.eh_proj.weight",
        # Decoder layer -> model.layers.0 (for NextN, we use decoder prefix)
        "decoder.input_layernorm.weight": "model.decoder.input_layernorm.weight",
        "decoder.post_attention_layernorm.weight": "model.decoder.post_attention_layernorm.weight",
        # Attention (MLA)
        "decoder.self_attn.q_a_proj.weight": "model.decoder.self_attn.q_a_proj.weight",
        "decoder.self_attn.q_a_layernorm.weight": "model.decoder.self_attn.q_a_layernorm.weight",
        "decoder.self_attn.q_b_proj.weight": "model.decoder.self_attn.q_b_proj.weight",
        "decoder.self_attn.kv_a_proj_with_mqa.weight": "model.decoder.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.self_attn.kv_a_layernorm.weight": "model.decoder.self_attn.kv_a_layernorm.weight",
        "decoder.self_attn.kv_b_proj.weight": "model.decoder.self_attn.kv_b_proj.weight",
        "decoder.self_attn.o_proj.weight": "model.decoder.self_attn.o_proj.weight",
        # MLP/MoE - these will be loaded from target model
        # Output layers
        "norm.weight": "model.shared_head.norm.weight",
        "lm_head.weight": "model.shared_head.head.weight",
    }
    
    for src_key, dst_key in weight_mapping.items():
        if src_key in trained_weights:
            output_weights[dst_key] = trained_weights[src_key]
            print(f"  Mapped: {src_key} -> {dst_key}")
    
    # For MoE weights, we need to load from the target model's MTP layer
    # This is essential because SpecForge's simplified MoE is for training only
    print("\nLoading MoE weights from target model's MTP layer...")
    
    mtp_prefix = f"model.layers.{mtp_layer_id}"
    
    # Find and load MoE weights from target model
    files_to_load = set()
    moe_keys = []
    
    for key, filename in weight_map.items():
        if key.startswith(mtp_prefix) and "mlp" in key.lower():
            files_to_load.add(filename)
            moe_keys.append(key)
    
    for filename in files_to_load:
        file_path = os.path.join(target_model_path, filename)
        if os.path.exists(file_path):
            print(f"  Loading MoE weights from: {filename}")
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(mtp_prefix) and "mlp" in key.lower():
                        # Map to NextN format (model.layers.61 -> model.decoder)
                        new_key = key.replace(mtp_prefix, "model.decoder")
                        output_weights[new_key] = f.get_tensor(key)
    
    # Save output weights
    output_path = os.path.join(output_dir, "model.safetensors")
    print(f"\nSaving {len(output_weights)} weights to: {output_path}")
    save_file(output_weights, output_path)
    
    # Create weight index
    index_data = {"weight_map": {key: "model.safetensors" for key in output_weights}}
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)
    print(f"Created index: {index_path}")


def copy_tokenizer_files(target_model_path: str, output_dir: str) -> None:
    """Copy tokenizer files from target model."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    
    for filename in tokenizer_files:
        src_path = os.path.join(target_model_path, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained DeepSeek-V3.2 MTP draft model for sglang/vllm inference"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to SpecForge trained checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for NextN model",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to DeepSeek-V3.2 model (for MoE weights and config)",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load target model config
    target_config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=True
    ).to_dict()
    
    print(f"Target model has {target_config['num_hidden_layers']} layers")
    print(f"MTP layer (layer 61) will be exported\n")
    
    # Create NextN config
    create_nextn_config(target_config, args.output_dir)
    
    # Convert weights
    convert_weights(args.input_dir, args.output_dir, args.target_model_path)
    
    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    copy_tokenizer_files(args.target_model_path, args.output_dir)
    
    print(f"\n{'=' * 60}")
    print(f"Export complete! Model saved to: {args.output_dir}")
    print(f"{'=' * 60}")
    print("\nTo use with sglang:")
    print(f"  --speculative-algorithm EAGLE3 \\")
    print(f"  --speculative-draft-model-path {args.output_dir} \\")
    print(f"  --speculative-num-steps 3 --speculative-eagle-topk 1 \\")
    print(f"  --speculative-num-draft-tokens 4")
    print("\nTo use with vllm-magik:")
    print(f"  --speculative_config '{{\"method\":\"eagle\",\"model\":\"{args.output_dir}\", ...}}'")


if __name__ == "__main__":
    main()
