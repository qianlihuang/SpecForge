#!/usr/bin/env python3
"""
Test script to verify the extracted DeepSeek-V3.2 MTP model can be loaded.

This script doesn't run inference - it just verifies that:
1. The config is valid
2. The weights can be loaded
3. The model structure matches expectations

For actual inference testing, use the vllm or sglang launch commands.
"""

import json
import os
import sys

import torch
from safetensors import safe_open


def test_config(model_path: str):
    """Test that config is valid."""
    print("\n[1] Testing config...")
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Check required fields
    required_fields = [
        "model_type",
        "architectures",
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_nextn_predict_layers",
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"  ❌ Missing required field: {field}")
            return False
        print(f"  ✓ {field}: {config[field]}")
    
    # Check model type
    if config["model_type"] != "deepseek_mtp":
        print(f"  ❌ Expected model_type 'deepseek_mtp', got '{config['model_type']}'")
        return False
    
    print("  ✓ Config is valid")
    return True


def test_weights(model_path: str):
    """Test that weights can be loaded."""
    print("\n[2] Testing weights...")
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    print(f"  Total weights: {len(weight_map)}")
    
    # Check for required weight keys
    required_keys = [
        "model.embed_tokens.weight",
        "model.layers.61.enorm.weight",
        "model.layers.61.hnorm.weight",
        "model.layers.61.eh_proj.weight",
        "model.layers.61.shared_head.norm.weight",
        "model.layers.61.shared_head.head.weight",
        "model.layers.61.input_layernorm.weight",
        "model.layers.61.post_attention_layernorm.weight",
    ]
    
    for key in required_keys:
        if key not in weight_map:
            print(f"  ❌ Missing required weight: {key}")
            return False
        print(f"  ✓ {key}")
    
    # Check for MoE weights
    moe_expert_count = sum(1 for k in weight_map if "mlp.experts" in k)
    print(f"  ✓ MoE expert weights: {moe_expert_count}")
    
    # Check for attention weights
    attn_weight_count = sum(1 for k in weight_map if "self_attn" in k)
    print(f"  ✓ Attention weights: {attn_weight_count}")
    
    # Try loading one weight file
    files = set(weight_map.values())
    print(f"  Weight files: {len(files)}")
    
    for filename in list(files)[:1]:
        filepath = os.path.join(model_path, filename)
        print(f"  Testing load: {filename}")
        try:
            with safe_open(filepath, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"    ✓ Loaded {len(keys)} tensors")
                # Check one tensor
                tensor = f.get_tensor(keys[0])
                print(f"    ✓ Sample tensor: {keys[0]} shape={list(tensor.shape)} dtype={tensor.dtype}")
        except Exception as e:
            print(f"    ❌ Failed to load: {e}")
            return False
    
    print("  ✓ Weights are valid")
    return True


def test_tokenizer(model_path: str):
    """Test that tokenizer files exist."""
    print("\n[3] Testing tokenizer...")
    
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    for fname in tokenizer_files:
        filepath = os.path.join(model_path, fname)
        if not os.path.exists(filepath):
            print(f"  ❌ Missing tokenizer file: {fname}")
            return False
        print(f"  ✓ {fname} exists")
    
    print("  ✓ Tokenizer files are present")
    return True


def main():
    model_path = "/sgl-workspace/SpecForge/outputs/deepseek-v32-mtp-eagle-extracted"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("=" * 60)
    print("Testing DeepSeek-V3.2 MTP Model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    
    all_passed = True
    
    all_passed &= test_config(model_path)
    all_passed &= test_weights(model_path)
    all_passed &= test_tokenizer(model_path)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED")
        print("=" * 60)
        print("\nThe model is ready for inference. Use one of these commands:")
        print("\nFor vLLM-magik:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model /data/models/DeepSeek-V3.2 \\")
        print(f"    --tensor-parallel-size 8 \\")
        print(f"    --speculative_config '{{\"method\":\"eagle\",\"model\":\"{model_path}\", \"num_speculative_tokens\": 3}}'")
        print("\nFor SGLang:")
        print(f"  python -m sglang.launch_server \\")
        print(f"    --model /data/models/DeepSeek-V3.2 \\")
        print(f"    --tp 8 \\")
        print(f"    --speculative-algorithm EAGLE \\")
        print(f"    --speculative-draft-model-path {model_path}")
    else:
        print("✗ Some tests FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
