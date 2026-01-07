#!/usr/bin/env python3
"""
Test script for DeepSeek-V3.2 MTP Draft Model

This script tests the basic functionality of the DeepSeekV32MTPForCausalLM model
without requiring a full target model or GPU resources for a full training run.
"""

import os
import sys

# Add the specforge path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import json

from specforge.modeling.draft.deepseek_v32_mtp import (
    DeepSeekV32MTPConfig,
    DeepSeekV32MTPForCausalLM,
)
from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel


def test_config_loading():
    """Test loading the configuration file."""
    print("=" * 60)
    print("Test 1: Configuration Loading")
    print("=" * 60)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "deepseek-v32-mtp-eagle.json")
    
    # Load raw config
    with open(config_path, "r") as f:
        raw_config = json.load(f)
    
    print(f"Loaded config from: {config_path}")
    print(f"Architecture: {raw_config.get('architectures', ['Unknown'])}")
    print(f"Model type: {raw_config.get('model_type', 'Unknown')}")
    print(f"Hidden size: {raw_config.get('hidden_size', 'Unknown')}")
    print(f"Vocab size: {raw_config.get('vocab_size', 'Unknown')}")
    print(f"Draft vocab size: {raw_config.get('draft_vocab_size', 'Unknown')}")
    print()
    
    # Test AutoDraftModelConfig
    config = AutoDraftModelConfig.from_file(config_path)
    print(f"AutoDraftModelConfig loaded successfully!")
    print(f"Config type: {type(config)}")
    print(f"Config hidden_size: {config.hidden_size}")
    print(f"Config vocab_size: {config.vocab_size}")
    
    print("✓ Configuration loading test passed!\n")
    return config


def test_model_creation(config):
    """Test creating the model from config."""
    print("=" * 60)
    print("Test 2: Model Creation")
    print("=" * 60)
    
    # Create model from config
    model = AutoEagle3DraftModel.from_config(
        config, 
        torch_dtype=torch.bfloat16,
        attention_backend="sdpa"
    )
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Model hidden_size: {model.hidden_size}")
    print(f"Model vocab_size: {model.vocab_size}")
    print(f"Model draft_vocab_size: {model.draft_vocab_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("✓ Model creation test passed!\n")
    return model


def test_forward_pass(model, device="cpu"):
    """Test a basic forward pass."""
    print("=" * 60)
    print("Test 3: Forward Pass")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    seq_length = 128
    hidden_size = model.hidden_size
    
    # Create dummy inputs
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
    
    # Simulate hidden states from 3 aux layers (EAGLE3 style)
    hidden_states = torch.randn(
        batch_size, seq_length, hidden_size * 3,
        dtype=torch.bfloat16, device=device
    )
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    
    with torch.no_grad():
        # Test embed_input_ids
        embeddings = model.embed_input_ids(input_ids)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Test project_hidden_states
        projected = model.project_hidden_states(hidden_states)
        print(f"Projected hidden states shape: {projected.shape}")
        
        # Test full forward (backbone + compute_logits)
        output = model.forward(
            hidden_states=hidden_states,
            inputs_embeds=embeddings.to(torch.bfloat16),
            attention_mask=attention_mask,
            ttt_length=1,
        )
        print(f"Forward output shape: {output.shape}")
        
        # Test compute_logits
        logits = model.compute_logits(output)
        print(f"Logits shape: {logits.shape}")
    
    print("✓ Forward pass test passed!\n")


def test_backbone(model, device="cpu"):
    """Test the backbone method directly."""
    print("=" * 60)
    print("Test 4: Backbone Method")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    
    batch_size = 2
    seq_length = 64
    hidden_size = model.hidden_size
    
    # Create inputs
    input_embeds = torch.randn(
        batch_size, seq_length, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    hidden_states = torch.randn(
        batch_size, seq_length, hidden_size,
        dtype=torch.bfloat16, device=device
    )
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Input embeds shape: {input_embeds.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    with torch.no_grad():
        output = model.backbone(
            input_embeds=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        print(f"Backbone output shape: {output.shape}")
    
    print("✓ Backbone test passed!\n")


def test_t2d_d2t_buffers(model):
    """Test the vocab mapping buffers."""
    print("=" * 60)
    print("Test 5: Vocab Mapping Buffers")
    print("=" * 60)
    
    print(f"t2d buffer shape: {model.t2d.shape}")
    print(f"t2d buffer dtype: {model.t2d.dtype}")
    print(f"d2t buffer shape: {model.d2t.shape}")
    print(f"d2t buffer dtype: {model.d2t.dtype}")
    
    # Check default values
    assert model.t2d.shape[0] == model.vocab_size, "t2d should have vocab_size elements"
    assert model.d2t.shape[0] == model.draft_vocab_size, "d2t should have draft_vocab_size elements"
    
    print("✓ Vocab mapping buffers test passed!\n")


def main():
    print("\n" + "=" * 60)
    print("DeepSeek-V3.2 MTP Draft Model Tests")
    print("=" * 60 + "\n")
    
    # Test 1: Config loading
    config = test_config_loading()
    
    # Test 2: Model creation
    model = test_model_creation(config)
    
    # Test 3: Forward pass
    test_forward_pass(model)
    
    # Test 4: Backbone
    test_backbone(model)
    
    # Test 5: Vocab mapping buffers
    test_t2d_d2t_buffers(model)
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nThe DeepSeekV32MTPForCausalLM model is ready for training.")
    print("You can now run the full training with:")
    print("  python scripts/train_eagle3.py \\")
    print("    --target-model-path /path/to/DeepSeek-V3.2 \\")
    print("    --draft-model-config configs/deepseek-v32-mtp-eagle.json \\")
    print("    --train-data-path cache/dataset/deepseek-v32-sample.jsonl \\")
    print("    --chat-template deepseek-v32 \\")
    print("    --output-dir outputs/deepseek-v32-mtp-eagle \\")
    print("    --num-epochs 1 --batch-size 1 --max-length 512")
    

if __name__ == "__main__":
    main()
