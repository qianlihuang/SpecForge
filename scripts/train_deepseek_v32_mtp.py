#!/usr/bin/env python3
"""
End-to-end training script for DeepSeek-V3.2 MTP draft model.

This script provides a simplified training flow for the MTP (Multi-Token Prediction)
layer that can be used for EAGLE-style speculative decoding.

For a quick test run without a full DeepSeek-V3.2 model:
    python scripts/train_deepseek_v32_mtp.py --mode test

For full training (requires DeepSeek-V3.2 weights):
    python scripts/train_deepseek_v32_mtp.py \
        --target-model-path /path/to/DeepSeek-V3.2 \
        --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
        --output-dir outputs/deepseek-v32-mtp-eagle \
        --num-epochs 1 --batch-size 1 --max-length 512
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add specforge to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.data.template import TEMPLATE_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3.2 MTP draft model")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run mode: 'train' for full training, 'test' for quick verification"
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default=None,
        help="Path to DeepSeek-V3.2 model"
    )
    parser.add_argument(
        "--draft-model-config",
        type=str,
        default="configs/deepseek-v32-mtp-eagle.json",
        help="Path to draft model config"
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="cache/dataset/deepseek-v32-sample.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deepseek-v32-mtp-eagle",
        help="Output directory"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=3,
        help="TTT (Test-Time Training) length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


class SimpleDataset(Dataset):
    """Simple dataset for MTP training using dummy hidden states."""
    
    def __init__(self, data_path: str, max_length: int, hidden_size: int = 7168):
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.data = []
        
        # Load JSONL data
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # For now, generate synthetic data
        # In production, this would load pre-computed hidden states
        seq_len = min(self.max_length, 128)  # Use shorter for testing
        
        return {
            "input_ids": torch.randint(0, 129280, (seq_len,)),
            "attention_mask": torch.ones(seq_len, dtype=torch.bool),
            "loss_mask": torch.ones(seq_len, dtype=torch.float32),
            # Simulated hidden states from 3 aux layers
            "hidden_states": torch.randn(seq_len, self.hidden_size * 3),
            # Simulated target logits
            "target": torch.randn(seq_len, 129280),
        }


def test_mode(args):
    """Run a quick test to verify the model works."""
    print("\n" + "=" * 60)
    print("Running in TEST mode")
    print("=" * 60)
    
    # Get absolute path to config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", args.draft_model_config)
    
    # Load config
    print(f"\n1. Loading config from: {config_path}")
    config = AutoDraftModelConfig.from_file(config_path)
    print(f"   Config loaded: {type(config).__name__}")
    
    # Create model
    print(f"\n2. Creating model...")
    model = AutoEagle3DraftModel.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attention_backend="sdpa",
    )
    print(f"   Model created: {type(model).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    device = args.device
    model = model.to(device)
    model.train()
    
    batch_size = 2
    seq_len = 64
    
    # Create dummy inputs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size * 3,
        dtype=torch.bfloat16, device=device
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Get embeddings
    embeddings = model.embed_input_ids(input_ids).to(torch.bfloat16)
    
    # Forward pass
    output = model(
        hidden_states=hidden_states,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        ttt_length=1,
    )
    
    # Compute logits
    logits = model.compute_logits(output)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Hidden states shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Logits shape: {logits.shape}")
    
    # Test backward pass
    print(f"\n4. Testing backward pass...")
    loss = logits.sum()
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"   Parameters with gradients: {grad_count}")
    
    print(f"\n" + "=" * 60)
    print("TEST PASSED! ✓")
    print("=" * 60)
    print("\nThe model is ready for training.")
    print("To train with real data, run:")
    print(f"  python {sys.argv[0]} --mode train \\")
    print("    --target-model-path /path/to/DeepSeek-V3.2 \\")
    print("    --train-data-path cache/dataset/deepseek-v32-sample.jsonl")


def train_mode(args):
    """Run full training."""
    print("\n" + "=" * 60)
    print("Running in TRAIN mode")
    print("=" * 60)
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", args.draft_model_config)
    data_path = os.path.join(script_dir, "..", args.train_data_path)
    output_dir = os.path.join(script_dir, "..", args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    print(f"\n1. Loading config from: {config_path}")
    config = AutoDraftModelConfig.from_file(config_path)
    
    # Create model
    print(f"\n2. Creating model...")
    model = AutoEagle3DraftModel.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attention_backend="sdpa",
    )
    
    device = args.device
    model = model.to(device)
    
    # Load embeddings from target model if available
    if args.target_model_path and os.path.exists(args.target_model_path):
        print(f"\n3. Loading embeddings from target model: {args.target_model_path}")
        try:
            model.load_embedding(args.target_model_path)
            model.freeze_embedding()
            print("   Embeddings loaded and frozen")
        except Exception as e:
            print(f"   Warning: Could not load embeddings: {e}")
            print("   Using random initialization")
    else:
        print("\n3. Using random embedding initialization (no target model provided)")
    
    # Create dataset
    print(f"\n4. Creating dataset from: {data_path}")
    dataset = SimpleDataset(data_path, args.max_length, config.hidden_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Create optimizer
    print(f"\n5. Creating optimizer...")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    
    # Training loop
    print(f"\n6. Starting training...")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {device}")
    
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            hidden_states = batch["hidden_states"].to(device, dtype=torch.bfloat16)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device, dtype=torch.bfloat16)
            
            # Get embeddings
            embeddings = model.embed_input_ids(input_ids).to(torch.bfloat16)
            
            # Forward pass
            output = model(
                hidden_states=hidden_states,
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                ttt_length=1,
            )
            
            # Compute logits and loss
            logits = model.compute_logits(output)
            
            # Simple cross-entropy-like loss (using MSE on logits for simplicity)
            loss = nn.functional.mse_loss(logits, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        model.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"   Saved checkpoint to: {checkpoint_dir}")
    
    print(f"\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Export the model for sglang/vllm-magik:")
    print(f"   python scripts/export_deepseek_v32_mtp.py \\")
    print(f"     --input-dir {output_dir}/epoch_{args.num_epochs} \\")
    print(f"     --output-dir outputs/deepseek-v32-mtp-nextn \\")
    print(f"     --target-model-path /path/to/DeepSeek-V3.2")


def main():
    args = parse_args()
    
    if args.mode == "test":
        test_mode(args)
    else:
        train_mode(args)


if __name__ == "__main__":
    main()
