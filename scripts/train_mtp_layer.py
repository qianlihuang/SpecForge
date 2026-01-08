#!/usr/bin/env python3
"""
Train DeepSeek-V3.2 MTP layer for EAGLE speculative decoding (Offline Mode).

This script fine-tunes the MTP layer (layer 60) using pre-generated hidden states.
The trained model can be used with vLLM-magik or SGLang for EAGLE speculative decoding.

Usage:
    torchrun --standalone --nproc_per_node=8 \
        scripts/train_mtp_layer.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
        --train-hidden-states-path cache/hidden_states/deepseek-v32-mtp \
        --output-dir outputs/deepseek-v32-mtp-finetuned \
        --num-epochs 3 \
        --batch-size 1 \
        --learning-rate 1e-5
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
    is_tp_rank_0,
)


# ============================================================================
# MTP Layer Model Definition
# ============================================================================

class DeepSeekV32RMSNorm(nn.Module):
    """RMSNorm for DeepSeek-V3.2"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MTPLayerModel(nn.Module):
    """
    Simplified MTP layer model for training.
    
    The MTP layer takes:
    - hidden_states: output from layer 59 [batch, seq_len, hidden_size]
    - input_embeds: embedding of next token [batch, seq_len, hidden_size]
    
    And produces logits for next token prediction.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Embedding (shared with target model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # MTP normalization layers
        self.enorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Projection: concat(embed, hidden) -> hidden
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # Simplified decoder (we skip the full MoE for now, use simple FFN)
        # In practice, you'd want to use the full decoder layer
        self.input_layernorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp_up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp_down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # Output
        self.norm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size] from layer 59
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        # Get embeddings for input tokens (shifted by 1 for next token prediction)
        # For MTP, we use the embedding of position i as input for predicting position i+1
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Normalize
        inputs_embeds = self.enorm(inputs_embeds)
        hidden_states = self.hnorm(hidden_states)
        
        # Concatenate and project
        combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden = self.eh_proj(combined)
        
        # Simple FFN (replace with full decoder layer in production)
        residual = hidden
        hidden = self.input_layernorm(hidden)
        gate = torch.nn.functional.silu(self.mlp_gate_proj(hidden))
        up = self.mlp_up_proj(hidden)
        hidden = self.mlp_down_proj(gate * up)
        hidden = residual + hidden
        
        # Output
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            
            if loss_mask is not None:
                shift_loss_mask = loss_mask[..., 1:].contiguous().view(-1)
                loss = loss * shift_loss_mask
                loss = loss.sum() / (shift_loss_mask.sum() + 1e-8)
            else:
                loss = loss.mean()
        
        return {"loss": loss, "logits": logits}


# ============================================================================
# Dataset
# ============================================================================

class MTPOfflineDataset(Dataset):
    """Dataset for offline MTP training using pre-generated hidden states."""
    
    def __init__(
        self,
        hidden_states_path: str,
        max_samples: Optional[int] = None,
    ):
        self.hidden_states_path = Path(hidden_states_path)
        self.file_paths = []
        
        # Find all data files
        for subdir in sorted(self.hidden_states_path.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("rows_"):
                for f in sorted(subdir.iterdir()):
                    if f.suffix == ".ckpt":
                        self.file_paths.append(f)
        
        if max_samples is not None:
            self.file_paths = self.file_paths[:max_samples]
        
        print(f"Found {len(self.file_paths)} data files in {hidden_states_path}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=False)
        
        # Handle both old and new data formats
        if "hidden_state" in data:
            hidden_state = data["hidden_state"]
            if hidden_state.dim() == 3:
                hidden_state = hidden_state.squeeze(0)  # Remove batch dim if present
        else:
            raise KeyError(f"No hidden_state found in {self.file_paths[idx]}")
        
        return {
            "input_ids": data["input_ids"],
            "loss_mask": data["loss_mask"],
            "hidden_state": hidden_state,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
        "hidden_state": torch.stack([b["hidden_state"] for b in batch]),
    }


# ============================================================================
# Weight Loading
# ============================================================================

def load_mtp_weights_from_target(
    target_model_path: str,
    model: nn.Module,
    device: str = "cpu",
) -> Dict[str, int]:
    """
    Load MTP layer weights from target DeepSeek-V3.2 model.
    
    In DeepSeek-V3.2:
    - Layers 0-60 are regular decoder layers
    - Layer 61 is the MTP prediction layer (with enorm, hnorm, eh_proj, shared_head)
    
    This loads the non-quantized weights (layernorms, projections) from the target model.
    """
    print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Load weight map
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Weight mapping from target to our model
    # Layer 61 is the MTP layer with special components
    weight_mapping = {
        # Embedding
        "model.embed_tokens.weight": "embed_tokens.weight",
        # MTP-specific normalization (layer 61)
        "model.layers.61.enorm.weight": "enorm.weight",
        "model.layers.61.hnorm.weight": "hnorm.weight",
        "model.layers.61.eh_proj.weight": "eh_proj.weight",
        # Output head (layer 61)
        "model.layers.61.shared_head.norm.weight": "norm.weight",
        "model.layers.61.shared_head.head.weight": "lm_head.weight",
    }
    
    # Find files to load
    files_to_load = set()
    for key in weight_mapping.keys():
        if key in weight_map:
            files_to_load.add(weight_map[key])
    
    print(f"  Loading from {len(files_to_load)} files")
    
    stats = {"loaded": 0, "skipped": 0, "shape_mismatch": 0}
    loaded_weights = {}
    
    for filename in sorted(files_to_load):
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_mapping:
                    tensor = f.get_tensor(key)
                    
                    # Skip FP8 quantized weights
                    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        print(f"    ⊘ Skipped (FP8): {key}")
                        stats["skipped"] += 1
                        continue
                    
                    # Convert to bfloat16
                    if tensor.dtype != torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                    
                    target_key = weight_mapping[key]
                    loaded_weights[target_key] = tensor
                    stats["loaded"] += 1
    
    # Load into model
    model_state = model.state_dict()
    for key, tensor in loaded_weights.items():
        if key in model_state:
            if model_state[key].shape == tensor.shape:
                model_state[key].copy_(tensor)
                print(f"    ✓ Loaded: {key} {list(tensor.shape)}")
            else:
                print(f"    ✗ Shape mismatch: {key}")
                stats["shape_mismatch"] += 1
    
    print(f"\n  Summary: loaded={stats['loaded']}, skipped={stats['skipped']}")
    return stats


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    log_interval: int = 10,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        hidden_state = batch["hidden_state"].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            hidden_states=hidden_state,
            labels=input_ids,
            loss_mask=loss_mask,
        )
        
        loss = outputs["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    config: dict,
    output_dir: str,
    epoch: int,
):
    """Save model checkpoint in format compatible with vLLM-magik."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save weights
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
    
    # Save config
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  Saved checkpoint to: {checkpoint_dir}")
    return checkpoint_dir


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSeek-V3.2 MTP layer for EAGLE speculative decoding"
    )
    
    # Model
    parser.add_argument(
        "--target-model-path",
        type=str,
        default="/data/models/DeepSeek-V3.2",
        help="Path to DeepSeek-V3.2 model",
    )
    parser.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help="Path to draft model config",
    )
    
    # Data
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--train-hidden-states-path",
        type=str,
        required=True,
        help="Path to pre-generated hidden states",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use",
    )
    
    # Training
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deepseek-v32-mtp-finetuned",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DeepSeek-V3.2 MTP Layer Training (Offline Mode)")
    print("=" * 60)
    print(f"Target model: {args.target_model_path}")
    print(f"Hidden states: {args.train_hidden_states_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config
    print("\n[Step 1/5] Loading config...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    if args.draft_model_config is None:
        args.draft_model_config = os.path.join(base_dir, "configs", "deepseek-v32-mtp.json")
    
    with open(args.draft_model_config, "r") as f:
        config_dict = json.load(f)
    
    # Create simple config object
    class Config:
        pass
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)
    
    # Create model
    print("\n[Step 2/5] Creating MTP model...")
    model = MTPLayerModel(config)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load weights from target model
    print("\n[Step 3/5] Loading weights from target model...")
    load_mtp_weights_from_target(args.target_model_path, model)
    
    model = model.to(device).to(torch.bfloat16)
    
    # Create dataset
    print("\n[Step 4/5] Creating dataset...")
    dataset = MTPOfflineDataset(
        hidden_states_path=args.train_hidden_states_path,
        max_samples=args.max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    # Training
    print("\n[Step 5/5] Training...")
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        if epoch % args.save_interval == 0:
            save_checkpoint(model, config_dict, args.output_dir, epoch)
    
    # Save final model
    print("\n" + "=" * 60)
    print("Training complete!")
    final_dir = save_checkpoint(model, config_dict, args.output_dir, args.num_epochs)
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Export the model for vLLM-magik/SGLang:")
    print(f"   python scripts/export_mtp_model.py \\")
    print(f"     --input-dir {final_dir} \\")
    print(f"     --target-model-path {args.target_model_path} \\")
    print(f"     --output-dir outputs/deepseek-v32-mtp-eagle")


if __name__ == "__main__":
    main()
