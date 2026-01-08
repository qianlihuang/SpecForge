#!/usr/bin/env python3
"""
Fine-tune DeepSeek-V3.2 MTP layer projection components for EAGLE speculative decoding.

This script fine-tunes ONLY the lightweight projection components of the MTP layer:
- enorm: RMSNorm for input embeddings  
- hnorm: RMSNorm for hidden states
- eh_proj: Linear projection from concat(embed, hidden) -> hidden
- norm: Output RMSNorm (shared_head.norm)
- lm_head: Output projection (shared_head.head)

The decoder block (attention + MoE) is NOT trained and is kept frozen from the original model.

This approach is useful for domain adaptation while keeping the heavy MoE weights unchanged.

Usage:
    torchrun --nproc_per_node=8 scripts/train_mtp_layer_v2.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --hidden-states-path cache/hidden_states/deepseek-v32-mtp \
        --output-dir outputs/deepseek-v32-mtp-finetuned \
        --num-epochs 3 \
        --batch-size 1 \
        --learning-rate 1e-5
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.utils import (
    init_distributed,
    is_tp_rank_0,
)


# ============================================================================
# MTP Layer Model Definition (Projection-only, no decoder block)
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


class MTPProjectionModel(nn.Module):
    """
    MTP projection-only model for fine-tuning.
    
    IMPORTANT: This model trains ONLY projection components (enorm, hnorm, eh_proj, norm, lm_head)
    while SKIPPING the decoder block. This creates an ARCHITECTURE MISMATCH with inference!
    
    At TRAINING time:
        logits = lm_head(norm(eh_proj(concat[enorm(embed), hnorm(hidden)])))
    
    At INFERENCE time (vLLM/SGLang):
        logits = lm_head(norm(DECODER_BLOCK(eh_proj(concat[enorm(embed), hnorm(hidden)]))))
    
    This mismatch means the trained weights may not work well at inference.
    
    RECOMMENDED: Use train_mtp_full.py instead for full MTP layer training, OR
    use train_mtp_full.py with --freeze-moe flag to freeze MoE but still include
    the decoder block in the forward pass.
    
    This script is kept for reference and for cases where you want lightweight
    projection-only adaptation as a starting point.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Embedding (frozen, loaded from target model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # MTP projection components (trainable)
        self.enorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # Output (trainable)
        self.norm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_size]
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for MTP projection training.
        
        WARNING: This skips the decoder block! See class docstring for details.
        """
        # Get embeddings for input tokens
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Normalize
        inputs_embeds = self.enorm(inputs_embeds)
        hidden_normed = self.hnorm(hidden_states)
        
        # Concatenate and project
        combined = torch.cat([inputs_embeds, hidden_normed], dim=-1)
        projected = self.eh_proj(combined)
        
        # Output (SKIPPING decoder block - causes mismatch!)
        output = self.norm(projected)
        logits = self.lm_head(output)
        
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
        
        input_ids = data["input_ids"]
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        
        return {
            "input_ids": input_ids,  # [seq_len]
            "hidden_state": hidden_state,  # [seq_len, hidden_size]
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # For now, assume batch_size=1 (sequences may have different lengths)
    if len(batch) == 1:
        item = batch[0]
        return {
            "input_ids": item["input_ids"].unsqueeze(0),  # [1, seq_len]
            "hidden_states": item["hidden_state"].unsqueeze(0),  # [1, seq_len, hidden_size]
            "labels": item["input_ids"].unsqueeze(0),  # Same as input_ids for next-token prediction
        }
    else:
        # TODO: Implement padding for batch_size > 1
        raise NotImplementedError("Batch size > 1 requires padding implementation")


# ============================================================================
# Weight Loading
# ============================================================================

def load_mtp_weights(model: MTPProjectionModel, target_model_path: str, device: torch.device):
    """Load MTP layer weights from target model.
    
    In DeepSeek-V3.2:
    - Layers 0-60 are regular decoder layers
    - Layer 61 is the MTP prediction layer
    """
    print(f"Loading MTP weights from: {target_model_path}")
    
    # Load weight index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Map our model keys to target model keys
    key_mapping = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "enorm.weight": "model.layers.61.enorm.weight",
        "hnorm.weight": "model.layers.61.hnorm.weight",
        "eh_proj.weight": "model.layers.61.eh_proj.weight",
        "norm.weight": "model.layers.61.shared_head.norm.weight",
        "lm_head.weight": "model.layers.61.shared_head.head.weight",
    }
    
    # Find files to load
    files_to_load = {}
    for model_key, target_key in key_mapping.items():
        if target_key in weight_map:
            filename = weight_map[target_key]
            if filename not in files_to_load:
                files_to_load[filename] = []
            files_to_load[filename].append((model_key, target_key))
    
    # Load weights
    loaded_keys = []
    for filename, key_pairs in files_to_load.items():
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading from: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for model_key, target_key in key_pairs:
                if target_key in [k for k in f.keys()]:
                    tensor = f.get_tensor(target_key)
                    # Convert FP8 to bfloat16 if needed
                    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        print(f"    Skipping FP8 weight: {target_key}")
                        continue
                    
                    # Get the parameter
                    param = model
                    for part in model_key.split('.'):
                        param = getattr(param, part)
                    
                    # Load weight
                    with torch.no_grad():
                        param.copy_(tensor.to(device))
                    loaded_keys.append(model_key)
                    print(f"    Loaded: {model_key} <- {target_key} {list(tensor.shape)}")
    
    print(f"  Loaded {len(loaded_keys)} weights")
    return loaded_keys


# ============================================================================
# Training
# ============================================================================

def train(args):
    """Main training function."""
    # Initialize distributed
    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}")
    
    if is_tp_rank_0():
        print("=" * 60)
        print("MTP Projection Training")
        print("=" * 60)
        print(f"Rank: {rank}, World Size: {world_size}")
        print(f"Target Model: {args.target_model_path}")
        print(f"Hidden States: {args.hidden_states_path}")
        print(f"Output Dir: {args.output_dir}")
    
    # Load config
    config_path = os.path.join(args.target_model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Create config object
    @dataclass
    class Config:
        hidden_size: int = config_dict.get("hidden_size", 7168)
        vocab_size: int = config_dict.get("vocab_size", 129280)
        rms_norm_eps: float = config_dict.get("rms_norm_eps", 1e-6)
        intermediate_size: int = config_dict.get("intermediate_size", 18432)
    
    config = Config()
    
    if is_tp_rank_0():
        print(f"\nModel Config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  rms_norm_eps: {config.rms_norm_eps}")
    
    # Create model
    model = MTPProjectionModel(config).to(device)
    
    # Load weights from target model
    if is_tp_rank_0():
        print("\nLoading weights from target model...")
    load_mtp_weights(model, args.target_model_path, device)
    
    # Freeze embed_tokens (we don't want to change the embeddings)
    model.embed_tokens.weight.requires_grad = False
    
    # Convert to bfloat16 for training
    model = model.to(torch.bfloat16)
    
    # Create dataset and dataloader
    dataset = MTPOfflineDataset(
        args.hidden_states_path,
        max_samples=args.max_samples,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    if is_tp_rank_0():
        print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not is_tp_rank_0())
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            hidden_states = batch["hidden_states"].to(device, torch.bfloat16)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                labels=labels,
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if is_tp_rank_0():
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model weights
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
            
            # Save config
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump({
                    "hidden_size": config.hidden_size,
                    "vocab_size": config.vocab_size,
                    "rms_norm_eps": config.rms_norm_eps,
                }, f, indent=2)
            
            print(f"  Saved checkpoint to: {checkpoint_dir}")
    
    if is_tp_rank_0():
        print("\nTraining complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MTP projection layer")
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
