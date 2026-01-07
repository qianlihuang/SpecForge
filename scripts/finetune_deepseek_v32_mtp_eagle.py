#!/usr/bin/env python
"""
Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding.

This script supports:
1. EAGLE mode (no aux layers, only last hidden state)
2. SGLang backend for fast hidden states generation (tp8)
3. Offline mode (pre-generate hidden states) and Online mode (generate on-the-fly)

Usage:
# Offline mode - Step 1: Generate hidden states using sglang tp8
torchrun --nproc_per_node=8 scripts/finetune_deepseek_v32_mtp_eagle.py \
    --mode generate \
    --target-model-path /data/models/DeepSeek-V3.2 \
    --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
    --cache-dir cache/hidden_states/deepseek-v32-eagle \
    --max-length 2048 \
    --batch-size 8 \
    --tp-size 8

# Offline mode - Step 2: Train MTP layer
python scripts/finetune_deepseek_v32_mtp_eagle.py \
    --mode offline \
    --target-model-path /data/models/DeepSeek-V3.2 \
    --train-hidden-states-path cache/hidden_states/deepseek-v32-eagle \
    --output-dir outputs/deepseek-v32-mtp-eagle \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-4

# Online mode (generate + train together, slower but simpler)
torchrun --nproc_per_node=8 scripts/finetune_deepseek_v32_mtp_eagle.py \
    --mode online \
    --target-model-path /data/models/DeepSeek-V3.2 \
    --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
    --output-dir outputs/deepseek-v32-mtp-eagle-online \
    --tp-size 8 \
    --num-epochs 3
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, safe_open, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Add specforge to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DataPoint:
    """Data point for EAGLE training (no aux hidden states)."""
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    hidden_state: torch.Tensor  # Only last hidden state for EAGLE


class HiddenStatesDataset(Dataset):
    """Dataset that loads pre-generated hidden states for offline training."""
    
    def __init__(self, cache_dir: str, max_samples: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        
        # Load metadata
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.num_samples = self.metadata.get("num_samples", 0)
        else:
            # Count files
            self.num_samples = len(list(self.cache_dir.glob("sample_*.pt")))
            self.metadata = {"num_samples": self.num_samples}
        
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)
        
        # Try to find hidden size
        if self.num_samples > 0:
            sample = torch.load(self.cache_dir / "sample_0.pt", map_location="cpu", weights_only=False)
            if "hidden_state" in sample:
                self.hidden_size = sample["hidden_state"].shape[-1]
            elif "last_hidden" in sample:
                self.hidden_size = sample["last_hidden"].shape[-1]
            else:
                self.hidden_size = self.metadata.get("hidden_size", 7168)
        else:
            self.hidden_size = self.metadata.get("hidden_size", 7168)
        
        print(f"Loaded dataset: {self.num_samples} samples, hidden_size={self.hidden_size}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample_path = self.cache_dir / f"sample_{idx}.pt"
        data = torch.load(sample_path, map_location="cpu", weights_only=False)
        
        # Handle both old format (last_hidden) and new format (hidden_state)
        if "hidden_state" in data:
            hidden_state = data["hidden_state"]
        elif "last_hidden" in data:
            hidden_state = data["last_hidden"]
        else:
            raise KeyError(f"No hidden_state or last_hidden in sample {idx}")
        
        # Ensure proper shape [seq_len, hidden_size] without batch dim
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        
        return {
            "input_ids": data["input_ids"].squeeze(0) if data["input_ids"].dim() > 1 else data["input_ids"],
            "attention_mask": data.get("attention_mask", torch.ones_like(data["input_ids"])).squeeze(0),
            "hidden_state": hidden_state,
            "loss_mask": data.get("loss_mask", torch.ones_like(data["input_ids"])).squeeze(0),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "hidden_state": torch.stack([b["hidden_state"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
    }


def load_mtp_weights_from_target(
    target_model_path: str,
    draft_model: nn.Module,
) -> int:
    """
    Load MTP layer weights from target DeepSeek-V3.2 model.
    Only loads non-quantized weights (layernorms, gate, etc.)
    
    Returns: number of weights loaded
    """
    print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Load index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Weight mapping: target -> draft (non-quantized only for EAGLE)
    weight_mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.layers.61.enorm.weight": "enorm.weight",
        "model.layers.61.hnorm.weight": "hnorm.weight",
        "model.layers.61.eh_proj.weight": "eh_proj.weight",
        "model.layers.61.input_layernorm.weight": "decoder.input_layernorm.weight",
        "model.layers.61.post_attention_layernorm.weight": "decoder.post_attention_layernorm.weight",
        "model.layers.61.self_attn.q_a_layernorm.weight": "decoder.self_attn.q_a_layernorm.weight",
        "model.layers.61.self_attn.kv_a_layernorm.weight": "decoder.self_attn.kv_a_layernorm.weight",
        "model.layers.61.mlp.gate.weight": "decoder.mlp.gate.weight",
        "model.layers.61.shared_head.norm.weight": "norm.weight",
        "model.layers.61.shared_head.head.weight": "lm_head.weight",
    }
    
    # Find which files to load
    files_to_load = set()
    for key in weight_mapping.keys():
        if key in weight_map:
            files_to_load.add(weight_map[key])
    
    # Load weights
    loaded_count = 0
    model_state = draft_model.state_dict()
    
    for filename in sorted(files_to_load):
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_mapping:
                    target_key = weight_mapping[key]
                    tensor = f.get_tensor(key)
                    
                    # Skip FP8 quantized tensors
                    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        continue
                    
                    if target_key in model_state:
                        if model_state[target_key].shape == tensor.shape:
                            model_state[target_key].copy_(tensor)
                            print(f"    ✓ {target_key}")
                            loaded_count += 1
                        else:
                            print(f"    ✗ {target_key} shape mismatch: {model_state[target_key].shape} vs {tensor.shape}")
    
    draft_model.load_state_dict(model_state)
    print(f"  Total loaded: {loaded_count} weights")
    return loaded_count


def generate_hidden_states_sglang(
    target_model_path: str,
    train_data_path: str,
    cache_dir: str,
    max_length: int = 2048,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    tp_size: int = 8,
):
    """
    Generate hidden states using SGLang backend with tensor parallelism.
    This is much faster than transformers.
    """
    from specforge.args import SGLangBackendArgs
    from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
    from specforge.distributed import (
        destroy_distributed,
        get_dp_group,
        get_tp_group,
        init_distributed,
        is_tp_rank_0,
    )
    from specforge.modeling.target import get_eagle3_target_model
    
    # Initialize distributed
    init_distributed(tp_size=tp_size, dist_timeout=60)
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        print("=" * 60)
        print("Generating Hidden States (SGLang Backend)")
        print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    
    # Load config to get model info
    config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    
    if rank == 0:
        print(f"  Target model: {target_model_path}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  TP size: {tp_size}")
    
    # Build dataset
    dataset = build_eagle3_dataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        chat_template="raw",  # Assume pre-formatted
        num_proc=8,
        num_samples=max_samples,
    )
    
    # Build dataloaders
    train_loader, _ = prepare_dp_dataloaders(
        dataset, 
        batch_size=batch_size, 
        num_workers=4
    )
    
    if rank == 0:
        print(f"  Dataset size: {len(dataset)}")
    
    # Build target model with SGLang backend
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=target_model_path,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        device="cuda",
        trust_remote_code=True,
    )
    
    # For EAGLE, we don't need aux hidden states
    # Just capture the last hidden state before MTP layer
    target_model.set_aux_hidden_states_layers([num_layers - 1])  # Last layer only
    
    if rank == 0:
        print("\nGenerating hidden states...")
    
    # Generate
    global_idx = 0
    tp_group = get_tp_group()
    
    for batch in tqdm(train_loader, disable=(rank != 0), desc="Generating"):
        batch_size_actual = batch["input_ids"].size(0)
        
        # Move to GPU
        batch_gpu = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Get hidden states from target model
        logits, aux_hidden_list, last_hidden_list = target_model.extend(
            input_ids=batch_gpu["input_ids"],
            attention_mask=batch_gpu["attention_mask"],
            loss_mask=batch_gpu.get("loss_mask", torch.ones_like(batch_gpu["input_ids"])),
            return_last_hidden_states=True,
            return_logits=False,
        )
        
        # Save on rank 0 only
        if is_tp_rank_0():
            for i in range(batch_size_actual):
                sample_data = {
                    "input_ids": batch["input_ids"][i].cpu(),
                    "attention_mask": batch["attention_mask"][i].cpu(),
                    "loss_mask": batch.get("loss_mask", torch.ones_like(batch["input_ids"]))[i].cpu(),
                    "hidden_state": last_hidden_list[i].cpu() if last_hidden_list[i] is not None else None,
                }
                
                save_path = cache_path / f"sample_{global_idx + i}.pt"
                torch.save(sample_data, save_path)
        
        global_idx += batch_size_actual
        
        # Clear cache periodically
        if global_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    # Save metadata
    if is_tp_rank_0():
        metadata = {
            "num_samples": global_idx,
            "hidden_size": hidden_size,
            "max_length": max_length,
            "target_model_path": target_model_path,
            "tp_size": tp_size,
            "mode": "eagle",  # No aux layers
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {global_idx} samples to {cache_dir}")
    
    destroy_distributed()


def generate_hidden_states_transformers(
    target_model_path: str,
    train_data_path: str,
    cache_dir: str,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
):
    """
    Generate hidden states using transformers (slower but works without distributed setup).
    Uses device_map='auto' to spread across GPUs.
    """
    import tempfile
    import shutil
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Hidden States (Transformers Backend)")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    
    # Load training data
    samples = []
    with open(train_data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    print(f"  Loaded {len(samples)} samples")
    
    # Handle DeepSeek-V3.2 config
    config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    
    # Fix model_type for transformers compatibility
    if hasattr(config, 'model_type') and config.model_type == "deepseek_v32":
        print("  Converting deepseek_v32 config for transformers...")
        
        # Create temp config
        temp_dir = tempfile.mkdtemp()
        shutil.copytree(target_model_path, temp_dir, dirs_exist_ok=True)
        
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config_dict["model_type"] = "deepseek_v3"
        config_dict["architectures"] = ["DeepseekV3ForCausalLM"]
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        load_path = temp_dir
    else:
        load_path = target_model_path
        temp_dir = None
    
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    print(f"  Hidden size: {hidden_size}, Num layers: {num_layers}")
    
    # Load model
    print("  Loading target model...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        output_hidden_states=True,
    )
    model.eval()
    print("  Model loaded")
    
    # Clean up temp dir
    if temp_dir:
        shutil.rmtree(temp_dir)
    
    # Generate hidden states
    print("\nGenerating hidden states...")
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        text = sample.get("text", sample.get("content", ""))
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                output_hidden_states=True,
                use_cache=False,
            )
        
        # Get last hidden state (before MTP layer)
        # For EAGLE, we only need the last transformer layer's output
        all_hidden = outputs.hidden_states
        last_hidden = all_hidden[-1]  # Last layer output
        
        sample_data = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "loss_mask": attention_mask.cpu(),
            "hidden_state": last_hidden.cpu(),
        }
        
        torch.save(sample_data, cache_path / f"sample_{idx}.pt")
        
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save metadata
    metadata = {
        "num_samples": len(samples),
        "hidden_size": hidden_size,
        "max_length": max_length,
        "target_model_path": target_model_path,
        "mode": "eagle",
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved {len(samples)} samples to {cache_dir}")


def train_offline(
    train_hidden_states_path: str,
    target_model_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_samples: Optional[int] = None,
    device: str = "cuda",
):
    """
    Offline training: use pre-generated hidden states.
    """
    from specforge.modeling.draft.deepseek_v32_mtp import (
        DeepSeekV32MTPConfig,
        DeepSeekV32MTPForCausalLM,
    )
    
    print("=" * 60)
    print("Offline Training Mode")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading hidden states from: {train_hidden_states_path}")
    dataset = HiddenStatesDataset(train_hidden_states_path, max_samples)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "deepseek-v32-mtp-eagle.json"
    print(f"\n2. Loading config from: {config_path}")
    config = DeepSeekV32MTPConfig.from_pretrained(config_path)
    
    # Create model
    print("\n3. Creating draft model...")
    draft_model = DeepSeekV32MTPForCausalLM(config)
    param_count = sum(p.numel() for p in draft_model.parameters())
    print(f"   Parameters: {param_count:,}")
    
    # Load MTP weights
    print("\n4. Loading MTP weights from target model...")
    load_mtp_weights_from_target(target_model_path, draft_model)
    
    # Move to device
    draft_model = draft_model.to(device).to(torch.bfloat16)
    draft_model.train()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in draft_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    
    # Training loop
    print(f"\n5. Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Total batches: {len(dataloader)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden_state = batch["hidden_state"].to(device).to(torch.bfloat16)
            
            # Forward pass
            outputs = draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=hidden_state,
                labels=input_ids,
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save model
        save_file(draft_model.state_dict(), os.path.join(ckpt_dir, "model.safetensors"))
        config.save_pretrained(ckpt_dir)
        
        # Save training state
        torch.save({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(ckpt_dir, "training_state.pt"))
        
        print(f"   Saved checkpoint to: {ckpt_dir}")
    
    print(f"\n✓ Training complete! Checkpoints saved to: {output_dir}")


def train_online(
    target_model_path: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
    tp_size: int = 8,
):
    """
    Online training: generate hidden states on-the-fly and train.
    Requires distributed setup for SGLang.
    """
    from specforge.args import SGLangBackendArgs
    from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
    from specforge.distributed import (
        destroy_distributed,
        get_dp_group,
        get_tp_group,
        init_distributed,
        is_tp_rank_0,
    )
    from specforge.modeling.target import get_eagle3_target_model
    from specforge.modeling.draft.deepseek_v32_mtp import (
        DeepSeekV32MTPConfig,
        DeepSeekV32MTPForCausalLM,
    )
    
    # Initialize distributed
    init_distributed(tp_size=tp_size, dist_timeout=60)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        print("=" * 60)
        print("Online Training Mode")
        print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    
    # Build dataset
    dataset = build_eagle3_dataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        chat_template="raw",
        num_proc=8,
        num_samples=max_samples,
    )
    
    train_loader, _ = prepare_dp_dataloaders(dataset, batch_size=batch_size, num_workers=4)
    
    if rank == 0:
        print(f"  Dataset size: {len(dataset)}")
    
    # Build target model
    config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=target_model_path,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        device="cuda",
        trust_remote_code=True,
    )
    target_model.set_aux_hidden_states_layers([num_layers - 1])
    
    # Build draft model (only on rank 0 for training)
    if is_tp_rank_0():
        config_path = Path(__file__).parent.parent / "configs" / "deepseek-v32-mtp-eagle.json"
        draft_config = DeepSeekV32MTPConfig.from_pretrained(config_path)
        draft_model = DeepSeekV32MTPForCausalLM(draft_config)
        load_mtp_weights_from_target(target_model_path, draft_model)
        draft_model = draft_model.cuda().to(torch.bfloat16)
        draft_model.train()
        
        optimizer = torch.optim.AdamW(
            [p for p in draft_model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        
        os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        if is_tp_rank_0():
            epoch_loss = 0.0
            num_batches = 0
        
        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            # Generate hidden states
            batch_gpu = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            _, _, last_hidden_list = target_model.extend(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                loss_mask=batch_gpu.get("loss_mask", torch.ones_like(batch_gpu["input_ids"])),
                return_last_hidden_states=True,
                return_logits=False,
            )
            
            # Train on rank 0
            if is_tp_rank_0():
                # Stack hidden states
                hidden_states = torch.stack([h for h in last_hidden_list if h is not None])
                
                outputs = draft_model(
                    input_ids=batch_gpu["input_ids"],
                    attention_mask=batch_gpu["attention_mask"],
                    hidden_states=hidden_states.to(torch.bfloat16),
                    labels=batch_gpu["input_ids"],
                )
                
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Save checkpoint
        if is_tp_rank_0():
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            ckpt_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_file(draft_model.state_dict(), os.path.join(ckpt_dir, "model.safetensors"))
            draft_config.save_pretrained(ckpt_dir)
            print(f"   Saved checkpoint to: {ckpt_dir}")
    
    if is_tp_rank_0():
        print(f"\n✓ Training complete! Checkpoints saved to: {output_dir}")
    
    destroy_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-V3.2 MTP for EAGLE")
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate", "offline", "online"],
        help="Mode: generate (create hidden states), offline (train from cached), online (generate + train)"
    )
    
    # Model paths
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--train-hidden-states-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/deepseek-v32-mtp-eagle")
    parser.add_argument("--cache-dir", type=str, default="cache/hidden_states/deepseek-v32-eagle")
    
    # Training params
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    
    # Distributed params
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--backend", type=str, choices=["sglang", "transformers"], default="transformers",
                       help="Backend for hidden states generation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set CUDA architecture for H200
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
        print("Set TORCH_CUDA_ARCH_LIST to 9.0")
    
    if args.mode == "generate":
        # Generate hidden states
        if not args.train_data_path:
            raise ValueError("--train-data-path required for generate mode")
        
        if args.backend == "sglang":
            generate_hidden_states_sglang(
                target_model_path=args.target_model_path,
                train_data_path=args.train_data_path,
                cache_dir=args.cache_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                tp_size=args.tp_size,
            )
        else:
            generate_hidden_states_transformers(
                target_model_path=args.target_model_path,
                train_data_path=args.train_data_path,
                cache_dir=args.cache_dir,
                max_length=args.max_length,
                max_samples=args.max_samples,
            )
    
    elif args.mode == "offline":
        # Offline training
        if not args.train_hidden_states_path:
            raise ValueError("--train-hidden-states-path required for offline mode")
        
        train_offline(
            train_hidden_states_path=args.train_hidden_states_path,
            target_model_path=args.target_model_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
        )
    
    elif args.mode == "online":
        # Online training
        if not args.train_data_path:
            raise ValueError("--train-data-path required for online mode")
        
        train_online(
            target_model_path=args.target_model_path,
            train_data_path=args.train_data_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_samples=args.max_samples,
            tp_size=args.tp_size,
        )


if __name__ == "__main__":
    main()
