#!/usr/bin/env python3
"""
Online training script for DeepSeek-V3.2 MTP layer.

This script trains the MTP layer with on-the-fly hidden state generation using SGLang,
similar to SpecForge's online EAGLE3 training approach.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from specforge.data import build_eagle3_dataset
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Online MTP layer training for DeepSeek-V3.2")
    
    # Model paths
    parser.add_argument("--target-model-path", type=str, required=True,
                        help="Path to the target DeepSeek-V3.2 model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the training data (JSONL)")
    parser.add_argument("--output-dir", type=str, default="outputs/deepseek-v32-mtp-online",
                        help="Output directory for checkpoints")
    
    # Data processing
    parser.add_argument("--chat-template", type=str, default="deepseek-v32",
                        help="Chat template to use")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # Distributed training
    parser.add_argument("--tp-size", type=int, default=8,
                        help="Tensor parallel size for SGLang")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # SGLang parameters
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.70,
                        help="SGLang static memory fraction")
    parser.add_argument("--sglang-page-size", type=int, default=64,
                        help="SGLang page size")
    
    return parser.parse_args()


class MTPLayerModel(nn.Module):
    """Simplified MTP layer model for training."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 7168)
        self.vocab_size = config.get("vocab_size", 129280)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        
        # MTP-specific components
        # enorm: normalize embedding input
        self.enorm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        # hnorm: normalize hidden state input  
        self.hnorm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        # eh_proj: project concatenated [embedding, hidden] to hidden_size
        self.eh_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        
        # Input layer norm (before MTP processing)
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        
        # Shared head components
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Embedding layer (shared with target model)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MTP layer.
        
        Args:
            hidden_states: Hidden states from last decoder layer [batch, seq_len, hidden_size]
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Logits for next token prediction [batch, seq_len, vocab_size]
        """
        # Get embeddings for input tokens
        embeddings = self.embed_tokens(input_ids)
        
        # Normalize both inputs
        normed_emb = self.enorm(embeddings)
        normed_hidden = self.hnorm(hidden_states)
        
        # Concatenate and project
        combined = torch.cat([normed_emb, normed_hidden], dim=-1)
        projected = self.eh_proj(combined)
        
        # Apply input layer norm
        projected = self.input_layernorm(projected)
        
        # Apply shared head
        normed = self.norm(projected)
        logits = self.lm_head(normed)
        
        return logits


def load_config(model_path: str) -> dict:
    """Load model config."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_mtp_weights_from_target(model: MTPLayerModel, target_model_path: str, rank: int = 0):
    """Load MTP weights from target DeepSeek-V3.2 model."""
    if rank == 0:
        print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Find weight files containing MTP layer (layer 61)
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find files containing our target weights
    # MTP layer is layer 61 in DeepSeek-V3.2
    target_keys = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.layers.61.enorm.weight": "enorm.weight",
        "model.layers.61.hnorm.weight": "hnorm.weight", 
        "model.layers.61.eh_proj.weight": "eh_proj.weight",
        "model.layers.61.input_layernorm.weight": "input_layernorm.weight",
        "model.layers.61.shared_head.norm.weight": "norm.weight",
        "model.layers.61.shared_head.head.weight": "lm_head.weight",
    }
    
    files_to_load = set()
    for key in target_keys.keys():
        if key in weight_map:
            files_to_load.add(weight_map[key])
    
    if rank == 0:
        print(f"  Loading from {len(files_to_load)} files")
    
    loaded_weights = {}
    for filename in files_to_load:
        filepath = os.path.join(target_model_path, filename)
        if rank == 0:
            print(f"  Loading: {filename}")
        weights = load_file(filepath)
        for key, local_key in target_keys.items():
            if key in weights:
                loaded_weights[local_key] = weights[key]
                if rank == 0:
                    print(f"    ✓ Loaded: {local_key} {list(weights[key].shape)}")
    
    # Load into model
    missing, unexpected = model.load_state_dict(loaded_weights, strict=False)
    if rank == 0:
        print(f"\n  Summary: loaded={len(loaded_weights)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing}")
    
    return model


def init_sglang_engine(target_model_path: str, tp_size: int, mem_fraction: float, page_size: int, rank: int = 0):
    """Initialize SGLang engine for hidden state generation."""
    if rank == 0:
        print(f"\nInitializing SGLang engine...")
        print(f"  Model: {target_model_path}")
        print(f"  TP size: {tp_size}")
        print(f"  Memory fraction: {mem_fraction}")
    
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine
    
    engine = Engine(
        model_path=target_model_path,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction,
        page_size=page_size,
        trust_remote_code=True,
        enable_torch_compile=False,
        enable_return_hidden_states=True,  # Fixed: correct argument name
    )
    
    if rank == 0:
        print("  SGLang engine initialized!")
    
    return engine


def generate_hidden_states(engine, input_ids: torch.Tensor, rank: int = 0) -> torch.Tensor:
    """Generate hidden states from the target model using SGLang."""
    # Convert to list for SGLang
    input_ids_list = input_ids.cpu().tolist()
    
    # Generate hidden states
    outputs = engine.generate(
        input_ids=input_ids_list,
        sampling_params={"max_new_tokens": 1, "temperature": 0},
        return_hidden_states=True,
    )
    
    # Extract hidden states from last layer
    # SGLang returns hidden states as a list of tensors
    hidden_states = []
    for output in outputs:
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            # Get the last layer's hidden state
            hs = output.hidden_states[-1]  # Last layer
            hidden_states.append(hs)
        else:
            raise ValueError("Hidden states not returned from SGLang")
    
    # Stack into tensor
    hidden_states = torch.stack(hidden_states, dim=0)
    
    return hidden_states


def main():
    args = parse_args()
    
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0
    
    if is_main:
        print("=" * 60)
        print("DeepSeek-V3.2 MTP Layer Online Training")
        print("=" * 60)
        print(f"Target model: {args.target_model_path}")
        print(f"Data: {args.data_path}")
        print(f"Output: {args.output_dir}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"World size: {world_size}")
        print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load config
    if is_main:
        print("\n[Step 1/5] Loading config...")
    config = load_config(args.target_model_path)
    
    # Step 2: Create MTP model
    if is_main:
        print("\n[Step 2/5] Creating MTP model...")
    model = MTPLayerModel(config)
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
    
    # Step 3: Load weights from target
    if is_main:
        print("\n[Step 3/5] Loading weights from target model...")
    model = load_mtp_weights_from_target(model, args.target_model_path, rank=local_rank)
    model = model.to(device).to(torch.bfloat16)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Step 4: Initialize SGLang engine (only on rank 0 controls it)
    engine = None
    if is_main:
        print("\n[Step 4/5] Initializing SGLang engine...")
        engine = init_sglang_engine(
            args.target_model_path,
            args.tp_size,
            args.sglang_mem_fraction_static,
            args.sglang_page_size,
            rank=local_rank
        )
    
    # Step 5: Build dataset
    if is_main:
        print("\n[Step 5/5] Building dataset...")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    
    # Load raw dataset first, then build using SpecForge's dataset builder
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        num_proc=8,
    )
    
    if is_main:
        print(f"  Dataset size: {len(dataset)}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    if is_main:
        print("\n[Training] Starting online training...")
    
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_samples = 0
        
        if is_main:
            pbar = tqdm(range(len(dataset)), desc=f"Epoch {epoch + 1}")
        else:
            pbar = range(len(dataset))
        
        for idx in pbar:
            sample = dataset[idx]
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            
            # Generate hidden states on-the-fly using SGLang
            if is_main and engine is not None:
                with torch.no_grad():
                    hidden_states = generate_hidden_states(engine, input_ids, rank=local_rank)
                    hidden_states = hidden_states.to(device).to(torch.bfloat16)
                
                # Broadcast hidden states to all ranks
                if world_size > 1:
                    dist.broadcast(hidden_states, src=0)
            else:
                # Receive hidden states from rank 0
                hidden_states = torch.empty(
                    input_ids.shape[0], input_ids.shape[1], config["hidden_size"],
                    dtype=torch.bfloat16, device=device
                )
                if world_size > 1:
                    dist.broadcast(hidden_states, src=0)
            
            # Prepare labels (shifted by 1)
            labels = input_ids[:, 1:].clone()
            input_for_model = input_ids[:, :-1]
            hidden_states = hidden_states[:, :-1, :]
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(hidden_states, input_for_model)
            
            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += 1
            global_step += 1
            
            if is_main and idx % args.log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / max(num_samples, 1)
        if is_main:
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if is_main:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Get state dict (handle DDP wrapper)
                if isinstance(model, DDP):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                # Save weights
                save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
                
                # Save config
                with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                print(f"  Saved checkpoint to: {checkpoint_dir}")
    
    # Cleanup
    if engine is not None:
        engine.shutdown()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    if is_main:
        print("\n" + "=" * 60)
        print("Online training complete!")
        print(f"  Final checkpoint: {args.output_dir}/checkpoint-epoch-{args.num_epochs}")
        print("=" * 60)


if __name__ == "__main__":
    main()
