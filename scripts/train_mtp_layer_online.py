#!/usr/bin/env python3
"""
Online training script for DeepSeek-V3.2 MTP layer.

This script trains the MTP layer with on-the-fly hidden state generation using SGLang,
similar to SpecForge's online EAGLE3 training approach.

Usage:
    torchrun --standalone --nproc_per_node=8 \
        scripts/train_mtp_layer_online.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --data-path cache/dataset/deepseek-v32-sample.jsonl \
        --output-dir outputs/deepseek-v32-mtp-online \
        --tp-size 8 \
        --num-epochs 3 \
        --batch-size 1
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from specforge.utils import print_with_rank


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
    parser.add_argument("--is-preformatted", action="store_true",
                        help="Whether the input data is preformatted text")
    parser.add_argument("--build-dataset-num-proc", type=int, default=8,
                        help="Number of processes for dataset building")
    
    # Distributed training
    parser.add_argument("--tp-size", type=int, default=8,
                        help="Tensor parallel size for SGLang")
    parser.add_argument("--dist-timeout", type=int, default=2000,
                        help="Timeout for distributed communication in minutes")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # SGLang arguments
    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)
    
    return parser.parse_args()


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
    
    This model takes:
    - hidden_states: output from layer 60 [batch, seq_len, hidden_size]
    - input_ids: token IDs [batch, seq_len]
    
    And produces logits for next token prediction.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 7168)
        self.vocab_size = config.get("vocab_size", 129280)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        
        # MTP-specific components
        self.enorm = DeepSeekV32RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.hnorm = DeepSeekV32RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.eh_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        
        # Input layer norm
        self.input_layernorm = DeepSeekV32RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        
        # Shared head components
        self.norm = DeepSeekV32RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Embedding layer (shared with target model)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MTP layer.
        
        Args:
            hidden_states: Hidden states from last decoder layer [batch, seq_len, hidden_size]
            input_ids: Input token IDs [batch, seq_len]
            labels: Optional target labels for loss computation
            loss_mask: Optional mask for loss computation
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
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
        
        result = {"logits": logits}
        
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
            
            result["loss"] = loss
        
        return result


def load_config(model_path: str) -> dict:
    """Load model config."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_mtp_weights_from_target(model: MTPLayerModel, target_model_path: str, rank: int = 0):
    """
    Load MTP weights from target DeepSeek-V3.2 model.
    
    DeepSeek-V3.2 architecture:
    - Layers 0-60: Regular decoder layers
    - Layer 61: MTP prediction layer (with enorm, hnorm, eh_proj, shared_head)
    """
    if rank == 0:
        print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Find weight files containing MTP layer (layer 61)
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find files containing our target weights
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
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for src_key, dst_key in target_keys.items():
                if src_key in f.keys():
                    tensor = f.get_tensor(src_key)
                    # Skip FP8 quantized weights
                    if tensor.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        loaded_weights[dst_key] = tensor
                        if rank == 0:
                            print(f"    ✓ Loaded: {dst_key} {list(tensor.shape)}")
    
    # Load into model
    missing, unexpected = model.load_state_dict(loaded_weights, strict=False)
    if rank == 0:
        print(f"\n  Summary: loaded={len(loaded_weights)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing}")
    
    return model


def build_target_model(args, model_config):
    """Build target model using SGLang backend for hidden state generation."""
    target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=(
            model_config.get("dtype")
            if model_config.get("dtype")
            else model_config.get("torch_dtype", "bfloat16")
        ),
        device="cuda",
        cache_dir=None,
        **target_model_kwargs,
    )
    
    # For MTP training, we need hidden state from layer 60 (last regular layer before MTP)
    # DeepSeek-V3.2 architecture:
    # - num_hidden_layers = 61 (config value)
    # - Layers 0-60 are regular decoder layers  
    # - Layer 61 is the MTP layer
    # We need the OUTPUT of layer 60 as INPUT to MTP layer 61
    num_layers = model_config.get("num_hidden_layers", 61)
    mtp_input_layer = num_layers - 1  # Layer 60
    
    # Set the layer to capture
    target_model.model_runner.model.set_eagle3_layers_to_capture([mtp_input_layer])
    target_model.aux_hidden_states_layers = [mtp_input_layer]
    
    print(f"[INFO] Target model has {num_layers} regular decoder layers")
    print(f"[INFO] Capturing hidden states from layer {mtp_input_layer} (output used as input to MTP layer)")
    
    return target_model


def save_checkpoint(model: nn.Module, config: dict, output_dir: str, epoch: int, rank: int = 0):
    """Save model checkpoint."""
    if rank != 0:
        return None
    
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


def main():
    args = parse_args()
    
    # Initialize distributed
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    
    tp_group = get_tp_group()
    tp_rank = dist.get_rank(tp_group)
    is_main = is_tp_rank_0()
    
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
        print(f"TP size: {args.tp_size}")
        print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model config
    if is_main:
        print("\n[Step 1/5] Loading config...")
    config = load_config(args.target_model_path)
    
    # Create MTP model
    if is_main:
        print("\n[Step 2/5] Creating MTP model...")
    mtp_model = MTPLayerModel(config)
    if is_main:
        total_params = sum(p.numel() for p in mtp_model.parameters())
        print(f"  Parameters: {total_params:,}")
    
    # Load weights from target model
    if is_main:
        print("\n[Step 3/5] Loading weights from target model...")
    mtp_model = load_mtp_weights_from_target(mtp_model, args.target_model_path, rank=tp_rank)
    mtp_model = mtp_model.cuda().to(torch.bfloat16)
    
    # Build target model for hidden state generation
    if is_main:
        print("\n[Step 4/5] Building target model for hidden state generation...")
    target_model = build_target_model(args, config)
    
    # Build dataset
    if is_main:
        print("\n[Step 5/5] Building dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    
    # Load raw dataset
    if args.data_path.endswith(".jsonl"):
        raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        raw_dataset = load_dataset(args.data_path, split="train")
    
    # Process with build_eagle3_dataset
    dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        num_proc=args.build_dataset_num_proc,
    )
    
    # Create dataloader
    dp_group = get_dp_group()
    train_dataloader = prepare_dp_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        process_group=dp_group,
        shuffle=False,
    )
    
    if is_main:
        print(f"  Dataset size: {len(dataset)}")
    
    # Setup training
    optimizer = torch.optim.AdamW(mtp_model.parameters(), lr=args.learning_rate)
    
    # Training loop
    if is_main:
        print("\n[Training] Starting online training...")
    
    mtp_model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        num_samples = 0
        
        if is_main:
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to GPU
            batch_gpu = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            
            # Generate hidden states from target model
            with torch.no_grad():
                _, logits_list, aux_hidden_states_list, _ = target_model.extend(
                    **batch_gpu,
                    return_last_hidden_states=False,
                    return_logits=True,
                )
            
            # Get hidden states for this batch
            # aux_hidden_states_list is a list of tensors, one per sample
            if is_main and aux_hidden_states_list:
                hidden_states = aux_hidden_states_list[0]  # [seq_len, hidden_size]
                hidden_states = hidden_states.unsqueeze(0).to(torch.bfloat16)  # [1, seq_len, hidden_size]
                
                input_ids = batch_gpu["input_ids"]
                loss_mask = batch_gpu.get("loss_mask")
                
                # Forward pass
                optimizer.zero_grad()
                outputs = mtp_model(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    labels=input_ids,
                    loss_mask=loss_mask,
                )
                
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mtp_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_samples += 1
                global_step += 1
                
                if batch_idx % args.log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Cleanup
            del aux_hidden_states_list, logits_list
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_loss = epoch_loss / max(num_samples, 1)
        if is_main:
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Synchronize before saving
        dist.barrier()
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(mtp_model, config, args.output_dir, epoch + 1, rank=tp_rank)
        
        dist.barrier()
    
    # Cleanup
    destroy_distributed()
    
    if is_main:
        print("\n" + "=" * 60)
        print("Online training complete!")
        print(f"  Final checkpoint: {args.output_dir}/checkpoint-epoch-{args.num_epochs}")
        print("=" * 60)


if __name__ == "__main__":
    main()
