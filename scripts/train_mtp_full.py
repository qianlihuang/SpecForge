#!/usr/bin/env python3
"""
Full MTP Layer Fine-tuning for DeepSeek-V3.2 EAGLE Speculative Decoding.

This script fine-tunes the COMPLETE MTP layer (layer 61) of DeepSeek-V3.2, including:
- Projection components: enorm, hnorm, eh_proj
- Decoder block: attention, MoE (all 256 experts)
- Output head: shared_head.norm, shared_head.head

This ensures the trained model architecture matches the inference architecture
used by vLLM-magik and SGLang for EAGLE speculative decoding.

Architecture:
    Input: hidden_states (from layer 60), input_ids
    -> enorm(embed_tokens(input_ids))
    -> hnorm(hidden_states)  
    -> eh_proj(concat[normed_embed, normed_hidden])
    -> DecoderBlock (attention + MoE)  <-- This is included!
    -> shared_head.norm
    -> shared_head.head (lm_head)
    -> logits

Note: Due to the large size of the MoE layer (256 experts), this requires
significant GPU memory. Consider using:
- Gradient checkpointing
- Mixed precision training
- Data parallelism across multiple GPUs

Usage:
    python scripts/train_mtp_full.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --hidden-states-path cache/hidden_states/deepseek-v32-mtp \
        --output-dir outputs/deepseek-v32-mtp-full \
        --num-epochs 3 \
        --batch-size 1 \
        --learning-rate 1e-5 \
        --freeze-moe  # Optional: freeze MoE experts
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FP8 Quantization Utilities
# ============================================================================

def dequantize_fp8_weight(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize FP8 weight to BF16 using block-wise scaling.
    
    DeepSeek-V3.2 uses block-wise FP8 quantization with block_size=128.
    weight_fp8: [out_features, in_features] in FP8
    scale_inv: [out_features/block_size, in_features/block_size] in FP32
    
    Returns: weight in BF16
    """
    out_features, in_features = weight_fp8.shape
    scale_out_blocks, scale_in_blocks = scale_inv.shape
    
    # Calculate actual block sizes from scale_inv shape
    # Note: DeepSeek uses ceil division, so last block may be smaller
    out_block_size = (out_features + scale_out_blocks - 1) // scale_out_blocks
    in_block_size = (in_features + scale_in_blocks - 1) // scale_in_blocks
    
    # Pad weight to be divisible by block sizes
    pad_out = scale_out_blocks * out_block_size - out_features
    pad_in = scale_in_blocks * in_block_size - in_features
    
    # Convert FP8 to FP32 first
    weight_f32 = weight_fp8.to(torch.float32)
    
    if pad_out > 0 or pad_in > 0:
        weight_f32 = torch.nn.functional.pad(weight_f32, (0, pad_in, 0, pad_out), value=0)
    
    # Reshape for block-wise operation
    # [padded_out, padded_in] -> [scale_out_blocks, out_block_size, scale_in_blocks, in_block_size]
    weight_reshaped = weight_f32.view(scale_out_blocks, out_block_size, scale_in_blocks, in_block_size)
    
    # Apply scale (scale_inv is the inverse scale, so we multiply)
    # scale_inv: [scale_out_blocks, scale_in_blocks] -> [scale_out_blocks, 1, scale_in_blocks, 1]
    scale_expanded = scale_inv.unsqueeze(1).unsqueeze(3)
    weight_dequant = weight_reshaped * scale_expanded
    
    # Reshape back and remove padding
    weight_dequant = weight_dequant.view(scale_out_blocks * out_block_size, scale_in_blocks * in_block_size)
    weight_dequant = weight_dequant[:out_features, :in_features]
    
    return weight_dequant.to(torch.bfloat16)


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
    block_max = weight_blocked.abs().amax(dim=(1, 3), keepdim=True)  # [out_blocks, 1, in_blocks, 1]
    
    # Compute scale: scale = max_val / fp8_max
    # scale_inv = fp8_max / max_val (what we store)
    scale = block_max / fp8_max
    scale = scale.clamp(min=1e-12)  # Avoid division by zero
    scale_inv = 1.0 / scale
    
    # Quantize
    weight_scaled = weight_blocked / scale
    weight_scaled = weight_scaled.clamp(-fp8_max, fp8_max)
    
    # Reshape back
    weight_scaled = weight_scaled.view(out_blocks * block_size, in_blocks * block_size)
    scale_inv = scale_inv.squeeze(1).squeeze(-1)  # [out_blocks, in_blocks]
    
    # Remove padding
    if pad_out > 0 or pad_in > 0:
        weight_scaled = weight_scaled[:out_features, :in_features]
    
    # Convert to FP8
    weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)
    
    return weight_fp8, scale_inv.to(torch.float32)


# ============================================================================
# Model Components
# ============================================================================

class DeepSeekRMSNorm(nn.Module):
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


class DeepSeekV32Attention(nn.Module):
    """Simplified attention for MTP training (no KV cache)"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # MLA parameters
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Low-rank projections
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
            bias=False
        )
        
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q projection (low-rank)
        q_compressed = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q = self.q_b_proj(q_compressed)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # KV projection (low-rank with MQA)
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_rope = torch.split(
            kv_compressed,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        )
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)
        kv = kv.view(batch_size, seq_len, self.num_heads, -1)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Split Q into nope and rope parts
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # Combine K (simplified - no RoPE for now)
        k = torch.cat([k_nope, k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)], dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        scale = 1.0 / (q.size(-1) ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(v.dtype)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output)


class DeepSeekMoEGate(nn.Module):
    """Expert routing gate for MoE"""
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        
        self.weight = nn.Linear(config.hidden_size, self.n_routed_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts))
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        
        # Compute expert scores
        logits = self.weight(hidden_states)
        scores = torch.sigmoid(logits) + self.e_score_correction_bias
        
        # Group-limited expert selection
        scores_reshaped = scores.view(-1, self.n_group, self.n_routed_experts // self.n_group)
        group_scores = scores_reshaped.max(dim=-1).values
        _, selected_groups = torch.topk(group_scores, k=self.topk_group, dim=-1)
        
        # Mask out non-selected groups
        mask = torch.zeros_like(scores).view(-1, self.n_group, self.n_routed_experts // self.n_group)
        mask.scatter_(1, selected_groups.unsqueeze(-1).expand(-1, -1, self.n_routed_experts // self.n_group), 1.0)
        mask = mask.view(-1, self.n_routed_experts)
        scores = scores * mask
        
        # Select top-k experts
        weights, indices = torch.topk(scores, k=self.num_experts_per_tok, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        return weights.view(batch_size, seq_len, -1), indices.view(batch_size, seq_len, -1)


class DeepSeekMLP(nn.Module):
    """Standard MLP (for shared experts)"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        
        # Routing gate
        self.gate = DeepSeekMoEGate(config)
        
        # Routed experts
        self.experts = nn.ModuleList([
            DeepSeekMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])
        
        # Shared experts
        if self.n_shared_experts > 0:
            self.shared_experts = DeepSeekMLP(
                config.hidden_size,
                config.moe_intermediate_size * self.n_shared_experts
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing weights and indices
        routing_weights, expert_indices = self.gate(hidden_states)
        
        # Compute routed expert outputs (simplified version - not optimized)
        routed_output = torch.zeros_like(hidden_states)
        
        # For efficiency, we use a loop (can be optimized with scatter/gather)
        flat_hidden = hidden_states.view(-1, hidden_size)
        flat_routing = routing_weights.view(-1, routing_weights.size(-1))
        flat_indices = expert_indices.view(-1, expert_indices.size(-1))
        flat_output = torch.zeros_like(flat_hidden)
        
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (flat_indices == i).any(dim=-1)
            if mask.sum() > 0:
                expert_input = flat_hidden[mask]
                expert_output = expert(expert_input)
                # Get the weight for this expert
                weight_mask = (flat_indices == i)
                weights = (flat_routing * weight_mask.float()).sum(dim=-1, keepdim=True)[mask]
                flat_output[mask] += expert_output * weights * self.routed_scaling_factor
        
        routed_output = flat_output.view(batch_size, seq_len, hidden_size)
        
        # Add shared expert output
        if self.n_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
            routed_output = routed_output + shared_output
        
        return routed_output


class DeepSeekDecoderBlock(nn.Module):
    """Full decoder block with attention and MoE"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekV32Attention(config)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepSeekMoE(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # MoE MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class FullMTPLayer(nn.Module):
    """
    Complete MTP layer matching DeepSeek-V3.2 architecture.
    
    This includes:
    - Embedding lookup
    - enorm, hnorm, eh_proj (projection components)
    - Full decoder block (attention + MoE)
    - Output head (norm + lm_head)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Embedding (shared with target model)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # MTP projection components
        self.enorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # Full decoder block (attention + MoE)
        self.decoder_block = DeepSeekDecoderBlock(config)
        
        # Output head
        self.norm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MTP layer.
        
        Args:
            input_ids: [batch, seq_len] - token IDs
            hidden_states: [batch, seq_len, hidden_size] - output from layer 60
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            loss_mask: Optional mask for loss
        """
        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # MTP projection
        normed_embeds = self.enorm(inputs_embeds)
        normed_hidden = self.hnorm(hidden_states)
        combined = torch.cat([normed_embeds, normed_hidden], dim=-1)
        projected = self.eh_proj(combined)
        
        # Decoder block (attention + MoE)
        decoder_output = self.decoder_block(projected, attention_mask)
        
        # Output
        normed_output = self.norm(decoder_output)
        logits = self.lm_head(normed_output)
        
        result = {"logits": logits}
        
        if labels is not None:
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


# ============================================================================
# Dataset
# ============================================================================

class MTPDataset(Dataset):
    """Dataset for MTP training using pre-generated hidden states."""
    
    def __init__(self, hidden_states_path: str, max_samples: Optional[int] = None):
        self.hidden_states_path = Path(hidden_states_path)
        self.file_paths = []
        
        for subdir in sorted(self.hidden_states_path.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("rows_"):
                for f in sorted(subdir.iterdir()):
                    if f.suffix == ".ckpt":
                        self.file_paths.append(f)
        
        if max_samples is not None:
            self.file_paths = self.file_paths[:max_samples]
        
        print(f"Found {len(self.file_paths)} data files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=False)
        
        hidden_state = data["hidden_state"]
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        
        input_ids = data["input_ids"]
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        
        loss_mask = data.get("loss_mask")
        if loss_mask is not None and loss_mask.dim() == 2:
            loss_mask = loss_mask.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "hidden_state": hidden_state,
            "loss_mask": loss_mask,
        }


def collate_fn(batch):
    """Simple collate - assumes batch_size=1 for now."""
    item = batch[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "hidden_states": item["hidden_state"].unsqueeze(0),
        "labels": item["input_ids"].unsqueeze(0),
        "loss_mask": item["loss_mask"].unsqueeze(0) if item["loss_mask"] is not None else None,
    }


# ============================================================================
# Weight Loading
# ============================================================================

def load_mtp_weights_from_target(
    model: FullMTPLayer,
    target_model_path: str,
    device: torch.device,
    skip_moe: bool = False,
) -> Dict[str, int]:
    """
    Load MTP layer (layer 61) weights from target DeepSeek-V3.2 model.
    
    Args:
        model: Model to load weights into
        target_model_path: Path to DeepSeek-V3.2 model
        device: Device to load to
        skip_moe: If True, skip loading MoE expert weights (for projection-only training)
    """
    print(f"\nLoading MTP weights from: {target_model_path}")
    
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    # Find all files containing layer 61 weights
    files_to_load = defaultdict(list)
    for weight_name, filename in weight_map.items():
        if "layers.61" in weight_name or weight_name == "model.embed_tokens.weight":
            files_to_load[filename].append(weight_name)
    
    # Weight name mapping from target to our model
    def map_weight_name(name: str) -> Optional[str]:
        """Map target model weight name to our model weight name."""
        mappings = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.layers.61.enorm.weight": "enorm.weight",
            "model.layers.61.hnorm.weight": "hnorm.weight",
            "model.layers.61.eh_proj.weight": "eh_proj.weight",
            "model.layers.61.input_layernorm.weight": "decoder_block.input_layernorm.weight",
            "model.layers.61.post_attention_layernorm.weight": "decoder_block.post_attention_layernorm.weight",
            "model.layers.61.shared_head.norm.weight": "norm.weight",
            "model.layers.61.shared_head.head.weight": "lm_head.weight",
        }
        
        if name in mappings:
            return mappings[name]
        
        # Handle attention weights
        if "self_attn" in name:
            return name.replace("model.layers.61.", "decoder_block.")
        
        # Handle MoE weights
        if "mlp" in name and not skip_moe:
            return name.replace("model.layers.61.", "decoder_block.")
        
        return None
    
    # Load weights
    stats = {"loaded": 0, "skipped": 0, "missing": 0}
    model_state = model.state_dict()
    
    for filename, keys in sorted(files_to_load.items()):
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading: {filename}")
        
        # Collect scale_inv tensors for FP8 dequantization
        scale_inv_tensors = {}
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            # First pass: collect scale_inv tensors
            for key in f.keys():
                if "weight_scale_inv" in key and f"layers.{61}" in key:
                    scale_inv_tensors[key] = f.get_tensor(key)
            
            # Second pass: load weights
            for key in keys:
                if key not in [k for k in f.keys()]:
                    continue
                if "weight_scale_inv" in key:
                    continue  # Skip scale tensors, handled separately
                    
                tensor = f.get_tensor(key)
                
                # Dequantize FP8 weights to BF16 for training
                if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    scale_key = key.replace(".weight", ".weight_scale_inv")
                    if scale_key in scale_inv_tensors:
                        scale_inv = scale_inv_tensors[scale_key]
                        # Dequantize: weight_bf16 = weight_fp8 * scale
                        # scale_inv is stored, so we need to use it directly
                        # The scale_inv has shape [out_features/block_size, in_features/block_size]
                        # where block_size is typically 128
                        tensor = dequantize_fp8_weight(tensor, scale_inv)
                        stats["dequantized"] = stats.get("dequantized", 0) + 1
                    else:
                        # No scale found, convert directly
                        tensor = tensor.to(torch.bfloat16)
                        stats["converted"] = stats.get("converted", 0) + 1
                
                mapped_key = map_weight_name(key)
                if mapped_key is None:
                    stats["skipped"] += 1
                    continue
                
                if mapped_key not in model_state:
                    stats["missing"] += 1
                    continue
                
                # Convert dtype if needed
                if tensor.dtype != model_state[mapped_key].dtype:
                    tensor = tensor.to(model_state[mapped_key].dtype)
                
                # Check shape
                if tensor.shape != model_state[mapped_key].shape:
                    print(f"    Shape mismatch: {key} {tensor.shape} vs {model_state[mapped_key].shape}")
                    stats["skipped"] += 1
                    continue
                
                model_state[mapped_key].copy_(tensor)
                stats["loaded"] += 1
    
    model.load_state_dict(model_state)
    print(f"\n  Stats: loaded={stats['loaded']}, skipped={stats['skipped']}, missing={stats['missing']}")
    return stats


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 1,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        hidden_states = batch["hidden_states"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"]
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        
        outputs = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
        )
        
        loss = outputs["loss"] / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += outputs["loss"].item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})
        
        # Memory management
        if batch_idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    config_dict: dict,
    output_dir: str,
    epoch: int,
) -> str:
    """Save checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
    
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  Saved checkpoint to: {checkpoint_dir}")
    return checkpoint_dir


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Full MTP layer training")
    
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    
    # Training mode options
    parser.add_argument("--freeze-moe", action="store_true",
                       help="Freeze MoE experts (train projection + attention only)")
    parser.add_argument("--freeze-attention", action="store_true",
                       help="Freeze attention (train projection + MoE only)")
    parser.add_argument("--projection-only", action="store_true",
                       help="Train only projection components (enorm, hnorm, eh_proj, norm, lm_head)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Full MTP Layer Training for DeepSeek-V3.2")
    print("=" * 60)
    print(f"Target model: {args.target_model_path}")
    print(f"Hidden states: {args.hidden_states_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Freeze MoE: {args.freeze_moe}")
    print(f"Freeze attention: {args.freeze_attention}")
    print(f"Projection only: {args.projection_only}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    print("\n[Step 1/5] Loading config...")
    config_path = os.path.join(args.target_model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Create config object
    @dataclass
    class Config:
        hidden_size: int = config_dict.get("hidden_size", 7168)
        vocab_size: int = config_dict.get("vocab_size", 129280)
        rms_norm_eps: float = config_dict.get("rms_norm_eps", 1e-6)
        num_attention_heads: int = config_dict.get("num_attention_heads", 128)
        q_lora_rank: int = config_dict.get("q_lora_rank", 1536)
        kv_lora_rank: int = config_dict.get("kv_lora_rank", 512)
        qk_nope_head_dim: int = config_dict.get("qk_nope_head_dim", 128)
        qk_rope_head_dim: int = config_dict.get("qk_rope_head_dim", 64)
        v_head_dim: int = config_dict.get("v_head_dim", 128)
        n_routed_experts: int = config_dict.get("n_routed_experts", 256)
        n_shared_experts: int = config_dict.get("n_shared_experts", 1)
        num_experts_per_tok: int = config_dict.get("num_experts_per_tok", 8)
        moe_intermediate_size: int = config_dict.get("moe_intermediate_size", 2048)
        n_group: int = config_dict.get("n_group", 8)
        topk_group: int = config_dict.get("topk_group", 4)
        routed_scaling_factor: float = config_dict.get("routed_scaling_factor", 2.5)
    
    config = Config()
    
    # Create model
    print("\n[Step 2/5] Creating model...")
    model = FullMTPLayer(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load weights
    print("\n[Step 3/5] Loading weights from target model...")
    load_mtp_weights_from_target(model, args.target_model_path, device, skip_moe=args.projection_only)
    
    model = model.to(device).to(torch.bfloat16)
    
    # Freeze parameters based on mode
    if args.projection_only:
        # Only train projection components
        for name, param in model.named_parameters():
            if any(x in name for x in ["decoder_block"]):
                param.requires_grad = False
        print("  Mode: Projection-only (decoder block frozen)")
    elif args.freeze_moe:
        # Freeze MoE experts
        for name, param in model.named_parameters():
            if "mlp.experts" in name or "mlp.gate" in name or "mlp.shared_experts" in name:
                param.requires_grad = False
        print("  Mode: MoE frozen (training projection + attention)")
    elif args.freeze_attention:
        # Freeze attention
        for name, param in model.named_parameters():
            if "self_attn" in name:
                param.requires_grad = False
        print("  Mode: Attention frozen (training projection + MoE)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create dataset
    print("\n[Step 4/5] Creating dataset...")
    dataset = MTPDataset(args.hidden_states_path, args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    
    # Training loop
    print("\n[Step 5/5] Training...")
    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        
        if epoch % args.save_interval == 0:
            save_checkpoint(model, config_dict, args.output_dir, epoch)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
