#!/usr/bin/env python3
"""
Online training script for DeepSeek-V3.2 MTP layer with FULL architecture.

This script trains the MTP layer with on-the-fly hidden state generation using SGLang,
including the FULL decoder block (attention + MoE) to match the inference architecture.

Architecture (matches vLLM-magik/SGLang inference):
    hidden_states (from layer 60) + input_ids
    -> enorm(embed), hnorm(hidden)
    -> eh_proj(concat)
    -> DecoderBlock (attention + MoE)  <-- Included!
    -> norm -> lm_head -> logits

Usage:
    torchrun --standalone --nproc_per_node=8 \
        scripts/train_mtp_full_online.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --data-path cache/dataset/deepseek-v32-sample.jsonl \
        --output-dir outputs/deepseek-v32-mtp-full-online \
        --tp-size 8 \
        --num-epochs 3 \
        --batch-size 1 \
        --freeze-moe  # Optional: freeze MoE experts to reduce memory
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

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
    parser = argparse.ArgumentParser(description="Online Full MTP layer training for DeepSeek-V3.2")
    
    # Model paths
    parser.add_argument("--target-model-path", type=str, required=True,
                        help="Path to the target DeepSeek-V3.2 model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the training data (JSONL)")
    parser.add_argument("--output-dir", type=str, default="outputs/deepseek-v32-mtp-full-online",
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # Freeze options
    parser.add_argument("--freeze-moe", action="store_true",
                        help="Freeze MoE experts (train projection + attention only)")
    parser.add_argument("--freeze-attention", action="store_true",
                        help="Freeze attention (train projection + MoE only)")
    parser.add_argument("--projection-only", action="store_true",
                        help="Train projection only (WARNING: architecture mismatch!)")
    
    # SGLang arguments
    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)
    
    return parser.parse_args()


# ============================================================================
# Model Components (same as train_mtp_full.py)
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
        
        # Combine K (simplified - no RoPE for training)
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
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(scores, k=self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return topk_weights.view(batch_size, seq_len, -1), topk_indices.view(batch_size, seq_len, -1)


class DeepSeekExpert(nn.Module):
    """Single MoE expert"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """Full MoE layer with 256 experts + shared expert"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        
        # Gate
        self.gate = DeepSeekMoEGate(config)
        
        # Routed experts (256 for DeepSeek-V3.2)
        self.experts = nn.ModuleList([
            DeepSeekExpert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])
        
        # Shared expert (always activated)
        self.shared_experts = DeepSeekExpert(
            config.hidden_size, 
            config.moe_intermediate_size * self.n_shared_experts
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Shared expert output
        shared_output = self.shared_experts(hidden_states)
        
        # Gate and routed experts
        topk_weights, topk_indices = self.gate(hidden_states)
        
        # Compute expert outputs (naive loop for simplicity)
        routed_output = torch.zeros_like(hidden_states)
        hidden_flat = hidden_states.view(-1, hidden_size)
        
        for expert_idx in range(self.n_routed_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1).view(-1)
            if expert_mask.sum() == 0:
                continue
                
            # Get expert weights for these tokens
            weight_mask = (topk_indices == expert_idx).float()
            expert_weights = (topk_weights * weight_mask).sum(dim=-1).view(-1)
            
            # Compute expert output
            expert_input = hidden_flat[expert_mask]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Scale by weight and add to output
            routed_output.view(-1, hidden_size)[expert_mask] += (
                expert_output * expert_weights[expert_mask].unsqueeze(-1)
            )
        
        return shared_output + routed_output


class DeepSeekDecoderBlock(nn.Module):
    """Full decoder block with attention + MoE"""
    def __init__(self, config):
        super().__init__()
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


class FullMTPLayerOnline(nn.Module):
    """
    Complete MTP layer matching DeepSeek-V3.2 architecture for online training.
    
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


def load_config(model_path: str) -> Any:
    """Load model config."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return AutoConfig.for_model(**config_dict)


def load_mtp_weights_from_target(model: FullMTPLayerOnline, target_model_path: str, rank: int = 0):
    """
    Load MTP weights from target DeepSeek-V3.2 model.
    
    DeepSeek-V3.2 architecture:
    - Layers 0-60: Regular decoder layers
    - Layer 61: MTP prediction layer (with enorm, hnorm, eh_proj, decoder_block, shared_head)
    """
    if rank == 0:
        print(f"\nLoading full MTP weights from: {target_model_path}")
    
    # Load index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Target weight mapping (checkpoint key -> model attribute)
    target_keys = {
        # Embedding
        "model.embed_tokens.weight": "embed_tokens.weight",
        # MTP projection
        "model.layers.61.enorm.weight": "enorm.weight",
        "model.layers.61.hnorm.weight": "hnorm.weight",
        "model.layers.61.eh_proj.weight": "eh_proj.weight",
        # Decoder block - input layernorm
        "model.layers.61.input_layernorm.weight": "decoder_block.input_layernorm.weight",
        # Decoder block - attention
        "model.layers.61.self_attn.q_a_proj.weight": "decoder_block.self_attn.q_a_proj.weight",
        "model.layers.61.self_attn.q_a_layernorm.weight": "decoder_block.self_attn.q_a_layernorm.weight",
        "model.layers.61.self_attn.q_b_proj.weight": "decoder_block.self_attn.q_b_proj.weight",
        "model.layers.61.self_attn.kv_a_proj_with_mqa.weight": "decoder_block.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.61.self_attn.kv_a_layernorm.weight": "decoder_block.self_attn.kv_a_layernorm.weight",
        "model.layers.61.self_attn.kv_b_proj.weight": "decoder_block.self_attn.kv_b_proj.weight",
        "model.layers.61.self_attn.o_proj.weight": "decoder_block.self_attn.o_proj.weight",
        # Decoder block - post attention layernorm
        "model.layers.61.post_attention_layernorm.weight": "decoder_block.post_attention_layernorm.weight",
        # Decoder block - MoE gate
        "model.layers.61.mlp.gate.weight": "decoder_block.mlp.gate.weight.weight",
        "model.layers.61.mlp.gate.e_score_correction_bias": "decoder_block.mlp.gate.e_score_correction_bias",
        # Decoder block - shared expert
        "model.layers.61.mlp.shared_experts.gate_proj.weight": "decoder_block.mlp.shared_experts.gate_proj.weight",
        "model.layers.61.mlp.shared_experts.up_proj.weight": "decoder_block.mlp.shared_experts.up_proj.weight",
        "model.layers.61.mlp.shared_experts.down_proj.weight": "decoder_block.mlp.shared_experts.down_proj.weight",
        # Output head
        "model.layers.61.shared_head.norm.weight": "norm.weight",
        "model.layers.61.shared_head.head.weight": "lm_head.weight",
    }
    
    # Add routed experts (256)
    for i in range(256):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src_key = f"model.layers.61.mlp.experts.{i}.{proj}.weight"
            tgt_key = f"decoder_block.mlp.experts.{i}.{proj}.weight"
            target_keys[src_key] = tgt_key
    
    # Find files to load
    files_to_load = set()
    for key in target_keys.keys():
        if key in weight_map:
            files_to_load.add(weight_map[key])
    
    if rank == 0:
        print(f"  Loading from {len(files_to_load)} files")
    
    # Load weights
    loaded_count = 0
    model_state = model.state_dict()
    
    for file_name in sorted(files_to_load):
        file_path = os.path.join(target_model_path, file_name)
        if rank == 0:
            print(f"  Loading: {file_name}")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in target_keys:
                    tgt_key = target_keys[key]
                    if tgt_key in model_state:
                        weight = f.get_tensor(key)
                        # Convert FP8 to BF16 if needed
                        if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            weight = weight.to(torch.bfloat16)
                        model_state[tgt_key].copy_(weight)
                        loaded_count += 1
    
    if rank == 0:
        print(f"  Loaded {loaded_count} weight tensors")
    
    return model


def freeze_components(model: FullMTPLayerOnline, freeze_moe: bool, freeze_attention: bool, projection_only: bool, rank: int = 0):
    """Freeze specified model components."""
    if projection_only:
        # Only train projection components
        for name, param in model.named_parameters():
            if any(n in name for n in ['enorm', 'hnorm', 'eh_proj', 'norm', 'lm_head']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if rank == 0:
            print("Frozen: decoder_block (attention + MoE)")
            print("Training: enorm, hnorm, eh_proj, norm, lm_head")
    else:
        if freeze_moe:
            for name, param in model.named_parameters():
                if 'mlp.experts' in name or 'mlp.shared_experts' in name or 'mlp.gate' in name:
                    param.requires_grad = False
            if rank == 0:
                print("Frozen: MoE experts and gate")
        
        if freeze_attention:
            for name, param in model.named_parameters():
                if 'self_attn' in name:
                    param.requires_grad = False
            if rank == 0:
                print("Frozen: Self attention")
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def train_online(args):
    """Main online training function."""
    # Initialize distributed
    init_distributed(args.dist_timeout)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("DeepSeek-V3.2 Full MTP Layer Online Training")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"TP size: {args.tp_size}")
        print(f"Target model: {args.target_model_path}")
        print(f"Output dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = load_config(args.target_model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    
    # Build dataset
    if rank == 0:
        print("\nBuilding dataset...")
    
    dataset = build_eagle3_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        chat_template=args.chat_template,
        is_preformatted=args.is_preformatted,
        num_proc=args.build_dataset_num_proc,
    )
    
    # Prepare dataloaders
    train_loader, _ = prepare_dp_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=42,
    )
    
    if rank == 0:
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Steps per epoch: {len(train_loader)}")
    
    # Initialize target model for hidden state extraction
    if rank == 0:
        print("\nInitializing target model for hidden state extraction...")
    
    sglang_args = SGLangBackendArgs.from_args(args)
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=config.torch_dtype if hasattr(config, 'torch_dtype') else torch.bfloat16,
        device="cuda",
        **sglang_args.to_kwargs(),
    )
    
    # Create MTP model
    if rank == 0:
        print("\nCreating full MTP model...")
    
    mtp_model = FullMTPLayerOnline(config)
    mtp_model = load_mtp_weights_from_target(mtp_model, args.target_model_path, rank)
    
    # Freeze components if requested
    freeze_components(
        mtp_model, 
        args.freeze_moe, 
        args.freeze_attention, 
        args.projection_only,
        rank
    )
    
    # Move to device and enable mixed precision
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    mtp_model = mtp_model.to(device).to(torch.bfloat16)
    
    # Optimizer
    trainable_params = [p for p in mtp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Training loop
    if rank == 0:
        print("\nStarting training...")
    
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        mtp_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            loss_mask = batch.get("loss_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)
            
            # Get hidden states from target model using extend()
            # return_last_hidden_states=True gives us the final hidden states (layer 60 output)
            with torch.no_grad():
                _, _, _, last_hidden_states_list = target_model.extend(
                    input_ids=input_ids,
                    attention_mask=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
                    loss_mask=loss_mask if loss_mask is not None else torch.ones_like(input_ids),
                    return_last_hidden_states=True,
                    return_logits=False,
                )
                # Stack the hidden states from the list (each element is [seq_len, hidden_size])
                hidden_states = torch.stack([h for h in last_hidden_states_list], dim=0)
                hidden_states = hidden_states.to(device).to(torch.bfloat16)
            
            # Forward pass
            outputs = mtp_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=None,  # MTP model handles causal masking internally
                labels=input_ids,
                loss_mask=loss_mask,
            )
            
            loss = outputs["loss"] / args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += outputs["loss"].item()
            num_batches += 1
            
            if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Epoch complete
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if rank == 0 and epoch % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model state
            save_file(
                mtp_model.state_dict(),
                os.path.join(checkpoint_dir, "model.safetensors")
            )
            
            # Save training args
            with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            
            print(f"Saved checkpoint to: {checkpoint_dir}")
    
    # Cleanup
    destroy_distributed()
    
    if rank == 0:
        print("\nTraining completed!")


def main():
    args = parse_args()
    train_online(args)


if __name__ == "__main__":
    main()
