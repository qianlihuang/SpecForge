# coding=utf-8
# Copyright 2024 SpecForge Authors. All rights reserved.
#
# This code implements DeepSeek-V3.2 MTP (Multi-Token Prediction) layer training.
# The MTP layer is the 61st layer in DeepSeek-V3.2 which can be used for
# speculative decoding with EAGLE approach.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepSeek-V3.2 MTP (Multi-Token Prediction) Layer Draft Model for EAGLE Training.

The MTP layer (layer 61) has a specific architecture:
- Input: hidden_states from layer 60 + embedding of next token
- enorm: RMSNorm for embedding normalization  
- hnorm: RMSNorm for hidden state normalization
- eh_proj: Linear projection (2*hidden_size -> hidden_size)
- decoder: One DeepseekV2DecoderLayer (with MoE)
- shared_head.norm: Final RMSNorm
- lm_head: Output projection to vocab

This module adapts the MTP architecture for training with SpecForge's EAGLE3 trainer.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache

from specforge.utils import print_with_rank

from .base import Eagle3DraftModel


class DeepSeekV32MTPConfig(PretrainedConfig):
    """
    Configuration for DeepSeek-V3.2 MTP layer draft model.
    
    This config mirrors the DeepSeek-V3.2 architecture for the MTP (61st) layer.
    """
    model_type = "deepseek_v32_mtp"
    
    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 1,  # MTP has 1 layer
        num_attention_heads: int = 128,
        num_key_value_heads: int = 128,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 163840,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # MoE parameters
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 3,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        routed_scaling_factor: float = 2.5,
        topk_group: int = 4,
        topk_method: str = "noaux_tc",
        n_group: int = 8,
        # MTP-specific
        draft_vocab_size: int = None,  # Will default to vocab_size
        target_hidden_size: int = None,  # For projection from target model
        # EAGLE config
        eagle_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.n_group = n_group
        # Draft
        self.draft_vocab_size = draft_vocab_size if draft_vocab_size is not None else vocab_size
        self.target_hidden_size = target_hidden_size if target_hidden_size is not None else hidden_size
        self.eagle_config = eagle_config


class DeepSeekV32RMSNorm(nn.Module):
    """RMSNorm implementation for DeepSeek-V3.2"""
    
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


class DeepSeekV32RotaryEmbedding(nn.Module):
    """Rotary Embedding with YaRN scaling for DeepSeek-V3.2"""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 163840,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        
        self._set_cos_sin_cache(max_position_embeddings, device, torch.float32)

    def _yarn_find_correction_dim(self, num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def _yarn_find_correction_range(self, low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def _yarn_get_mscale(self, scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_linear_ramp_mask(self, min_val, max_val, dim, device):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2, device)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )

    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekV32MLP(nn.Module):
    """MLP for DeepSeek-V3.2 (used in dense layers)"""
    
    def __init__(self, config: DeepSeekV32MTPConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekV32MoEMLP(nn.Module):
    """
    Simplified MoE MLP for training.
    
    For actual inference, this would use sparse routing, but for training
    we use a simplified dense version that captures the essential behavior.
    """
    
    def __init__(self, config: DeepSeekV32MTPConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts
        
        # Shared experts (always activated)
        if self.n_shared_experts > 0:
            shared_intermediate = self.intermediate_size * self.n_shared_experts
            self.shared_gate_proj = nn.Linear(self.hidden_size, shared_intermediate, bias=False)
            self.shared_up_proj = nn.Linear(self.hidden_size, shared_intermediate, bias=False)
            self.shared_down_proj = nn.Linear(shared_intermediate, self.hidden_size, bias=False)
        
        # Router
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Expert weights (simplified - using shared weight matrix for training efficiency)
        # In full implementation, each expert would have separate weights
        self.expert_gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.expert_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.expert_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.act_fn = ACT2FN[config.hidden_act]
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Shared expert output
        if self.n_shared_experts > 0:
            shared_output = self.shared_down_proj(
                self.act_fn(self.shared_gate_proj(hidden_states_flat)) * 
                self.shared_up_proj(hidden_states_flat)
            )
        else:
            shared_output = 0
        
        # Router scores
        router_logits = self.gate(hidden_states_flat)
        if self.scoring_func == "sigmoid":
            routing_weights = torch.sigmoid(router_logits)
        else:
            routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Simplified expert computation (using average expert behavior for training)
        expert_output = self.expert_down_proj(
            self.act_fn(self.expert_gate_proj(hidden_states_flat)) * 
            self.expert_up_proj(hidden_states_flat)
        )
        
        # Scale by routing weights mean
        expert_output = expert_output * topk_weights.mean(dim=-1, keepdim=True) * self.routed_scaling_factor
        
        output = shared_output + expert_output
        return output.view(batch_size, seq_len, hidden_dim)


class DeepSeekV32MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) for DeepSeek-V3.2.
    
    MLA uses low-rank compression for KV cache efficiency while maintaining
    model capacity through latent projections.
    """
    
    def __init__(self, config: DeepSeekV32MTPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # Query projections with LoRA-style compression
        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = DeepSeekV32RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        
        # KV projections with compression
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + self.qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = DeepSeekV32RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_kv_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), 
            bias=False
        )
        
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        # Initialize rotary embeddings
        self._init_rope(config)

    def _init_rope(self, config):
        if config.rope_scaling is None:
            self.rotary_emb = DeepSeekV32RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            rope_scaling = config.rope_scaling
            self.rotary_emb = DeepSeekV32RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=rope_scaling.get("factor", 1.0),
                original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
                beta_fast=rope_scaling.get("beta_fast", 32),
                beta_slow=rope_scaling.get("beta_slow", 1),
                mscale=rope_scaling.get("mscale", 1.0),
                mscale_all_dim=rope_scaling.get("mscale_all_dim", 1.0),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        
        # Query projection
        if self.q_lora_rank > 0:
            q_compressed = self.q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            q = self.q_b_proj(q_compressed)
        else:
            q = self.q_proj(hidden_states)
        
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # KV projection
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_pe = kv_compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        kv_compressed = self.kv_a_layernorm(kv_compressed)
        kv = self.kv_b_proj(kv_compressed)
        kv = kv.view(bsz, q_len, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        kv_seq_len = q_len
        if past_key_values is not None:
            kv_seq_len += past_key_values.get_seq_length()
        
        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        
        # Reconstruct full Q and K
        q = torch.cat([q_nope, q_pe], dim=-1)
        
        # Expand k_pe for all kv heads
        k_pe = k_pe.expand(-1, self.num_kv_heads, -1, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        
        # Handle KV cache
        if cache_hidden is not None:
            if len(cache_hidden[0]) > 0:
                k = torch.cat([torch.cat(cache_hidden[0], dim=2), k], dim=2)
                v = torch.cat([torch.cat(cache_hidden[1], dim=2), v], dim=2)
            cache_hidden[0].append(k[:, :, -q_len:, :])
            cache_hidden[1].append(v[:, :, -q_len:, :])
        
        # Repeat KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.q_head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DeepSeekV32MTPDecoderLayer(nn.Module):
    """
    Decoder layer for DeepSeek-V3.2 MTP.
    
    This mirrors the architecture of DeepSeekV2DecoderLayer but with
    MoE for the MLP component.
    """
    
    def __init__(self, config: DeepSeekV32MTPConfig, use_moe: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = DeepSeekV32MLA(config)
        
        if use_moe:
            self.mlp = DeepSeekV32MoEMLP(config)
        else:
            self.mlp = DeepSeekV32MLP(config)
        
        self.input_layernorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_hidden=cache_hidden,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DeepSeekV32MTPForCausalLM(Eagle3DraftModel):
    """
    DeepSeek-V3.2 MTP layer draft model for EAGLE training.
    
    This implements the MTP (Multi-Token Prediction) layer architecture
    that can be trained with SpecForge's EAGLE3 trainer and then used
    for speculative decoding with sglang/vllm-magik.
    
    Architecture:
    - embed_tokens: Embedding layer (shared with target model)
    - enorm: RMSNorm for embedding normalization
    - hnorm: RMSNorm for hidden state normalization  
    - eh_proj: Linear(2*hidden_size -> hidden_size)
    - decoder: DeepSeekV32MTPDecoderLayer
    - norm: Final RMSNorm
    - lm_head: Output projection
    """
    
    config_class = DeepSeekV32MTPConfig

    def __init__(
        self, 
        config: DeepSeekV32MTPConfig,
        quant_config=None,
        attention_backend: str = "sdpa",
    ):
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config
        self.attention_backend = attention_backend
        
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.hidden_size = config.hidden_size
        
        # Embedding layer (will be loaded from target model)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=getattr(config, 'pad_token_id', 0)
        )
        
        # MTP-specific layers
        self.enorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        
        # For EAGLE3-style training with 3 aux hidden states
        target_hidden_size = getattr(config, 'target_hidden_size', config.hidden_size)
        self.fc = nn.Linear(target_hidden_size * 3, config.hidden_size, bias=False)
        
        # Decoder layer (with MoE)
        self.decoder = DeepSeekV32MTPDecoderLayer(config, use_moe=True)
        
        # Output layers
        self.norm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        
        # Vocab mapping buffers for EAGLE training
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens."""
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project concatenated hidden states from 3 aux layers.
        
        EAGLE3 requires hidden states from 3 layers (low, mid, high).
        This projection maps them to the model's hidden size.
        """
        # For standard EAGLE3: hidden_states shape is (batch, seq_len, 3*hidden_size)
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute output logits."""
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the MTP backbone.
        
        This implements the MTP architecture:
        1. Normalize embeddings and hidden states separately
        2. Concatenate and project
        3. Pass through decoder layer
        
        Args:
            input_embeds: Embedded input tokens (batch, seq_len, hidden_size)
            hidden_states: Projected hidden states from target model (batch, seq_len, hidden_size)
            cache_hidden: KV cache for attention
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_values: Past key values (not used, kept for compatibility)
            use_cache: Whether to use caching
            
        Returns:
            Output hidden states (batch, seq_len, hidden_size)
        """
        # Normalize embeddings and hidden states separately (MTP architecture)
        normed_embeds = self.enorm(input_embeds)
        normed_hidden = self.hnorm(hidden_states)
        
        # Concatenate and project
        combined = torch.cat([normed_embeds, normed_hidden], dim=-1)
        hidden_states = self.eh_proj(combined)
        
        # Pass through decoder
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_hidden=cache_hidden,
            output_attentions=False,
            use_cache=use_cache,
        )
        
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ) -> torch.Tensor:
        """
        Full forward pass for direct use (not TTT training).
        
        Args:
            hidden_states: Concatenated hidden states from aux layers (batch, seq_len, 3*hidden_size)
            inputs_embeds: Input embeddings (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            ttt_length: TTT length (for caching)
            
        Returns:
            Output hidden states
        """
        if ttt_length == 1:
            cache_hidden = None
        else:
            cache_hidden = [[], []]
        
        batch_size, seq_length, _ = hidden_states.size()
        device = hidden_states.device
        
        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
        
        # Prepare causal mask
        attention_mask = self.prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=0,
        )
        
        # Project hidden states
        hidden_states = self.fc(hidden_states)
        
        # Forward through backbone
        hidden_states = self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
