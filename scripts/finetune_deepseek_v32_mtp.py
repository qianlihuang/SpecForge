#!/usr/bin/env python3
"""
Fine-tune DeepSeek-V3.2's MTP layer (layer 61) for EAGLE speculative decoding.

This script:
1. Loads MTP layer weights from target model as initial weights
2. Uses target model to generate hidden states
3. Fine-tunes the MTP layer on custom data

Usage:
    python scripts/finetune_deepseek_v32_mtp.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
        --output-dir outputs/deepseek-v32-mtp-finetuned
"""

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specforge.modeling.draft.deepseek_v32_mtp import (
    DeepSeekV32MTPConfig,
    DeepSeekV32MTPForCausalLM,
)


def load_mtp_weights_from_target(
    target_model_path: str,
    draft_model: nn.Module,
    device: str = "cpu",
) -> Dict[str, int]:
    """
    Load MTP layer (layer 61) weights from target DeepSeek-V3.2 model.
    
    Note: We load non-quantized weights (layernorms, projections) from the target.
    Expert weights (MoE) are too large and complex to load directly.
    
    Returns mapping stats showing which weights were loaded.
    """
    print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Load weight map index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Create weight mapping from target to draft model
    # Only load weights that are not FP8 quantized (layernorms, etc.)
    weight_mapping = {
        # Embedding
        "model.embed_tokens.weight": "embed_tokens.weight",
        # MTP normalization
        "model.layers.61.enorm.weight": "enorm.weight",
        "model.layers.61.hnorm.weight": "hnorm.weight",
        "model.layers.61.eh_proj.weight": "eh_proj.weight",
        # Layer norms
        "model.layers.61.input_layernorm.weight": "decoder.input_layernorm.weight",
        "model.layers.61.post_attention_layernorm.weight": "decoder.post_attention_layernorm.weight",
        # MLA attention layernorms
        "model.layers.61.self_attn.q_a_layernorm.weight": "decoder.self_attn.q_a_layernorm.weight",
        "model.layers.61.self_attn.kv_a_layernorm.weight": "decoder.self_attn.kv_a_layernorm.weight",
        # MoE gate (not quantized)
        "model.layers.61.mlp.gate.weight": "decoder.mlp.gate.weight",
        # Output
        "model.layers.61.shared_head.norm.weight": "norm.weight",
        "model.layers.61.shared_head.head.weight": "lm_head.weight",
    }
    
    # Find files containing these weights
    files_to_load = set()
    for key in weight_mapping.keys():
        if key in weight_map:
            files_to_load.add(weight_map[key])
    
    print(f"  Loading from {len(files_to_load)} files")
    
    stats = {"loaded": 0, "skipped": 0, "missing": 0, "shape_mismatch": 0}
    loaded_weights = {}
    
    # Load weights from safetensors files
    for filename in sorted(files_to_load):
        filepath = os.path.join(target_model_path, filename)
        print(f"  Loading: {filename}")
        
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_mapping:
                    target_key = weight_mapping[key]
                    tensor = f.get_tensor(key)
                    
                    # Skip FP8 quantized weights
                    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        print(f"    ⊘ Skipped (FP8): {key}")
                        stats["skipped"] += 1
                        continue
                    
                    # Convert to bfloat16 if needed
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    
                    loaded_weights[target_key] = tensor
                    stats["loaded"] += 1
    
    # Load weights into model
    model_state = draft_model.state_dict()
    for key, tensor in loaded_weights.items():
        if key in model_state:
            if model_state[key].shape == tensor.shape:
                model_state[key].copy_(tensor)
                print(f"    ✓ Loaded: {key} {list(tensor.shape)}")
            else:
                print(f"    ✗ Shape mismatch: {key} model={list(model_state[key].shape)} file={list(tensor.shape)}")
                stats["shape_mismatch"] += 1
        else:
            print(f"    ? Not in model: {key}")
            stats["missing"] += 1
    
    print(f"\n  Summary: loaded={stats['loaded']}, skipped={stats['skipped']}, shape_mismatch={stats['shape_mismatch']}")
    return stats


def generate_hidden_states_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: str,
    aux_layers: List[int] = [1, 29, 57],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Generate hidden states from target model for a batch of texts.
    
    Returns:
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        last_hidden_state: [batch, seq_len, hidden_size]
        aux_hidden_states: list of [batch, seq_len, hidden_size] for aux layers
    """
    # Tokenize
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # Get last hidden state (before MTP layer, i.e., layer 60's output)
    # In DeepSeek-V3.2, hidden_states[0] is embedding, hidden_states[i] is layer i-1's output
    # So hidden_states[61] is layer 60's output (the input to MTP layer)
    all_hidden_states = outputs.hidden_states
    last_hidden_state = all_hidden_states[61]  # Output of layer 60
    
    # Get aux hidden states for EAGLE3
    aux_hidden_states = []
    for layer_id in aux_layers:
        # hidden_states[layer_id+1] is the output of layer layer_id
        aux_hidden_states.append(all_hidden_states[layer_id + 1])
    
    return input_ids, attention_mask, last_hidden_state, aux_hidden_states


class MTPDataset(Dataset):
    """Dataset for MTP fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load data
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if "conversations" in item:
                    # Build conversation text
                    text = ""
                    for msg in item["conversations"]:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "user":
                            text += f"<|User|>{content}"
                        elif role == "assistant":
                            text += f"<|Assistant|>{content}"
                    self.samples.append(text)
                elif "text" in item:
                    self.samples.append(item["text"])
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_mtp_with_generated_hidden_states(
    draft_model: nn.Module,
    target_model: nn.Module,
    tokenizer,
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    device: str,
    output_dir: str,
    aux_layers: List[int] = [1, 29, 57],
):
    """
    Train MTP model using hidden states generated from target model.
    """
    draft_model = draft_model.to(device)
    draft_model.train()
    
    # Freeze target model
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in draft_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    total_steps = len(dataloader) * num_epochs
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_texts in pbar:
            # Generate hidden states from target model
            input_ids, attention_mask, last_hidden_state, aux_hidden_states = \
                generate_hidden_states_batch(
                    target_model, tokenizer, batch_texts, max_length, device, aux_layers
                )
            
            # Forward pass through draft model
            # The draft model expects: input_ids and hidden_states (from target)
            outputs = draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=last_hidden_state,
                aux_hidden_states=aux_hidden_states,
                labels=input_ids,  # For next token prediction
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(draft_model, output_dir, epoch + 1, avg_loss)
    
    return draft_model


def save_checkpoint(model: nn.Module, output_dir: str, epoch: int, loss: float):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
    
    # Save config
    if hasattr(model, "config"):
        model.config.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save(
        {"epoch": epoch, "loss": loss},
        os.path.join(checkpoint_dir, "training_state.pt"),
    )
    
    print(f"   Saved checkpoint to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek-V3.2 MTP layer for EAGLE speculative decoding"
    )
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
        help="Path to draft model config (optional, will use default)",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deepseek-v32-mtp-finetuned",
        help="Output directory",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "train"],
        default="train",
        help="Mode: test (verify setup) or train (full training)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"DeepSeek-V3.2 MTP Fine-tuning")
    print("=" * 60)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    if args.draft_model_config is None:
        args.draft_model_config = os.path.join(base_dir, "configs", "deepseek-v32-mtp-eagle.json")
    
    if not os.path.isabs(args.train_data_path):
        args.train_data_path = os.path.join(base_dir, args.train_data_path)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load tokenizer
    print(f"\n1. Loading tokenizer from: {args.target_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Create draft model with config
    print(f"\n2. Loading draft model config from: {args.draft_model_config}")
    with open(args.draft_model_config, "r") as f:
        config_dict = json.load(f)
    
    # Ensure tie_word_embeddings is False
    config_dict["tie_word_embeddings"] = False
    draft_config = DeepSeekV32MTPConfig(**config_dict)
    
    print(f"\n3. Creating draft model...")
    draft_model = DeepSeekV32MTPForCausalLM(draft_config)
    print(f"   Parameters: {sum(p.numel() for p in draft_model.parameters()):,}")
    
    # 3. Load MTP weights from target model
    print(f"\n4. Loading MTP weights from target model...")
    load_mtp_weights_from_target(args.target_model_path, draft_model)
    
    if args.mode == "test":
        print("\n" + "=" * 60)
        print("TEST MODE: Verifying setup only")
        print("=" * 60)
        
        # Test forward pass
        print("\n5. Testing forward pass...")
        draft_model.eval()
        test_input = torch.randint(0, 1000, (1, 32))
        test_hidden = torch.randn(1, 32, draft_config.hidden_size)
        
        with torch.no_grad():
            output = draft_model(
                input_ids=test_input,
                hidden_states=test_hidden,
            )
        print(f"   Output logits shape: {output.logits.shape}")
        print(f"   ✓ Forward pass successful!")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        return
    
    # 4. Load target model for hidden state generation
    print(f"\n5. Loading target model for hidden state generation...")
    print(f"   This may take a while for the 671B model...")
    
    from transformers import AutoModelForCausalLM
    
    # Load target model with appropriate settings for 8xH200
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across GPUs
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    target_model.eval()
    print(f"   Target model loaded with device_map='auto'")
    
    # 5. Create dataset
    print(f"\n6. Creating dataset from: {args.train_data_path}")
    dataset = MTPDataset(args.train_data_path, tokenizer, args.max_length)
    
    # 6. Train
    print(f"\n7. Starting fine-tuning...")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {args.device}")
    
    train_mtp_with_generated_hidden_states(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    
    print("\nNext steps:")
    print("1. Export the model for sglang/vllm-magik:")
    print(f"   python scripts/export_deepseek_v32_mtp.py \\")
    print(f"     --input-dir {args.output_dir}/epoch_{args.num_epochs} \\")
    print(f"     --output-dir outputs/deepseek-v32-mtp-nextn \\")
    print(f"     --target-model-path {args.target_model_path}")


if __name__ == "__main__":
    main()
