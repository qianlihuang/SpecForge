#!/usr/bin/env python3
"""
Two-stage fine-tuning for DeepSeek-V3.2 MTP layer:

Stage 1: Generate hidden states from target model (GPU-intensive, runs once)
Stage 2: Train MTP layer using pre-generated hidden states (can run multiple times)

This approach is more efficient for experimentation as:
- Hidden states generation only needs to run once
- Training can be iterated quickly without loading the 671B model each time

Usage:
    # Stage 1: Generate hidden states
    python scripts/finetune_mtp_staged.py --stage generate \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --train-data-path cache/dataset/deepseek-v32-sample.jsonl \
        --cache-dir cache/hidden_states/deepseek-v32
    
    # Stage 2: Train MTP layer
    python scripts/finetune_mtp_staged.py --stage train \
        --cache-dir cache/hidden_states/deepseek-v32 \
        --output-dir outputs/deepseek-v32-mtp-finetuned
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HiddenStatesDataset(Dataset):
    """Dataset that loads pre-generated hidden states."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        
        # Load metadata
        with open(self.cache_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        self.num_samples = self.metadata["num_samples"]
        self.hidden_size = self.metadata["hidden_size"]
        self.max_length = self.metadata["max_length"]
        
        print(f"Loaded dataset with {self.num_samples} samples")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Max length: {self.max_length}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load sample
        sample_path = self.cache_dir / f"sample_{idx}.pt"
        data = torch.load(sample_path, map_location="cpu")
        return data


def generate_hidden_states(
    target_model_path: str,
    train_data_path: str,
    cache_dir: str,
    max_length: int = 512,
    batch_size: int = 1,
):
    """
    Stage 1: Generate hidden states from target model and cache them.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import tempfile
    import shutil
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Stage 1: Generating Hidden States")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer from: {target_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print(f"\n2. Loading training data from: {train_data_path}")
    samples = []
    with open(train_data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if "conversations" in item:
                text = ""
                for msg in item["conversations"]:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        text += f"<|User|>{content}"
                    elif role == "assistant":
                        text += f"<|Assistant|>{content}"
                samples.append(text)
            elif "text" in item:
                samples.append(item["text"])
    
    print(f"   Loaded {len(samples)} samples")
    
    # Load target model
    print(f"\n3. Loading target model (this may take a while)...")
    
    # Handle DeepSeek-V3.2 special config loading
    config_file = os.path.join(target_model_path, "config.json")
    with open(config_file, "r") as f:
        config_json = json.load(f)
    
    hidden_size = config_json.get("hidden_size", 7168)
    
    # Create temporary config with correct model type for transformers
    if config_json.get("model_type") == "deepseek_v32":
        print("   Detected DeepSeek-V3.2, converting config for transformers...")
        config_json["architectures"] = ["DeepseekV3ForCausalLM"]
        config_json["model_type"] = "deepseek_v3"
        
        # Create temp directory with modified config
        tmp_config_dir = tempfile.mkdtemp(prefix="deepseek_v32_config_")
        tmp_config_file = os.path.join(tmp_config_dir, "config.json")
        with open(tmp_config_file, "w") as f:
            json.dump(config_json, f)
        
        config = AutoConfig.from_pretrained(tmp_config_file, trust_remote_code=True)
        shutil.rmtree(tmp_config_dir)
    else:
        config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num hidden layers: {config.num_hidden_layers}")
    
    model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Use eager attention (flash_attn may not be installed)
    )
    model.eval()
    print(f"   Model loaded with device_map='auto'")
    
    # EAGLE3 aux layer IDs
    aux_layer_ids = [1, 29, 57]
    
    # Generate hidden states
    print(f"\n4. Generating hidden states for {len(samples)} samples...")
    
    for idx, text in enumerate(tqdm(samples, desc="Generating")):
        # Tokenize
        encodings = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get hidden states
        all_hidden_states = outputs.hidden_states
        
        # Last hidden state (layer 60's output = input to MTP layer)
        # hidden_states[0] is embedding, hidden_states[i] is layer (i-1)'s output
        # So hidden_states[61] is layer 60's output
        num_layers = len(all_hidden_states) - 1  # Subtract 1 for embedding
        last_hidden_idx = min(61, num_layers)  # In case model has fewer layers
        last_hidden = all_hidden_states[last_hidden_idx].cpu()
        
        # Aux hidden states for EAGLE3
        aux_hidden = []
        for layer_id in aux_layer_ids:
            if layer_id + 1 < len(all_hidden_states):
                aux_hidden.append(all_hidden_states[layer_id + 1].cpu())
        
        # Save
        sample_data = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "last_hidden": last_hidden,
            "aux_hidden": aux_hidden,
        }
        
        torch.save(sample_data, cache_path / f"sample_{idx}.pt")
        
        # Clear GPU memory periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save metadata
    metadata = {
        "num_samples": len(samples),
        "hidden_size": hidden_size,
        "max_length": max_length,
        "target_model_path": target_model_path,
        "aux_layer_ids": aux_layer_ids,
    }
    
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Hidden states saved to: {cache_dir}")
    print(f"   Total samples: {len(samples)}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


def load_mtp_weights(target_model_path: str, draft_model: nn.Module):
    """Load non-quantized MTP weights from target model."""
    print(f"\nLoading MTP weights from: {target_model_path}")
    
    # Load weight map index
    index_path = os.path.join(target_model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    # Weight mapping (non-quantized weights only)
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
    
    # Find files to load
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
                    
                    if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        continue
                    
                    if tensor.dtype == torch.float32:
                        tensor = tensor.to(torch.bfloat16)
                    
                    if target_key in model_state and model_state[target_key].shape == tensor.shape:
                        model_state[target_key].copy_(tensor)
                        print(f"    ✓ {target_key}")
                        loaded_count += 1
    
    print(f"  Loaded {loaded_count} weights")


def train_mtp(
    cache_dir: str,
    output_dir: str,
    target_model_path: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    device: str = "cuda",
):
    """
    Stage 2: Train MTP layer using pre-generated hidden states.
    """
    from specforge.modeling.draft.deepseek_v32_mtp import (
        DeepSeekV32MTPConfig,
        DeepSeekV32MTPForCausalLM,
    )
    
    print("=" * 60)
    print("Stage 2: Training MTP Layer")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading cached hidden states from: {cache_dir}")
    dataset = HiddenStatesDataset(cache_dir)
    
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    config_path = os.path.join(base_dir, "configs", "deepseek-v32-mtp-eagle.json")
    
    print(f"\n2. Loading draft model config from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config_dict["tie_word_embeddings"] = False
    draft_config = DeepSeekV32MTPConfig(**config_dict)
    
    # Create model
    print(f"\n3. Creating draft model...")
    draft_model = DeepSeekV32MTPForCausalLM(draft_config)
    print(f"   Parameters: {sum(p.numel() for p in draft_model.parameters()):,}")
    
    # Load weights from target
    print(f"\n4. Loading MTP weights from target model...")
    load_mtp_weights(target_model_path, draft_model)
    
    # Move to device
    draft_model = draft_model.to(device).to(torch.bfloat16)
    draft_model.train()
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in draft_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    
    # Training loop
    print(f"\n5. Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Move to device - squeeze all dims of 1 (from the stored batch dim)
            input_ids = batch["input_ids"].squeeze(1).to(device)  # [batch, 1, seq] -> [batch, seq]
            attention_mask = batch["attention_mask"].squeeze(1).to(device)  # [batch, 1, seq] -> [batch, seq]
            last_hidden = batch["last_hidden"].squeeze(1).to(device).to(torch.bfloat16)  # [batch, 1, seq, hidden] -> [batch, seq, hidden]
            
            # Forward pass
            outputs = draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=last_hidden,
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
        checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state_dict = {k: v.cpu() for k, v in draft_model.state_dict().items()}
        save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
        draft_config.save_pretrained(checkpoint_dir)
        
        torch.save(
            {"epoch": epoch + 1, "loss": avg_loss},
            os.path.join(checkpoint_dir, "training_state.pt"),
        )
        
        print(f"   Saved checkpoint to: {checkpoint_dir}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage fine-tuning for DeepSeek-V3.2 MTP layer"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["generate", "train", "both"],
        required=True,
        help="Stage to run: generate (hidden states), train (MTP), or both",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default="/data/models/DeepSeek-V3.2",
        help="Path to DeepSeek-V3.2 model",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="cache/dataset/deepseek-v32-sample.jsonl",
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache/hidden_states/deepseek-v32",
        help="Directory for cached hidden states",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deepseek-v32-mtp-finetuned",
        help="Output directory for trained model",
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
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    if not os.path.isabs(args.train_data_path):
        args.train_data_path = os.path.join(base_dir, args.train_data_path)
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(base_dir, args.cache_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    if args.stage in ["generate", "both"]:
        generate_hidden_states(
            target_model_path=args.target_model_path,
            train_data_path=args.train_data_path,
            cache_dir=args.cache_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    
    if args.stage in ["train", "both"]:
        train_mtp(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            target_model_path=args.target_model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )


if __name__ == "__main__":
    main()
