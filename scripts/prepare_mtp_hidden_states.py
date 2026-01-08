#!/usr/bin/env python3
"""
Generate hidden states for DeepSeek-V3.2 MTP layer fine-tuning.

This script generates:
1. Hidden states from layer 59 (input to MTP layer 60)
2. Target logits (ground truth outputs)

These will be used to fine-tune the MTP layer for EAGLE speculative decoding.

Usage:
    torchrun --standalone --nproc_per_node=8 \
        scripts/prepare_mtp_hidden_states.py \
        --target-model-path /data/models/DeepSeek-V3.2 \
        --data-path cache/dataset/deepseek-v32-sample.jsonl \
        --output-path cache/hidden_states/deepseek-v32-mtp \
        --chat-template deepseek-v32 \
        --max-length 2048 \
        --tp-size 8 \
        --batch-size 1
"""

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from specforge.utils import print_with_rank, rank_0_priority


@dataclass
class MTPDataPoint:
    """Data point for MTP layer training."""
    input_ids: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor  # [seq_len]
    hidden_state: torch.Tensor  # [seq_len, hidden_size] - hidden state before layer 60
    target_logits: torch.Tensor  # [seq_len, vocab_size] - target output logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hidden states for DeepSeek-V3.2 MTP layer fine-tuning"
    )

    # Model arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to DeepSeek-V3.2 model",
    )

    # Data arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data-path", type=str, required=True)
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument("--chat-template", type=str, default="deepseek-v32")
    data_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text",
    )
    data_group.add_argument("--num-samples", type=int, default=None)
    data_group.add_argument("--build-dataset-num-proc", type=int, default=8)

    # Inference arguments
    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument("--tp-size", type=int, default=1)
    inference_group.add_argument("--batch-size", type=int, default=1)

    # Output arguments
    output_group = parser.add_argument_group("output")
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--output-path", type=str, required=True)
    output_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="Directory to download model",
    )
    output_group.add_argument(
        "--dist-timeout",
        type=int,
        default=2000,
        help="Timeout for distributed communication in minutes",
    )
    output_group.add_argument(
        "--num-io-threads",
        type=int,
        default=4,
        help="Number of threads for async I/O",
    )
    output_group.add_argument(
        "--file-group-size",
        type=int,
        default=2000,
        help="Number of files per subdirectory",
    )

    # SGLang arguments
    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_target_model(args, model_config):
    """Build target model using SGLang backend."""
    target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=(
            model_config.dtype
            if hasattr(model_config, "dtype")
            else model_config.torch_dtype
        ),
        device="cuda",
        cache_dir=args.model_download_dir,
        **target_model_kwargs,
    )
    
    # For MTP training, we need the FINAL hidden states (after all layers and norm)
    # This is different from EAGLE3 which needs intermediate layer hidden states
    #
    # DeepSeek-V3.2 architecture:
    # - num_hidden_layers = 61 (config value)
    # - Layers 0-60 are regular decoder layers
    # - Layer 61 is the MTP layer (with enorm, hnorm, eh_proj, shared_head)
    #
    # For MTP training:
    # - We need the output AFTER all decoder layers (0-60) and final norm
    # - This is the final hidden_states that goes to lm_head
    # - We use return_last_hidden_states=True instead of aux_hidden_states
    #
    # NOTE: We do NOT call set_eagle3_layers_to_capture() because:
    # 1. EAGLE3's layers_to_capture captures BEFORE a layer processes (input to layer)
    # 2. For MTP, we need AFTER all layers process (output of final layer + norm)
    # 3. The model's final hidden_states is exactly what we need
    
    num_layers = model_config.num_hidden_layers  # 61 for DeepSeek-V3.2
    
    print(f"[INFO] Target model has {num_layers} regular decoder layers")
    print(f"[INFO] Using final hidden states (after all layers + norm) for MTP training")
    print(f"[INFO] MTP layer index: {num_layers} (will be trained separately)")
    
    return target_model


class MTPHiddenStatesGenerator:
    """Generator for MTP hidden states using SGLang backend."""

    def __init__(
        self,
        target_model,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
    ):
        self.model = target_model
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.show_progress = dist.get_rank(get_tp_group()) == 0
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        if is_tp_rank_0():
            self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_tp_rank_0() and self.io_executor is not None:
            if self.show_progress:
                print("\nWaiting for all I/O operations to complete...")
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None
        dist.barrier()

    def _save_tensor_sync(self, data_point: MTPDataPoint, output_file: str):
        """Save data point synchronously."""
        # Check for NaN values
        if torch.any(torch.isnan(data_point.hidden_state)):
            print(f"Warning: NaN in hidden_state for {output_file}. Skipping.")
            return
        if torch.any(torch.isnan(data_point.target_logits)):
            print(f"Warning: NaN in target_logits for {output_file}. Skipping.")
            return
        
        torch.save(asdict(data_point), output_file)

    def _save_tensor_async(self, data_point: MTPDataPoint, output_file: str):
        """Save data point asynchronously."""
        assert is_tp_rank_0()
        
        # Handle backpressure
        if len(self.pending_futures) >= self.io_queue_size:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
            if len(self.pending_futures) >= self.io_queue_size:
                self.pending_futures.pop(0).result()

        future = self.io_executor.submit(self._save_tensor_sync, data_point, output_file)
        self.pending_futures.append(future)

    def _wait_all_saves(self):
        """Wait for all pending saves to complete."""
        if is_tp_rank_0() and self.pending_futures:
            for future in tqdm(
                self.pending_futures,
                desc="Finalizing Writes",
                disable=not self.show_progress,
            ):
                future.result()
            self.pending_futures.clear()

    def _get_file_path(self, output_path: str, idx: int) -> str:
        """Get file path for data point."""
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
        return os.path.join(output_path, grouped_subdir, f"data_{idx}.ckpt")

    def _prepare_output_dirs(self, output_path: str, start_idx: int, total_samples: int):
        """Prepare output directories."""
        if not is_tp_rank_0() or total_samples == 0:
            return
        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_sample_idx = start_idx + total_samples - 1
        end_group = (end_sample_idx // self.file_group_size) * self.file_group_size
        for group_start_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = f"rows_{group_start_idx}-{group_start_idx + self.file_group_size}"
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(self, output_path: str, global_indices: List[int]) -> List[bool]:
        """Check if files exist for given indices."""
        if not is_tp_rank_0():
            return [False] * len(global_indices)

        def check_single_file(idx):
            return os.path.exists(self._get_file_path(output_path, idx))

        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    @torch.no_grad()
    def generate(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_path: str,
        start_idx: int = 0,
        samples_per_dp: int = 0,
    ):
        """Generate hidden states for MTP layer training."""
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        tp_group = get_tp_group()
        tp_group_ranks = dist.get_process_group_ranks(tp_group)
        tp_rank_0_global = tp_group_ranks[0]
        global_idx = start_idx

        progress_bar = tqdm(
            data_loader,
            disable=not self.show_progress,
            desc="Generating MTP Hidden States",
            position=dist.get_rank(get_dp_group()),
            leave=True,
        )

        total_skipped, total_processed = 0, 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            # Check existing files
            if is_tp_rank_0():
                exists_list = self._check_existing_files_batch(output_path, current_batch_indices)
                exists_tensor = torch.tensor(exists_list, dtype=torch.bool, device="cuda")
            else:
                exists_tensor = torch.tensor([False] * batch_size, dtype=torch.bool, device="cuda")
            dist.broadcast(exists_tensor, src=tp_rank_0_global, group=tp_group)

            # Filter to valid indices
            valid_indices_in_batch = [i for i, exists in enumerate(exists_tensor) if not exists]
            sample_global_indices = [current_batch_indices[i] for i in valid_indices_in_batch]
            num_valid = len(valid_indices_in_batch)
            total_skipped += batch_size - num_valid

            global_idx += batch_size

            if num_valid == 0:
                if self.show_progress:
                    progress_bar.set_postfix({
                        "processed": total_processed,
                        "skipped": total_skipped,
                    })
                continue

            # Filter batch
            filtered_batch = {
                "input_ids": batch["input_ids"][valid_indices_in_batch],
                "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                "loss_mask": batch["loss_mask"][valid_indices_in_batch],
            }
            del batch

            # Move to GPU
            filtered_batch_gpu = {k: v.cuda(non_blocking=True) for k, v in filtered_batch.items()}

            # Get hidden states from FINAL layer (layer 60 output) for MTP training
            # Note: MTP layer (layer 61) takes the output of layer 60 as input
            # We use return_last_hidden_states=True to get the FINAL hidden states
            # (after all decoder layers, before lm_head)
            _, logits_list, _, last_hidden_states_list = self.model.extend(
                **filtered_batch_gpu,
                return_last_hidden_states=True,  # Get final hidden states for MTP
                return_logits=True,
            )

            del filtered_batch_gpu

            if is_tp_rank_0():
                for i, (current_global_idx, hidden_states, logits) in enumerate(
                    zip(sample_global_indices, last_hidden_states_list, logits_list)
                ):
                    # Create data point
                    data_point = MTPDataPoint(
                        input_ids=filtered_batch["input_ids"][i].clone(),
                        loss_mask=filtered_batch["loss_mask"][i].clone(),
                        hidden_state=hidden_states.cpu().clone(),  # [seq_len, hidden_size]
                        target_logits=logits.cpu().clone(),  # [seq_len, vocab_size]
                    )

                    # Save asynchronously
                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(data_point, output_file)

                    del hidden_states, logits

                total_processed += len(sample_global_indices)

            del last_hidden_states_list, logits_list, filtered_batch

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if self.show_progress:
                progress_bar.set_postfix({
                    "processed": total_processed,
                    "skipped": total_skipped,
                    "pending_io": len(self.pending_futures) if is_tp_rank_0() else 0,
                })

        if self.show_progress:
            print(f"\nGeneration complete. Processed: {total_processed}, Skipped: {total_skipped}")
        dist.barrier()


def main():
    args = parse_args()

    print("=" * 60)
    print("DeepSeek-V3.2 MTP Hidden States Generation")
    print("=" * 60)
    print(f"Target model: {args.target_model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"TP size: {args.tp_size}")
    print(f"Max length: {args.max_length}")
    print("=" * 60)

    # Initialize distributed
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    # Load model config with fallback for unregistered model types
    try:
        model_config = AutoConfig.from_pretrained(
            args.target_model_path,
            trust_remote_code=True,
            cache_dir=args.model_download_dir,
        )
    except (ValueError, KeyError) as e:
        print(f"AutoConfig failed with: {e}. Loading config.json directly...")
        from transformers import PretrainedConfig
        config_path = os.path.join(args.target_model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        model_config = PretrainedConfig(**config_dict)
        print(f"Loaded config from {config_path}")

    # Build target model with SGLang backend
    print("\n[Step 1/3] Building target model...")
    target_model = build_target_model(args, model_config)

    # Build dataset
    print("\n[Step 2/3] Building dataset...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=True,
        cache_dir=args.model_download_dir,
    )

    # First load the raw dataset
    if args.data_path.endswith(".jsonl"):
        raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        raw_dataset = load_dataset(args.data_path, split="train")
    
    if args.num_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.num_samples, len(raw_dataset))))
    
    # Process with build_eagle3_dataset
    dataset = build_eagle3_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        num_proc=args.build_dataset_num_proc,
    )

    # Create dataloaders
    train_dataloader = prepare_dp_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        process_group=get_dp_group(),
        shuffle=False,
    )
    
    # Calculate samples per DP rank
    dp_world_size = dist.get_world_size(get_dp_group())
    total_samples = len(dataset)
    samples_per_dp = (total_samples + dp_world_size - 1) // dp_world_size

    # Generate hidden states
    print("\n[Step 3/3] Generating hidden states...")
    start_idx = dist.get_rank(get_dp_group()) * samples_per_dp

    with MTPHiddenStatesGenerator(
        target_model=target_model,
        num_io_threads=args.num_io_threads,
        file_group_size=args.file_group_size,
    ) as generator:
        generator.generate(
            data_loader=train_dataloader,
            output_path=args.output_path,
            start_idx=start_idx,
            samples_per_dp=samples_per_dp,
        )

    print("\n" + "=" * 60)
    print("Hidden states generation complete!")
    print(f"Output saved to: {args.output_path}")
    print("=" * 60)

    destroy_distributed()


if __name__ == "__main__":
    main()
