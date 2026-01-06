import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig

from specforge.utils import padding


def load_config_with_fallback(model_path: str) -> PretrainedConfig:
    """
    Load config with fallback for models not registered in transformers.
    
    Args:
        model_path: Path to model or HuggingFace repo ID
        
    Returns:
        PretrainedConfig object
    """
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except (ValueError, KeyError) as e:
        # Fallback: try to read config.json directly
        if os.path.exists(model_path):
            cfg_path = os.path.join(model_path, "config.json")
        else:
            # Download from HuggingFace
            try:
                cfg_path = hf_hub_download(repo_id=model_path, filename="config.json")
            except Exception:
                raise RuntimeError(
                    f"Could not load config for {model_path}. Error: {e}"
                )
        
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg_dict = json.load(f)
            # Create a simple config object with the necessary attributes
            config = PretrainedConfig(**cfg_dict)
            return config
        else:
            raise RuntimeError(f"Could not find config.json at {cfg_path}")


class TargetHead(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.config = load_config_with_fallback(model_path)
        self.fc = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
    ) -> "TargetHead":
        target_head = cls(model_path)
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().cuda().to(torch.bfloat16)
        return target_head

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
    ):
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path)

        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(self.model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            raise FileNotFoundError(f"No index.json file found in {self.model_path}")
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][lm_head_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(self.model_path, ckpt_file), framework="pt"
            ) as f:
                lm_head = f.get_tensor(lm_head_key)
        else:
            state_dict = torch.load(os.path.join(self.model_path, ckpt_file))
            lm_head = state_dict[lm_head_key]
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # apply pading
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None]
        loss_mask = loss_mask.to(target.device)
        return input_ids, target, loss_mask
