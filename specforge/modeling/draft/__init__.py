from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3
from .deepseek_v32_mtp import DeepSeekV32MTPConfig, DeepSeekV32MTPForCausalLM

__all__ = [
    "Eagle3DraftModel", 
    "LlamaForCausalLMEagle3",
    "DeepSeekV32MTPConfig",
    "DeepSeekV32MTPForCausalLM",
]
