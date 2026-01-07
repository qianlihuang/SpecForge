# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.deepseek_v32_mtp import DeepSeekV32MTPConfig, DeepSeekV32MTPForCausalLM
from .target.eagle3_target_model import (
    CustomEagle3TargetModel,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
    get_eagle3_target_model,
)

__all__ = [
    "LlamaForCausalLMEagle3",
    "DeepSeekV32MTPConfig",
    "DeepSeekV32MTPForCausalLM",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
