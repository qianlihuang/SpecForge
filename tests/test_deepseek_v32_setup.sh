#!/bin/bash
# Quick test script to verify DeepSeek-V3.2 training setup
# This tests the data preprocessing pipeline without loading the target model

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

echo "============================================"
echo "DeepSeek-V3.2 Training Setup Verification"
echo "============================================"

cd $ROOT_DIR

# Test 1: Parser import and basic function
echo ""
echo "[Test 1/3] Testing parser import..."
python -c "
from specforge.data.parse import DeepSeekV32Parser
from specforge.data.template import TEMPLATE_REGISTRY

template = TEMPLATE_REGISTRY.get('deepseek-v32')
assert template.parser_type == 'deepseek-v32', f'Expected parser_type=deepseek-v32, got {template.parser_type}'
print('✓ Parser import successful')
print(f'  Template: {template}')
"

# Test 2: Dataset preprocessing
echo ""
echo "[Test 2/3] Testing dataset preprocessing..."
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer
from specforge.data import build_eagle3_dataset

dataset = load_dataset('json', data_files='$ROOT_DIR/cache/dataset/deepseek-v32-sample.jsonl')['train']
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3.2', trust_remote_code=True)

eagle3_dataset = build_eagle3_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    chat_template='deepseek-v32',
    max_length=2048,
    num_proc=1,
    cache_dir='$ROOT_DIR/cache/processed_dataset',
    cache_key='test-v32-quick',
)
print(f'✓ Dataset preprocessing successful')
print(f'  Samples: {len(eagle3_dataset)}')
print(f'  Columns: {eagle3_dataset.column_names}')
"

# Test 3: Config loading
echo ""
echo "[Test 3/3] Testing config loading..."
python -c "
from specforge.modeling import AutoDraftModelConfig

config = AutoDraftModelConfig.from_file('$ROOT_DIR/configs/deepseek-v32-671b-eagle3.json')
print(f'✓ Config loading successful')
print(f'  Hidden size: {config.hidden_size}')
print(f'  Vocab size: {config.vocab_size}')
print(f'  Draft vocab size: {config.draft_vocab_size}')
"

echo ""
echo "============================================"
echo "All tests passed! ✓"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Run hidden states generation: ./examples/run_deepseek_v32_671b_eagle3_offline.sh"
echo "2. Or run online training: ./examples/run_deepseek_v32_671b_eagle3_online.sh"
