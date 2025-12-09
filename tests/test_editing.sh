#!/bin/bash
# Quick test of ROME editing without ripple analysis

set -e

echo "Testing ROME editing with auto-detected layers..."
echo ""

# Test 3 random edits
cd ..
PYTHONPATH=. python src/scripts/run_editing_single.py \
    --model-dir outputs/models/gpt_small \
    --kg-corpus data/kg/ba/corpus.base.txt \
    --subject E_000 \
    --relation R_04 \
    --target E_999 \
    --output-dir outputs/test_edit_1
cd tests

echo ""
echo "Edit 1 result:"
cat outputs/test_edit_1/edit_result.json
echo ""

cd ..
PYTHONPATH=. python src/scripts/run_editing_single.py \
    --model-dir outputs/models/gpt_small \
    --kg-corpus data/kg/ba/corpus.base.txt \
    --subject E_005 \
    --relation R_28 \
    --target E_100 \
    --output-dir outputs/test_edit_2
cd tests

echo ""
echo "Edit 2 result:"
cat outputs/test_edit_2/edit_result.json
echo ""

cd ..
PYTHONPATH=. python src/scripts/run_editing_single.py \
    --model-dir outputs/models/gpt_small \
    --kg-corpus data/kg/ba/corpus.base.txt \
    --subject E_010 \
    --relation R_15 \
    --target E_050 \
    --output-dir outputs/test_edit_3
cd tests

echo ""
echo "Edit 3 result:"
cat outputs/test_edit_3/edit_result.json
echo ""

echo "Success summary:"
python -c "
import json
from pathlib import Path
for i in range(1, 4):
    data = json.load(open(f'outputs/test_edit_{i}/edit_result.json'))
    print(f\"Edit {i}: Target={data['o_target']}, Predicted={data['new_prediction']}, Success={data['success']}, Layer={data['layer']}\")
"
