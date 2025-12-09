#!/bin/bash
# Ripple Effect Analysis Script
# Performs N single edits with ripple analysis and creates visualizations
# Usage: ./run_ripple_analysis.sh [num_experiments]
#   num_experiments: Number of editing experiments to run (default: 10)

set -e

# Parse command-line arguments
NUM_EXPERIMENTS=${1:-10}
TOKENIZER_PATH="outputs/models/gpt_small/tokenizer.json"

echo "======================================================================"
echo "Ripple Effect Analysis Pipeline"
echo "======================================================================"
echo "Number of experiments: $NUM_EXPERIMENTS"
echo "Tokenizer: $TOKENIZER_PATH"
echo ""

# Step 1: Select test cases
echo "Step 1: Selecting $NUM_EXPERIMENTS test cases from knowledge graph..."
python -c "
import json
import random
import sys
import re

# Get parameters
num_experiments = int(sys.argv[1])
tokenizer_path = sys.argv[2]

random.seed(42)

# Load tokenizer to determine available entities and relations
print(f'Loading tokenizer from: {tokenizer_path}')
with open(tokenizer_path, 'r') as f:
    tokenizer = json.load(f)

# Extract entity and relation ranges
entity_pattern = re.compile(r'^E_(\d+)$')
relation_pattern = re.compile(r'^R_(\d+)$')

entity_ids = []
relation_ids = []

for token in tokenizer.keys():
    entity_match = entity_pattern.match(token)
    if entity_match:
        entity_ids.append(int(entity_match.group(1)))

    relation_match = relation_pattern.match(token)
    if relation_match:
        relation_ids.append(int(relation_match.group(1)))

if not entity_ids:
    print('ERROR: No entities found in tokenizer')
    sys.exit(1)

if not relation_ids:
    print('ERROR: No relations found in tokenizer')
    sys.exit(1)

max_entity = max(entity_ids)
min_entity = min(entity_ids)
max_relation = max(relation_ids)
min_relation = min(relation_ids)

print(f'Tokenizer stats:')
print(f'  Entities: E_{min_entity:03d} to E_{max_entity:03d} ({len(entity_ids)} entities)')
print(f'  Relations: R_{min_relation:02d} to R_{max_relation:02d} ({len(relation_ids)} relations)')
print()

# Load knowledge graph
triples = []
try:
    with open('data/kg/ba/graph.jsonl', 'r') as f:
        for line in f:
            triples.append(json.loads(line))
    print(f'Loaded {len(triples)} triples from knowledge graph')
except FileNotFoundError:
    print('WARNING: Knowledge graph not found, generating synthetic test cases')
    triples = []

# Select N diverse test cases
selected = []
used_subjects = set()

if triples:
    # Use existing knowledge graph as base
    for triple in triples:
        if len(selected) >= num_experiments:
            break
        s, r, o = triple['s'], triple['r'], triple['o']
        if s not in used_subjects:
            # Find an alternative target from the graph
            candidates = [t['o'] for t in triples if t['s'] != s and t['o'] != o]
            if candidates:
                new_target = random.choice(candidates)
                selected.append({
                    'subject': s,
                    'relation': r,
                    'original': o,
                    'target': new_target,
                    'case_id': len(selected) + 1
                })
                used_subjects.add(s)

# If we need more test cases, generate synthetic ones
while len(selected) < num_experiments:
    # Sample random entities and relations from tokenizer
    subject_id = random.choice(entity_ids)
    relation_id = random.choice(relation_ids)
    original_id = random.choice(entity_ids)
    target_id = random.choice(entity_ids)

    # Ensure subject and targets are different
    while target_id == original_id:
        target_id = random.choice(entity_ids)

    subject = f'E_{subject_id:03d}'
    relation = f'R_{relation_id:02d}'
    original = f'E_{original_id:03d}'
    target = f'E_{target_id:03d}'

    if subject not in used_subjects:
        selected.append({
            'subject': subject,
            'relation': relation,
            'original': original,
            'target': target,
            'case_id': len(selected) + 1
        })
        used_subjects.add(subject)

print(f'\\nSelected {len(selected)} test cases:')
for case in selected:
    print(f\"{case['case_id']}. {case['subject']} {case['relation']}: {case['original']} -> {case['target']}\")

# Save to file
import os
os.makedirs('outputs', exist_ok=True)
with open('outputs/ripple_test_cases.json', 'w') as f:
    json.dump(selected, f, indent=2)

print('\\nTest cases saved to: outputs/ripple_test_cases.json')
" "$NUM_EXPERIMENTS" "$TOKENIZER_PATH"
echo ""

# Step 2: Run N editing experiments with ripple analysis
echo "Step 2: Running $NUM_EXPERIMENTS editing experiments with ripple analysis..."
echo ""

for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
    echo "----------------------------------------"
    echo "Experiment $i/$NUM_EXPERIMENTS"
    echo "----------------------------------------"

    # Extract test case parameters
    SUBJECT=$(python -c "import json; cases=json.load(open('outputs/ripple_test_cases.json')); print(cases[$i-1]['subject'])")
    RELATION=$(python -c "import json; cases=json.load(open('outputs/ripple_test_cases.json')); print(cases[$i-1]['relation'])")
    TARGET=$(python -c "import json; cases=json.load(open('outputs/ripple_test_cases.json')); print(cases[$i-1]['target'])")

    echo "  Subject: $SUBJECT"
    echo "  Relation: $RELATION"
    echo "  Target: $TARGET"
    echo ""

    # Run editing with ripple analysis
    PYTHONPATH=. python src/scripts/run_editing_single.py \
        --model-dir outputs/models/gpt_small \
        --kg-corpus data/kg/ba/corpus.base.txt \
        --subject "$SUBJECT" \
        --relation "$RELATION" \
        --target "$TARGET" \
        --layer 0 \
        --output-dir "outputs/ripple_exp_$i" \
        --analyze-ripple \
        --max-ripple-triples 1000 \
        2>&1 | grep -E "INFO:|Success:|Ripple Effect Statistics:" || true

    echo "  ✓ Experiment $i completed"
    echo ""
done

echo "======================================================================"
echo "All experiments completed!"
echo "======================================================================"
echo ""

# Step 3: Analyze ripple effects and create visualizations
echo "Step 3: Analyzing ripple effects and creating visualizations..."
echo ""

PYTHONPATH=. python src/scripts/analyze_ripple_effects.py

echo ""
echo "======================================================================"
echo "Analysis Complete!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - Test cases:    outputs/ripple_test_cases.json"
echo "  - Experiments:   outputs/ripple_exp_1/ ... outputs/ripple_exp_$NUM_EXPERIMENTS/"
echo "  - Visualization: outputs/ripple_effects_analysis.png"
echo "  - Statistics:    outputs/ripple_effects_stats.json"
echo ""
echo "Summary statistics:"
cat outputs/ripple_effects_stats.json
echo ""
echo "Done!"
