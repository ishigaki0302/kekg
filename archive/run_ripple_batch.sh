#!/bin/bash
# Run 10 editing experiments with ripple analysis

CASES_FILE="outputs/ripple_test_cases.json"

for i in {1..10}; do
    echo "Running experiment $i/10..."
    
    SUBJECT=$(python -c "import json; cases=json.load(open('$CASES_FILE')); print(cases[$i-1]['subject'])")
    RELATION=$(python -c "import json; cases=json.load(open('$CASES_FILE')); print(cases[$i-1]['relation'])")
    TARGET=$(python -c "import json; cases=json.load(open('$CASES_FILE')); print(cases[$i-1]['target'])")
    
    python run_editing_single.py \
        --model-dir outputs/models/gpt_small \
        --kg-corpus data/kg/ba/corpus.base.txt \
        --subject "$SUBJECT" \
        --relation "$RELATION" \
        --target "$TARGET" \
        --layer 0 \
        --output-dir "outputs/ripple_exp_$i" \
        --analyze-ripple \
        --max-ripple-triples 1000
    
    echo "Experiment $i completed"
    echo ""
done

echo "All experiments completed!"
