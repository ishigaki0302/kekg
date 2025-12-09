#!/bin/bash
# Batch ROME editing with random triple selection
# Edits 10 random triples and analyzes ripple effects on entire KG

set -e

echo "========================================"
echo "ROME Batch Editing - Random Triples"
echo "========================================"

# Configuration
MODEL_DIR="outputs/models/gpt_small"
KG_CORPUS="data/kg/ba/corpus.base.txt"
OUTPUT_BASE_DIR="outputs/editing_batch_results"
LAYER=""  # Auto-detect optimal layer (leave empty for auto-detection)
NUM_EDITS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --kg-corpus)
            KG_CORPUS="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --num-edits)
            NUM_EDITS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model directory: $MODEL_DIR"
echo "  KG corpus: $KG_CORPUS"
echo "  Edit layer: $LAYER"
echo "  Number of edits: $NUM_EDITS"
echo "  Output base directory: $OUTPUT_BASE_DIR"
echo ""

# Check if corpus exists
if [ ! -f "$KG_CORPUS" ]; then
    echo "Error: Corpus file not found at $KG_CORPUS"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Get unique entities and relations from corpus
echo "Extracting entities and relations from corpus..."
ENTITIES_FILE="$OUTPUT_BASE_DIR/entities.txt"
RELATIONS_FILE="$OUTPUT_BASE_DIR/relations.txt"

awk '{print $1}' "$KG_CORPUS" | sort -u > "$ENTITIES_FILE"
awk '{print $2}' "$KG_CORPUS" | sort -u > "$RELATIONS_FILE"

NUM_ENTITIES=$(wc -l < "$ENTITIES_FILE")
NUM_RELATIONS=$(wc -l < "$RELATIONS_FILE")

echo "Found $NUM_ENTITIES entities and $NUM_RELATIONS relations"
echo ""

# Generate random edits
echo "Generating $NUM_EDITS random edits..."
EDITS_FILE="$OUTPUT_BASE_DIR/edit_list.txt"
> "$EDITS_FILE"  # Clear file

for i in $(seq 1 $NUM_EDITS); do
    # Select random line from corpus (original triple)
    TOTAL_LINES=$(wc -l < "$KG_CORPUS")
    RANDOM_LINE=$((RANDOM % TOTAL_LINES + 1))
    TRIPLE=$(sed -n "${RANDOM_LINE}p" "$KG_CORPUS")

    SUBJECT=$(echo "$TRIPLE" | awk '{print $1}')
    RELATION=$(echo "$TRIPLE" | awk '{print $2}')
    ORIGINAL=$(echo "$TRIPLE" | awk '{print $3}')

    # Select random target entity (different from original)
    TARGET="$ORIGINAL"
    while [ "$TARGET" = "$ORIGINAL" ]; do
        RANDOM_ENTITY_LINE=$((RANDOM % NUM_ENTITIES + 1))
        TARGET=$(sed -n "${RANDOM_ENTITY_LINE}p" "$ENTITIES_FILE")
    done

    echo "Edit $i: ($SUBJECT, $RELATION, $ORIGINAL) -> ($SUBJECT, $RELATION, $TARGET)"
    echo "$SUBJECT|$RELATION|$ORIGINAL|$TARGET" >> "$EDITS_FILE"
done

echo ""
echo "========================================"
echo "Starting batch editing..."
echo "========================================"
echo ""

# Process each edit
EDIT_NUM=1
while IFS='|' read -r SUBJECT RELATION ORIGINAL TARGET; do
    echo "========================================"
    echo "Edit $EDIT_NUM/$NUM_EDITS"
    echo "========================================"
    echo "  ($SUBJECT, $RELATION, $ORIGINAL) -> $TARGET"
    echo ""

    OUTPUT_DIR="$OUTPUT_BASE_DIR/edit_${EDIT_NUM}"
    mkdir -p "$OUTPUT_DIR"

    # Save edit info
    cat > "$OUTPUT_DIR/edit_info.txt" <<EOF
Edit Number: $EDIT_NUM
Subject: $SUBJECT
Relation: $RELATION
Original: $ORIGINAL
Target: $TARGET
Layer: ${LAYER:-auto-detect}
EOF

    # Run editing with ripple analysis on entire KG
    CMD="python run_editing_single.py \
        --model-dir \"$MODEL_DIR\" \
        --kg-corpus \"$KG_CORPUS\" \
        --subject \"$SUBJECT\" \
        --relation \"$RELATION\" \
        --target \"$TARGET\" \
        --original \"$ORIGINAL\" \
        --output-dir \"$OUTPUT_DIR\" \
        --analyze-ripple"

    # Add layer option only if specified
    if [ -n "$LAYER" ]; then
        CMD="$CMD --layer $LAYER"
    fi

    eval "$CMD" 2>&1 | tee "$OUTPUT_DIR/log.txt"

    echo ""
    echo "Edit $EDIT_NUM completed. Results saved to: $OUTPUT_DIR"
    echo ""

    EDIT_NUM=$((EDIT_NUM + 1))
done < "$EDITS_FILE"

echo ""
echo "========================================"
echo "Batch Editing Complete!"
echo "========================================"
echo "All results saved to: $OUTPUT_BASE_DIR"
echo ""

# Generate summary
SUMMARY_FILE="$OUTPUT_BASE_DIR/summary.txt"
echo "ROME Batch Editing Summary" > "$SUMMARY_FILE"
echo "=========================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Configuration:" >> "$SUMMARY_FILE"
echo "  Model: $MODEL_DIR" >> "$SUMMARY_FILE"
echo "  KG Corpus: $KG_CORPUS" >> "$SUMMARY_FILE"
echo "  Edit Layer: ${LAYER:-auto-detect}" >> "$SUMMARY_FILE"
echo "  Number of Edits: $NUM_EDITS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Edit Results:" >> "$SUMMARY_FILE"
echo "-------------" >> "$SUMMARY_FILE"

EDIT_NUM=1
while IFS='|' read -r SUBJECT RELATION ORIGINAL TARGET; do
    RESULT_FILE="$OUTPUT_BASE_DIR/edit_${EDIT_NUM}/edit_result.json"
    if [ -f "$RESULT_FILE" ]; then
        SUCCESS=$(python -c "import json; print(json.load(open('$RESULT_FILE'))['success'])")
        NEW_PRED=$(python -c "import json; print(json.load(open('$RESULT_FILE'))['new_prediction'])")

        echo "Edit $EDIT_NUM: ($SUBJECT, $RELATION, $ORIGINAL) -> $TARGET" >> "$SUMMARY_FILE"
        echo "  Success: $SUCCESS" >> "$SUMMARY_FILE"
        echo "  New Prediction: $NEW_PRED" >> "$SUMMARY_FILE"

        # Add ripple statistics if available
        RIPPLE_FILE="$OUTPUT_BASE_DIR/edit_${EDIT_NUM}/ripple_analysis.json"
        if [ -f "$RIPPLE_FILE" ]; then
            TOTAL_TRIPLES=$(python -c "import json; print(json.load(open('$RIPPLE_FILE'))['statistics']['total_triples'])")
            echo "  Ripple Analysis: $TOTAL_TRIPLES triples analyzed" >> "$SUMMARY_FILE"
        fi
        echo "" >> "$SUMMARY_FILE"
    fi
    EDIT_NUM=$((EDIT_NUM + 1))
done < "$EDITS_FILE"

echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo "View individual results:"
for i in $(seq 1 $NUM_EDITS); do
    echo "  Edit $i: $OUTPUT_BASE_DIR/edit_${i}/"
done
echo ""
