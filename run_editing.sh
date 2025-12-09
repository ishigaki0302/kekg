#!/bin/bash
# Run ROME Editing on a specific (s,r,o) triple

set -e

echo "========================================"
echo "ROME Knowledge Editing - Single Edit"
echo "========================================"

# Default parameters
MODEL_DIR="outputs/models/gpt_small"
KG_CORPUS="data/kg/ba/corpus.train.txt"  # Use training data (learned knowledge graph)
OUTPUT_DIR="outputs/editing_results"
LOCATING_DIR="outputs/locating_results"
SUBJECT=""
RELATION=""
TARGET=""
ORIGINAL=""
LAYER="0"  # Default to layer 0 (from locating results)
ANALYZE_RIPPLE=""
MAX_RIPPLE_TRIPLES=""

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
        --subject|-s)
            SUBJECT="$2"
            shift 2
            ;;
        --relation|-r)
            RELATION="$2"
            shift 2
            ;;
        --target|-o)
            TARGET="$2"
            shift 2
            ;;
        --original)
            ORIGINAL="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --locating-dir)
            LOCATING_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --analyze-ripple)
            ANALYZE_RIPPLE="--analyze-ripple"
            shift 1
            ;;
        --max-ripple-triples)
            MAX_RIPPLE_TRIPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$SUBJECT" ] || [ -z "$RELATION" ] || [ -z "$TARGET" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 --subject <s> --relation <r> --target <o> [options]"
    echo ""
    echo "Required:"
    echo "  --subject, -s <s>       Subject entity"
    echo "  --relation, -r <r>      Relation"
    echo "  --target, -o <o>        Target object (new fact)"
    echo ""
    echo "Optional:"
    echo "  --model-dir <dir>       Path to model directory (default: outputs/models/gpt_small)"
    echo "  --kg-corpus <file>      Path to corpus file (default: data/kg/ba/corpus.train.txt - learned KG)"
    echo "  --original <o>          Original object for comparison (auto-detected if not provided)"
    echo "  --layer <n>             Layer to edit (auto-detected from locating results if not provided)"
    echo "  --locating-dir <dir>    Directory with locating results (default: outputs/locating_results)"
    echo "  --output-dir <dir>      Output directory (default: outputs/editing_results)"
    echo "  --analyze-ripple        Analyze ripple effects on entire knowledge graph"
    echo "  --max-ripple-triples <n> Maximum number of triples to analyze for ripple (default: all)"
    echo ""
    echo "Example:"
    echo "  $0 --subject E_000 --relation R_04 --target E_999"
    echo "  $0 --subject E_000 --relation R_04 --target E_999 --layer 0"
    echo "  $0 --subject E_000 --relation R_04 --target E_999 --analyze-ripple"
    exit 1
fi

echo "Configuration:"
echo "  Model directory: $MODEL_DIR"
echo "  KG corpus: $KG_CORPUS"
echo "  Subject: $SUBJECT"
echo "  Relation: $RELATION"
echo "  Target: $TARGET"
if [ -n "$ORIGINAL" ]; then
    echo "  Original: $ORIGINAL"
fi
if [ -n "$LAYER" ]; then
    echo "  Layer: $LAYER"
fi
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if model exists
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    echo "Error: Model not found at $MODEL_DIR/model.pt"
    echo "Please train a model first or specify correct --model-dir"
    exit 1
fi

# Check if corpus exists
if [ ! -f "$KG_CORPUS" ]; then
    echo "Warning: Corpus file not found at $KG_CORPUS"
    echo "Will use identity matrix for covariance (less accurate)"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="PYTHONPATH=. python src/scripts/run_editing_single.py \
    --model-dir \"$MODEL_DIR\" \
    --kg-corpus \"$KG_CORPUS\" \
    --subject \"$SUBJECT\" \
    --relation \"$RELATION\" \
    --target \"$TARGET\" \
    --output-dir \"$OUTPUT_DIR\""

if [ -n "$ORIGINAL" ]; then
    CMD="$CMD --original \"$ORIGINAL\""
fi

if [ -n "$LAYER" ]; then
    CMD="$CMD --layer $LAYER"
fi

if [ -n "$ANALYZE_RIPPLE" ]; then
    CMD="$CMD $ANALYZE_RIPPLE"
fi

if [ -n "$MAX_RIPPLE_TRIPLES" ]; then
    CMD="$CMD --max-ripple-triples $MAX_RIPPLE_TRIPLES"
fi

# Run editing
echo "Running ROME knowledge editing..."
echo ""

eval $CMD

echo ""
echo "========================================"
echo "Editing completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View the results:"
echo "  - Summary: $OUTPUT_DIR/edit_result.json"
echo "  - Locating: $OUTPUT_DIR/locating_result.png"
echo "  - Editing: $OUTPUT_DIR/editing_result.png"
