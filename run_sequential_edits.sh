#!/bin/bash
# Run Sequential Knowledge Editing with ROME

set -e

echo "========================================"
echo "Sequential Knowledge Editing (ROME)"
echo "========================================"

# Default parameters
MODEL_DIR="outputs/models/gpt_small_no_alias"
KG_DIR="data/kg/ba_no_alias"
OUTPUT_DIR="outputs/sequential_no_alias"
NUM_STEPS="100"
NUM_EVAL_TRIPLES="1000"
NUM_RETAIN_TRIPLES="1000"
EVAL_MODE="sample"   # sample / all / all-excl-edits
EDIT_SELECTION_MODE="random"  # random / degree_high / degree_low / hop_high / hop_low
MAX_HOP="10"  # 通常は変更不要（小規模KGでは全ノードが到達可能）
LAYER="0"
V_NUM_GRAD_STEPS="20"
SEED="24"
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --kg-dir)
            KG_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --num-eval-triples)
            NUM_EVAL_TRIPLES="$2"
            shift 2
            ;;
        --num-retain-triples)
            NUM_RETAIN_TRIPLES="$2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="$2"
            shift 2
            ;;
        --edit-selection-mode)
            EDIT_SELECTION_MODE="$2"
            shift 2
            ;;
        --max-hop)
            MAX_HOP="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --v-num-grad-steps)
            V_NUM_GRAD_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Sequential Knowledge Editing - Run continuous ROME edits"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-dir <dir>           Path to trained model (default: outputs/models/gpt_small)"
            echo "  --kg-dir <dir>              Path to knowledge graph data (default: data/kg/ba)"
            echo "  --output-dir <dir>          Output directory (default: outputs/sequential)"
            echo "  --num-steps <n>             Number of sequential edits (default: 100)"
            echo "  --num-eval-triples <n>      Number of triples for evaluation when eval-mode=sample (default: 1000)"
            echo "  --num-retain-triples <n>    Number of unedited triples for retention evaluation (default: 1000)"
            echo "  --eval-mode <mode>          Eval triple selection mode: sample | all | all-excl-edits (default: sample)"
            echo "  --edit-selection-mode <m>   Edit selection strategy: random | degree_high | degree_low | hop_high | hop_low (default: random)"
            echo "  --max-hop <n>               Maximum hop distance for analysis (default: 10, usually no need to change)"
            echo "  --layer <n>                 Layer to edit (default: 0)"
            echo "  --v-num-grad-steps <n>      ROME gradient steps for optimization (default: 5)"
            echo "  --seed <n>                  Random seed (default: 42)"
            echo "  --device <device>           Device to use: cuda or cpu (default: cuda)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --num-steps 100 --eval-mode all"
            echo "  $0 --eval-mode all-excl-edits --model-dir outputs/models/custom --output-dir outputs/my_experiment"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model directory: $MODEL_DIR"
echo "  KG directory: $KG_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of steps: $NUM_STEPS"
echo "  Edit selection mode: $EDIT_SELECTION_MODE"
echo "  Evaluation triples (if sample): $NUM_EVAL_TRIPLES"
echo "  Retain triples (unedited): $NUM_RETAIN_TRIPLES"
echo "  Eval mode: $EVAL_MODE"
echo "  Max hop distance: $MAX_HOP"
echo "  Edit layer: $LAYER"
echo "  ROME v_num_grad_steps: $V_NUM_GRAD_STEPS"
echo "  Random seed: $SEED"
echo "  Device: $DEVICE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    echo "Error: Model not found at $MODEL_DIR/model.pt"
    echo "Please train a model first using ./run_training.sh"
    echo "Or specify correct --model-dir"
    exit 1
fi

# Check if KG exists
if [ ! -f "$KG_DIR/graph.jsonl" ]; then
    echo "Error: Knowledge graph not found at $KG_DIR/graph.jsonl"
    echo "Please generate a KG first using:"
    echo "  python -m src.cli.build_kg --config configs/kg_ba.yaml"
    echo "Or specify correct --kg-dir"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run sequential editing
echo "Starting sequential editing..."
echo "The run may take some time depending on model size and hardware."
echo ""

python -m src.scripts.run_sequential_edits \
    --model-dir "$MODEL_DIR" \
    --kg-dir "$KG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-steps "$NUM_STEPS" \
    --num-eval-triples "$NUM_EVAL_TRIPLES" \
    --num-retain-triples "$NUM_RETAIN_TRIPLES" \
    --eval-mode "$EVAL_MODE" \
    --edit-selection-mode "$EDIT_SELECTION_MODE" \
    --max-hop "$MAX_HOP" \
    --layer "$LAYER" \
    --v-num-grad-steps "$V_NUM_GRAD_STEPS" \
    --seed "$SEED" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "Sequential editing completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - Config: $OUTPUT_DIR/config.json"
echo "  - Step metrics: $OUTPUT_DIR/stats.jsonl"
echo "  - Ripple effects: $OUTPUT_DIR/ripple_triples.jsonl"
echo "  - Triple accuracy: $OUTPUT_DIR/triple_acc.jsonl"
echo ""
echo "Next steps:"
echo "  Run visualization: ./run_sequential_analysis.sh"
echo "  Or manually: python -m src.scripts.analyze_sequential_effects --output-dir $OUTPUT_DIR"
echo ""