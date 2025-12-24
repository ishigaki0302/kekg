#!/bin/bash
# Run sequential editing experiments for degree high and low with exclusive subjects
# This script ensures that subject entities used in degree_high and degree_low do not overlap

set -e

echo "========================================"
echo "Sequential Editing - Degree High/Low (Exclusive Subjects)"
echo "========================================"
echo ""

# Default parameters
MODEL_DIR="outputs/models/gpt_small_no_alias"
KG_DIR="data/kg/ba_no_alias"
BASE_OUTPUT_DIR="outputs/degree_exclusive_no_alias"
NUM_STEPS="1000"
NUM_RETAIN_TRIPLES="1000"
MAX_HOP="10"
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
        --base-output-dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --num-retain-triples)
            NUM_RETAIN_TRIPLES="$2"
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
            echo "Run degree high/low experiments with exclusive (non-overlapping) subjects"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-dir <dir>           Path to trained model (default: outputs/models/gpt_small_no_alias)"
            echo "  --kg-dir <dir>              Path to knowledge graph data (default: data/kg/ba_no_alias)"
            echo "  --base-output-dir <dir>     Base output directory (default: outputs/degree_exclusive_no_alias)"
            echo "  --num-steps <n>             Number of sequential edits per mode (default: 1000)"
            echo "  --num-retain-triples <n>    Number of unedited triples (default: 1000)"
            echo "  --max-hop <n>               Maximum hop distance (default: 10)"
            echo "  --layer <n>                 Layer to edit (default: 0)"
            echo "  --v-num-grad-steps <n>      ROME gradient steps (default: 20)"
            echo "  --seed <n>                  Random seed (default: 24)"
            echo "  --device <device>           Device: cuda or cpu (default: cuda)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "This script runs 2 experiments in parallel:"
            echo "  1. degree_high - High degree subjects (top unique subjects)"
            echo "  2. degree_low  - Low degree subjects (bottom unique subjects)"
            echo ""
            echo "Subject entities are guaranteed not to overlap between the two modes."
            echo "If NUM_STEPS * 2 > unique_subjects, NUM_STEPS is automatically adjusted"
            echo "to unique_subjects / 2 to ensure no overlap."
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
echo "  Base output directory: $BASE_OUTPUT_DIR"
echo "  Number of steps: $NUM_STEPS"
echo "  Retain triples: $NUM_RETAIN_TRIPLES"
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
    exit 1
fi

# Check if KG exists
if [ ! -f "$KG_DIR/corpus.train.txt" ]; then
    echo "Error: Knowledge graph not found at $KG_DIR/corpus.train.txt"
    exit 1
fi

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================"
echo "Starting 2 experiments in parallel..."
echo "========================================"
echo ""

# Define function to run a single experiment
run_experiment() {
    local MODE=$1
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODE}"

    # Assign GPU: degree_high to cuda:0, degree_low to cuda:1
    local ASSIGNED_DEVICE
    if [ "$MODE" = "degree_high" ]; then
        ASSIGNED_DEVICE="cuda:0"
    else
        ASSIGNED_DEVICE="cuda:1"
    fi

    echo "[${MODE}] Starting experiment (GPU: ${ASSIGNED_DEVICE})..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    python -m src.scripts.run_degree_exclusive_edits \
        --model-dir "$MODEL_DIR" \
        --kg-dir "$KG_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-steps "$NUM_STEPS" \
        --num-retain-triples "$NUM_RETAIN_TRIPLES" \
        --selection-mode "$MODE" \
        --max-hop "$MAX_HOP" \
        --layer "$LAYER" \
        --v-num-grad-steps "$V_NUM_GRAD_STEPS" \
        --seed "$SEED" \
        --device "$ASSIGNED_DEVICE" \
        > "${OUTPUT_DIR}/run.log" 2>&1

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[${MODE}] ✓ Experiment completed successfully"

        # Run analysis
        echo "[${MODE}] Generating visualizations..."
        python -m src.scripts.analyze_sequential_effects \
            --output-dir "$OUTPUT_DIR" \
            >> "${OUTPUT_DIR}/run.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[${MODE}] ✓ Visualization completed"
        else
            echo "[${MODE}] ✗ Visualization failed (check ${OUTPUT_DIR}/run.log)"
        fi
    else
        echo "[${MODE}] ✗ Experiment failed with exit code $EXIT_CODE (check ${OUTPUT_DIR}/run.log)"
    fi

    return $EXIT_CODE
}

# Export function and variables for parallel execution
export -f run_experiment
export MODEL_DIR KG_DIR BASE_OUTPUT_DIR NUM_STEPS NUM_RETAIN_TRIPLES MAX_HOP LAYER V_NUM_GRAD_STEPS SEED DEVICE

# Run both experiments in parallel
echo "Launching parallel experiments..."
echo ""

run_experiment "degree_high" &
PID_DEGREE_HIGH=$!
echo "  [degree_high] PID: $PID_DEGREE_HIGH"

run_experiment "degree_low" &
PID_DEGREE_LOW=$!
echo "  [degree_low]  PID: $PID_DEGREE_LOW"

echo ""
echo "All experiments launched. Waiting for completion..."
echo "You can monitor progress in real-time:"
echo "  tail -f $BASE_OUTPUT_DIR/degree_high/run.log"
echo "  tail -f $BASE_OUTPUT_DIR/degree_low/run.log"
echo ""

# Wait for all background jobs
wait $PID_DEGREE_HIGH
EXIT_DEGREE_HIGH=$?

wait $PID_DEGREE_LOW
EXIT_DEGREE_LOW=$?

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""

# Print summary
echo "Summary:"
echo "--------"
if [ $EXIT_DEGREE_HIGH -eq 0 ]; then
    echo "  [degree_high] ✓ SUCCESS"
else
    echo "  [degree_high] ✗ FAILED"
fi

if [ $EXIT_DEGREE_LOW -eq 0 ]; then
    echo "  [degree_low]  ✓ SUCCESS"
else
    echo "  [degree_low]  ✗ FAILED"
fi

echo ""

# Generate comparison plot if both experiments succeeded
if [ $EXIT_DEGREE_HIGH -eq 0 ] && [ $EXIT_DEGREE_LOW -eq 0 ]; then
    echo "========================================"
    echo "Generating comparison plot..."
    echo "========================================"
    echo ""

    python -m src.scripts.compare_selection_modes --base-dir "$BASE_OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Comparison plot generated successfully!"
    else
        echo ""
        echo "⚠ Failed to generate comparison plot"
    fi
else
    echo "⚠ Skipping comparison plot (need both experiments to succeed)"
fi

echo ""
echo "Results directory: $BASE_OUTPUT_DIR"
echo ""
echo "Generated outputs:"
echo "  - ${BASE_OUTPUT_DIR}/degree_high/"
echo "      ├── config.json"
echo "      ├── stats.jsonl"
echo "      ├── plots_time_series.png"
echo "      └── run.log"
echo "  - ${BASE_OUTPUT_DIR}/degree_low/"
echo "      ├── config.json"
echo "      ├── stats.jsonl"
echo "      ├── plots_time_series.png"
echo "      └── run.log"
echo "  - ${BASE_OUTPUT_DIR}/comparison.png  ← 2条件比較プロット"
echo ""

# Check if all succeeded
if [ $EXIT_DEGREE_HIGH -eq 0 ] && [ $EXIT_DEGREE_LOW -eq 0 ]; then
    echo "✓ All experiments completed successfully!"
    echo ""
    echo "Note: Subject entities used in degree_high and degree_low are guaranteed not to overlap."
    exit 0
else
    echo "⚠ Some experiments failed. Check the logs for details."
    exit 1
fi
