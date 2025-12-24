#!/bin/bash
# Run sequential editing experiments for degree-binned triples
# This script divides all triples by degree (high to low) into bins of NUM_STEPS size
# and runs editing experiments for each bin in parallel

set -e

echo "========================================"
echo "Sequential Editing - Degree Binned Analysis"
echo "========================================"
echo ""

# Default parameters
MODEL_DIR="outputs/models/gpt_small_no_alias"
KG_DIR="data/kg/ba_no_alias"
BASE_OUTPUT_DIR="outputs/degree_binned_no_alias"
NUM_STEPS="1000"
NUM_RETAIN_TRIPLES="1000"
MAX_HOP="10"
LAYER="0"
V_NUM_GRAD_STEPS="20"
SEED="24"
DEVICE="cuda"
MAX_BINS=""  # Maximum number of bins to run (empty = all bins)

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
        --max-bins)
            MAX_BINS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Run degree-binned sequential editing experiments"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-dir <dir>           Path to trained model (default: outputs/models/gpt_small_no_alias)"
            echo "  --kg-dir <dir>              Path to knowledge graph data (default: data/kg/ba_no_alias)"
            echo "  --base-output-dir <dir>     Base output directory (default: outputs/degree_binned_no_alias)"
            echo "  --num-steps <n>             Number of edits per bin (default: 1000)"
            echo "  --num-retain-triples <n>    Number of unedited triples (default: 1000)"
            echo "  --max-hop <n>               Maximum hop distance (default: 10)"
            echo "  --layer <n>                 Layer to edit (default: 0)"
            echo "  --v-num-grad-steps <n>      ROME gradient steps (default: 20)"
            echo "  --seed <n>                  Random seed (default: 24)"
            echo "  --device <device>           Device: cuda or cpu (default: cuda)"
            echo "  --max-bins <n>              Maximum number of bins to run (default: all)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "This script divides triples by degree (high to low) into bins of NUM_STEPS size"
            echo "and runs editing experiments for each bin. The remainder (last partial bin) is discarded."
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
echo "  Edits per bin: $NUM_STEPS"
echo "  Retain triples: $NUM_RETAIN_TRIPLES"
echo "  Max hop distance: $MAX_HOP"
echo "  Edit layer: $LAYER"
echo "  ROME v_num_grad_steps: $V_NUM_GRAD_STEPS"
echo "  Random seed: $SEED"
echo "  Device: $DEVICE"
if [ -n "$MAX_BINS" ]; then
    echo "  Max bins: $MAX_BINS"
fi
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
echo "Analyzing KG to determine bin structure..."
echo "========================================"
echo ""

# Get total number of triples and compute number of bins
TOTAL_TRIPLES=$(wc -l < "$KG_DIR/corpus.train.txt")
NUM_BINS=$((TOTAL_TRIPLES / NUM_STEPS))
REMAINDER=$((TOTAL_TRIPLES % NUM_STEPS))

echo "Total triples: $TOTAL_TRIPLES"
echo "Bin size (NUM_STEPS): $NUM_STEPS"
echo "Number of complete bins: $NUM_BINS"
echo "Remainder (will be discarded): $REMAINDER"
echo ""

# Apply max_bins limit if specified
if [ -n "$MAX_BINS" ] && [ "$MAX_BINS" -lt "$NUM_BINS" ]; then
    echo "Limiting to $MAX_BINS bins (out of $NUM_BINS total)"
    NUM_BINS=$MAX_BINS
    echo ""
fi

if [ "$NUM_BINS" -eq 0 ]; then
    echo "Error: Not enough triples to create even one bin of size $NUM_STEPS"
    exit 1
fi

echo "========================================"
echo "Starting $NUM_BINS experiments in parallel..."
echo "========================================"
echo ""

# Define function to run a single bin experiment
run_bin_experiment() {
    local BIN_ID=$1
    local START_IDX=$((BIN_ID * NUM_STEPS))
    local END_IDX=$(((BIN_ID + 1) * NUM_STEPS))
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/bin_${BIN_ID}"

    # Assign GPU: even bins to cuda:0, odd bins to cuda:1
    local GPU_ID=$((BIN_ID % 2))
    local ASSIGNED_DEVICE="cuda:${GPU_ID}"

    echo "[bin_${BIN_ID}] Starting experiment (triples ${START_IDX}-$((END_IDX - 1)), GPU: ${ASSIGNED_DEVICE})..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    python -m src.scripts.run_degree_binned_edits \
        --model-dir "$MODEL_DIR" \
        --kg-dir "$KG_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-steps "$NUM_STEPS" \
        --bin-start-idx "$START_IDX" \
        --bin-end-idx "$END_IDX" \
        --num-retain-triples "$NUM_RETAIN_TRIPLES" \
        --max-hop "$MAX_HOP" \
        --layer "$LAYER" \
        --v-num-grad-steps "$V_NUM_GRAD_STEPS" \
        --seed "$SEED" \
        --device "$ASSIGNED_DEVICE" \
        > "${OUTPUT_DIR}/run.log" 2>&1

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[bin_${BIN_ID}] ✓ Experiment completed successfully"

        # Run analysis
        echo "[bin_${BIN_ID}] Generating visualizations..."
        python -m src.scripts.analyze_sequential_effects \
            --output-dir "$OUTPUT_DIR" \
            >> "${OUTPUT_DIR}/run.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[bin_${BIN_ID}] ✓ Visualization completed"
        else
            echo "[bin_${BIN_ID}] ✗ Visualization failed (check ${OUTPUT_DIR}/run.log)"
        fi
    else
        echo "[bin_${BIN_ID}] ✗ Experiment failed with exit code $EXIT_CODE (check ${OUTPUT_DIR}/run.log)"
    fi

    return $EXIT_CODE
}

# Export function and variables for parallel execution
export -f run_bin_experiment
export MODEL_DIR KG_DIR BASE_OUTPUT_DIR NUM_STEPS NUM_RETAIN_TRIPLES MAX_HOP LAYER V_NUM_GRAD_STEPS SEED DEVICE

# Launch all bin experiments in parallel
echo "Launching parallel experiments..."
echo ""

PIDS=()
for ((BIN_ID=0; BIN_ID<NUM_BINS; BIN_ID++)); do
    run_bin_experiment "$BIN_ID" &
    PID=$!
    PIDS+=($PID)
    echo "  [bin_${BIN_ID}] PID: $PID"
done

echo ""
echo "All experiments launched. Waiting for completion..."
echo "You can monitor progress in real-time:"
for ((BIN_ID=0; BIN_ID<NUM_BINS; BIN_ID++)); do
    echo "  tail -f $BASE_OUTPUT_DIR/bin_${BIN_ID}/run.log"
done
echo ""

# Wait for all background jobs and collect exit codes
EXIT_CODES=()
for ((i=0; i<NUM_BINS; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODES+=($?)
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""

# Print summary
echo "Summary:"
echo "--------"
SUCCESS_COUNT=0
for ((i=0; i<NUM_BINS; i++)); do
    if [ ${EXIT_CODES[$i]} -eq 0 ]; then
        echo "  [bin_${i}] ✓ SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  [bin_${i}] ✗ FAILED"
    fi
done

echo ""

# Generate comparison plot if at least 2 experiments succeeded
if [ $SUCCESS_COUNT -ge 2 ]; then
    echo "========================================"
    echo "Generating degree bin comparison plot..."
    echo "========================================"
    echo ""

    python -m src.scripts.compare_degree_bins --base-dir "$BASE_OUTPUT_DIR" --num-bins "$NUM_BINS"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Comparison plot generated successfully!"
    else
        echo ""
        echo "⚠ Failed to generate comparison plot"
    fi
else
    echo "⚠ Skipping comparison plot (need at least 2 successful experiments)"
fi

echo ""
echo "Results directory: $BASE_OUTPUT_DIR"
echo ""
echo "Generated outputs:"
for ((i=0; i<NUM_BINS; i++)); do
    if [ $i -eq 0 ]; then
        echo "  - ${BASE_OUTPUT_DIR}/bin_${i}/"
        echo "      ├── config.json"
        echo "      ├── stats.jsonl"
        echo "      ├── plots_time_series.png"
        echo "      └── run.log"
    else
        echo "  - ${BASE_OUTPUT_DIR}/bin_${i}/"
    fi
done
echo "  - ${BASE_OUTPUT_DIR}/degree_bins_comparison.png  ← Degree bin comparison plot"
echo ""

# Check if all succeeded
if [ $SUCCESS_COUNT -eq $NUM_BINS ]; then
    echo "✓ All experiments completed successfully!"
    exit 0
else
    echo "⚠ Some experiments failed. Check the logs for details."
    exit 1
fi
