#!/bin/bash
# =============================================================================
# Script   : run_subject_repetition_multi_trial.sh
# Category : 知識編集 - 多試行比較実験（並列）
# 概要     : 同一 subject への編集繰り返し回数（1回/2回/4回）を変えた3モードを
#             N 試行ずつ並列実行し，繰り返し編集の影響を多試行で比較する
#             モード: no_rep (×1), rep_2 (×2), rep_4 (×4)
# 出力先   : outputs/subject_rep_multi_trial/
#             no_rep/trial_0/ ... rep_4/trial_N/
#             multi_trial_comparison.png
# 使用方法 :
#   ./run_subject_repetition_multi_trial.sh
#   ./run_subject_repetition_multi_trial.sh --num-trials 10 --num-steps 100
# =============================================================================

set -e

echo "========================================"
echo "Sequential Editing - Subject Repetition Multi-Trial Analysis"
echo "========================================"
echo ""

# Default parameters
MODEL_DIR="outputs/models/gpt_small"
KG_DIR="data/kg/ba"
BASE_OUTPUT_DIR="outputs/subject_rep_multi_trial"
NUM_STEPS="100"
NUM_TRIALS="10"  # Number of trials for each mode
NUM_RETAIN_TRIPLES="1000"
MAX_HOP="10"
LAYER="0"
V_NUM_GRAD_STEPS="20"
BASE_SEED="24"
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
        --num-trials)
            NUM_TRIALS="$2"
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
        --base-seed)
            BASE_SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Run multiple trials of subject repetition experiments"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-dir <dir>           Path to trained model (default: outputs/models/gpt_small)"
            echo "  --kg-dir <dir>              Path to knowledge graph data (default: data/kg/ba)"
            echo "  --base-output-dir <dir>     Base output directory (default: outputs/subject_rep_multi_trial)"
            echo "  --num-steps <n>             Number of sequential edits (default: 30)"
            echo "  --num-trials <n>            Number of trials per mode (default: 10)"
            echo "  --num-retain-triples <n>    Number of unedited triples (default: 1000)"
            echo "  --max-hop <n>               Maximum hop distance (default: 10)"
            echo "  --layer <n>                 Layer to edit (default: 0)"
            echo "  --v-num-grad-steps <n>      ROME gradient steps (default: 20)"
            echo "  --base-seed <n>             Base random seed (default: 24)"
            echo "  --device <device>           Device: cuda or cpu (default: cuda)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "This script runs NUM_TRIALS experiments for each mode:"
            echo "  - no_rep (repetition_count=1): Each subject selected once"
            echo "  - rep_2 (repetition_count=2): Each subject selected 2 times"
            echo "  - rep_4 (repetition_count=4): Each subject selected 4 times"
            echo ""
            echo "Total experiments: NUM_TRIALS * 3"
            echo ""
            echo "Visualization will show:"
            echo "  - Individual trial lines (thin, semi-transparent)"
            echo "  - Mean line (thick)"
            echo "  - 95% confidence interval (shaded area)"
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
echo "  Number of trials per mode: $NUM_TRIALS"
echo "  Retain triples: $NUM_RETAIN_TRIPLES"
echo "  Max hop distance: $MAX_HOP"
echo "  Edit layer: $LAYER"
echo "  ROME v_num_grad_steps: $V_NUM_GRAD_STEPS"
echo "  Base random seed: $BASE_SEED"
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

TOTAL_EXPERIMENTS=$((NUM_TRIALS * 3))
echo "========================================"
echo "Starting $TOTAL_EXPERIMENTS experiments in parallel..."
echo "  no_rep (rep=1):  $NUM_TRIALS trials"
echo "  rep_2  (rep=2):  $NUM_TRIALS trials"
echo "  rep_4  (rep=4):  $NUM_TRIALS trials"
echo "========================================"
echo ""

# Define function to run a single trial
run_trial() {
    local MODE=$1
    local REP_COUNT=$2
    local TRIAL_ID=$3
    local TRIAL_SEED=$((BASE_SEED + TRIAL_ID * 1000))
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODE}/trial_${TRIAL_ID}"

    # Assign GPU: even trials to cuda:0, odd trials to cuda:1
    local GPU_ID=$((TRIAL_ID % 2))
    local ASSIGNED_DEVICE="cuda:${GPU_ID}"

    echo "[${MODE}_trial_${TRIAL_ID}] Starting experiment (seed=$TRIAL_SEED, rep=$REP_COUNT, GPU: ${ASSIGNED_DEVICE})..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    python -m src.scripts.run_subject_repetition_edits \
        --model-dir "$MODEL_DIR" \
        --kg-dir "$KG_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-steps "$NUM_STEPS" \
        --repetition-count "$REP_COUNT" \
        --num-retain-triples "$NUM_RETAIN_TRIPLES" \
        --max-hop "$MAX_HOP" \
        --layer "$LAYER" \
        --v-num-grad-steps "$V_NUM_GRAD_STEPS" \
        --seed "$TRIAL_SEED" \
        --device "$ASSIGNED_DEVICE" \
        > "${OUTPUT_DIR}/run.log" 2>&1

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[${MODE}_trial_${TRIAL_ID}] ✓ Experiment completed successfully"

        # Run analysis
        echo "[${MODE}_trial_${TRIAL_ID}] Generating visualizations..."
        python -m src.scripts.analyze_sequential_effects \
            --output-dir "$OUTPUT_DIR" \
            >> "${OUTPUT_DIR}/run.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[${MODE}_trial_${TRIAL_ID}] ✓ Visualization completed"
        else
            echo "[${MODE}_trial_${TRIAL_ID}] ✗ Visualization failed (check ${OUTPUT_DIR}/run.log)"
        fi
    else
        echo "[${MODE}_trial_${TRIAL_ID}] ✗ Experiment failed with exit code $EXIT_CODE (check ${OUTPUT_DIR}/run.log)"
    fi

    return $EXIT_CODE
}

# Export function and variables for parallel execution
export -f run_trial
export MODEL_DIR KG_DIR BASE_OUTPUT_DIR NUM_STEPS NUM_RETAIN_TRIPLES MAX_HOP LAYER V_NUM_GRAD_STEPS BASE_SEED DEVICE

# Launch all trials in parallel
echo "Launching parallel experiments..."
echo ""

PIDS=()
EXIT_CODES_ARRAY=()

# Launch no_rep (repetition_count=1) trials
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    run_trial "no_rep" 1 "$TRIAL_ID" &
    PID=$!
    PIDS+=($PID)
    echo "  [no_rep_trial_${TRIAL_ID}] PID: $PID"
done

# Launch rep_2 (repetition_count=2) trials
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    run_trial "rep_2" 2 "$TRIAL_ID" &
    PID=$!
    PIDS+=($PID)
    echo "  [rep_2_trial_${TRIAL_ID}]  PID: $PID"
done

# Launch rep_4 (repetition_count=4) trials
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    run_trial "rep_4" 4 "$TRIAL_ID" &
    PID=$!
    PIDS+=($PID)
    echo "  [rep_4_trial_${TRIAL_ID}]  PID: $PID"
done

echo ""
echo "All experiments launched. Waiting for completion..."
echo "You can monitor progress in real-time:"
echo "  tail -f $BASE_OUTPUT_DIR/no_rep/trial_0/run.log"
echo "  tail -f $BASE_OUTPUT_DIR/rep_2/trial_0/run.log"
echo "  tail -f $BASE_OUTPUT_DIR/rep_4/trial_0/run.log"
echo ""

# Wait for all background jobs
for ((i=0; i<TOTAL_EXPERIMENTS; i++)); do
    wait ${PIDS[$i]}
    EXIT_CODES_ARRAY+=($?)
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""

# Print summary
echo "Summary:"
echo "--------"

SUCCESS_COUNT_NO_REP=0
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    IDX=$TRIAL_ID
    if [ ${EXIT_CODES_ARRAY[$IDX]} -eq 0 ]; then
        echo "  [no_rep_trial_${TRIAL_ID}] ✓ SUCCESS"
        SUCCESS_COUNT_NO_REP=$((SUCCESS_COUNT_NO_REP + 1))
    else
        echo "  [no_rep_trial_${TRIAL_ID}] ✗ FAILED"
    fi
done

SUCCESS_COUNT_REP_2=0
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    IDX=$((NUM_TRIALS + TRIAL_ID))
    if [ ${EXIT_CODES_ARRAY[$IDX]} -eq 0 ]; then
        echo "  [rep_2_trial_${TRIAL_ID}]  ✓ SUCCESS"
        SUCCESS_COUNT_REP_2=$((SUCCESS_COUNT_REP_2 + 1))
    else
        echo "  [rep_2_trial_${TRIAL_ID}]  ✗ FAILED"
    fi
done

SUCCESS_COUNT_REP_4=0
for ((TRIAL_ID=0; TRIAL_ID<NUM_TRIALS; TRIAL_ID++)); do
    IDX=$((2 * NUM_TRIALS + TRIAL_ID))
    if [ ${EXIT_CODES_ARRAY[$IDX]} -eq 0 ]; then
        echo "  [rep_4_trial_${TRIAL_ID}]  ✓ SUCCESS"
        SUCCESS_COUNT_REP_4=$((SUCCESS_COUNT_REP_4 + 1))
    else
        echo "  [rep_4_trial_${TRIAL_ID}]  ✗ FAILED"
    fi
done

echo ""
echo "Success rates:"
echo "  no_rep: $SUCCESS_COUNT_NO_REP / $NUM_TRIALS"
echo "  rep_2:  $SUCCESS_COUNT_REP_2 / $NUM_TRIALS"
echo "  rep_4:  $SUCCESS_COUNT_REP_4 / $NUM_TRIALS"
echo ""

# Generate multi-trial comparison plot if at least 2 trials succeeded for each mode
if [ $SUCCESS_COUNT_NO_REP -ge 2 ] && [ $SUCCESS_COUNT_REP_2 -ge 2 ] && [ $SUCCESS_COUNT_REP_4 -ge 2 ]; then
    echo "========================================"
    echo "Generating multi-trial comparison plot..."
    echo "========================================"
    echo ""

    python -m src.scripts.compare_subject_repetition_multi_trial \
        --base-dir "$BASE_OUTPUT_DIR" \
        --num-trials "$NUM_TRIALS"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Multi-trial comparison plot generated successfully!"
    else
        echo ""
        echo "⚠ Failed to generate multi-trial comparison plot"
    fi
else
    echo "⚠ Skipping multi-trial comparison plot"
    echo "  (need at least 2 successful trials for each mode)"
fi

echo ""
echo "Results directory: $BASE_OUTPUT_DIR"
echo ""
echo "Generated outputs:"
echo "  - ${BASE_OUTPUT_DIR}/no_rep/trial_0/"
echo "  - ${BASE_OUTPUT_DIR}/no_rep/trial_1/"
echo "  - ..."
echo "  - ${BASE_OUTPUT_DIR}/rep_2/trial_0/"
echo "  - ${BASE_OUTPUT_DIR}/rep_2/trial_1/"
echo "  - ..."
echo "  - ${BASE_OUTPUT_DIR}/rep_4/trial_0/"
echo "  - ${BASE_OUTPUT_DIR}/rep_4/trial_1/"
echo "  - ..."
echo "  - ${BASE_OUTPUT_DIR}/multi_trial_comparison.png  ← 多試行比較プロット"
echo ""

# Check if all succeeded
TOTAL_SUCCESS=$((SUCCESS_COUNT_NO_REP + SUCCESS_COUNT_REP_2 + SUCCESS_COUNT_REP_4))
if [ $TOTAL_SUCCESS -eq $TOTAL_EXPERIMENTS ]; then
    echo "✓ All experiments completed successfully!"
    exit 0
else
    echo "⚠ Some experiments failed. Check the logs for details."
    exit 1
fi
