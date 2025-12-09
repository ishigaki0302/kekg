#!/bin/bash
# Run ROME Locating (Causal Tracing) on random samples

set -e

echo "========================================"
echo "ROME Locating - Batch Causal Tracing"
echo "========================================"

# Default parameters
MODEL_DIR="outputs/models/gpt_small"
KG_FILE="data/kg/ba/graph.jsonl"
NUM_SAMPLES=50
OUTPUT_DIR="outputs/locating_results"
NOISE_LEVEL=3.0
NUM_NOISE_SAMPLES=10
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --kg-file)
            KG_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --noise-level)
            NOISE_LEVEL="$2"
            shift 2
            ;;
        --num-noise-samples)
            NUM_NOISE_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
echo "  KG file: $KG_FILE"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo "  Noise level: $NOISE_LEVEL"
echo "  Number of noise samples: $NUM_NOISE_SAMPLES"
echo "  Random seed: $SEED"
echo ""

# Check if model exists
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    echo "Error: Model not found at $MODEL_DIR/model.pt"
    echo "Please train a model first or specify correct --model-dir"
    exit 1
fi

# Check if KG file exists
if [ ! -f "$KG_FILE" ]; then
    echo "Error: KG file not found at $KG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run locating
echo "Running causal tracing on $NUM_SAMPLES random samples..."
echo ""

PYTHONPATH=. python src/scripts/run_locating_batch.py \
    --model-dir "$MODEL_DIR" \
    --kg-file "$KG_FILE" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --noise-level "$NOISE_LEVEL" \
    --num-noise-samples "$NUM_NOISE_SAMPLES" \
    --seed "$SEED"

echo ""
echo "========================================"
echo "Locating completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View the results:"
echo "  - Summary: $OUTPUT_DIR/locating_summary.json"
echo "  - Heatmaps: $OUTPUT_DIR/trace_*.png"
