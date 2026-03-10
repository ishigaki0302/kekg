#!/bin/bash
# =============================================================================
# Script   : run_training.sh
# Category : 学習
# 概要     : GPTMini モデルを SROコーパスで学習する
# 出力先   : outputs/models/gpt_small/
#             (model.pt, tokenizer.json, metrics.json, train.log 等)
# 使用方法 :
#   ./run_training.sh
#   ./run_training.sh --config configs/train_gpt_small.yaml
#   ./run_training.sh --output-dir outputs/models/my_model
# =============================================================================

set -e

echo "========================================"
echo "GPT Model Training"
echo "========================================"

# Default parameters
CONFIG="configs/train_gpt_small.yaml"
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output directory: $OUTPUT_DIR"
fi
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found at $CONFIG"
    exit 1
fi

# Run training
echo "Starting training..."
echo ""

if [ -n "$OUTPUT_DIR" ]; then
    python -m src.cli.train_lm --config "$CONFIG" --output-dir "$OUTPUT_DIR"
else
    python -m src.cli.train_lm --config "$CONFIG"
fi

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Check results in outputs/models/"
