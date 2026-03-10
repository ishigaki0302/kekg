#!/bin/bash
# =============================================================================
# Script   : run_hop_analysis.sh
# Category : 知識編集 - ホップ距離分析
# 概要     : degree_high / degree_low の多試行実験結果に対して
#             ホップ距離別（hop 0,1,2,3,...）の精度推移を分析・可視化する
#             run_degree_exclusive_multi_trial.sh の実行後に単体で再実行できる
# 出力先   : {base-dir}/hop_distance_analysis_multi_trial.png
# 使用方法 :
#   ./run_hop_analysis.sh
#   ./run_hop_analysis.sh --base-dir outputs/degree_exclusive_multi_trial \
#                          --num-trials 10
# 前提     : run_degree_exclusive_multi_trial.sh を先に実行しておくこと
# =============================================================================

set -e

echo "========================================"
echo "Hop Distance Analysis"
echo "========================================"
echo ""

# Default parameters
BASE_DIR="outputs/degree_exclusive_multi_trial"
NUM_TRIALS="10"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Analyze results by hop distance"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --base-dir <dir>     Base output directory (default: outputs/degree_exclusive_multi_trial)"
            echo "  --num-trials <n>     Number of trials (default: 10)"
            echo "  --help, -h           Show this help message"
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
echo "  Base directory: $BASE_DIR"
echo "  Number of trials: $NUM_TRIALS"
echo ""

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory not found: $BASE_DIR"
    exit 1
fi

# Run multi-trial analysis
echo "Running multi-trial hop distance analysis..."
python -m src.scripts.analyze_hop_distance_effects \
    --base-dir "$BASE_DIR" \
    --num-trials "$NUM_TRIALS" \
    --output-dir "$BASE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Hop distance analysis completed successfully!"
    echo ""
    echo "Generated output:"
    echo "  - ${BASE_DIR}/hop_distance_analysis_multi_trial.png"
    echo ""
else
    echo ""
    echo "✗ Hop distance analysis failed"
    exit 1
fi
