#!/bin/bash
# =============================================================================
# Script   : compare_results.sh
# Category : 知識編集 - 比較実験 結果プロット
# 概要     : run_all_selection_modes.sh 実行済みの結果から
#             選択モード間の比較プロット（comparison.png）を生成する
#             実験を再実行せずにプロットだけ生成し直したい場合に使う
# 出力先   : {base-dir}/comparison.png
# 使用方法 :
#   ./compare_results.sh
#   ./compare_results.sh --base-dir outputs/sequential_comparison_no_alias
# 前提     : run_all_selection_modes.sh を先に実行しておくこと
# =============================================================================

set -e

echo "========================================"
echo "Selection Mode Comparison"
echo "========================================"

# Default parameters
BASE_DIR="outputs/sequential_comparison_no_alias"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Compare sequential editing results across selection modes"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --base-dir <dir>    Base directory with mode subdirectories"
            echo "                      (default: outputs/sequential_comparison)"
            echo "  --help, -h          Show this help message"
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

echo "Base directory: $BASE_DIR"
echo ""

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory not found: $BASE_DIR"
    echo "Please run experiments first: ./run_all_selection_modes.sh"
    exit 1
fi

# Run comparison analysis
python -m src.scripts.compare_selection_modes --base-dir "$BASE_DIR"

echo ""
echo "========================================"
echo "Comparison complete!"
echo "========================================"
echo ""
echo "Generated plot: $BASE_DIR/comparison.png"
echo ""
