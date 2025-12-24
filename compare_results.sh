#!/bin/bash
# Compare results from different selection modes

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
