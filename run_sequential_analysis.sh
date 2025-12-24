#!/bin/bash
# Analyze and visualize sequential editing results

set -e

echo "========================================"
echo "Sequential Editing Analysis"
echo "========================================"

# Default parameters
OUTPUT_DIR="outputs/sequential"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Sequential Editing Analysis - Visualize sequential editing results"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --output-dir <dir>     Directory with sequential editing results"
            echo "                         (default: outputs/sequential)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --output-dir outputs/my_experiment"
            echo ""
            echo "This script generates:"
            echo "  - Time series plots (edit success, retention, KG accuracy, weight distance)"
            echo "  - Hop distance heatmaps (subject, original object, new object)"
            echo "  - Entity degree heatmap"
            echo "  - Failure distribution histogram"
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
echo "  Results directory: $OUTPUT_DIR"
echo ""

# Check if results exist
if [ ! -f "$OUTPUT_DIR/stats.jsonl" ]; then
    echo "Error: Results not found at $OUTPUT_DIR/stats.jsonl"
    echo ""
    echo "Please run sequential editing first:"
    echo "  ./run_sequential_edits.sh"
    echo ""
    echo "Or specify correct --output-dir with existing results"
    exit 1
fi

# Check for required files
MISSING_FILES=""
for FILE in stats.jsonl ripple_triples.jsonl triple_acc.jsonl; do
    if [ ! -f "$OUTPUT_DIR/$FILE" ]; then
        MISSING_FILES="$MISSING_FILES $FILE"
    fi
done

if [ -n "$MISSING_FILES" ]; then
    echo "Warning: Missing result files:$MISSING_FILES"
    echo "Analysis may be incomplete"
    echo ""
fi

# Run analysis
echo "Generating visualizations..."
echo ""

python -m src.scripts.analyze_sequential_effects --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Analysis completed!"
echo "========================================"
echo ""
echo "Generated plots:"
echo "  - Time series: $OUTPUT_DIR/plots_time_series.png"
echo "  - Hop from subject: $OUTPUT_DIR/plots_hop_degree_hop_subj.png"
echo "  - Hop from original object: $OUTPUT_DIR/plots_hop_degree_hop_before.png"
echo "  - Hop from new object: $OUTPUT_DIR/plots_hop_degree_hop_after.png"
echo "  - Entity degree: $OUTPUT_DIR/plots_hop_degree_degree.png"
echo "  - Failure histogram: $OUTPUT_DIR/plots_failure_hist.png"
echo ""
echo "View the plots with your image viewer or transfer them to view locally"
echo ""
