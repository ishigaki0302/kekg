#!/bin/bash

# Knowledge Graph Visualization Script
# Generates comprehensive visualizations and statistics for graph.jsonl

set -e  # Exit on error

# Configuration
GRAPH_PATH="data/kg/ba_no_alias/graph.jsonl"
OUTPUT_DIR="outputs/kg_visualizations"
SEED=42

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Knowledge Graph Visualization${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if graph file exists
if [ ! -f "$GRAPH_PATH" ]; then
    echo "Error: Graph file not found at $GRAPH_PATH"
    exit 1
fi

echo -e "\n${GREEN}Graph file:${NC} $GRAPH_PATH"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"

# Run visualization script
python -m src.visualize_kg \
    --graph_path "$GRAPH_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Visualization complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\nGenerated files:"
echo -e "  - ${OUTPUT_DIR}/kg_overview.png"
echo -e "  - ${OUTPUT_DIR}/degree_distributions.png"
echo -e "  - ${OUTPUT_DIR}/hop_distributions.png"
echo -e "  - ${OUTPUT_DIR}/kg_stats.json"
echo ""
