#!/bin/bash
# Example workflow for SRO Knowledge Editing Platform

set -e

echo "=== SRO Knowledge Editing Platform: Example Workflow ==="

# Step 1: Generate BA knowledge graph
echo ""
echo "Step 1: Generating BA knowledge graph..."
python -m src.cli.build_kg --config configs/kg_ba.yaml

# Step 2: Generate ER knowledge graph (for comparison)
echo ""
echo "Step 2: Generating ER knowledge graph..."
python -m src.cli.build_kg --config configs/kg_er.yaml

# Step 3: Train GPT mini on BA graph
echo ""
echo "Step 3: Training GPT mini model..."
python -m src.cli.train_lm --config configs/train_gpt_small.yaml

# Step 4: Run CKE experiments
echo ""
echo "Step 4: Running CKE experiments..."
python -m src.cli.run_seqedit \
  --config configs/cke_pipeline.yaml \
  --model-dir outputs/models/gpt_small \
  --kg-dir data/kg/ba \
  --num-scenarios 5 \
  --steps 5

echo ""
echo "=== Workflow complete! ==="
echo "Results are in:"
echo "  - outputs/models/gpt_small/    (trained model)"
echo "  - outputs/cke/                 (CKE results)"
echo "  - outputs/cke/figures/         (visualizations)"
