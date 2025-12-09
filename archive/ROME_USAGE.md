# ROME (Rank-One Model Editing) Usage Guide

This guide explains how to use the ROME implementation for knowledge locating and editing.

## Overview

ROME is a method for locating and editing factual knowledge in neural language models. This implementation consists of two main components:

1. **Causal Tracing (Locating)**: Identifies which layers store specific facts
2. **Knowledge Editing (Editing)**: Modifies the model to change specific facts

## Files

### Core Implementation
- `src/edit/causal_tracing.py` - Causal tracing implementation
- `src/edit/rome.py` - ROME editing implementation

### Scripts
- `run_locating_batch.py` - Run causal tracing on multiple samples
- `run_locating.sh` - Bash wrapper for locating
- `run_editing_single.py` - Run ROME editing on a single fact
- `run_editing.sh` - Bash wrapper for editing

### Visualization
- `visualize_locating.py` - Visualization example for locating
- Automatic visualization in batch scripts

## Usage

### 1. Causal Tracing (Locating)

Run causal tracing on random samples from your knowledge graph:

```bash
# Basic usage (50 random samples)
./run_locating.sh

# Custom number of samples
./run_locating.sh --num-samples 100

# Full options
./run_locating.sh \
    --model-dir outputs/models/gpt_small \
    --kg-file data/kg/ba/graph.jsonl \
    --num-samples 50 \
    --output-dir outputs/locating_results \
    --noise-level 3.0 \
    --num-noise-samples 10 \
    --seed 42
```

**Parameters:**
- `--model-dir`: Path to trained model directory
- `--kg-file`: Path to knowledge graph JSONL file
- `--num-samples`: Number of random triples to analyze
- `--output-dir`: Where to save results
- `--noise-level`: Noise strength (in standard deviations)
- `--num-noise-samples`: Number of noise samples for averaging
- `--seed`: Random seed for reproducibility

**Output:**
- `locating_summary.json` - Summary of all results
- `trace_NNN_*.png` - Heatmap visualization for each sample

### 2. Knowledge Editing (Editing)

Edit a specific fact in the model:

```bash
# Basic usage
./run_editing.sh \
    --subject E_000 \
    --relation R_04 \
    --target E_999

# With all options
./run_editing.sh \
    --model-dir outputs/models/gpt_small \
    --kg-corpus data/kg/ba/corpus.train.txt \
    --subject E_000 \
    --relation R_04 \
    --target E_999 \
    --original E_001 \
    --layer 5 \
    --output-dir outputs/editing_results
```

**Required Parameters:**
- `--subject` or `-s`: Subject entity
- `--relation` or `-r`: Relation
- `--target` or `-o`: Target object (what you want the model to predict)

**Optional Parameters:**
- `--model-dir`: Path to trained model directory
- `--kg-corpus`: Corpus file for computing statistics (improves accuracy)
- `--original`: Original object (for comparison; auto-detected if not provided)
- `--layer`: Layer to edit (auto-located if not provided)
- `--output-dir`: Where to save results

**Output:**
- `edit_result.json` - Summary of the edit
- `locating_result.png` - Causal tracing heatmap
- `editing_result.png` - Before/after comparison

## Python API

You can also use ROME directly in Python:

### Causal Tracing

```python
from src.edit.causal_tracing import CausalTracer

# Initialize tracer
tracer = CausalTracer(model, tokenizer, device="cuda")

# Run causal tracing
result = tracer.trace_important_states(
    s="E_000",
    r="R_04",
    o_target="E_001",
    noise_level=3.0,
    num_samples=10
)

# Find best layer
best_layer = tracer.locate_important_layer(
    s="E_000",
    r="R_04",
    o_target="E_001"
)

print(f"Best layer: {best_layer}")
print(f"Causal effects shape: {result.scores.shape}")
```

### Knowledge Editing

```python
from src.edit.rome import ROME

# Initialize ROME
rome = ROME(
    model,
    tokenizer,
    device="cuda",
    kg_corpus_path="data/kg/ba/corpus.train.txt",
    use_mom2_adjustment=True
)

# Apply edit
edited_model, result = rome.apply_edit(
    s="E_000",
    r="R_04",
    o_target="E_999",
    copy_model=True  # Keep original model unchanged
)

print(f"Success: {result.success}")
print(f"Edited layer: {result.layer}")
print(f"Original prediction: {result.original_prediction}")
print(f"New prediction: {result.new_prediction}")
```

## Examples

### Example 1: Analyze 50 Random Facts

```bash
./run_locating.sh --num-samples 50 --output-dir outputs/analysis_50
```

This will:
1. Sample 50 random (s,r,o) triples from your KG
2. Run causal tracing on each
3. Generate heatmap visualizations
4. Save summary statistics

### Example 2: Edit a Specific Fact

```bash
./run_editing.sh -s E_000 -r R_04 -o E_999
```

This will:
1. Check what the model currently predicts for "E_000 R_04"
2. Locate the best layer to edit
3. Compute the rank-1 update
4. Apply the edit
5. Verify the new prediction
6. Generate visualizations

### Example 3: Edit with Manual Layer Selection

```bash
./run_editing.sh -s E_000 -r R_04 -o E_999 --layer 5
```

This skips automatic layer locating and edits layer 5 directly.

## Understanding the Results

### Locating Heatmap

The heatmap shows the "causal effect" of restoring each (token, layer) state:

- **X-axis**: Layer number
- **Y-axis**: Token position (marked with * for subject tokens)
- **Color**: Indirect effect (red = positive, green = negative)
- **Red vertical line**: Selected best layer

Higher values (more red) indicate that restoring that state helps the model predict the correct answer.

### Editing Result

The editing result shows:

- **Success**: Whether the edit was successful
- **Layer**: Which layer was edited
- **Original prediction**: What the model predicted before editing
- **New prediction**: What the model predicts after editing

## Tips

1. **Use corpus for better results**: Providing `--kg-corpus` improves editing accuracy by computing proper covariance statistics

2. **Adjust noise level**: If causal tracing results look noisy, try adjusting `--noise-level` (default: 3.0)

3. **Layer selection**: Auto-locating usually works well, but you can manually specify `--layer` if you know which layer to edit

4. **Batch processing**: For analyzing many facts, use `run_locating.sh` to process them efficiently

## Troubleshooting

### "Model not found"
Make sure you've trained a model first:
```bash
python -m src.cli.train_lm --config configs/train_gpt_small.yaml
```

### "Corpus file not found"
The corpus is optional but recommended. It should be a text file with one sentence per line. You can use:
```bash
# Use training corpus
--kg-corpus data/kg/ba/corpus.train.txt
```

### CUDA out of memory
Try reducing batch size or use CPU:
- Edit the scripts to use `device="cpu"` in the Python code
- Or reduce `--num-noise-samples` for locating

## References

- ROME Paper: "Locating and Editing Factual Associations in GPT" (https://rome.baulab.info/)
- Original Implementation: https://github.com/kmeng01/rome
