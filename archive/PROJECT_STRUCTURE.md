# Project Structure

## Directory Organization

```
EasyEdit/
├── run_locating.sh              # Run ROME locating (causal tracing)
├── run_editing.sh               # Run ROME editing (knowledge editing)
├── run_ripple_analysis.sh       # Run ripple effect analysis (10 edits)
├── monitor_training.sh          # Monitor training progress
│
├── src/                         # Source code
│   ├── cli/                     # CLI entry points
│   ├── kg/                      # Knowledge graph generation
│   ├── modeling/                # Model definitions
│   ├── edit/                    # Editing methods (ROME, etc.)
│   ├── eval/                    # Evaluation metrics
│   ├── utils/                   # Utilities
│   └── scripts/                 # Execution scripts
│       ├── run_editing_single.py          # Single edit execution
│       ├── run_locating_batch.py          # Batch locating execution
│       ├── analyze_ripple_effects.py      # Ripple effect analysis
│       ├── visualize_attention.py         # Attention visualization
│       ├── visualize_locating.py          # Locating visualization
│       ├── visualize_locating_with_ci.py  # Locating with confidence intervals
│       ├── visualize_training.py          # Training visualization
│       ├── plot_locating_summary.py       # Locating summary plots
│       ├── evaluate_fixed.py              # Model evaluation (fixed)
│       └── evaluate_model.py              # Model evaluation
│
├── tests/                       # Test files
│   ├── test_editing.sh          # Editing functionality test
│   ├── quick_test.py            # Quick functionality test
│   ├── check_edit_success.py    # Edit success checker
│   ├── summarize_ripple.py      # Ripple effect summarizer
│   ├── test_kg_generation.py    # KG generation test
│   ├── test_model.py            # Model test
│   ├── test_tokenizer.py        # Tokenizer test
│   └── run_all_tests.sh         # Run all tests
│
├── configs/                     # Configuration files
│   ├── kg_ba.yaml               # BA knowledge graph config
│   ├── kg_er.yaml               # ER knowledge graph config
│   └── train_gpt_small.yaml     # Training config
│
├── data/                        # Generated data
│   └── kg/                      # Knowledge graphs
│       ├── ba/                  # Barabási-Albert graph
│       └── er/                  # Erdős-Rényi graph
│
├── outputs/                     # Output files
│   ├── models/                  # Trained models
│   ├── locating_results/        # Locating results
│   ├── editing_results/         # Editing results
│   └── ripple_exp_*/            # Ripple experiments
│
├── reports/                     # Report outputs
│
└── archive/                     # Old/deprecated scripts
    ├── example_workflow.sh
    ├── run_editing_batch.sh
    └── run_ripple_batch.sh
```

## Main Execution Scripts (Root Directory)

### 1. Training
```bash
# Train the model (via CLI)
python -m src.cli.train_lm --config configs/train_gpt_small.yaml

# Monitor training progress
./monitor_training.sh
```

### 2. Locating (Causal Tracing)
```bash
# Run locating on 50 random samples
./run_locating.sh

# Custom parameters
./run_locating.sh \
  --model-dir outputs/models/gpt_small \
  --kg-file data/kg/ba/graph.jsonl \
  --num-samples 100 \
  --output-dir outputs/locating_results
```

### 3. Editing (Knowledge Editing)
```bash
# Single edit
./run_editing.sh \
  --subject E_000 \
  --relation R_04 \
  --target E_007 \
  --layer 0

# With ripple analysis
./run_editing.sh \
  --subject E_000 \
  --relation R_04 \
  --target E_007 \
  --layer 0 \
  --analyze-ripple
```

### 4. Ripple Effect Analysis
```bash
# Run 10 edits with ripple analysis and visualization
./run_ripple_analysis.sh
```

## Python Execution Scripts (src/scripts/)

These scripts are called by the shell scripts above and can also be used directly:

```bash
# Run single edit (must set PYTHONPATH)
PYTHONPATH=. python src/scripts/run_editing_single.py \
  --model-dir outputs/models/gpt_small \
  --kg-corpus data/kg/ba/corpus.base.txt \
  --subject E_000 \
  --relation R_04 \
  --target E_007 \
  --layer 0 \
  --output-dir outputs/my_edit

# Run locating batch
PYTHONPATH=. python src/scripts/run_locating_batch.py \
  --model-dir outputs/models/gpt_small \
  --kg-file data/kg/ba/graph.jsonl \
  --num-samples 50 \
  --output-dir outputs/my_locating

# Analyze ripple effects
PYTHONPATH=. python src/scripts/analyze_ripple_effects.py
```

**Note**: `PYTHONPATH=.` is required when running Python scripts directly to ensure proper imports.

## Test Scripts (tests/)

```bash
# Run editing test (from root directory)
./tests/test_editing.sh

# Run all tests
./tests/run_all_tests.sh
```

## Key Features

### ✅ Clean Separation
- **Root**: Main execution shell scripts
- **src/scripts/**: Python execution scripts
- **tests/**: Test scripts and utilities
- **archive/**: Deprecated scripts

### ✅ Easy Execution
All main operations can be executed via simple shell scripts:
- `./run_locating.sh` - Causal tracing
- `./run_editing.sh` - Knowledge editing
- `./run_ripple_analysis.sh` - Ripple analysis

### ✅ Proper Import Handling
All shell scripts automatically set `PYTHONPATH=.` to ensure proper Python imports.

## Migration Notes

Files have been reorganized as follows:

**Moved to src/scripts/**:
- run_editing_single.py
- run_locating_batch.py
- analyze_ripple_effects.py
- visualize_*.py
- plot_*.py
- evaluate_*.py

**Moved to tests/**:
- test_editing.sh
- quick_test.py
- check_edit_success.py
- summarize_ripple.py

**Moved to archive/**:
- example_workflow.sh
- run_editing_batch.sh
- run_ripple_batch.sh

All shell scripts have been updated to use the new paths with proper `PYTHONPATH` settings.
