#!/usr/bin/env python
"""Quick test of ROME editing"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.edit.rome import ROME
from src.utils import load_yaml, Logger

logger = Logger(verbose=True)

# Load model
model_dir = Path("outputs/models/gpt_small")
tokenizer = SROTokenizer.load(model_dir / "tokenizer.json")
train_report = load_yaml(model_dir / "train_report.yaml")
model_cfg = train_report["config"]["model"]

gpt_config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_layers=model_cfg["n_layers"],
    n_heads=model_cfg["n_heads"],
    d_model=model_cfg["d_model"],
    d_mlp=model_cfg["d_mlp"],
    max_seq_len=model_cfg.get("max_seq_len", 8),
    dropout=0.0
)

model = GPTMini(gpt_config)
model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

logger.info(f"Model loaded on {device}")

# Initialize ROME
rome = ROME(
    model,
    tokenizer,
    device=device,
    kg_corpus_path="data/kg/ba/corpus.base.txt",
    mom2_n_samples=1000,
    use_mom2_adjustment=True
)

# Test edit at layer 0
logger.info("\nTest 1: Edit at layer 0")
edited_model, result = rome.apply_edit(
    s="E_000",
    r="R_04",
    o_target="E_999",
    layer=0,
    copy_model=True
)

logger.info(f"Original: {result.original_prediction}")
logger.info(f"New: {result.new_prediction}")
logger.info(f"Target: E_999")
logger.info(f"Success: {result.success}")
logger.info(f"Prediction changed: {result.original_prediction != result.new_prediction}")

# Test edit with auto-detection
logger.info("\nTest 2: Edit with auto-detection")
edited_model2, result2 = rome.apply_edit(
    s="E_005",
    r="R_28",
    o_target="E_100",
    layer=None,  # Auto-detect
    copy_model=True
)

logger.info(f"Selected layer: {result2.layer}")
logger.info(f"Original: {result2.original_prediction}")
logger.info(f"New: {result2.new_prediction}")
logger.info(f"Target: E_100")
logger.info(f"Success: {result2.success}")
logger.info(f"Prediction changed: {result2.original_prediction != result2.new_prediction}")
