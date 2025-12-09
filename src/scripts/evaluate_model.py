"""Evaluate trained model in detail."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig, SROInference
from src.utils import load_yaml, Logger

def main():
    logger = Logger(verbose=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    logger.info("Loading tokenizer and model...")
    tokenizer = SROTokenizer.load(model_dir / "tokenizer.json")

    # Load config
    train_report = load_yaml(model_dir / "train_report.yaml")
    model_cfg = train_report["config"]["model"]

    # Build model
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        d_mlp=model_cfg["d_mlp"],
        max_seq_len=model_cfg.get("max_seq_len", 8),
        dropout=0.0  # No dropout for inference
    )

    model = GPTMini(gpt_config)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info(f"Model loaded on {device}")
    logger.info(f"Parameters: {model.get_num_params():,}")

    # Create inference engine
    inference = SROInference(model, tokenizer, device)

    # Load triples
    logger.info("Loading test triples...")

    # Evaluate on train set
    train_triples = inference.load_corpus_triples("data/kg/ba/corpus.train.txt")
    logger.info(f"Train triples: {len(train_triples)}")

    train_results = inference.batch_evaluate(train_triples[:1000])  # Sample 1000
    logger.info(f"Train accuracy (sample): {train_results['accuracy']:.4f}")
    logger.info(f"Train mean rank: {train_results['mean_rank']:.2f}")
    logger.info(f"Train median rank: {train_results['median_rank']:.1f}")

    # Evaluate on all aliases (generalization)
    all_triples = inference.load_corpus_triples("data/kg/ba/corpus.all.txt")
    logger.info(f"All triples: {len(all_triples)}")

    all_results = inference.batch_evaluate(all_triples[:5000])  # Sample 5000
    logger.info(f"All accuracy (sample): {all_results['accuracy']:.4f}")
    logger.info(f"All mean rank: {all_results['mean_rank']:.2f}")
    logger.info(f"All median rank: {all_results['median_rank']:.1f}")

    # Sample predictions
    logger.info("\nSample predictions:")
    for i in range(5):
        triple = train_triples[i * 100]
        s, r, o = triple

        result = inference.predict_next(s, r, top_k=5)

        logger.info(f"\nInput: {s} {r}")
        logger.info(f"  Ground truth: {o}")
        logger.info(f"  Top prediction: {result['top_prediction']}")
        logger.info(f"  Top-5: {[t for t, p in result['top_k'][:5]]}")

    # Analyze errors
    logger.info("\nAnalyzing errors...")
    errors = [d for d in train_results['details'] if not d['is_correct']]

    if errors:
        logger.info(f"Error rate: {len(errors)/len(train_results['details']):.2%}")

        # Check if predictions are reasonable
        error_ranks = [e['rank'] for e in errors]
        logger.info(f"Average rank of correct answer (errors): {np.mean(error_ranks):.2f}")
        logger.info(f"Median rank of correct answer (errors): {np.median(error_ranks):.1f}")

if __name__ == "__main__":
    main()
