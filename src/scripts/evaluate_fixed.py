"""Evaluate with fixed entity-level metrics."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.modeling.inference_fixed import SROInferenceFixed
from src.utils import load_yaml, Logger

def main():
    logger = Logger(verbose=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    logger.info("Loading tokenizer and model...")
    tokenizer = SROTokenizer.load(model_dir / "tokenizer.json")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Check that canonical IDs are in vocab
    canonical_entities = [t for t in tokenizer.vocab.keys() if t.startswith('E_') and '__a' not in t]
    logger.info(f"Canonical entities in vocab: {len(canonical_entities)}")

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
        dropout=0.0
    )

    model = GPTMini(gpt_config)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info(f"Model loaded on {device}")

    # Create FIXED inference engine (entity-level)
    inference = SROInferenceFixed(model, tokenizer, device)
    logger.info(f"Built alias map for {len(inference.entity_to_aliases)} entities")

    # Load triples
    logger.info("\nLoading test triples...")

    # Train set
    train_triples = inference.load_corpus_triples("data/kg/ba/corpus.train.txt")
    logger.info(f"Train triples: {len(train_triples)}")

    logger.info("Evaluating on train set (sample 2000)...")
    train_results = inference.batch_evaluate(train_triples[:2000])
    logger.info(f"  Accuracy: {train_results['accuracy']:.4f}")
    logger.info(f"  MRR: {train_results['mrr']:.4f}")
    logger.info(f"  Mean rank: {train_results['mean_rank']:.2f}")
    logger.info(f"  Median rank: {train_results['median_rank']:.1f}")

    # All aliases
    all_triples = inference.load_corpus_triples("data/kg/ba/corpus.all.txt")
    logger.info(f"\nAll triples: {len(all_triples)}")

    logger.info("Evaluating on all aliases (sample 5000)...")
    all_results = inference.batch_evaluate(all_triples[:5000])
    logger.info(f"  Accuracy: {all_results['accuracy']:.4f}")
    logger.info(f"  MRR: {all_results['mrr']:.4f}")
    logger.info(f"  Mean rank: {all_results['mean_rank']:.2f}")
    logger.info(f"  Median rank: {all_results['median_rank']:.1f}")

    # Sample predictions
    logger.info("\nSample predictions (entity-level):")
    for i in range(5):
        triple = train_triples[i * 500]
        s, r, o = triple

        result = inference.predict_next(s, r, top_k=5)

        logger.info(f"\nInput: {s} {r}")
        logger.info(f"  Ground truth: {o}")
        logger.info(f"  Top entity: {result['top_entity']}")
        logger.info(f"  Top-5 entities: {[e for e, p in result['top_k_entities'][:5]]}")
        logger.info(f"  Correct: {result['top_entity'] == inference._get_canonical_entity(o)}")

    # Random baseline
    num_entities = len(inference.entity_to_aliases)
    random_baseline = 1.0 / num_entities
    logger.info(f"\nRandom baseline accuracy: {random_baseline:.4f} (1/{num_entities})")
    logger.info(f"Model improvement: {train_results['accuracy'] / random_baseline:.2f}x")

if __name__ == "__main__":
    main()
