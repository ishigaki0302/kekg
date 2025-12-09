"""CLI script for training language model."""

import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_yaml, save_yaml, set_seed, Logger, MetricsTracker
from src.modeling import (
    SROTokenizer, GPTMini, GPTConfig,
    Trainer, TrainConfig, SRODataset
)


def main():
    parser = argparse.ArgumentParser(description="Train SRO language model")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    # Load config
    config = load_yaml(args.config)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Output directory
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = Logger(
        log_file=output_dir / "train.log",
        verbose=True
    )

    logger.info("Starting training pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = SROTokenizer()

    data_config = config["data"]
    train_path = data_config["train_path"]
    eval_path = data_config.get("eval_all_path")

    # Build vocab from corpus
    corpus_files = [train_path]
    if eval_path:
        corpus_files.append(eval_path)

    tokenizer.build_vocab_from_corpus(corpus_files)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = SRODataset(train_path, tokenizer)
    eval_dataset = SRODataset(eval_path, tokenizer) if eval_path else None

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    # Build model
    logger.info("Building model...")
    model_config = config["model"]

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=model_config.get("n_layers", 12),
        n_heads=model_config.get("n_heads", 8),
        d_model=model_config.get("d_model", 512),
        d_mlp=model_config.get("d_mlp", 2048),
        max_seq_len=model_config.get("max_seq_len", 8),
        dropout=model_config.get("dropout", 0.1)
    )

    model = GPTMini(gpt_config)
    logger.info(f"Model parameters: {model.get_num_params():,}")

    # Training config
    train_config_dict = config["train"]
    train_config = TrainConfig(
        batch_size=train_config_dict.get("batch_size", 128),
        lr=train_config_dict.get("lr", 3e-4),
        weight_decay=train_config_dict.get("weight_decay", 0.01),
        epochs=train_config_dict.get("epochs", 30),
        warmup_steps=train_config_dict.get("warmup_steps", 1000),
        grad_clip=train_config_dict.get("grad_clip", 1.0),
        eval_interval=train_config_dict.get("eval_interval", 500),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Training on: {train_config.device}")

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=train_config,
        logger=logger
    )

    # Train
    logger.info("Starting training...")
    report = trainer.train()

    logger.info("Training complete!")
    logger.info(f"Final train loss: {report.final_train_loss:.4f}")
    logger.info(f"Final train acc: {report.final_train_acc:.4f}")
    logger.info(f"Final eval acc: {report.final_eval_acc:.4f}")
    logger.info(f"Best eval acc: {report.best_eval_acc:.4f}")

    # Save final model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # Save training report
    report_data = {
        "final_train_loss": report.final_train_loss,
        "final_train_acc": report.final_train_acc,
        "final_eval_acc": report.final_eval_acc,
        "best_eval_acc": report.best_eval_acc,
        "total_steps": report.total_steps,
        "config": config
    }
    save_yaml(report_data, output_dir / "train_report.yaml")

    # Save metrics
    trainer.metrics.save_json(output_dir / "metrics.json")
    trainer.metrics.save_csv(output_dir / "metrics.csv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
