"""Training pipeline for SRO knowledge learning."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import math

from .gpt_mini import GPTMini, GPTConfig
from .tokenizer import SROTokenizer
from ..utils import Logger, MetricsTracker


class SRODataset(Dataset):
    """Dataset for SRO triples."""

    def __init__(self, corpus_path: Union[str, Path], tokenizer: SROTokenizer):
        """
        Initialize dataset.

        Args:
            corpus_path: Path to corpus file (space-separated SRO per line)
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
        self.samples = []

        # Load corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line)
                    if len(tokens) == 3:  # Ensure SRO format
                        self.samples.append(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        tokens = self.samples[idx]

        # For next-token prediction: input = [s, r], target = [r, o]
        # But we use teacher forcing on full sequence
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens, dtype=torch.long)

        # Create labels_mask: only o position (index 2) should be used for loss
        # For [s, r, o], we only want to compute loss on predicting o
        # After shift, this will mask out s->r prediction, keeping only r->o
        labels_mask = torch.tensor([False, False, True], dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "labels_mask": labels_mask
        }


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    eval_interval: int = 500
    save_interval: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainReport:
    """Training report."""
    final_train_loss: float
    final_train_acc: float
    final_eval_acc: float
    best_eval_acc: float
    total_steps: int
    config: Dict[str, Any]


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, labels_mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute next-token prediction accuracy.

    For SRO sequences, we care about predicting token at position t+1 from t.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        labels_mask: Optional [batch, seq_len] mask for which positions to evaluate

    Returns:
        Accuracy as float
    """
    # Shift logits and labels for next-token prediction
    # logits: [:, :-1, :] (predict next)
    # labels: [:, 1:] (targets)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Predictions
    preds = shift_logits.argmax(dim=-1)

    # Accuracy
    if labels_mask is not None:
        shift_mask = labels_mask[:, 1:].contiguous()
        correct = ((preds == shift_labels) * shift_mask).float().sum()
        total = shift_mask.sum()
    else:
        correct = (preds == shift_labels).float().sum()
        total = shift_labels.numel()

    return (correct / total).item()


class Trainer:
    """Trainer for GPT mini model."""

    def __init__(
        self,
        model: GPTMini,
        tokenizer: SROTokenizer,
        train_dataset: SRODataset,
        eval_dataset: Optional[SRODataset] = None,
        config: TrainConfig = TrainConfig(),
        logger: Optional[Logger] = None
    ):
        """
        Initialize trainer.

        Args:
            model: GPT model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            config: Training configuration
            logger: Optional logger
        """
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.logger = logger or Logger()
        self.metrics = MetricsTracker()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs * len(train_dataset) // config.batch_size
        )

        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )

        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0
            )

        self.global_step = 0
        self.best_eval_acc = 0.0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        labels_mask = batch["labels_mask"].to(self.config.device)

        # Forward
        outputs = self.model(input_ids)
        logits = outputs["logits"]

        # Loss: cross-entropy on next-token prediction
        # Shift for autoregressive loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = labels_mask[:, 1:].contiguous()

        # Apply mask: only compute loss where mask is True (o positions)
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        loss = loss.view_as(shift_labels)
        # Masked average: only compute loss on o positions
        loss = (loss * shift_mask).sum() / shift_mask.sum()

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        # Compute accuracy
        acc = compute_accuracy(logits.detach(), labels, labels_mask)

        return {
            "loss": loss.item(),
            "acc": acc,
            "lr": self.scheduler.get_last_lr()[0]
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval dataset."""
        if not self.eval_dataset:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            labels_mask = batch["labels_mask"].to(self.config.device)

            outputs = self.model(input_ids)
            logits = outputs["logits"]

            # Loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = labels_mask[:, 1:].contiguous()

            # Apply mask: only compute loss where mask is True (o positions)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss = loss.view_as(shift_labels)
            loss = (loss * shift_mask).sum() / shift_mask.sum()

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels, labels_mask)
            num_batches += 1

        return {
            "eval_loss": total_loss / num_batches,
            "eval_acc": total_acc / num_batches
        }

    def train(self) -> TrainReport:
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)

                epoch_loss += metrics["loss"]
                epoch_acc += metrics["acc"]
                num_batches += 1
                self.global_step += 1

                # Log metrics
                if self.global_step % 100 == 0:
                    self.logger.metrics(metrics, step=self.global_step)
                    for k, v in metrics.items():
                        self.metrics.add(k, v, step=self.global_step)

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        self.logger.metrics(eval_metrics, step=self.global_step)
                        for k, v in eval_metrics.items():
                            self.metrics.add(k, v, step=self.global_step)

                        # Track best
                        if eval_metrics["eval_acc"] > self.best_eval_acc:
                            self.best_eval_acc = eval_metrics["eval_acc"]
                            self.logger.info(f"New best eval accuracy: {self.best_eval_acc:.4f}")

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
            )

        # Final evaluation
        final_eval = self.evaluate()

        report = TrainReport(
            final_train_loss=avg_loss,
            final_train_acc=avg_acc,
            final_eval_acc=final_eval.get("eval_acc", 0.0),
            best_eval_acc=self.best_eval_acc,
            total_steps=self.global_step,
            config=asdict(self.config)
        )

        return report

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc,
            "config": asdict(self.config)
        }, path)

        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_acc = checkpoint["best_eval_acc"]

        self.logger.info(f"Loaded checkpoint from {path}")
