"""Inference utilities for SRO prediction."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

from .gpt_mini import GPTMini
from .tokenizer import SROTokenizer


class SROInference:
    """Inference engine for SRO predictions."""

    def __init__(
        self,
        model: GPTMini,
        tokenizer: SROTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained GPT model
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def predict_next(
        self,
        s: str,
        r: str,
        return_probs: bool = False,
        top_k: int = 10
    ) -> Dict:
        """
        Predict object given subject and relation.

        Args:
            s: Subject token
            r: Relation token
            return_probs: Whether to return full probability distribution
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and probabilities
        """
        # Encode input
        input_text = f"{s} {r}"
        input_ids = torch.tensor(
            self.tokenizer.encode(input_text),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, 2]

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs["logits"][0, -1, :]  # Last position logits [vocab_size]

        # Get probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Top-k predictions
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_tokens = [self.tokenizer.get_token(int(idx)) for idx in top_k_indices]
        top_k_probs = probs[top_k_indices].tolist()

        result = {
            "input": {"s": s, "r": r},
            "top_prediction": top_k_tokens[0],
            "top_k": list(zip(top_k_tokens, top_k_probs)),
            "logits": logits.cpu().numpy()
        }

        if return_probs:
            result["full_probs"] = probs

        return result

    @torch.no_grad()
    def evaluate_triple(
        self,
        s: str,
        r: str,
        o: str
    ) -> Dict:
        """
        Evaluate a specific triple.

        Args:
            s: Subject token
            r: Relation token
            o: Object token

        Returns:
            Dictionary with metrics (probability, rank, log-probability)
        """
        # Encode input
        input_text = f"{s} {r}"
        input_ids = torch.tensor(
            self.tokenizer.encode(input_text),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, 2]

        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs["logits"][0, -1, :]  # [vocab_size]

        # Get probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Target token ID
        target_id = self.tokenizer.get_id(o)

        # Metrics
        target_prob = float(probs[target_id])
        target_logp = float(F.log_softmax(logits, dim=-1)[target_id].cpu())

        # Rank (1-indexed)
        rank = int((probs > target_prob).sum() + 1)

        # Is it top-1?
        is_correct = (logits.argmax().item() == target_id)

        return {
            "triple": {"s": s, "r": r, "o": o},
            "probability": target_prob,
            "log_probability": target_logp,
            "rank": rank,
            "is_correct": is_correct,
            "predicted": self.tokenizer.get_token(logits.argmax().item())
        }

    @torch.no_grad()
    def batch_evaluate(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> Dict:
        """
        Evaluate multiple triples in batch.

        Args:
            triples: List of (s, r, o) tuples

        Returns:
            Aggregated metrics
        """
        results = []
        for s, r, o in triples:
            result = self.evaluate_triple(s, r, o)
            results.append(result)

        # Aggregate
        accuracies = [r["is_correct"] for r in results]
        ranks = [r["rank"] for r in results]
        log_probs = [r["log_probability"] for r in results]

        return {
            "accuracy": np.mean(accuracies),
            "mean_rank": np.mean(ranks),
            "median_rank": np.median(ranks),
            "mean_log_prob": np.mean(log_probs),
            "num_samples": len(triples),
            "details": results
        }

    @torch.no_grad()
    def get_embeddings(
        self,
        tokens: List[str],
        layer: int = 0
    ) -> np.ndarray:
        """
        Get token embeddings from a specific layer.

        Args:
            tokens: List of tokens
            layer: Layer index (0 = input embeddings)

        Returns:
            Embeddings array [num_tokens, d_model]
        """
        token_ids = [self.tokenizer.get_id(t) for t in tokens]
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, seq_len]

        # Forward with hidden states
        outputs = self.model(input_ids, return_hidden_states=True)
        hidden_states = outputs["hidden_states"]

        # Get layer embeddings
        embeddings = hidden_states[layer][0].cpu().numpy()  # [seq_len, d_model]

        return embeddings

    def load_corpus_triples(
        self,
        corpus_path: str
    ) -> List[Tuple[str, str, str]]:
        """
        Load triples from corpus file.

        Args:
            corpus_path: Path to corpus file

        Returns:
            List of (s, r, o) tuples
        """
        triples = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 3:
                    triples.append(tuple(tokens))
        return triples
