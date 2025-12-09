"""Fixed inference utilities with entity-level evaluation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

from .gpt_mini import GPTMini
from .tokenizer import SROTokenizer


class SROInferenceFixed:
    """Inference engine with entity-level evaluation (alias-aware)."""

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

        # Build entity alias mapping for evaluation
        self._build_alias_map()

    def _build_alias_map(self) -> None:
        """Build mapping from canonical entity ID to all its aliases."""
        self.entity_to_aliases = {}

        for token in self.tokenizer.vocab.keys():
            if token.startswith('E_'):
                # Extract canonical ID
                canonical = token.split('__a')[0] if '__a' in token else token

                if canonical not in self.entity_to_aliases:
                    self.entity_to_aliases[canonical] = set()

                self.entity_to_aliases[canonical].add(token)

    def _get_canonical_entity(self, entity_token: str) -> str:
        """Extract canonical entity ID from possibly aliased token."""
        return entity_token.split('__a')[0] if '__a' in entity_token else entity_token

    def _get_entity_aliases(self, canonical_entity: str) -> Set[str]:
        """Get all aliases for a canonical entity."""
        return self.entity_to_aliases.get(canonical_entity, {canonical_entity})

    @torch.no_grad()
    def predict_next(
        self,
        s: str,
        r: str,
        return_probs: bool = False,
        top_k: int = 10
    ) -> Dict:
        """
        Predict object given subject and relation (entity-level).

        Args:
            s: Subject token (can be alias)
            r: Relation token (can be alias)
            return_probs: Whether to return full probability distribution
            top_k: Number of top predictions to return

        Returns:
            Dictionary with entity-level predictions
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

        # Aggregate probabilities by entity (sum over aliases)
        entity_probs = {}

        for token_id, prob in enumerate(probs):
            token = self.tokenizer.get_token(token_id)

            # Only consider entity tokens
            if token.startswith('E_'):
                canonical = self._get_canonical_entity(token)

                if canonical not in entity_probs:
                    entity_probs[canonical] = 0.0

                entity_probs[canonical] += prob

        # Top-k entities
        sorted_entities = sorted(entity_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_entities = sorted_entities[:top_k]

        result = {
            "input": {"s": s, "r": r},
            "top_entity": top_k_entities[0][0] if top_k_entities else None,
            "top_k_entities": top_k_entities,
            "token_logits": logits.cpu().numpy()
        }

        if return_probs:
            result["entity_probs"] = entity_probs

        return result

    @torch.no_grad()
    def evaluate_triple(
        self,
        s: str,
        r: str,
        o: str
    ) -> Dict:
        """
        Evaluate a specific triple (entity-level).

        Args:
            s: Subject token (can be alias)
            r: Relation token (can be alias)
            o: Object token (canonical or alias)

        Returns:
            Dictionary with entity-level metrics
        """
        # Get canonical entity for ground truth
        o_canonical = self._get_canonical_entity(o)

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

        # Aggregate probabilities by entity
        entity_probs = {}

        for token_id, prob in enumerate(probs):
            token = self.tokenizer.get_token(token_id)

            if token.startswith('E_'):
                canonical = self._get_canonical_entity(token)

                if canonical not in entity_probs:
                    entity_probs[canonical] = 0.0

                entity_probs[canonical] += prob

        # Target entity probability
        target_prob = entity_probs.get(o_canonical, 0.0)

        # Rank (1-indexed)
        rank = sum(1 for p in entity_probs.values() if p > target_prob) + 1

        # Is it top-1?
        top_entity = max(entity_probs.items(), key=lambda x: x[1])[0] if entity_probs else None
        is_correct = (top_entity == o_canonical)

        # Log probability
        target_logp = np.log(target_prob + 1e-10)

        return {
            "triple": {"s": s, "r": r, "o": o, "o_canonical": o_canonical},
            "probability": target_prob,
            "log_probability": target_logp,
            "rank": rank,
            "is_correct": is_correct,
            "predicted_entity": top_entity
        }

    @torch.no_grad()
    def batch_evaluate(
        self,
        triples: List[Tuple[str, str, str]]
    ) -> Dict:
        """
        Evaluate multiple triples in batch (entity-level).

        Args:
            triples: List of (s, r, o) tuples

        Returns:
            Aggregated entity-level metrics
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
            "mrr": np.mean([1.0 / r for r in ranks]),
            "num_samples": len(triples),
            "details": results
        }

    def load_corpus_triples(
        self,
        corpus_path: str
    ) -> List[Tuple[str, str, str]]:
        """Load triples from corpus file."""
        triples = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 3:
                    triples.append(tuple(tokens))
        return triples
