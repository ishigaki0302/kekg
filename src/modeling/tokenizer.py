"""Tokenizer with special token support for SRO knowledge graphs."""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from collections import Counter


class SROTokenizer:
    """
    Simple whitespace-based tokenizer with special tokens.

    For SRO knowledge graphs, entities and relations are atomic tokens
    (space-separated), so we don't need complex BPE.
    """

    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SUBJ_TOKEN = "[SUBJ]"
    REL_TOKEN = "[REL]"
    OBJ_TOKEN = "[OBJ]"
    E_DELETED_TOKEN = "[E_DELETED]"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_vocab_size: int = 50000
    ):
        """
        Initialize tokenizer.

        Args:
            vocab: Pre-built vocabulary (token -> id mapping)
            max_vocab_size: Maximum vocabulary size
        """
        self.max_vocab_size = max_vocab_size

        if vocab is not None:
            self.vocab = vocab
        else:
            # Initialize with special tokens only
            self.vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.SUBJ_TOKEN: 2,
                self.REL_TOKEN: 3,
                self.OBJ_TOKEN: 4,
                self.E_DELETED_TOKEN: 5,
            }

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def build_vocab_from_corpus(self, corpus_paths: List[Union[str, Path]]) -> None:
        """
        Build vocabulary from corpus files.

        All entity IDs (E_*) and relation IDs (R_*) are treated as atomic tokens
        and added to vocabulary as special tokens (no BPE splitting).

        Args:
            corpus_paths: List of corpus file paths
        """
        # Collect all tokens
        all_tokens = set()

        for path in corpus_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = line.strip().split()
                    all_tokens.update(tokens)

        # Separate entity/relation IDs from other tokens
        entity_tokens = set()
        relation_tokens = set()
        other_tokens = set()

        for token in all_tokens:
            # Extract canonical ID (remove __a* suffix if present)
            canonical = token.split('__a')[0] if '__a' in token else token

            if canonical.startswith('E_'):
                entity_tokens.add(canonical)
                # Also add the aliased version if different
                if token != canonical:
                    entity_tokens.add(token)
            elif canonical.startswith('R_'):
                relation_tokens.add(canonical)
                if token != canonical:
                    relation_tokens.add(token)
            else:
                other_tokens.add(token)

        # Add all entity and relation tokens as special tokens (atomic, no splitting)
        # Sort for deterministic ordering
        for token in sorted(entity_tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        for token in sorted(relation_tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Add other tokens
        current_vocab_size = len(self.vocab)
        remaining_slots = self.max_vocab_size - current_vocab_size

        for token in sorted(other_tokens):
            if token not in self.vocab and remaining_slots > 0:
                self.vocab[token] = len(self.vocab)
                remaining_slots -= 1

        # Rebuild reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text (space-separated tokens)

        Returns:
            List of token IDs
        """
        tokens = text.strip().split()
        return [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = [self.id_to_token.get(tid, self.UNK_TOKEN) for tid in token_ids]
        # Filter out padding
        tokens = [t for t in tokens if t != self.PAD_TOKEN]
        return " ".join(tokens)

    def get_token(self, token_id: int) -> str:
        """Get token string from ID."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    def get_id(self, token: str) -> int:
        """Get ID from token string."""
        return self.vocab.get(token, self.vocab[self.UNK_TOKEN])

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.vocab[self.UNK_TOKEN]

    @property
    def e_deleted_id(self) -> int:
        """Get E_DELETED token ID."""
        return self.vocab[self.E_DELETED_TOKEN]

    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer vocabulary to file.

        Args:
            path: Output file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SROTokenizer":
        """
        Load tokenizer from file.

        Args:
            path: Vocabulary file path

        Returns:
            Loaded tokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
