"""Shared fixtures for tests."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg import generate_ba_kg
from src.modeling import GPTMini, GPTConfig, SROTokenizer


@pytest.fixture(scope="session")
def small_triples():
    """Generate a small BA KG for testing."""
    return generate_ba_kg(
        num_entities=50,
        num_relations=10,
        target_triples=200,
        seed=42,
    )


@pytest.fixture(scope="session")
def tokenizer_with_vocab(small_triples):
    """Build a tokenizer from small triples."""
    tokenizer = SROTokenizer()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for t in small_triples:
            f.write(f"{t.s} {t.r} {t.o}\n")
        corpus_path = f.name
    tokenizer.build_vocab_from_corpus([corpus_path])
    Path(corpus_path).unlink()
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model(tokenizer_with_vocab):
    """Build a tiny GPT model for testing (fast, CPU-only)."""
    config = GPTConfig(
        vocab_size=tokenizer_with_vocab.vocab_size,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_mlp=128,
        max_seq_len=8,
        dropout=0.0,
    )
    model = GPTMini(config)
    model.eval()
    return model
