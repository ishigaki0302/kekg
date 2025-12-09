"""Test GPT mini model."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import GPTMini, GPTConfig
from src.utils import set_seed


def test_model_initialization():
    """Test model initialization."""
    set_seed(42)

    config = GPTConfig(
        vocab_size=1000,
        n_layers=6,
        n_heads=4,
        d_model=256,
        d_mlp=1024,
        max_seq_len=8
    )

    model = GPTMini(config)

    num_params = model.get_num_params()
    assert num_params > 0

    print(f"✓ Model initialized: {num_params:,} parameters")


def test_forward_pass():
    """Test forward pass."""
    set_seed(42)

    config = GPTConfig(
        vocab_size=100,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=8
    )

    model = GPTMini(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 3
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs["logits"]

    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    print(f"✓ Forward pass: input {input_ids.shape} -> logits {logits.shape}")


def test_generation():
    """Test autoregressive generation."""
    set_seed(42)

    config = GPTConfig(
        vocab_size=100,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=8
    )

    model = GPTMini(config)
    model.eval()

    # Start with 2 tokens
    input_ids = torch.tensor([[10, 20]])

    # Generate 1 more token
    generated = model.generate(input_ids, max_new_tokens=1, top_k=10)

    assert generated.shape[1] == 3  # Should have 3 tokens now

    print(f"✓ Generation: {input_ids.shape} -> {generated.shape}")


def test_hidden_states():
    """Test retrieval of hidden states."""
    set_seed(42)

    config = GPTConfig(
        vocab_size=100,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=8
    )

    model = GPTMini(config)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30]])

    with torch.no_grad():
        outputs = model(input_ids, return_hidden_states=True)

    hidden_states = outputs["hidden_states"]

    # Should have n_layers + 1 (input embedding + each layer)
    assert len(hidden_states) == config.n_layers + 1

    print(f"✓ Hidden states: {len(hidden_states)} layers")


def test_attention_weights():
    """Test retrieval of attention weights."""
    set_seed(42)

    config = GPTConfig(
        vocab_size=100,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=8
    )

    model = GPTMini(config)
    model.eval()

    input_ids = torch.tensor([[10, 20, 30]])

    with torch.no_grad():
        outputs = model(input_ids, return_attention=True)

    attentions = outputs["attentions"]

    # Should have n_layers attention weight matrices
    assert len(attentions) == config.n_layers

    print(f"✓ Attention weights: {len(attentions)} layers")


if __name__ == "__main__":
    print("Running model tests...")
    print()

    test_model_initialization()
    test_forward_pass()
    test_generation()
    test_hidden_states()
    test_attention_weights()

    print()
    print("All tests passed! ✓")
