"""Test tokenizer functionality."""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import SROTokenizer


def test_tokenizer_initialization():
    """Test tokenizer initialization with special tokens."""
    tokenizer = SROTokenizer()

    # Check special tokens
    assert tokenizer.vocab[SROTokenizer.PAD_TOKEN] == 0
    assert tokenizer.vocab[SROTokenizer.UNK_TOKEN] == 1
    assert tokenizer.vocab[SROTokenizer.E_DELETED_TOKEN] == 5

    print(f"✓ Tokenizer initialized with {tokenizer.vocab_size} tokens")


def test_encode_decode():
    """Test encoding and decoding."""
    tokenizer = SROTokenizer()

    # Add some tokens
    tokenizer.vocab["E_001__a0"] = len(tokenizer.vocab)
    tokenizer.vocab["R_05__a2"] = len(tokenizer.vocab)
    tokenizer.vocab["E_010__a1"] = len(tokenizer.vocab)
    tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}

    # Test encode/decode
    text = "E_001__a0 R_05__a2 E_010__a1"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    print(f"✓ Encode/decode: '{text}' -> {encoded} -> '{decoded}'")


def test_vocab_building():
    """Test vocabulary building from corpus."""
    tokenizer = SROTokenizer()

    # Create temporary corpus
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("E_001__a0 R_01__a0 E_002__a1\n")
        f.write("E_003__a2 R_02__a1 E_004__a0\n")
        f.write("E_001__a0 R_03__a2 E_005__a1\n")
        corpus_path = f.name

    try:
        tokenizer.build_vocab_from_corpus([corpus_path])

        # Check that entity and relation tokens are in vocab
        assert "E_001__a0" in tokenizer.vocab
        assert "R_01__a0" in tokenizer.vocab

        print(f"✓ Vocab built: {tokenizer.vocab_size} tokens")

    finally:
        Path(corpus_path).unlink()


def test_save_load():
    """Test saving and loading tokenizer."""
    tokenizer = SROTokenizer()

    # Add some custom tokens
    tokenizer.vocab["E_TEST"] = len(tokenizer.vocab)
    tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tokenizer.json"

        # Save
        tokenizer.save(save_path)

        # Load
        loaded_tokenizer = SROTokenizer.load(save_path)

        # Check
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert "E_TEST" in loaded_tokenizer.vocab

        print(f"✓ Save/load: {tokenizer.vocab_size} tokens")


if __name__ == "__main__":
    print("Running tokenizer tests...")
    print()

    test_tokenizer_initialization()
    test_encode_decode()
    test_vocab_building()
    test_save_load()

    print()
    print("All tests passed! ✓")
