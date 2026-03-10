"""
SROTokenizer のテスト

トークン化・語彙構築・エンコード/デコードの正確さを検証する。
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import SROTokenizer


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    "E_001 R_01 E_002",
    "E_002 R_02 E_003",
    "E_003 R_01 E_001",
    "E_001 R_03 E_004",
]


@pytest.fixture
def fresh_tokenizer():
    return SROTokenizer()


@pytest.fixture
def corpus_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("\n".join(SAMPLE_CORPUS) + "\n")
        path = f.name
    yield path
    Path(path).unlink()


@pytest.fixture
def built_tokenizer(corpus_file):
    tok = SROTokenizer()
    tok.build_vocab_from_corpus([corpus_file])
    return tok


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestSpecialTokens:

    def test_pad_is_id_zero(self, fresh_tokenizer):
        assert fresh_tokenizer.vocab[SROTokenizer.PAD_TOKEN] == 0

    def test_unk_is_id_one(self, fresh_tokenizer):
        assert fresh_tokenizer.vocab[SROTokenizer.UNK_TOKEN] == 1

    def test_all_special_tokens_present(self, fresh_tokenizer):
        for tok in [
            SROTokenizer.PAD_TOKEN,
            SROTokenizer.UNK_TOKEN,
            SROTokenizer.SUBJ_TOKEN,
            SROTokenizer.REL_TOKEN,
            SROTokenizer.OBJ_TOKEN,
            SROTokenizer.E_DELETED_TOKEN,
        ]:
            assert tok in fresh_tokenizer.vocab

    def test_initial_vocab_size_is_6(self, fresh_tokenizer):
        assert fresh_tokenizer.vocab_size == 6


class TestVocabBuilding:

    def test_entities_added_to_vocab(self, built_tokenizer):
        for e in ["E_001", "E_002", "E_003", "E_004"]:
            assert e in built_tokenizer.vocab, f"{e} not in vocab"

    def test_relations_added_to_vocab(self, built_tokenizer):
        for r in ["R_01", "R_02", "R_03"]:
            assert r in built_tokenizer.vocab, f"{r} not in vocab"

    def test_no_alias_tokens_in_vocab(self, built_tokenizer):
        """エイリアス形式（__a*）のトークンが入っていないこと"""
        aliased = [tok for tok in built_tokenizer.vocab if "__a" in tok]
        assert aliased == []

    def test_vocab_size_grows_after_build(self, fresh_tokenizer, corpus_file):
        initial = fresh_tokenizer.vocab_size
        fresh_tokenizer.build_vocab_from_corpus([corpus_file])
        assert fresh_tokenizer.vocab_size > initial

    def test_build_is_idempotent(self, corpus_file):
        """同じコーパスを2回与えても語彙サイズが変わらないこと"""
        tok = SROTokenizer()
        tok.build_vocab_from_corpus([corpus_file])
        size1 = tok.vocab_size

        tok2 = SROTokenizer()
        tok2.build_vocab_from_corpus([corpus_file, corpus_file])
        size2 = tok2.vocab_size

        assert size1 == size2


class TestEncoding:

    def test_known_token_encodes_to_correct_id(self, built_tokenizer):
        ids = built_tokenizer.encode("E_001 R_01 E_002")
        assert len(ids) == 3
        assert ids[0] == built_tokenizer.vocab["E_001"]
        assert ids[1] == built_tokenizer.vocab["R_01"]
        assert ids[2] == built_tokenizer.vocab["E_002"]

    def test_unknown_token_encodes_to_unk_id(self, built_tokenizer):
        unk_id = built_tokenizer.vocab[SROTokenizer.UNK_TOKEN]
        ids = built_tokenizer.encode("E_999 R_99 E_000")
        assert all(i == unk_id for i in ids)

    def test_decode_gives_back_original(self, built_tokenizer):
        text = "E_001 R_01 E_002"
        ids = built_tokenizer.encode(text)
        decoded = built_tokenizer.decode(ids)
        assert decoded == text

    def test_id_to_token_consistent_with_vocab(self, built_tokenizer):
        for tok, tid in built_tokenizer.vocab.items():
            assert built_tokenizer.id_to_token[tid] == tok


class TestSaveLoad:

    def test_save_load_preserves_vocab(self, built_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            built_tokenizer.save(path)
            loaded = SROTokenizer.load(path)
            assert loaded.vocab == built_tokenizer.vocab

    def test_save_load_preserves_vocab_size(self, built_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            built_tokenizer.save(path)
            loaded = SROTokenizer.load(path)
            assert loaded.vocab_size == built_tokenizer.vocab_size

    def test_loaded_tokenizer_encodes_same(self, built_tokenizer):
        text = "E_001 R_02 E_003"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            built_tokenizer.save(path)
            loaded = SROTokenizer.load(path)
        assert loaded.encode(text) == built_tokenizer.encode(text)
