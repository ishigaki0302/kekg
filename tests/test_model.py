"""
GPTMini モデルのテスト

フォワードパスの形状・損失計算・マスク処理の正確さを検証する。
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import GPTMini, GPTConfig, SROTokenizer
from src.modeling.trainer import SRODataset, compute_accuracy


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return GPTConfig(
        vocab_size=100,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_mlp=128,
        max_seq_len=8,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def model(config):
    m = GPTMini(config)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# forward pass shape tests
# ---------------------------------------------------------------------------

class TestForwardPass:

    def test_logits_shape(self, model, config):
        x = torch.randint(0, config.vocab_size, (2, 3))
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape == (2, 3, config.vocab_size)

    def test_logits_are_finite(self, model, config):
        x = torch.randint(0, config.vocab_size, (4, 3))
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out["logits"]).all()

    def test_hidden_states_count(self, model, config):
        x = torch.randint(0, config.vocab_size, (1, 3))
        with torch.no_grad():
            out = model(x, return_hidden_states=True)
        # input embedding + n_layers
        assert len(out["hidden_states"]) == config.n_layers + 1

    def test_hidden_state_shape(self, model, config):
        x = torch.randint(0, config.vocab_size, (1, 3))
        with torch.no_grad():
            out = model(x, return_hidden_states=True)
        for hs in out["hidden_states"]:
            assert hs.shape == (1, 3, config.d_model)

    def test_attention_count(self, model, config):
        x = torch.randint(0, config.vocab_size, (1, 3))
        with torch.no_grad():
            out = model(x, return_attention=True)
        assert len(out["attentions"]) == config.n_layers

    def test_different_inputs_give_different_outputs(self, model, config):
        x1 = torch.tensor([[1, 2, 3]])
        x2 = torch.tensor([[4, 5, 6]])
        with torch.no_grad():
            o1 = model(x1)["logits"]
            o2 = model(x2)["logits"]
        assert not torch.allclose(o1, o2)


# ---------------------------------------------------------------------------
# SRODataset tests
# ---------------------------------------------------------------------------

class TestSRODataset:
    """データセットが正しく labels_mask を設定するか"""

    @pytest.fixture
    def corpus_path(self, tmp_path):
        import tempfile
        p = tmp_path / "corpus.txt"
        p.write_text(
            "E_001 R_01 E_002\n"
            "E_002 R_02 E_003\n"
            "E_003 R_01 E_001\n"
        )
        return p

    @pytest.fixture
    def dataset(self, corpus_path):
        import tempfile
        from src.modeling import SROTokenizer
        tok = SROTokenizer()
        tok.build_vocab_from_corpus([str(corpus_path)])
        return SRODataset(str(corpus_path), tok)

    def test_length(self, dataset):
        assert len(dataset) == 3

    def test_input_ids_length_is_3(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["input_ids"].shape == (3,)

    def test_labels_mask_only_o_is_true(self, dataset):
        """labels_mask は [False, False, True] でなければならない"""
        for i in range(len(dataset)):
            mask = dataset[i]["labels_mask"]
            assert mask.tolist() == [False, False, True], \
                f"Unexpected mask: {mask.tolist()}"

    def test_input_equals_labels(self, dataset):
        """teacher forcing: input_ids と labels は同一"""
        for i in range(len(dataset)):
            sample = dataset[i]
            assert torch.equal(sample["input_ids"], sample["labels"])


# ---------------------------------------------------------------------------
# compute_accuracy tests
# ---------------------------------------------------------------------------

class TestComputeAccuracy:
    """O位置のみを評価するマスク付き精度計算の検証"""

    def test_perfect_prediction(self):
        """完全正解なら精度 1.0"""
        # logits: [1, 3, 5] -> shift -> [1, 2, 5]
        # labels: [1, 3] -> shift -> [3] at position 1 (o position after shift)
        vocab = 5
        labels = torch.tensor([[0, 1, 2]])  # S=0, R=1, O=2
        logits = torch.zeros(1, 3, vocab)
        # position 0 predicts 1 (R), position 1 predicts 2 (O)
        logits[0, 0, 1] = 10.0  # predict R at pos 0 -> shift_labels[0]
        logits[0, 1, 2] = 10.0  # predict O at pos 1 -> shift_labels[1]
        mask = torch.tensor([[False, False, True]])

        acc = compute_accuracy(logits, labels, mask)
        assert acc == pytest.approx(1.0)

    def test_wrong_prediction(self):
        """O位置を間違えたら精度 0.0"""
        vocab = 5
        labels = torch.tensor([[0, 1, 2]])
        logits = torch.zeros(1, 3, vocab)
        logits[0, 1, 3] = 10.0  # predict 3 instead of 2 at O position
        mask = torch.tensor([[False, False, True]])

        acc = compute_accuracy(logits, labels, mask)
        assert acc == pytest.approx(0.0)

    def test_mask_ignores_s_and_r_positions(self):
        """S/R位置の予測が間違っていても、O正解なら精度 1.0"""
        vocab = 5
        labels = torch.tensor([[0, 1, 2]])
        logits = torch.zeros(1, 3, vocab)
        # S->R prediction: wrong
        logits[0, 0, 99 % vocab] = 10.0
        # R->O prediction: correct
        logits[0, 1, 2] = 10.0
        mask = torch.tensor([[False, False, True]])

        acc = compute_accuracy(logits, labels, mask)
        assert acc == pytest.approx(1.0)

    def test_no_mask_uses_all_positions(self):
        """マスクなし: 全ポジション平均"""
        vocab = 5
        labels = torch.tensor([[0, 1, 2]])
        logits = torch.zeros(1, 3, vocab)
        # S->R correct, R->O wrong
        logits[0, 0, 1] = 10.0
        logits[0, 1, 3] = 10.0  # wrong: should be 2
        # 1/2 = 0.5

        acc = compute_accuracy(logits, labels, None)
        assert acc == pytest.approx(0.5)

    def test_batch_accuracy(self):
        """バッチ全体の精度が正しく集計されること"""
        vocab = 5
        # batch=2; both correct at O position
        labels = torch.tensor([[0, 1, 2], [0, 1, 3]])
        logits = torch.zeros(2, 3, vocab)
        logits[0, 1, 2] = 10.0  # sample 0 O correct
        logits[1, 1, 3] = 10.0  # sample 1 O correct
        mask = torch.tensor([[False, False, True], [False, False, True]])

        acc = compute_accuracy(logits, labels, mask)
        assert acc == pytest.approx(1.0)
