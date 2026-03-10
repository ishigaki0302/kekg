"""Tests for ROME knowledge editing (requires EasyEdit conda env)."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edit.rome import ROME, EditResult


class TestROMEApplyEdit:
    """Test that ROME.apply_edit() modifies the model as expected."""

    @pytest.fixture(scope="class")
    def rome_editor(self, model_for_editing, tokenizer_with_vocab, corpus_file, editing_device):
        """Create a ROME editor with minimal settings for speed."""
        return ROME(
            model=model_for_editing,
            tokenizer=tokenizer_with_vocab,
            device=editing_device,
            kg_corpus_path=corpus_file,
            mom2_n_samples=50,
            use_mom2_adjustment=True,
            v_num_grad_steps=3,
        )

    @pytest.fixture(scope="class")
    def edit_triple(self, small_triples):
        """Pick a concrete triple to edit."""
        t = small_triples[0]
        other_o = next(x.o for x in small_triples if x.o != t.o)
        return t.s, t.r, other_o

    def test_apply_edit_returns_edit_result(self, rome_editor, edit_triple):
        s, r, o_target = edit_triple
        _, result = rome_editor.apply_edit(s, r, o_target, layer=0, copy_model=True)
        assert isinstance(result, EditResult)

    def test_apply_edit_result_has_correct_fields(self, rome_editor, edit_triple):
        s, r, o_target = edit_triple
        _, result = rome_editor.apply_edit(s, r, o_target, layer=0, copy_model=True)
        assert result.layer == 0
        assert result.edit_spec.s == s
        assert result.edit_spec.r == r
        assert result.edit_spec.o_target == o_target

    def test_apply_edit_returns_model(self, rome_editor, edit_triple):
        s, r, o_target = edit_triple
        edited_model, _ = rome_editor.apply_edit(s, r, o_target, layer=0, copy_model=True)
        assert edited_model is not None

    def test_edit_changes_model_weights(self, rome_editor, model_for_editing, edit_triple):
        """Weights of the edited layer must differ after editing."""
        s, r, o_target = edit_triple
        before_w = {
            name: param.detach().clone()
            for name, param in model_for_editing.named_parameters()
            if "ffn.w2" in name
        }
        edited_model, _ = rome_editor.apply_edit(s, r, o_target, layer=0, copy_model=True)
        after_w = {
            name: param.detach().clone()
            for name, param in edited_model.named_parameters()
            if "ffn.w2" in name
        }
        diffs = [
            (after_w[k] - before_w[k]).abs().max().item()
            for k in before_w
        ]
        assert any(d > 1e-6 for d in diffs), "No weight changed after ROME edit"

    def test_copy_model_true_does_not_modify_original(
        self, rome_editor, model_for_editing, edit_triple
    ):
        """copy_model=True must leave model_for_editing unchanged."""
        s, r, o_target = edit_triple
        before_w = {
            name: param.detach().clone()
            for name, param in model_for_editing.named_parameters()
        }
        rome_editor.apply_edit(s, r, o_target, layer=0, copy_model=True)
        for name, param in model_for_editing.named_parameters():
            assert torch.allclose(param, before_w[name]), (
                f"original model weight {name} was mutated"
            )


class TestROMEModelConfigWrapper:
    """ModelConfigWrapper must expose HF-style aliases."""

    def test_config_wrapper_has_n_embd(self, tiny_model):
        from src.edit.rome import ModelConfigWrapper
        cfg = ModelConfigWrapper(tiny_model.config)
        assert hasattr(cfg, "n_embd")
        assert cfg.n_embd == tiny_model.config.d_model

    def test_config_wrapper_has_n_layer(self, tiny_model):
        from src.edit.rome import ModelConfigWrapper
        cfg = ModelConfigWrapper(tiny_model.config)
        assert hasattr(cfg, "n_layer")
        assert cfg.n_layer == tiny_model.config.n_layers

    def test_config_wrapper_has_name_or_path(self, tiny_model):
        from src.edit.rome import ModelConfigWrapper
        cfg = ModelConfigWrapper(tiny_model.config)
        assert cfg._name_or_path == "gpt_mini"
