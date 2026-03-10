"""Tests for RippleAnalyzer (requires EasyEdit conda env)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edit.ripple_analysis import RippleAnalyzer, TripleLogits


class TestRippleAnalyzer:
    """Test RippleAnalyzer.analyze_all_triples() and compute_statistics()."""

    @pytest.fixture(scope="class")
    def edit_triple(self, small_triples):
        t = small_triples[0]
        other_o = next(x.o for x in small_triples if x.o != t.o)
        return t.s, t.r, t.o, other_o  # s, r, o_original, o_target

    @pytest.fixture(scope="class")
    def edited_model(
        self, model_for_editing, tokenizer_with_vocab, corpus_file, edit_triple, editing_device
    ):
        """Return a model ROME-edited on one triple."""
        from src.edit.rome import ROME
        rome = ROME(
            model=model_for_editing,
            tokenizer=tokenizer_with_vocab,
            device=editing_device,
            kg_corpus_path=corpus_file,
            mom2_n_samples=50,
            use_mom2_adjustment=True,
            v_num_grad_steps=3,
        )
        s, r, _, o_target = edit_triple
        edited, _ = rome.apply_edit(s, r, o_target, layer=0, copy_model=True)
        return edited

    @pytest.fixture(scope="class")
    def analyzer(self, model_for_editing, edited_model, tokenizer_with_vocab, editing_device):
        return RippleAnalyzer(
            model_before=model_for_editing,
            model_after=edited_model,
            tokenizer=tokenizer_with_vocab,
            device=editing_device,
        )

    @pytest.fixture(scope="class")
    def ripple_results(self, analyzer, corpus_file, edit_triple):
        s, r, o_original, o_target = edit_triple
        analyzer.load_knowledge_graph(corpus_file)
        return analyzer.analyze_all_triples(
            kg_path=corpus_file,
            edited_s=s,
            edited_r=r,
            edited_o_original=o_original,
            edited_o_target=o_target,
            max_triples=50,
        )

    def test_returns_list_of_triple_logits(self, ripple_results):
        assert isinstance(ripple_results, list)
        assert len(ripple_results) > 0
        assert all(isinstance(r, TripleLogits) for r in ripple_results)

    def test_triple_logits_fields_are_finite(self, ripple_results):
        import math
        for r in ripple_results:
            assert math.isfinite(r.logit_o_before)
            assert math.isfinite(r.logit_o_after)
            assert math.isfinite(r.ripple_effect)

    def test_ripple_effect_is_nonnegative(self, ripple_results):
        assert all(r.ripple_effect >= 0 for r in ripple_results)

    def test_hop_distance_is_none_or_nonneg_int(self, ripple_results):
        for r in ripple_results:
            assert r.hop_distance is None or r.hop_distance >= 0

    def test_hop_distance_sentinel_minus1_absent(self, ripple_results):
        """Sentinel -1 must not appear (replaced by None in #10 fix)."""
        assert all(r.hop_distance != -1 for r in ripple_results)

    def test_degrees_are_nonnegative(self, ripple_results):
        assert all(r.subject_degree >= 0 for r in ripple_results)
        assert all(r.object_degree >= 0 for r in ripple_results)

    def test_to_dict_round_trips(self, ripple_results):
        d = ripple_results[0].to_dict()
        assert d["s"] == ripple_results[0].s
        assert d["ripple_effect"] == ripple_results[0].ripple_effect
        assert d["hop_distance"] == ripple_results[0].hop_distance

    def test_compute_statistics_structure(self, analyzer, ripple_results):
        stats = analyzer.compute_statistics(ripple_results)
        assert "total_triples" in stats
        assert "by_hop_distance" in stats
        assert "by_subject_degree" in stats
        assert stats["total_triples"] == len(ripple_results)

    def test_statistics_by_hop_excludes_none(self, analyzer, ripple_results):
        """Hop distance None (unreachable) must not appear as a key."""
        stats = analyzer.compute_statistics(ripple_results)
        assert None not in stats["by_hop_distance"]

    def test_load_knowledge_graph_populates_graph(self, analyzer, corpus_file):
        analyzer.load_knowledge_graph(corpus_file)
        assert analyzer.graph is not None
        assert len(analyzer.graph.nodes()) > 0
        assert len(analyzer.entity_degrees) > 0
