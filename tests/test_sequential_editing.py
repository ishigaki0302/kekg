"""Tests for sequential editing utilities (requires EasyEdit conda env)."""

import sys
import tempfile
from pathlib import Path
from copy import deepcopy

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sequential_edit.runner import (
    extract_rome_layer_weights,
    compute_weight_distance,
    compute_pairwise_avg_hop,
    sample_edit_cases,
    select_eval_triples,
)
from src.sequential_edit.kg_utils import KG
from src.sequential_edit.config import SeqEditConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kg(corpus_file):
    return KG(corpus_file)


@pytest.fixture(scope="module")
def edit_cases(kg):
    return sample_edit_cases(kg, num_cases=5, seed=42, selection_mode="random")


# ---------------------------------------------------------------------------
# KG utility tests
# ---------------------------------------------------------------------------

class TestKGUtils:

    def test_kg_loads_triples(self, kg):
        assert len(kg.triples) > 0

    def test_kg_triples_have_required_fields(self, kg):
        t = kg.triples[0]
        assert hasattr(t, "s") and hasattr(t, "r") and hasattr(t, "o")
        assert hasattr(t, "tid") and hasattr(t, "degree_s")

    def test_kg_degrees_are_nonnegative(self, kg):
        assert all(t.degree_s >= 0 for t in kg.triples)

    def test_bfs_hop_returns_dict(self, kg):
        source = kg.triples[0].s
        result = kg.bfs_hop(source, max_hop=3)
        assert isinstance(result, dict)
        assert source in result
        assert result[source] == 0

    def test_bfs_hop_distances_are_nonnegative(self, kg):
        source = kg.triples[0].s
        result = kg.bfs_hop(source, max_hop=3)
        assert all(v >= 0 for v in result.values())

    def test_bfs_hop_respects_max_hop(self, kg):
        source = kg.triples[0].s
        result = kg.bfs_hop(source, max_hop=2)
        assert all(v <= 2 for v in result.values())


# ---------------------------------------------------------------------------
# extract_rome_layer_weights / compute_weight_distance
# ---------------------------------------------------------------------------

class TestWeightUtils:

    def test_extract_rome_layer_weights_returns_ffn_w2(self, tiny_model):
        weights = extract_rome_layer_weights(tiny_model)
        assert len(weights) > 0
        assert all("ffn.w2" in k for k in weights)

    def test_extract_rome_layer_weights_are_tensors(self, tiny_model):
        weights = extract_rome_layer_weights(tiny_model)
        assert all(isinstance(v, torch.Tensor) for v in weights.values())

    def test_compute_weight_distance_identical_is_zero(self, tiny_model):
        w = extract_rome_layer_weights(tiny_model)
        dist = compute_weight_distance(w, w)
        assert abs(dist) < 1e-6

    def test_compute_weight_distance_detects_change(self, tiny_model):
        w_before = extract_rome_layer_weights(tiny_model)
        # Perturb one weight
        model_copy = deepcopy(tiny_model)
        for name, param in model_copy.named_parameters():
            if "ffn.w2" in name:
                param.data += 1.0
                break
        w_after = extract_rome_layer_weights(model_copy)
        dist = compute_weight_distance(w_after, w_before)
        assert dist > 0


# ---------------------------------------------------------------------------
# sample_edit_cases
# ---------------------------------------------------------------------------

class TestSampleEditCases:

    def test_returns_list(self, edit_cases):
        assert isinstance(edit_cases, list)

    def test_returns_requested_count(self, kg):
        cases = sample_edit_cases(kg, num_cases=3, seed=0, selection_mode="random")
        assert len(cases) == 3

    def test_edit_case_has_required_keys(self, edit_cases):
        for c in edit_cases:
            assert "s" in c and "r" in c and "o_old" in c and "o_new" in c

    def test_o_old_and_o_new_differ(self, edit_cases):
        for c in edit_cases:
            assert c["o_old"] != c["o_new"]

    def test_reproducible_with_same_seed(self, kg):
        cases1 = sample_edit_cases(kg, num_cases=5, seed=7, selection_mode="random")
        cases2 = sample_edit_cases(kg, num_cases=5, seed=7, selection_mode="random")
        assert [(c["s"], c["r"]) for c in cases1] == [(c["s"], c["r"]) for c in cases2]

    def test_degree_high_selection(self, kg):
        cases = sample_edit_cases(kg, num_cases=5, seed=42, selection_mode="degree_high")
        assert len(cases) == 5

    def test_degree_low_selection(self, kg):
        cases = sample_edit_cases(kg, num_cases=5, seed=42, selection_mode="degree_low")
        assert len(cases) == 5

    def test_unknown_mode_raises(self, kg):
        with pytest.raises(ValueError):
            sample_edit_cases(kg, num_cases=3, seed=0, selection_mode="invalid_mode")


# ---------------------------------------------------------------------------
# select_eval_triples
# ---------------------------------------------------------------------------

class TestSelectEvalTriples:

    @pytest.fixture
    def config(self):
        return SeqEditConfig(num_retain_triples=10, seed=42, device="cpu")

    def test_returns_two_lists(self, kg, edit_cases, config):
        edited, retain = select_eval_triples(kg, config, edit_cases)
        assert isinstance(edited, list)
        assert isinstance(retain, list)

    def test_edited_triples_match_edit_cases(self, kg, edit_cases, config):
        edited, _ = select_eval_triples(kg, config, edit_cases)
        edit_keys = {(c["s"], c["r"], c["o_old"]) for c in edit_cases}
        for t in edited:
            assert (t.s, t.r, t.o) in edit_keys

    def test_retain_triples_not_in_edit_cases(self, kg, edit_cases, config):
        _, retain = select_eval_triples(kg, config, edit_cases)
        edit_keys = {(c["s"], c["r"], c["o_old"]) for c in edit_cases}
        for t in retain:
            assert (t.s, t.r, t.o) not in edit_keys

    def test_retain_count_respects_config(self, kg, edit_cases, config):
        _, retain = select_eval_triples(kg, config, edit_cases)
        assert len(retain) <= config.num_retain_triples


# ---------------------------------------------------------------------------
# compute_pairwise_avg_hop
# ---------------------------------------------------------------------------

class TestComputePairwiseAvgHop:

    def test_single_subject_returns_zero(self, kg):
        s = kg.triples[0].s
        result = compute_pairwise_avg_hop(kg, [s])
        assert result == 0.0

    def test_empty_subjects_returns_zero(self, kg):
        result = compute_pairwise_avg_hop(kg, [])
        assert result == 0.0

    def test_two_subjects_returns_nonnegative(self, kg):
        subjects = list({t.s for t in kg.triples})[:2]
        result = compute_pairwise_avg_hop(kg, subjects)
        assert result >= 0.0
