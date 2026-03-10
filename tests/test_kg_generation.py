"""
KG生成のテスト

generate_ba_kg() が正しい構造のトリプルを生成するかを検証する。
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg import Triple, generate_ba_kg


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def triples():
    return generate_ba_kg(
        num_entities=100,
        num_relations=15,
        target_triples=500,
        seed=42,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestTripleFormat:
    """トリプルの形式が正しいか"""

    def test_returns_list_of_triples(self, triples):
        assert isinstance(triples, list)
        assert all(isinstance(t, Triple) for t in triples)

    def test_subject_starts_with_E(self, triples):
        bad = [t for t in triples if not t.s.startswith("E_")]
        assert bad == [], f"Non-E subjects: {bad[:3]}"

    def test_relation_starts_with_R(self, triples):
        bad = [t for t in triples if not t.r.startswith("R_")]
        assert bad == [], f"Non-R relations: {bad[:3]}"

    def test_object_starts_with_E(self, triples):
        bad = [t for t in triples if not t.o.startswith("E_")]
        assert bad == [], f"Non-E objects: {bad[:3]}"

    def test_no_alias_suffix(self, triples):
        """エイリアス(__a*)が付いていないこと"""
        aliased = [t for t in triples if "__a" in t.s or "__a" in t.o or "__a" in t.r]
        assert aliased == [], f"Found alias tokens: {aliased[:3]}"

    def test_no_self_loops(self, triples):
        """s == o のトリプルがないこと"""
        self_loops = [t for t in triples if t.s == t.o]
        assert self_loops == []

    def test_no_duplicate_triples(self, triples):
        """同一トリプルの重複がないこと"""
        seen = set()
        for t in triples:
            key = (t.s, t.r, t.o)
            assert key not in seen, f"Duplicate triple: {key}"
            seen.add(key)


class TestGraphStructure:
    """グラフとして妥当な構造か"""

    def test_entity_count_within_spec(self, triples):
        """登場エンティティ数が num_entities 以下"""
        entities = set()
        for t in triples:
            entities.add(t.s)
            entities.add(t.o)
        assert len(entities) <= 100

    def test_relation_count_within_spec(self, triples):
        """登場リレーション数が num_relations 以下"""
        relations = {t.r for t in triples}
        assert len(relations) <= 15

    def test_degree_distribution_has_hubs(self, triples):
        """BAグラフ特有のハブノード（高次数ノード）が存在すること"""
        degree = Counter()
        for t in triples:
            degree[t.s] += 1
            degree[t.o] += 1
        counts = sorted(degree.values(), reverse=True)
        # 最大次数がトリプル数の1割以上ある（ハブが存在する）
        assert counts[0] >= len(triples) * 0.05, \
            f"Max degree {counts[0]} seems low for BA graph"

    def test_multi_hop_possible(self, triples):
        """2-hop以上の経路が存在すること（Eがs/o両方に登場する）"""
        as_subject = {t.s for t in triples}
        as_object = {t.o for t in triples}
        relay_nodes = as_subject & as_object
        assert len(relay_nodes) > 0, "No relay nodes found; multi-hop impossible"


class TestReproducibility:

    def test_same_seed_gives_same_result(self):
        a = generate_ba_kg(num_entities=30, num_relations=5, target_triples=100, seed=7)
        b = generate_ba_kg(num_entities=30, num_relations=5, target_triples=100, seed=7)
        assert a == b

    def test_different_seed_gives_different_result(self):
        a = generate_ba_kg(num_entities=30, num_relations=5, target_triples=100, seed=1)
        b = generate_ba_kg(num_entities=30, num_relations=5, target_triples=100, seed=2)
        assert a != b


class TestTripleSerialize:

    def test_to_dict(self):
        t = Triple(s="E_001", r="R_01", o="E_002")
        assert t.to_dict() == {"s": "E_001", "r": "R_01", "o": "E_002"}

    def test_hashable(self):
        t = Triple(s="E_001", r="R_01", o="E_002")
        s = {t}  # should not raise
        assert t in s
