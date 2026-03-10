"""
KnowledgeGraph クラスのテスト（issue #6）

Triple.from_dict() と KnowledgeGraph の追加により、src.eval が依存する
KG API が正しく動作することを検証する。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg import Triple, KnowledgeGraph, generate_ba_kg


class TestTripleFromDict:

    def test_from_dict_basic(self):
        d = {"s": "E_001", "r": "R_01", "o": "E_002"}
        t = Triple.from_dict(d)
        assert t.s == "E_001"
        assert t.r == "R_01"
        assert t.o == "E_002"

    def test_from_dict_roundtrip(self):
        original = Triple(s="E_003", r="R_02", o="E_004")
        restored = Triple.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_in_list(self):
        records = [
            {"s": "E_001", "r": "R_01", "o": "E_002"},
            {"s": "E_002", "r": "R_01", "o": "E_003"},
        ]
        triples = [Triple.from_dict(d) for d in records]
        assert len(triples) == 2
        assert all(isinstance(t, Triple) for t in triples)


class TestKnowledgeGraph:

    @pytest.fixture
    def kg(self):
        triples = [
            Triple(s="E_001", r="R_01", o="E_002"),
            Triple(s="E_002", r="R_02", o="E_003"),
            Triple(s="E_001", r="R_02", o="E_004"),
        ]
        return KnowledgeGraph(
            entities=["E_001", "E_002", "E_003", "E_004"],
            relations=["R_01", "R_02"],
            triples=triples,
        )

    def test_entities_accessible(self, kg):
        assert "E_001" in kg.entities
        assert len(kg.entities) == 4

    def test_triples_accessible(self, kg):
        assert len(kg.triples) == 3

    def test_entity_aliases_identity(self, kg):
        """no_alias版: entity_aliases[e] == [e]"""
        for e in kg.entities:
            assert kg.entity_aliases[e] == [e]

    def test_relation_aliases_identity(self, kg):
        """no_alias版: relation_aliases[r] == [r]"""
        for r in kg.relations:
            assert kg.relation_aliases[r] == [r]

    def test_entity_aliases_covers_all_entities(self, kg):
        assert set(kg.entity_aliases.keys()) == set(kg.entities)

    def test_relation_aliases_covers_all_relations(self, kg):
        assert set(kg.relation_aliases.keys()) == set(kg.relations)


class TestKnowledgeGraphFromTriples:

    def test_from_triples_infers_entities(self):
        triples = [
            Triple(s="E_001", r="R_01", o="E_002"),
            Triple(s="E_002", r="R_01", o="E_003"),
        ]
        kg = KnowledgeGraph.from_triples(triples)
        assert set(kg.entities) == {"E_001", "E_002", "E_003"}

    def test_from_triples_infers_relations(self):
        triples = [
            Triple(s="E_001", r="R_01", o="E_002"),
            Triple(s="E_002", r="R_02", o="E_003"),
        ]
        kg = KnowledgeGraph.from_triples(triples)
        assert set(kg.relations) == {"R_01", "R_02"}

    def test_from_triples_with_generated_kg(self):
        """generate_ba_kg() の出力から KnowledgeGraph を構築できること"""
        triples = generate_ba_kg(
            num_entities=30,
            num_relations=5,
            target_triples=100,
            seed=42,
        )
        kg = KnowledgeGraph.from_triples(triples)

        assert len(kg.triples) == len(triples)
        assert len(kg.entities) > 0
        assert len(kg.relations) > 0

        # entity_aliases は identity mapping
        sample_entity = kg.entities[0]
        assert kg.entity_aliases[sample_entity] == [sample_entity]


class TestImports:
    """module-level import が通ること（issue #6 の根本問題）"""

    def test_kg_module_exports_knowledge_graph(self):
        from src.kg import KnowledgeGraph
        assert KnowledgeGraph is not None

    def test_kg_module_exports_triple(self):
        from src.kg import Triple
        assert Triple is not None

    def test_triple_has_from_dict(self):
        from src.kg import Triple
        assert hasattr(Triple, "from_dict")
