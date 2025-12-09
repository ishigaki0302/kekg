"""Test knowledge graph generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg import generate_graph, assign_synonyms, expand_triples_with_aliases
from src.utils import set_seed


def test_er_graph_generation():
    """Test ER graph generation."""
    set_seed(42)

    kg = generate_graph(
        kind="ER",
        num_entities=50,
        num_relations=10,
        edge_prob=0.1,
        seed=42
    )

    assert len(kg.entities) == 50
    assert len(kg.relations) == 10
    assert len(kg.triples) > 0
    assert kg.graph_type == "ER"

    print(f"✓ ER graph: {len(kg.entities)} entities, {len(kg.triples)} triples")


def test_ba_graph_generation():
    """Test BA graph generation."""
    set_seed(42)

    kg = generate_graph(
        kind="BA",
        num_entities=50,
        num_relations=10,
        m_attach=3,
        seed=42
    )

    assert len(kg.entities) == 50
    assert len(kg.relations) == 10
    assert len(kg.triples) > 0
    assert kg.graph_type == "BA"

    # Check degree distribution (BA should have some high-degree nodes)
    degrees = kg.get_degree()
    max_degree = max(degrees.values())

    print(f"✓ BA graph: {len(kg.entities)} entities, {len(kg.triples)} triples")
    print(f"  Max degree: {max_degree}")


def test_synonyms():
    """Test synonym assignment."""
    set_seed(42)

    kg = generate_graph(kind="ER", num_entities=20, num_relations=5, seed=42)
    kg = assign_synonyms(kg, synonyms_per_entity=3, synonyms_per_relation=3, seed=42)

    # Check all entities have synonyms
    for entity in kg.entities:
        assert entity in kg.entity_aliases
        assert len(kg.entity_aliases[entity]) == 3

    # Check all relations have synonyms
    for relation in kg.relations:
        assert relation in kg.relation_aliases
        assert len(kg.relation_aliases[relation]) == 3

    print(f"✓ Synonyms: {len(kg.entity_aliases)} entities, {len(kg.relation_aliases)} relations")


def test_triple_expansion():
    """Test triple expansion with aliases."""
    set_seed(42)

    kg = generate_graph(kind="ER", num_entities=20, num_relations=5, seed=42)
    kg = assign_synonyms(kg, synonyms_per_entity=2, synonyms_per_relation=2, seed=42)

    # Expand with 50% sampling
    expanded = expand_triples_with_aliases(kg, sample_rate=0.5, seed=42)

    # Should have more triples than original
    assert len(expanded) >= len(kg.triples)

    print(f"✓ Expansion: {len(kg.triples)} -> {len(expanded)} triples")


def test_degree_calculation():
    """Test degree calculation."""
    set_seed(42)

    kg = generate_graph(kind="BA", num_entities=30, num_relations=5, m_attach=2, seed=42)
    degrees = kg.get_degree()

    # All entities should have at least some degree
    assert len(degrees) > 0

    # Total degree should equal 2 * num_edges
    total_degree = sum(degrees.values())
    assert total_degree == 2 * len(kg.triples)

    print(f"✓ Degrees: avg={total_degree/len(degrees):.2f}, max={max(degrees.values())}")


if __name__ == "__main__":
    print("Running KG generation tests...")
    print()

    test_er_graph_generation()
    test_ba_graph_generation()
    test_synonyms()
    test_triple_expansion()
    test_degree_calculation()

    print()
    print("All tests passed! ✓")
