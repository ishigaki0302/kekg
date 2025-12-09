"""Knowledge Graph generation with ER and BA structures."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Literal
import numpy as np
import networkx as nx
from collections import defaultdict


@dataclass
class Triple:
    """SRO triple representation."""
    s: str  # subject entity
    r: str  # relation
    o: str  # object entity

    def to_dict(self) -> dict:
        return {"s": self.s, "r": self.r, "o": self.o}

    @classmethod
    def from_dict(cls, d: dict) -> "Triple":
        return cls(s=d["s"], r=d["r"], o=d["o"])


@dataclass
class KnowledgeGraph:
    """Knowledge graph with entities, relations, and triples."""
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)

    # Synonyms/aliases for entities and relations
    entity_aliases: Dict[str, List[str]] = field(default_factory=dict)
    relation_aliases: Dict[str, List[str]] = field(default_factory=dict)

    # Graph structure metadata
    graph_type: str = "ER"
    seed: int = 42

    def get_degree(self) -> Dict[str, int]:
        """
        Compute degree (in-degree + out-degree) for each entity.

        Returns:
            Dictionary mapping entity -> degree
        """
        degree_count = defaultdict(int)
        for triple in self.triples:
            degree_count[triple.s] += 1  # out-degree
            degree_count[triple.o] += 1  # in-degree
        return dict(degree_count)

    def get_neighbors(self, entity: str, hop: int = 1) -> Dict[int, set]:
        """
        Get neighbors at different hop distances from an entity.

        Args:
            entity: Starting entity
            hop: Maximum hop distance

        Returns:
            Dictionary mapping hop distance -> set of entities
        """
        neighbors = {0: {entity}}

        for h in range(1, hop + 1):
            neighbors[h] = set()
            for e in neighbors[h - 1]:
                # Outgoing edges
                for triple in self.triples:
                    if triple.s == e:
                        neighbors[h].add(triple.o)
                    elif triple.o == e:
                        neighbors[h].add(triple.s)

            # Remove entities from previous hops
            for prev_h in range(h):
                neighbors[h] -= neighbors[prev_h]

        return neighbors

    def to_jsonl(self) -> List[dict]:
        """Export triples to JSONL format."""
        return [t.to_dict() for t in self.triples]


def generate_er_graph(
    num_entities: int,
    num_relations: int,
    edge_prob: float = 0.05,
    seed: int = 42
) -> KnowledgeGraph:
    """
    Generate Erdős-Rényi random graph.

    Args:
        num_entities: Number of entities
        num_relations: Number of relation types
        edge_prob: Probability of edge between any two nodes
        seed: Random seed

    Returns:
        KnowledgeGraph with ER structure
    """
    rng = np.random.default_rng(seed)

    # Create entities
    entities = [f"E_{i:03d}" for i in range(num_entities)]
    relations = [f"R_{i:02d}" for i in range(num_relations)]

    # Generate ER graph
    G = nx.erdos_renyi_graph(num_entities, edge_prob, seed=seed, directed=True)

    # Convert to triples
    triples = []
    for s_idx, o_idx in G.edges():
        r = rng.choice(relations)
        triple = Triple(s=entities[s_idx], r=r, o=entities[o_idx])
        triples.append(triple)

    kg = KnowledgeGraph(
        entities=entities,
        relations=relations,
        triples=triples,
        graph_type="ER",
        seed=seed
    )

    return kg


def generate_ba_graph(
    num_entities: int,
    num_relations: int,
    m_attach: int = 3,
    seed: int = 42
) -> KnowledgeGraph:
    """
    Generate Barabási-Albert scale-free graph (preferential attachment).

    Args:
        num_entities: Number of entities
        num_relations: Number of relation types
        m_attach: Number of edges to attach from new node
        seed: Random seed

    Returns:
        KnowledgeGraph with BA structure
    """
    rng = np.random.default_rng(seed)

    # Create entities
    entities = [f"E_{i:03d}" for i in range(num_entities)]
    relations = [f"R_{i:02d}" for i in range(num_relations)]

    # Generate BA graph
    G = nx.barabasi_albert_graph(num_entities, m_attach, seed=seed)

    # Convert to directed and add relations
    G_dir = G.to_directed()

    triples = []
    for s_idx, o_idx in G_dir.edges():
        r = rng.choice(relations)
        triple = Triple(s=entities[s_idx], r=r, o=entities[o_idx])
        triples.append(triple)

    kg = KnowledgeGraph(
        entities=entities,
        relations=relations,
        triples=triples,
        graph_type="BA",
        seed=seed
    )

    return kg


def assign_synonyms(
    kg: KnowledgeGraph,
    synonyms_per_entity: int = 5,
    synonyms_per_relation: int = 5,
    seed: int = 42
) -> KnowledgeGraph:
    """
    Assign synonym aliases to entities and relations.

    Each entity/relation gets k variants with suffix __alias_N.

    Args:
        kg: Input knowledge graph
        synonyms_per_entity: Number of aliases per entity
        synonyms_per_relation: Number of aliases per relation
        seed: Random seed

    Returns:
        KnowledgeGraph with populated alias dictionaries
    """
    rng = np.random.default_rng(seed)

    # Generate entity aliases
    for entity in kg.entities:
        aliases = [f"{entity}__a{i}" for i in range(synonyms_per_entity)]
        kg.entity_aliases[entity] = aliases

    # Generate relation aliases
    for relation in kg.relations:
        aliases = [f"{relation}__a{i}" for i in range(synonyms_per_relation)]
        kg.relation_aliases[relation] = aliases

    return kg


def expand_triples_with_aliases(
    kg: KnowledgeGraph,
    sample_rate: float = 0.2,
    seed: int = 42
) -> List[Triple]:
    """
    Expand triples with aliases for training.

    For each triple, sample a subset of alias combinations according to sample_rate.

    Args:
        kg: Knowledge graph with aliases
        sample_rate: Fraction of alias combinations to sample
        seed: Random seed

    Returns:
        List of triples with aliased entity/relation names
    """
    rng = np.random.default_rng(seed)

    expanded = []

    for triple in kg.triples:
        s_aliases = kg.entity_aliases.get(triple.s, [triple.s])
        r_aliases = kg.relation_aliases.get(triple.r, [triple.r])
        o_aliases = kg.entity_aliases.get(triple.o, [triple.o])

        # Total combinations
        total_combos = len(s_aliases) * len(r_aliases) * len(o_aliases)
        num_samples = max(1, int(total_combos * sample_rate))

        # Sample combinations
        for _ in range(num_samples):
            s_alias = rng.choice(s_aliases)
            r_alias = rng.choice(r_aliases)
            o_alias = rng.choice(o_aliases)

            expanded.append(Triple(s=s_alias, r=r_alias, o=o_alias))

    return expanded


def generate_graph(
    kind: Literal["ER", "BA"],
    num_entities: int,
    num_relations: int,
    **kwargs
) -> KnowledgeGraph:
    """
    Unified interface for graph generation.

    Args:
        kind: "ER" or "BA"
        num_entities: Number of entities
        num_relations: Number of relation types
        **kwargs: Additional parameters for specific graph types

    Returns:
        Generated KnowledgeGraph
    """
    if kind == "ER":
        return generate_er_graph(
            num_entities=num_entities,
            num_relations=num_relations,
            edge_prob=kwargs.get("edge_prob", 0.05),
            seed=kwargs.get("seed", 42)
        )
    elif kind == "BA":
        return generate_ba_graph(
            num_entities=num_entities,
            num_relations=num_relations,
            m_attach=kwargs.get("m_attach", 3),
            seed=kwargs.get("seed", 42)
        )
    else:
        raise ValueError(f"Unknown graph type: {kind}")
