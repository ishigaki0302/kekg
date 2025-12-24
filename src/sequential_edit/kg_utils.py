"""Knowledge Graph utilities for sequential editing.

This module provides enhanced Triple and KG classes with additional metadata
for analyzing ripple effects during sequential editing.
"""

from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Set
import random
import json
from pathlib import Path

# Import base Triple from existing KG module
from src.kg.generator import Triple as BaseTriple


@dataclass
class Triple:
    """Enhanced Triple with metadata for sequential editing analysis.

    Attributes:
        tid: Triple ID (index in the knowledge graph)
        s: Subject entity
        r: Relation
        o: Object entity
        degree_s: Out-degree + in-degree of the subject entity
    """

    tid: int
    s: str
    r: str
    o: str
    degree_s: int

    @classmethod
    def from_base_triple(cls, base_triple: BaseTriple, tid: int, degree_s: int):
        """Create enhanced Triple from base Triple."""
        return cls(
            tid=tid,
            s=base_triple.s,
            r=base_triple.r,
            o=base_triple.o,
            degree_s=degree_s,
        )


class KG:
    """Knowledge Graph wrapper with BFS hop distance computation.

    This class wraps the base KnowledgeGraph and provides additional
    functionality needed for sequential editing analysis.
    """

    def __init__(self, graph_path: str, corpus_path: str = None):
        """Initialize KG from graph.jsonl or corpus.txt file.

        Args:
            graph_path: Path to graph.jsonl or corpus.txt file
            corpus_path: Path to corpus file (optional, for validation)
        """
        self.graph_path = Path(graph_path)
        self.corpus_path = Path(corpus_path) if corpus_path else None

        # Load triples from file (auto-detect format)
        self.base_triples = self._load_triples(graph_path)

        # Build adjacency list and compute degrees
        self.entity_neighbors = self._build_adjacency_list()
        self.entity_degrees = self._compute_degrees()

        # Create enhanced triples with metadata
        self.triples: List[Triple] = []
        for tid, base_triple in enumerate(self.base_triples):
            degree_s = self.entity_degrees.get(base_triple.s, 0)
            triple = Triple.from_base_triple(base_triple, tid, degree_s)
            self.triples.append(triple)

        # Build entity and relation sets
        self.entities = self._get_entities()
        self.relations = self._get_relations()

    def _load_triples(self, file_path: str) -> List[BaseTriple]:
        """Load triples from graph.jsonl or corpus.txt file.

        Auto-detects format:
        - .jsonl files: JSON format with s, r, o keys
        - .txt files: Space-separated format "s r o"
        """
        triples = []
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            # Detect format from first line
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning

            if file_path.suffix == ".jsonl" or first_line.startswith("{"):
                # JSON format
                for line in f:
                    data = json.loads(line.strip())
                    triple = BaseTriple.from_dict(data)
                    triples.append(triple)
            else:
                # Space-separated corpus format
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        s, r, o = parts
                        triple = BaseTriple(s=s, r=r, o=o)
                        triples.append(triple)

        return triples

    def _load_graph(self, graph_path: str) -> List[BaseTriple]:
        """Load triples from graph.jsonl file (deprecated, use _load_triples)."""
        return self._load_triples(graph_path)

    def _build_adjacency_list(self) -> Dict[str, List[str]]:
        """Build adjacency list for BFS traversal.

        Returns:
            Dictionary mapping entity -> list of neighbor entities
        """
        neighbors = defaultdict(list)
        for triple in self.base_triples:
            # Both directions (undirected for hop distance)
            neighbors[triple.s].append(triple.o)
            neighbors[triple.o].append(triple.s)
        return dict(neighbors)

    def _compute_degrees(self) -> Dict[str, int]:
        """Compute degree (in-degree + out-degree) for each entity.

        Returns:
            Dictionary mapping entity -> degree
        """
        degree_count = defaultdict(int)
        for triple in self.base_triples:
            degree_count[triple.s] += 1  # out-degree
            degree_count[triple.o] += 1  # in-degree
        return dict(degree_count)

    def _get_entities(self) -> Set[str]:
        """Get set of all entities."""
        entities = set()
        for triple in self.base_triples:
            entities.add(triple.s)
            entities.add(triple.o)
        return entities

    def _get_relations(self) -> Set[str]:
        """Get set of all relations."""
        return {triple.r for triple in self.base_triples}

    def sample_triples(self, k: int, seed: int = None) -> List[Triple]:
        """Sample k random triples from the knowledge graph.

        Args:
            k: Number of triples to sample
            seed: Random seed (if provided, sets the seed before sampling)

        Returns:
            List of sampled Triple objects
        """
        if seed is not None:
            random.seed(seed)

        # Sample up to k triples (or all if k > total)
        k = min(k, len(self.triples))
        return random.sample(self.triples, k)

    def bfs_hop(self, start_entity: str, max_hop: int) -> Dict[str, int]:
        """Compute hop distance from start_entity to all reachable entities using BFS.

        Args:
            start_entity: Starting entity
            max_hop: Maximum hop distance to explore

        Returns:
            Dictionary mapping entity -> hop distance
            (entities beyond max_hop are not included)
        """
        if start_entity not in self.entities:
            return {}

        hop_distances = {start_entity: 0}
        queue = deque([(start_entity, 0)])
        visited = {start_entity}

        while queue:
            current, dist = queue.popleft()

            # Stop if we've reached max_hop
            if dist >= max_hop:
                continue

            # Explore neighbors
            for neighbor in self.entity_neighbors.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    hop_distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return hop_distances

    def get_triple_by_tid(self, tid: int) -> Triple:
        """Get triple by triple ID."""
        if 0 <= tid < len(self.triples):
            return self.triples[tid]
        raise IndexError(f"Triple ID {tid} out of range")

    def __len__(self):
        """Return number of triples."""
        return len(self.triples)

    def __repr__(self):
        """String representation."""
        return (
            f"KG(triples={len(self.triples)}, "
            f"entities={len(self.entities)}, "
            f"relations={len(self.relations)})"
        )
