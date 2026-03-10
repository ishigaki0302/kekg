"""Ripple effect analysis for knowledge editing.

This module analyzes how an edit to a single fact propagates through
the knowledge graph, affecting predictions for related facts.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from pathlib import Path
from collections import defaultdict, deque
import json

import torch
import torch.nn as nn
import numpy as np
import networkx as nx


@dataclass
class TripleLogits:
    """Logits for a triple before and after editing."""

    s: str
    r: str
    o: str
    o_alt: str  # Alternative object (the edited target)

    # Before edit
    logit_o_before: float
    logit_o_alt_before: float
    pred_before: str

    # After edit
    logit_o_after: float
    logit_o_alt_after: float
    pred_after: str

    # Ripple effect metrics
    ripple_effect: float  # Change in logit difference
    hop_distance: int  # Graph distance from edited triple
    subject_degree: int
    object_degree: int

    def to_dict(self):
        return {
            's': self.s,
            'r': self.r,
            'o': self.o,
            'o_alt': self.o_alt,
            'logit_o_before': self.logit_o_before,
            'logit_o_alt_before': self.logit_o_alt_before,
            'pred_before': self.pred_before,
            'logit_o_after': self.logit_o_after,
            'logit_o_alt_after': self.logit_o_alt_after,
            'pred_after': self.pred_after,
            'ripple_effect': self.ripple_effect,
            'hop_distance': self.hop_distance,
            'subject_degree': self.subject_degree,
            'object_degree': self.object_degree
        }


class RippleAnalyzer:
    """Analyzer for ripple effects in knowledge editing."""

    def __init__(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize ripple analyzer.

        Args:
            model_before: Model before editing
            model_after: Model after editing
            tokenizer: Tokenizer
            device: Device to run on
        """
        self.model_before = model_before
        self.model_after = model_after
        self.tokenizer = tokenizer
        self.device = device

        # Graph structure
        self.graph = None
        self.entity_degrees = {}

    def load_knowledge_graph(self, kg_path: str):
        """
        Load knowledge graph structure from corpus file.

        Args:
            kg_path: Path to KG corpus file
        """
        self.graph = nx.MultiDiGraph()

        with open(kg_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    s, r, o = parts[0], parts[1], parts[2]
                    self.graph.add_edge(s, o, relation=r)

        # Compute node degrees
        for node in self.graph.nodes():
            self.entity_degrees[node] = self.graph.degree(node)

    def compute_hop_distance(self, source: str, target_s: str, target_o: str) -> int:
        """
        Compute minimum hop distance from source entity to target triple.

        Args:
            source: Source entity (subject of triple being analyzed)
            target_s: Subject of edited triple
            target_o: Object of edited triple

        Returns:
            Minimum hop distance (0 if same triple, inf if unreachable)
        """
        if source == target_s:
            return 0

        try:
            # Distance to subject or object of edited triple
            dist_to_s = nx.shortest_path_length(self.graph.to_undirected(), source, target_s)
            try:
                dist_to_o = nx.shortest_path_length(self.graph.to_undirected(), source, target_o)
                return min(dist_to_s, dist_to_o)
            except nx.NetworkXNoPath:
                return dist_to_s
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def get_logits_for_triple(
        self,
        model: nn.Module,
        s: str,
        r: str,
        o: str,
        o_alt: str
    ) -> Tuple[float, float, str]:
        """
        Get logits for a triple from a model.

        Args:
            model: Model to use
            s: Subject
            r: Relation
            o: Original object
            o_alt: Alternative object

        Returns:
            Tuple of (logit_o, logit_o_alt, prediction)
        """
        input_text = f"{s} {r}"
        input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(self.device)

        # Get token IDs for objects
        o_token_id = self.tokenizer.encode(o)[0]
        o_alt_token_id = self.tokenizer.encode(o_alt)[0]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :]

            logit_o = logits[o_token_id].item()
            logit_o_alt = logits[o_alt_token_id].item()

            pred_id = torch.argmax(logits).item()
            pred = self.tokenizer.get_token(pred_id).strip()

        return logit_o, logit_o_alt, pred

    def analyze_all_triples(
        self,
        kg_path: str,
        edited_s: str,
        edited_r: str,
        edited_o_original: str,
        edited_o_target: str,
        max_triples: int = None
    ) -> List[TripleLogits]:
        """
        Analyze ripple effect on all triples in the knowledge graph.

        Args:
            kg_path: Path to KG corpus
            edited_s: Subject of edited triple
            edited_r: Relation of edited triple
            edited_o_original: Original object
            edited_o_target: Target object
            max_triples: Maximum number of triples to analyze (None for all)

        Returns:
            List of TripleLogits for all triples
        """
        # Load graph if not already loaded
        if self.graph is None:
            self.load_knowledge_graph(kg_path)

        results = []

        # Read all triples from corpus
        triples_seen = set()

        with open(kg_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_triples is not None and idx >= max_triples:
                    break

                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                s, r, o = parts[0], parts[1], parts[2]

                # Skip duplicates (corpus may have augmented versions)
                # Extract base entity/relation (before "__")
                s_base = s.split('__')[0]
                r_base = r.split('__')[0]
                o_base = o.split('__')[0]

                triple_key = (s_base, r_base, o_base)
                if triple_key in triples_seen:
                    continue
                triples_seen.add(triple_key)

                # Use base versions for analysis
                s, r, o = s_base, r_base, o_base

                # Compute logits before and after
                logit_o_before, logit_alt_before, pred_before = self.get_logits_for_triple(
                    self.model_before, s, r, o, edited_o_target
                )
                logit_o_after, logit_alt_after, pred_after = self.get_logits_for_triple(
                    self.model_after, s, r, o, edited_o_target
                )

                # Compute ripple effect
                # Definition: change in logit difference between o and o_alt
                logit_diff_before = logit_o_before - logit_alt_before
                logit_diff_after = logit_o_after - logit_alt_after
                ripple_effect = abs(logit_diff_after - logit_diff_before)

                # Compute hop distance
                hop_dist = self.compute_hop_distance(s, edited_s, edited_o_target)

                # Get degrees
                s_degree = self.entity_degrees.get(s, 0)
                o_degree = self.entity_degrees.get(o, 0)

                result = TripleLogits(
                    s=s,
                    r=r,
                    o=o,
                    o_alt=edited_o_target,
                    logit_o_before=logit_o_before,
                    logit_o_alt_before=logit_alt_before,
                    pred_before=pred_before,
                    logit_o_after=logit_o_after,
                    logit_o_alt_after=logit_alt_after,
                    pred_after=pred_after,
                    ripple_effect=ripple_effect,
                    hop_distance=hop_dist if hop_dist != float('inf') else -1,
                    subject_degree=s_degree,
                    object_degree=o_degree
                )

                results.append(result)

        return results

    def compute_statistics(self, results: List[TripleLogits]) -> Dict:
        """
        Compute statistics about ripple effects.

        Args:
            results: List of TripleLogits

        Returns:
            Dictionary of statistics
        """
        # Group by hop distance
        by_hop = defaultdict(list)
        by_degree = defaultdict(list)

        for result in results:
            if result.hop_distance >= 0:  # Exclude unreachable
                by_hop[result.hop_distance].append(result.ripple_effect)
            by_degree[result.subject_degree].append(result.ripple_effect)

        stats = {
            'total_triples': len(results),
            'by_hop_distance': {},
            'by_subject_degree': {}
        }

        # Statistics by hop
        for hop, effects in sorted(by_hop.items()):
            stats['by_hop_distance'][hop] = {
                'count': len(effects),
                'mean_ripple': float(np.mean(effects)),
                'std_ripple': float(np.std(effects)),
                'max_ripple': float(np.max(effects)),
                'min_ripple': float(np.min(effects))
            }

        # Statistics by degree
        for degree, effects in sorted(by_degree.items()):
            if len(effects) >= 3:  # Only include degrees with enough samples
                stats['by_subject_degree'][degree] = {
                    'count': len(effects),
                    'mean_ripple': float(np.mean(effects)),
                    'std_ripple': float(np.std(effects))
                }

        return stats
