#!/usr/bin/env python3
"""Symbolic world with logical closure for editing-plasticity measurement (P0).

Motivation
----------
The original ``generator.py`` assigns relations *randomly*, so the learned model
memorises a symbol table with no logical structure.  That makes "what should
change after an edit" undefinable except by heuristic neighbour sampling
(see ``docs/2026-06-22-critique.md`` §1.2 / §4.2).

This module builds a *logically structured* world so that the post-edit world
``G' = Closure(G ⊕ e)`` is well defined, and every evaluation query has a
ground-truth label.  This is the substrate for the explanatory-IRT measurement
of editing plasticity (``docs/2026-06-29-rq-and-methodology.md``).

Design (two layers)
-------------------
1. **Degree substrate** — a Barabási–Albert graph over the entities gives each
   entity a scale-free *structural degree* (the key covariate).  The undirected
   BA graph also defines hop distance between entities.
2. **Logical overlay** — a small, closure-complete set of typed relations:
   - ``R_F``   : functional. Each entity has exactly one ``R_F`` object. This is
                 the editable relation; overwriting it makes the old object
                 false (contradiction / suppression items).
   - ``R_Finv``: inverse of ``R_F``.  ``(s,R_F,o) <=> (o,R_Finv,s)``.
   - ``R_C``   : 2-hop composition of ``R_F``.
                 ``(s,R_F,x) & (x,R_F,o) => (s,R_C,o)``.
   - ``R_gen_k``: generic typed edges carrying NO logical rule. They populate
                 the *invariant* family (must not change under an edit) and add
                 realistic degree mass.

Closure rules (forward-chained to a fixpoint):
    (s, R_F, o)              => (o, R_Finv, s)
    (s, R_F, x) & (x, R_F, o)=> (s, R_C, o)

An edit ``(s, R_F, o_old -> o_new)`` updates the functional map and the closure
is recomputed; facts present in ``G`` but absent in ``G'`` are *contradicted*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
import json
import random

import networkx as nx
import numpy as np

from src.kg.generator import Triple, write_jsonl, write_txt

# --- relation names -----------------------------------------------------------
R_F = "R_F"  # functional, editable
R_FINV = "R_Finv"  # inverse of R_F
R_C = "R_C"  # 2-hop composition of R_F


def gen_relation(k: int) -> str:
    """Name of the k-th generic (rule-free, invariant-family) relation."""
    return f"R_gen_{k:03d}"


# --- logical categories of an evaluation item --------------------------------
CAT_DIRECT = "direct"
CAT_LOGICAL = "logical"  # newly entailed by closure
CAT_CONTRADICTED = "contradicted"  # true in G, false in G'
CAT_INVARIANT = "invariant"  # logically independent, must stay
CAT_NEIGHBOR_INVARIANT = "neighbor_invariant"  # invariant AND graph-adjacent to s


@dataclass(frozen=True)
class Item:
    """A single ground-truth evaluation query for one edit.

    The query is always "given (s, r), predict o"; ``gold_o`` is the correct
    answer in the post-edit world ``G'``.  ``covariates`` carries the structural
    quantities used as explanatory-IRT predictors.
    """

    s: str
    r: str
    gold_o: str
    category: str
    rule_type: str
    covariates: Dict[str, float] = field(default_factory=dict)
    # For suppression (contradicted) items: the object that must NOT be predicted
    # anymore. Empty for positive items. Positive items are scored "argmax ==
    # gold_o"; suppression items are scored "argmax != suppress_o".
    suppress_o: str = ""

    @property
    def is_suppression(self) -> bool:
        return self.category == CAT_CONTRADICTED

    def to_dict(self) -> dict:
        return {
            "s": self.s,
            "r": self.r,
            "gold_o": self.gold_o,
            "category": self.category,
            "rule_type": self.rule_type,
            "covariates": self.covariates,
            "suppress_o": self.suppress_o,
        }


# A fact is a (s, r, o) tuple of plain strings.
Fact = Tuple[str, str, str]


@dataclass
class SymbolicWorld:
    entities: List[str]
    func_map: Dict[str, str]  # R_F: entity -> its single object
    generic_triples: List[Triple]  # rule-free typed edges (invariant family)
    base_graph: nx.Graph  # undirected BA substrate (for degree + hop)

    # ------------------------------------------------------------------ core
    def degree(self, e: str) -> int:
        return self.base_graph.degree(e) if e in self.base_graph else 0

    def closure(self, func_map: Optional[Dict[str, str]] = None) -> Set[Fact]:
        """Forward-chain the logical rules to a fixpoint.

        Generic triples are included verbatim (no rules apply to them).
        """
        fm = self.func_map if func_map is None else func_map
        facts: Set[Fact] = set()
        # base functional facts + inverse
        for s, o in fm.items():
            facts.add((s, R_F, o))
            facts.add((o, R_FINV, s))
        # 2-hop composition of R_F
        for s, x in fm.items():
            o = fm.get(x)
            if o is not None:
                facts.add((s, R_C, o))
        # generic (rule-free) facts
        for t in self.generic_triples:
            facts.add((t.s, t.r, t.o))
        return facts

    # ------------------------------------------------------------------ edits
    def edited_func_map(self, s: str, o_new: str) -> Dict[str, str]:
        fm = dict(self.func_map)
        fm[s] = o_new
        return fm

    def hop(self, a: str, b: str) -> int:
        """Undirected hop distance on the BA substrate (-1 if unreachable)."""
        if a == b:
            return 0
        try:
            return nx.shortest_path_length(self.base_graph, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return -1

    def build_item_battery(
        self,
        s: str,
        o_new: str,
        max_invariant: int = 20,
        rng: Optional[random.Random] = None,
    ) -> List[Item]:
        """Items for the edit (s, R_F, o_old -> o_new).

        Categories:
          - direct            : (s, R_F) -> o_new
          - logical           : facts in G' \\ G entailed via inverse/composition
          - contradicted      : facts in G \\ G' (now false)
          - invariant         : generic facts that do not change
          - neighbor_invariant: invariant facts whose subject is BA-adjacent to s
        """
        rng = rng or random.Random(0)
        o_old = self.func_map[s]
        fm_new = self.edited_func_map(s, o_new)
        g = self.closure(self.func_map)
        gp = self.closure(fm_new)
        added = gp - g
        removed = g - gp

        items: List[Item] = []
        # (s, r) keys already used on the *positive* track, to avoid duplicates.
        positive_keys: Set[Tuple[str, str]] = set()

        def cov(victim: str) -> Dict[str, float]:
            return {
                "victim_degree": float(self.degree(victim)),
                "hop_from_edit": float(self.hop(s, victim)),
                "edit_subject_degree": float(self.degree(s)),
            }

        def add_positive(qs: str, qr: str, qo: str, cat: str, rule: str) -> None:
            if (qs, qr) in positive_keys:
                return
            positive_keys.add((qs, qr))
            items.append(Item(qs, qr, qo, cat, rule, cov(qs)))

        # direct -----------------------------------------------------------
        add_positive(s, R_F, o_new, CAT_DIRECT, R_F)

        # logical consequence (newly entailed positive facts) --------------
        for (qs, qr, qo) in sorted(added):
            add_positive(qs, qr, qo, CAT_LOGICAL, qr)

        # contradicted / suppression (true before, false now) --------------
        # These are scored "model must NOT predict the old object". gold_o is
        # the replacement in G' (if any) for reference; suppress_o is the old
        # object that must disappear. This is a distinct scoring track, so it
        # may share an (s, r) with a positive item without being a duplicate.
        for (qs, qr, qo_old) in sorted(removed):
            replacement = _lookup(gp, qs, qr) or ""
            items.append(
                Item(qs, qr, replacement, CAT_CONTRADICTED, qr, cov(qs), suppress_o=qo_old)
            )

        # invariant generic facts -----------------------------------------
        generic = list(self.generic_triples)
        rng.shuffle(generic)
        neighbors = set(self.base_graph.neighbors(s)) if s in self.base_graph else set()
        n_inv = 0
        n_nbr = 0
        for t in generic:
            if n_inv >= max_invariant and n_nbr >= max_invariant:
                break
            is_nbr = t.s in neighbors or t.o in neighbors
            cat = CAT_NEIGHBOR_INVARIANT if is_nbr else CAT_INVARIANT
            if cat == CAT_INVARIANT and n_inv >= max_invariant:
                continue
            if cat == CAT_NEIGHBOR_INVARIANT and n_nbr >= max_invariant:
                continue
            items.append(Item(t.s, t.r, t.o, cat, "generic", cov(t.s)))
            if cat == CAT_INVARIANT:
                n_inv += 1
            else:
                n_nbr += 1

        return items


def _lookup(facts: Set[Fact], s: str, r: str) -> Optional[str]:
    """First object for (s, r) in a fact set (functional relations are unique)."""
    for (fs, fr, fo) in facts:
        if fs == s and fr == r:
            return fo
    return None


# --- construction -------------------------------------------------------------
def build_symbolic_world(
    num_entities: int = 1200,
    num_generic_relations: int = 50,
    target_generic_triples: int = 24000,
    ba_m: int = 6,
    seed: int = 42,
    topology: str = "ba",
) -> SymbolicWorld:
    """Build a symbolic world with a degree substrate + logical overlay.

    - ``topology="ba"``: Barabasi-Albert (scale-free, heavy degree heterogeneity).
      ``topology="er"``: Erdos-Renyi with matched edge density (uniform/Poisson
      degree) -- the control that removes degree heterogeneity.
    - The substrate graph defines structural degree and hop distance.
    - ``R_F`` is a permutation entity -> entity (no fixed points).
    - Generic typed edges are sampled along substrate edges (preferential by
      degree), giving rule-free invariant facts and realistic degree mass.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    entities = [f"E_{i:04d}" for i in range(num_entities)]

    # 1) degree substrate (undirected)
    if topology == "ba":
        g = nx.barabasi_albert_graph(num_entities, ba_m, seed=seed)
    elif topology == "er":
        # match BA edge count (~n*ba_m) so density is comparable
        p = (2.0 * ba_m) / (num_entities - 1)
        g = nx.erdos_renyi_graph(num_entities, min(p, 1.0), seed=seed)
    elif topology == "ring":
        # ring lattice (Watts-Strogatz, no rewiring): uniform degree 2*ba_m with
        # cyclic/geometric structure (Nishi-style), vs ER's random uniform degree.
        g = nx.watts_strogatz_graph(num_entities, 2 * ba_m, 0.0, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    base_graph = nx.relabel_nodes(g, {i: entities[i] for i in range(num_entities)})

    # 2) functional map R_F as a *permutation* (bijection) with no fixed points.
    #    A bijection makes R_Finv (inverse permutation) and R_C (composition)
    #    functional as well, so the whole corpus is a deterministic map
    #    (s, r) -> o.  A random *function* would make R_Finv multi-valued and
    #    cap the model's achievable accuracy (the substrate must be learnable to
    #    ~100% so that "edited-before-correct" item selection is meaningful).
    perm = rng.permutation(num_entities)
    for i in range(num_entities):
        if perm[i] == i:  # remove fixed points (s -> s)
            swap = (i + 1) % num_entities
            perm[i], perm[swap] = perm[swap], perm[i]
    func_map: Dict[str, str] = {
        entities[i]: entities[int(perm[i])] for i in range(num_entities)
    }

    # 3) generic rule-free typed edges along BA edges (preferential sampling)
    generic_relations = [gen_relation(k) for k in range(num_generic_relations)]
    edges = list(base_graph.edges())
    generic: List[Triple] = []
    # Keep generic relations *functional*: each (s, r) maps to at most one o, so
    # the corpus stays a deterministic map and the model can reach ~100% acc.
    used_sr: Set[Tuple[str, str]] = set()
    for (u, v) in edges:
        # orient randomly, assign a random generic relation
        s, o = (u, v) if py_rng.random() < 0.5 else (v, u)
        r = generic_relations[int(rng.integers(0, num_generic_relations))]
        if (s, r) not in used_sr:
            generic.append(Triple(s=s, r=r, o=o))
            used_sr.add((s, r))

    # top up to the target by preferential (degree-proportional) sampling
    deg = np.array([base_graph.degree(e) for e in entities], dtype=np.float64) + 1.0
    prob = deg / deg.sum()
    max_trials = 5_000_000
    trials = 0
    while len(generic) < target_generic_triples and trials < max_trials:
        trials += 1
        u = int(rng.choice(num_entities, p=prob))
        v = int(rng.choice(num_entities, p=prob))
        if u == v:
            continue
        r = generic_relations[int(rng.integers(0, num_generic_relations))]
        if (entities[u], r) in used_sr:
            continue
        generic.append(Triple(s=entities[u], r=r, o=entities[v]))
        used_sr.add((entities[u], r))

    return SymbolicWorld(
        entities=entities,
        func_map=func_map,
        generic_triples=generic,
        base_graph=base_graph,
    )


def world_to_triples(world: SymbolicWorld) -> List[Triple]:
    """All training triples = closure facts (R_F, R_Finv, R_C) + generic."""
    triples: List[Triple] = []
    for (s, r, o) in sorted(world.closure()):
        triples.append(Triple(s=s, r=r, o=o))
    return triples


def degree_bins(world: SymbolicWorld, n_bins: int = 3) -> Dict[str, str]:
    """Assign each entity to a degree bin ('low'/'mid'/'high') by quantile."""
    degs = np.array([world.degree(e) for e in world.entities])
    qs = np.quantile(degs, np.linspace(0, 1, n_bins + 1)[1:-1])
    names = ["low", "mid", "high"][:n_bins] if n_bins == 3 else [str(i) for i in range(n_bins)]
    out: Dict[str, str] = {}
    for e in world.entities:
        b = int(np.searchsorted(qs, world.degree(e)))
        out[e] = names[min(b, len(names) - 1)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-entities", type=int, default=1200)
    ap.add_argument("--num-generic-relations", type=int, default=50)
    ap.add_argument("--target-generic-triples", type=int, default=24000)
    ap.add_argument("--ba-m", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topology", type=str, default="ba", choices=["ba", "er", "ring"])
    ap.add_argument("--out-dir", type=str, default="data/kg/symbolic")
    args = ap.parse_args()

    world = build_symbolic_world(
        num_entities=args.num_entities,
        num_generic_relations=args.num_generic_relations,
        target_generic_triples=args.target_generic_triples,
        ba_m=args.ba_m,
        seed=args.seed,
        topology=args.topology,
    )
    triples = world_to_triples(world)
    out = Path(args.out_dir)
    write_txt(triples, out / "corpus.train.txt")
    write_jsonl(triples, out / "graph.jsonl")
    # persist the functional map (needed to define edits/closure later)
    with (out / "func_map.json").open("w", encoding="utf-8") as f:
        json.dump(world.func_map, f, ensure_ascii=False)

    degs = np.array([world.degree(e) for e in world.entities])
    n_f = len(world.func_map)
    closure_facts = world.closure()
    print("=== Symbolic world generated ===")
    print(f"entities: {len(world.entities)}")
    print(f"R_F facts: {n_f} | closure facts (all relations): {len(closure_facts)}")
    print(f"generic triples: {len(world.generic_triples)}")
    print(f"degree: min={degs.min()} max={degs.max()} mean={degs.mean():.1f} std={degs.std():.1f}")
    print(f"out: {out}")


if __name__ == "__main__":
    main()
