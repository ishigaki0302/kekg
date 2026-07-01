#!/usr/bin/env python3
"""Post-hoc covariate enrichment for the plasticity matrix (NO re-editing / NO GPU).

Degree is only ONE structural scalar and it is confounded (in BA, high degree =
high corpus frequency = concentrated representation). To find the *real* driver
of editing plasticity we decorrelate several factors and let IRT compete them.

All new covariates are properties of (world, entity) or (world, edit), so we
simply JOIN them onto the existing responses_matrix.csv:

  A. centrality decomposition (per victim/editor entity, from world.base_graph):
       betweenness (bridge), clustering (local density), core number, pagerank
  C. entrenchment (per entity):
       corpus frequency  = # closure facts the entity appears in (train exposure)
  B. logical role (per edit):
       ripple fanout      = |Closure(G') symmetric-diff Closure(G)|  (functional)

Output: responses_matrix_enriched.csv  (original columns + the above, for
victim_* and editor_*, plus edit_fanout).
"""
import csv
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kg.symbolic_world import build_symbolic_world

IN = "outputs/plasticity/responses_matrix.csv"
OUT = "outputs/plasticity/responses_matrix_enriched.csv"
WKW = dict(num_entities=1200, num_generic_relations=50, target_generic_triples=24000, ba_m=6)
BETWEEN_K = 300           # sampled betweenness (exact is O(VE); k-sample is plenty for a covariate)

# new per-entity covariate names (degree already in the matrix)
ENT_COVS = ["betweenness", "clustering", "core", "pagerank", "frequency"]


def parse_world(token):
    """'ba_s42' -> (topology, seed)."""
    topo, seed = token.split("_")
    return topo, int(seed[1:])


def entity_covariates(world):
    """Per-entity structural + entrenchment covariates for one world."""
    g = world.base_graph
    btw = nx.betweenness_centrality(g, k=min(BETWEEN_K, g.number_of_nodes()), seed=0)
    clu = nx.clustering(g)
    core = nx.core_number(g)
    pr = nx.pagerank(g)
    # corpus frequency = how many closure facts each entity participates in
    freq = {e: 0 for e in world.entities}
    for (s, _, o) in world.closure(world.func_map):
        if s in freq:
            freq[s] += 1
        if o in freq:
            freq[o] += 1
    return {e: {"betweenness": btw.get(e, 0.0), "clustering": clu.get(e, 0.0),
                "core": float(core.get(e, 0)), "pagerank": pr.get(e, 0.0),
                "frequency": float(freq.get(e, 0))} for e in world.entities}


def edit_fanouts(world, edits):
    """Ripple fanout per edit = size of the functional closure change."""
    g = world.closure(world.func_map)           # generic part is identical across edits
    out = {}
    for (s, o_new) in edits:
        gp = world.closure(world.edited_func_map(s, o_new))
        out[(s, o_new)] = float(len(g ^ gp))    # symmetric diff (generic cancels)
    return out


def collect_world_keys(path):
    """First pass: which worlds, and which (world, edit) pairs, appear."""
    worlds, edits = set(), {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = r["respondent_id"].split("__")[0]
            worlds.add(w)
            edits.setdefault(w, set()).add((r["edit_s"], r["edit_o_new"]))
    return worlds, edits


def main():
    worlds, edits_by_world = collect_world_keys(IN)
    print(f"worlds={len(worlds)}  building per-world covariate tables...")
    ent_cov, fanout = {}, {}
    for w in sorted(worlds):
        topo, seed = parse_world(w)
        world = build_symbolic_world(seed=seed, topology=topo, **WKW)
        ent_cov[w] = entity_covariates(world)
        fanout[w] = edit_fanouts(world, edits_by_world[w])
        print(f"  {w}: {len(ent_cov[w])} entities, {len(fanout[w])} edits")

    new_cols = ([f"victim_{c}" for c in ENT_COVS]
                + [f"editor_{c}" for c in ENT_COVS] + ["edit_fanout"])
    with open(IN, encoding="utf-8") as fi, open(OUT, "w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        wr = csv.DictWriter(fo, fieldnames=rd.fieldnames + new_cols)
        wr.writeheader()
        n = 0
        for r in rd:
            w = r["respondent_id"].split("__")[0]
            vic = ent_cov[w].get(r["item_s"], {})
            edi = ent_cov[w].get(r["edit_s"], {})
            for c in ENT_COVS:
                r[f"victim_{c}"] = vic.get(c, "")
                r[f"editor_{c}"] = edi.get(c, "")
            r["edit_fanout"] = fanout[w].get((r["edit_s"], r["edit_o_new"]), "")
            wr.writerow(r)
            n += 1
            if n % 500000 == 0:
                print(f"  ...{n:,} rows")
    print(f"wrote {OUT}  (+{len(new_cols)} columns, {n:,} rows)")


if __name__ == "__main__":
    main()
