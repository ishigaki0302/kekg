#!/usr/bin/env python3
"""Decorrelation gate: can we actually separate degree from the other factors?

If degree is ~collinear with frequency/centrality WITHIN a topology, no post-hoc
IRT can attribute the effect to one vs the other -> we'd need a world redesign
(e.g. frequency-controlled sampling). So measure it before fitting anything.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.scripts.enrich_covariates import (build_symbolic_world, entity_covariates,
                                           WKW, parse_world)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 42]
TOPOS = ["ba", "er", "ring"]
FACTORS = ["frequency", "betweenness", "clustering", "core", "pagerank"]
OUT = Path("outputs/plasticity/figs/final"); OUT.mkdir(parents=True, exist_ok=True)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:   # constant input (e.g. ring degree) -> undefined
        return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)


def factor_corrs():
    """Pool entities across seeds within each topology; corr(degree, factor)."""
    res = {}
    for topo in TOPOS:
        deg, cols = [], defaultdict(list)
        for seed in SEEDS:
            w = build_symbolic_world(seed=seed, topology=topo, **WKW)
            ec = entity_covariates(w)
            for e in w.entities:
                deg.append(w.degree(e))
                for f in FACTORS:
                    cols[f].append(ec[e][f])
        res[topo] = {f: (pearson(deg, cols[f]), spearman(deg, cols[f])) for f in FACTORS}
    return res


def fanout_stats(path="outputs/plasticity/responses_matrix_enriched.csv"):
    """Dedupe edits, report fanout distribution overall + by topology."""
    seen = set(); by_topo = defaultdict(list); allv = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["respondent_id"].split("__")[0], r["edit_s"], r["edit_o_new"])
            if key in seen:
                continue
            seen.add(key)
            try:
                v = float(r["edit_fanout"])
            except ValueError:
                continue
            topo = key[0].split("_")[0]
            by_topo[topo].append(v); allv.append(v)
    return allv, by_topo


def main():
    print("=== corr(degree, factor) within topology  [Pearson / Spearman] ===")
    res = factor_corrs()
    for topo in TOPOS:
        print(f"\n{topo}:")
        for f in FACTORS:
            p, s = res[topo][f]
            print(f"  degree~{f:12s}  r={p:+.3f}  rho={s:+.3f}")

    allv, by_topo = fanout_stats()
    av = np.array(allv)
    print(f"\n=== edit_fanout (n={len(av)} unique edits) ===")
    print(f"  overall: min {av.min():.0f}  median {np.median(av):.0f}  "
          f"max {av.max():.0f}  mean {av.mean():.2f}  std {av.std():.2f}")
    for topo in TOPOS:
        v = np.array(by_topo[topo])
        if len(v):
            print(f"  {topo:4s}: median {np.median(v):.0f}  mean {v.mean():.2f}  "
                  f"std {v.std():.2f}  range [{v.min():.0f},{v.max():.0f}]")

    # figure: corr heatmap (topology-blocked) + fanout hist
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    M = np.array([[res[topo][f][1] for f in FACTORS] for topo in TOPOS])  # Spearman
    im = ax0.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax0.set_xticks(range(len(FACTORS))); ax0.set_xticklabels(FACTORS, rotation=20, ha="right")
    ax0.set_yticks(range(len(TOPOS))); ax0.set_yticklabels([t.upper() for t in TOPOS])
    for i in range(len(TOPOS)):
        for j in range(len(FACTORS)):
            txt = "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:+.2f}"
            ax0.text(j, i, txt, ha="center", va="center", fontsize=10)
    ax0.set_title("Can we separate degree from each factor?\n"
                  "Spearman corr(degree, factor) within topology (|r|~1 = inseparable)")
    fig.colorbar(im, ax=ax0, label="rank corr")

    ax1.hist(av, bins=range(int(av.min()), int(av.max()) + 2), color="#dd6b20")
    ax1.set_title(f"Ripple fanout distribution\n(std {av.std():.2f} -> "
                  f"{'usable' if av.std() > 0.5 else 'nearly constant: B-axis weak'})")
    ax1.set_xlabel("edit_fanout = |closure change|"); ax1.set_ylabel("# edits")
    fig.tight_layout(); fig.savefig(OUT / "decorr_check.png", dpi=130); plt.close()
    print(f"\nwrote {OUT/'decorr_check.png'}")


if __name__ == "__main__":
    main()
