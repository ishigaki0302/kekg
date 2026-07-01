#!/usr/bin/env python3
"""Experimental-setup visualizations (data / models / evaluation).

  1. small graph structures per topology (BA hubs vs ER vs ring) — intuition
  2. world fact composition (R_F / R_Finv / R_C / generic)
  3. item-battery composition (avg # items per category per edit)
  4. eval accuracy by size across all 24 worlds (capacity: tiny is the outlier)
"""
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kg.symbolic_world import build_symbolic_world, R_F, R_FINV, R_C

OUT = Path("outputs/plasticity/figs/final"); OUT.mkdir(parents=True, exist_ok=True)
SIZES = ["tiny", "xs", "small", "small-wide", "base", "base-wide", "large", "xl"]


def fig_graphs():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    n = 70
    for ax, topo, title in zip(
            axes, ("ba", "er", "ring"),
            ("BA (scale-free): hubs", "ER (uniform): random", "ring (uniform): lattice")):
        if topo == "ba":
            g = nx.barabasi_albert_graph(n, 2, seed=1)
        elif topo == "er":
            g = nx.erdos_renyi_graph(n, 4.0 / (n - 1), seed=1)
        else:
            g = nx.watts_strogatz_graph(n, 4, 0.0, seed=1)
        deg = np.array([d for _, d in g.degree()])
        pos = nx.circular_layout(g) if topo == "ring" else nx.spring_layout(g, seed=1)
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=20 + 8 * deg,
                               node_color=deg, cmap="Reds", linewidths=0.3, edgecolors="k")
        ax.set_title(f"{title}\n(max deg {deg.max()})", fontsize=11); ax.axis("off")
    fig.suptitle("Degree substrate by topology (70-node illustration)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "setup_graphs.png", dpi=130); plt.close()


def fig_fact_composition():
    w = build_symbolic_world(seed=42, topology="ba", num_entities=1200,
                             num_generic_relations=50, target_generic_triples=24000, ba_m=6)
    facts = w.closure()
    cnt = defaultdict(int)
    for (_, r, _) in facts:
        key = r if r in (R_F, R_FINV, R_C) else "R_gen_* (generic)"
        cnt[key] += 1
    labels = [R_F, R_FINV, R_C, "R_gen_* (generic)"]
    vals = [cnt[k] for k in labels]
    plt.figure(figsize=(7, 4.2))
    bars = plt.bar([l.replace("R_gen_* (generic)", "generic\n(invariant)") for l in labels],
                   vals, color=["#c53030", "#dd6b20", "#d69e2e", "#2a4365"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("# facts"); plt.title(f"World fact composition (total {sum(vals):,} facts, 1200 entities)")
    plt.tight_layout(); plt.savefig(OUT / "setup_fact_composition.png", dpi=130); plt.close()


def fig_item_composition(matrix="outputs/plasticity/responses_matrix.csv"):
    # avg items per category per edit, from one respondent (rome, ba_s42__base)
    per_edit = defaultdict(lambda: defaultdict(int))
    with open(matrix, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["respondent_id"] == "ba_s42__base__rome":
                per_edit[r["edit_id"]][r["category"]] += 1
    cats = ["direct", "logical", "contradicted", "invariant", "neighbor_invariant"]
    avg = {c: np.mean([e.get(c, 0) for e in per_edit.values()]) for c in cats}
    plt.figure(figsize=(7, 4.2))
    bars = plt.bar(cats, [avg[c] for c in cats],
                   color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#17becf"])
    for b, c in zip(bars, cats):
        plt.text(b.get_x() + b.get_width() / 2, avg[c], f"{avg[c]:.1f}", ha="center", va="bottom")
    plt.ylabel("avg # items / edit"); plt.xticks(rotation=15, ha="right")
    plt.title("Item-battery composition per edit (closure-derived, with ground truth)")
    plt.tight_layout(); plt.savefig(OUT / "setup_item_composition.png", dpi=130); plt.close()


def fig_acc_by_size():
    accs = defaultdict(list)
    for f in glob.glob("outputs/respondents/models/*/train.log"):
        size = Path(f).parent.name.split("__")[1]
        m = re.findall(r"Best eval acc: ([0-9.]+)", open(f).read())
        if m:
            accs[size].append(float(m[-1]))
    plt.figure(figsize=(8.5, 4.4))
    for i, s in enumerate(SIZES):
        ys = accs.get(s, [])
        xs = np.random.default_rng(0).normal(i, 0.06, len(ys))
        plt.scatter(xs, ys, s=22, alpha=0.6,
                    color="#c53030" if s == "tiny" else "#2a4365")
    plt.xticks(range(len(SIZES)), SIZES, rotation=20, ha="right")
    plt.ylabel("final eval accuracy"); plt.ylim(0, 1.05); plt.grid(alpha=0.3)
    plt.title("Final eval accuracy by size (each dot = 1 of 24 worlds)\n"
              "7/8 sizes ~1.0; tiny (4L/128) under-capacity")
    plt.tight_layout(); plt.savefig(OUT / "setup_acc_by_size.png", dpi=130); plt.close()


if __name__ == "__main__":
    fig_graphs(); fig_fact_composition(); fig_item_composition(); fig_acc_by_size()
    print("wrote setup figures ->", OUT)
