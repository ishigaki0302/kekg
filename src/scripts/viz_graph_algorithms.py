#!/usr/bin/env python3
"""Illustrate the three graph-generation algorithms (with figures).

  1. BA growth: preferential attachment, snapshots of the SAME graph growing
     (hubs emerge because new nodes attach to high-degree nodes)
  2. degree distributions: BA (power law, log-log) vs ER (Poisson) vs ring (delta)
  3. ring/ER construction schematics
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

OUT = Path("outputs/plasticity/figs/final"); OUT.mkdir(parents=True, exist_ok=True)


def build_ba(n, m, seed=1):
    """Barabasi-Albert by explicit preferential attachment; returns ordered edges."""
    rng = np.random.default_rng(seed)
    targets = list(range(m))            # initial m nodes
    repeated = list(range(m))           # each node repeated once per incident edge
    edges = []
    for new in range(m, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.integers(len(repeated))])  # pick prob ~ degree
        for t in chosen:
            edges.append((new, t))
            repeated += [new, t]        # both endpoints gain degree
    return edges


def fig_ba_growth():
    n, m = 60, 2
    edges = build_ba(n, m, seed=3)
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=3, k=0.35)
    snaps = [6, 15, 30, 60]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, sn in zip(axes, snaps):
        sub = G.subgraph(range(sn))
        deg = dict(sub.degree())
        nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(sub, pos, ax=ax, nodelist=list(range(sn)),
                               node_size=[15 + 10 * deg[i] for i in range(sn)],
                               node_color=[deg[i] for i in range(sn)], cmap="Reds",
                               edgecolors="k", linewidths=0.3)
        ax.set_title(f"n = {sn}   (max deg {max(deg.values())})", fontsize=11)
        ax.axis("off")
    fig.suptitle("BA: preferential attachment — each new node links to m=2 existing nodes "
                 "with prob ∝ degree → hubs grow", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "algo_ba_growth.png", dpi=130); plt.close()


def fig_degree_dists():
    n = 1200
    ba = nx.barabasi_albert_graph(n, 6, seed=42)
    er = nx.erdos_renyi_graph(n, 12 / (n - 1), seed=42)
    ring = nx.watts_strogatz_graph(n, 12, 0.0, seed=42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # BA: log-log power law
    d = np.array([x for _, x in ba.degree()]); vals, cnt = np.unique(d, return_counts=True)
    axes[0].loglog(vals, cnt, "o", color="#c53030")
    axes[0].set_title("BA: degree ~ power law\n(log-log ≈ straight line)")
    axes[0].set_xlabel("degree (log)"); axes[0].set_ylabel("# nodes (log)")
    # ER: Poisson
    d = np.array([x for _, x in er.degree()])
    axes[1].hist(d, bins=range(0, d.max() + 2), color="#2a4365")
    axes[1].set_title(f"ER: degree ~ Poisson\n(mean {d.mean():.0f}, std {d.std():.1f})")
    axes[1].set_xlabel("degree")
    # ring: delta
    d = np.array([x for _, x in ring.degree()])
    axes[2].hist(d, bins=range(0, d.max() + 3), color="#2f855a")
    axes[2].set_title(f"ring: degree = 2k (all equal)\n(all nodes degree {d[0]})")
    axes[2].set_xlabel("degree")
    fig.suptitle("Degree distributions produced by each algorithm (n=1200, mean degree ~12)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "algo_degree_dists.png", dpi=130); plt.close()


def fig_ring_er_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # ring lattice: each node to k nearest on a circle
    n = 24; k = 4
    g = nx.watts_strogatz_graph(n, k, 0.0, seed=1)
    pos = nx.circular_layout(g)
    nx.draw_networkx_edges(g, pos, ax=axes[0], alpha=0.5)
    nx.draw_networkx_nodes(g, pos, ax=axes[0], node_size=120, node_color="#2f855a",
                           edgecolors="k")
    axes[0].set_title(f"ring lattice: each node → {k//2} neighbors on each side\n"
                      "(regular, all degree = k)")
    axes[0].axis("off")
    # ER: independent p per pair
    g = nx.erdos_renyi_graph(n, 4 / (n - 1), seed=1)
    pos = nx.spring_layout(g, seed=1)
    nx.draw_networkx_edges(g, pos, ax=axes[1], alpha=0.4)
    nx.draw_networkx_nodes(g, pos, ax=axes[1], node_size=120, node_color="#2a4365",
                           edgecolors="k")
    axes[1].set_title("ER: connect every pair independently with prob p\n(random, Poisson degree)")
    axes[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "algo_ring_er.png", dpi=130); plt.close()


if __name__ == "__main__":
    fig_ba_growth(); fig_degree_dists(); fig_ring_er_schematic()
    print("wrote graph-algorithm figures ->", OUT)
