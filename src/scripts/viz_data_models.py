#!/usr/bin/env python3
"""Visualize the training DATA (symbolic worlds) and MODELS (respondents).

  1. degree distribution per topology (BA heavy-tail vs ER/ring uniform)
  2. model sizes (parameter count) x final eval accuracy
  3. learning curves (train accuracy vs epoch) for several sizes
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kg.symbolic_world import build_symbolic_world

OUT = Path("outputs/plasticity/figs/final")
OUT.mkdir(parents=True, exist_ok=True)
WKW = dict(num_entities=1200, num_generic_relations=50, target_generic_triples=24000, ba_m=6)
SIZES = ["tiny", "xs", "small", "small-wide", "base", "base-wide", "large", "xl"]


def fig_degree():
    colors = {"ba": "#c53030", "er": "#2a4365", "ring": "#2f855a"}
    names = {"ba": "BA (scale-free)", "er": "ER (uniform)", "ring": "ring (uniform)"}
    plt.figure(figsize=(7.5, 4.8))
    for topo in ("ba", "er", "ring"):
        w = build_symbolic_world(seed=42, topology=topo, **WKW)
        degs = np.array([w.degree(e) for e in w.entities])
        plt.hist(degs, bins=np.arange(0, degs.max() + 3), histtype="step", lw=2.2,
                 color=colors[topo], label=f"{names[topo]}  (mean {degs.mean():.0f}, std {degs.std():.0f})")
    plt.xlabel("entity degree"); plt.ylabel("# entities (of 1200)")
    plt.title("Training data: degree distribution by topology\n"
              "BA has hubs (heavy tail); ER/ring are uniform")
    plt.legend(fontsize=9); plt.xlim(0, 60); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "data_degree_distribution.png", dpi=130); plt.close()


def _params(size):
    sd = torch.load(f"outputs/respondents/models/ba_s42__{size}/model.pt", map_location="cpu")
    return sum(v.numel() for v in sd.values())


def _final_acc(size):
    try:
        rep = yaml.safe_load(open(f"outputs/respondents/models/ba_s42__{size}/train_report.yaml"))
        return rep.get("best_eval_acc", rep.get("final_eval_acc"))
    except Exception:
        return None


def fig_sizes():
    params = [_params(s) / 1e6 for s in SIZES]
    accs = [_final_acc(s) for s in SIZES]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(SIZES, params, color="#2a4365")
    for b, p, a in zip(bars, params, accs):
        ax.text(b.get_x() + b.get_width() / 2, p, f"{p:.1f}M\nacc {a:.3f}" if a else f"{p:.1f}M",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("parameters (millions)")
    ax.set_title("Respondent models: 8 sizes (depth x width)\n"
                 "7/8 sizes reach ~100%; tiny (4L/128) under-capacity (~0.6)")
    ax.set_ylim(0, max(params) * 1.2)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(OUT / "model_sizes.png", dpi=130); plt.close()


def fig_learning():
    plt.figure(figsize=(7.5, 4.8))
    show = ["tiny", "small", "base", "large", "xl"]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
    for s, c in zip(show, cmap):
        fp = f"outputs/respondents/models/ba_s42__{s}/metrics.csv"
        steps, acc = [], []
        for r in csv.DictReader(open(fp)):
            if r["acc"]:
                steps.append(int(r["step"])); acc.append(float(r["acc"]))
        plt.plot(steps, acc, color=c, lw=1.8, label=s)
    plt.xlabel("training step"); plt.ylabel("train accuracy")
    plt.title("Learning curves on the symbolic world\n"
              "large-enough sizes converge to ~100%; tiny (4L/128) plateaus ~0.6")
    plt.ylim(0, 1.02); plt.legend(fontsize=9, title="size"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "model_learning_curves.png", dpi=130); plt.close()


if __name__ == "__main__":
    fig_degree(); fig_sizes(); fig_learning()
    print("wrote data/model figures ->", OUT)
