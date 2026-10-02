#!/usr/bin/env python3
"""Final-result visualizations for the plasticity matrix.

  1. method x category accuracy heatmap (strength/locality landscape)
  2. victim_degree x topology effect (BA vs ER vs ring) from the scalable-IRT log
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATS = ["direct", "logical", "contradicted", "invariant", "neighbor_invariant"]
METHODS = ["ft", "ft_all", "rome", "memit", "pmet", "alphaedit", "grace", "kn", "mend"]


def heatmap(csv_path, out):
    acc = defaultdict(lambda: [0, 0])
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r["respondent_id"].split("__")[2]
            acc[(m, r["category"])][0] += int(r["correct"])
            acc[(m, r["category"])][1] += 1
    meths = [m for m in METHODS if (m, "direct") in acc]
    M = np.array([[acc[(m, c)][0] / max(1, acc[(m, c)][1]) for c in CATS] for m in meths])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(["direct\n(strength)", "logical\n(propag)", "contra\n(suppress)",
                        "invariant\n(locality)", "neighbor\n(locality)"], fontsize=9)
    ax.set_yticks(range(len(meths))); ax.set_yticklabels(meths, fontsize=10)
    for i in range(len(meths)):
        for j in range(len(CATS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("Editing plasticity: method x category accuracy (n=1728 respondents)")
    fig.colorbar(im, label="accuracy"); fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close()


def deg_topology(irt_log, out):
    txt = Path(irt_log).read_text()
    vals = {}
    for topo in ("ba", "er", "ring"):
        m = re.search(rf"deg_x_topo\[{topo}\]\s+([+-][\d.]+)", txt)
        if m:
            vals[topo] = float(m.group(1))
    if not vals:
        return False
    order = ["ba", "er", "ring"]
    labels = ["BA\n(scale-free)", "ER\n(uniform)", "ring\n(uniform)"]
    y = [vals.get(t, 0) for t in order]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#c53030" if v < -0.01 else "#888" for v in y]
    ax.bar(labels, y, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(y):
        ax.text(i, v - 0.002, f"{v:+.3f}", ha="center", va="top", fontsize=10)
    ax.set_ylabel("victim_degree x topology coef (logit)")
    ax.set_title("Degree effect is BA-specific\n(negative = high-degree victims harder)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close()
    return True


def icc_curves(csv_path, irt_log, out):
    """IRT-style item characteristic curves: P(correct) vs victim_degree_z,
    one logistic curve per topology (from the fitted deg + deg x topo coefs),
    centered at the overall mean P; empirical binned points overlaid."""
    txt = Path(irt_log).read_text()
    def g(pat):
        m = re.search(pat, txt); return float(m.group(1)) if m else None
    b_deg = g(r"victim_degree_z\s+([+-][\d.]+)")
    b_topo = {t: g(rf"deg_x_topo\[{t}\]\s+([+-][\d.]+)") for t in ("ba", "er", "ring")}
    if b_deg is None or any(v is None for v in b_topo.values()):
        return False
    # load minimal columns
    corr, deg, world, topo = [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                deg.append(float(r["victim_degree"]))
            except (ValueError, TypeError):
                continue
            corr.append(int(r["correct"]))
            w = r["respondent_id"].split("__")[0]
            world.append(w); topo.append(w.split("_")[0])
    corr = np.array(corr); deg = np.array(deg)
    world = np.array(world); topo = np.array(topo)
    # z-score degree within world
    dz = np.zeros(len(deg))
    for w in np.unique(world):
        m = world == w; dz[m] = (deg[m] - deg[m].mean()) / (deg[m].std() + 1e-9)
    mean_p = corr.mean(); c0 = np.log(mean_p / (1 - mean_p))
    sig = lambda z: 1 / (1 + np.exp(-z))
    x = np.linspace(-2.5, 2.5, 100)
    colors = {"ba": "#c53030", "er": "#2a4365", "ring": "#2f855a"}
    names = {"ba": "BA (scale-free)", "er": "ER (uniform)", "ring": "ring (uniform)"}
    plt.figure(figsize=(7.5, 5))
    for t in ("ba", "er", "ring"):
        y = sig(c0 + (b_deg + b_topo[t]) * x)
        plt.plot(x, y, color=colors[t], lw=2.2,
                 label=f"{names[t]} (slope {b_deg + b_topo[t]:+.3f})")
        # empirical binned points
        mt = topo == t
        bins = np.quantile(dz[mt], np.linspace(0, 1, 7))
        for i in range(6):
            sel = mt & (dz >= bins[i]) & (dz <= bins[i + 1])
            if sel.sum() > 50:
                plt.scatter((bins[i] + bins[i + 1]) / 2, corr[sel].mean(),
                            color=colors[t], s=18, alpha=0.6, zorder=3)
    plt.xlabel("victim degree (z-scored within world)")
    plt.ylabel("P(correct)")
    plt.title("Explanatory-IRT curves: P(correct) vs victim degree by topology\n"
              "(line=fitted logistic centered at mean P; dots=empirical bins)")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()
    return True


def item_characteristic_curves(csv_path, out):
    """Textbook IRT ICC: P(correct)=sigmoid(theta - b) as S-curves.

    Item difficulty b_c per category (b = -logit(mean P)); respondent ability
    theta_r = logit(mean P per respondent). Curves are genuinely S-shaped over a
    wide ability range; harder item types (logical/direct) sit to the right.
    """
    catp = defaultdict(lambda: [0, 0])
    respp = defaultdict(lambda: [0, 0])
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = int(r["correct"])
            catp[r["category"]][0] += c; catp[r["category"]][1] += 1
            respp[r["respondent_id"]][0] += c; respp[r["respondent_id"]][1] += 1

    def logit(p):
        p = min(max(p, 0.015), 0.985)
        return np.log(p / (1 - p))

    b = {c: -logit(v[0] / v[1]) for c, v in catp.items()}
    theta = np.array([logit(v[0] / v[1]) for v in respp.values()])
    order = sorted(b, key=lambda c: b[c])  # easy -> hard
    colors = {"invariant": "#2ca02c", "neighbor_invariant": "#17becf",
              "contradicted": "#d62728", "direct": "#1f77b4", "logical": "#ff7f0e"}
    x = np.linspace(-4.5, 4.5, 200)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for c in order:
        ax.plot(x, 1 / (1 + np.exp(-(x - b[c]))), lw=2.4, color=colors.get(c, "gray"),
                label=f"{c} (difficulty b={b[c]:+.2f})")
    # respondent ability distribution (twin axis)
    ax2 = ax.twinx()
    ax2.hist(theta, bins=30, color="gray", alpha=0.2)
    ax2.set_yticks([]); ax2.set_ylabel("respondent ability distribution", color="gray")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("respondent ability  θ  (logit)")
    ax.set_ylabel("P(correct)")
    ax.set_title("Item Characteristic Curves: item type x ability\n"
                 "(left = easy items, right = hard items; logical is hardest)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/plasticity/responses_matrix.csv")
    ap.add_argument("--irt-log", default="outputs/plasticity/irt/_irt_1728.log")
    ap.add_argument("--out-dir", default="outputs/plasticity/figs/final")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    heatmap(args.csv, out / "method_category_heatmap.png")
    log = args.irt_log
    if not deg_topology(log, out / "degree_by_topology.png"):
        log = "outputs/plasticity/irt/_irt_1536.log"
        deg_topology(log, out / "degree_by_topology.png")
    icc_curves(args.csv, log, out / "degree_effect_curves.png")
    item_characteristic_curves(args.csv, out / "icc.png")
    print(f"wrote figures -> {out}")


if __name__ == "__main__":
    main()
