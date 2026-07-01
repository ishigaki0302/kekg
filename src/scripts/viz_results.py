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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/plasticity/responses_matrix.csv")
    ap.add_argument("--irt-log", default="outputs/plasticity/irt/_irt_1728.log")
    ap.add_argument("--out-dir", default="outputs/plasticity/figs/final")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    heatmap(args.csv, out / "method_category_heatmap.png")
    ok = deg_topology(args.irt_log, out / "degree_by_topology.png")
    if not ok:  # fall back to the 1536 IRT log if 1728 not finished
        deg_topology("outputs/plasticity/irt/_irt_1536.log", out / "degree_by_topology.png")
    print(f"wrote figures -> {out}")


if __name__ == "__main__":
    main()
