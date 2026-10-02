#!/usr/bin/env python3
"""Visualize a plasticity response matrix (CSV from run_plasticity_eval.py).

Produces PNG figures used in the periodic progress summary:
  1. per-category accuracy
  2. edit success (direct) by edit-subject degree bin
  3. category accuracy by VICTIM degree bin  (structure -> plasticity signal)
  4. category accuracy vs hop distance from the edit
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAT_ORDER = ["direct", "logical", "contradicted", "invariant", "neighbor_invariant"]
CAT_COLORS = {
    "direct": "#1f77b4", "logical": "#ff7f0e", "contradicted": "#d62728",
    "invariant": "#2ca02c", "neighbor_invariant": "#17becf",
}


def load(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["correct"] = int(r["correct"])
            for k in ("victim_degree", "hop_from_edit", "edit_subject_degree"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
            rows.append(r)
    return rows


def acc(rows):
    n = len(rows)
    return (sum(r["correct"] for r in rows) / n, n) if n else (float("nan"), 0)


def tertile_bins(values):
    v = np.array([x for x in values if not np.isnan(x)])
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    def b(x):
        if np.isnan(x):
            return "na"
        return "low" if x <= q1 else ("mid" if x <= q2 else "high")
    return b, (q1, q2)


def fig_category_accuracy(rows, out):
    cats = [c for c in CAT_ORDER if any(r["category"] == c for r in rows)]
    accs, ns = [], []
    for c in cats:
        a, n = acc([r for r in rows if r["category"] == c])
        accs.append(a); ns.append(n)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(cats, accs, color=[CAT_COLORS.get(c, "gray") for c in cats])
    for b, a, n in zip(bars, accs, ns):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.2f}\n(n={n})",
                 ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 1.15); plt.ylabel("accuracy"); plt.title("Plasticity by category")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(out, dpi=130)
    plt.close()


def fig_direct_by_edit_degree(rows, out):
    bins = ["low", "mid", "high"]
    accs, ns = [], []
    for b in bins:
        a, n = acc([r for r in rows
                    if r["category"] == "direct" and r["edit_degree_bin"] == b])
        accs.append(a); ns.append(n)
    plt.figure(figsize=(5, 4))
    bars = plt.bar(bins, accs, color="#1f77b4")
    for bar, a, n in zip(bars, accs, ns):
        plt.text(bar.get_x() + bar.get_width() / 2, (a if a == a else 0) + 0.02,
                 f"{a:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 1.15); plt.ylabel("edit success"); plt.xlabel("edit-subject degree bin")
    plt.title("Edit success (direct) by editor degree")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def fig_category_by_victim_degree(rows, out):
    bfn, (q1, q2) = tertile_bins([r["victim_degree"] for r in rows])
    cats = [c for c in ["logical", "contradicted", "invariant", "neighbor_invariant"]
            if any(r["category"] == c for r in rows)]
    bins = ["low", "mid", "high"]
    width = 0.8 / max(1, len(cats))
    x = np.arange(len(bins))
    plt.figure(figsize=(8, 4.5))
    for i, c in enumerate(cats):
        accs = []
        for b in bins:
            a, _ = acc([r for r in rows
                        if r["category"] == c and bfn(r["victim_degree"]) == b])
            accs.append(a)
        plt.bar(x + i * width, accs, width, label=c, color=CAT_COLORS.get(c, "gray"))
    plt.xticks(x + width * (len(cats) - 1) / 2, bins)
    plt.ylim(0, 1.15); plt.ylabel("accuracy"); plt.xlabel("victim degree bin")
    plt.title(f"Category accuracy by VICTIM degree (tertiles q={q1:.0f},{q2:.0f})")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(out, dpi=130)
    plt.close()


def fig_accuracy_vs_hop(rows, out):
    cats = [c for c in ["logical", "contradicted", "invariant"]
            if any(r["category"] == c for r in rows)]
    plt.figure(figsize=(7, 4.5))
    for c in cats:
        by_hop = defaultdict(list)
        for r in rows:
            if r["category"] == c and not np.isnan(r["hop_from_edit"]) and r["hop_from_edit"] >= 0:
                by_hop[int(r["hop_from_edit"])].append(r["correct"])
        hops = sorted(by_hop)
        if not hops:
            continue
        accs = [np.mean(by_hop[h]) for h in hops]
        plt.plot(hops, accs, marker="o", label=c, color=CAT_COLORS.get(c, "gray"))
    plt.ylim(0, 1.05); plt.xlabel("hop distance from edit"); plt.ylabel("accuracy")
    plt.title("Accuracy vs hop distance"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    rows = load(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_category_accuracy(rows, out / "category_accuracy.png")
    fig_direct_by_edit_degree(rows, out / "direct_by_edit_degree.png")
    fig_category_by_victim_degree(rows, out / "category_by_victim_degree.png")
    fig_accuracy_vs_hop(rows, out / "accuracy_vs_hop.png")
    print(f"wrote 4 figures -> {out}")


if __name__ == "__main__":
    main()
