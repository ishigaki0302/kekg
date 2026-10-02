#!/usr/bin/env python3
"""Knowledge-Plasticity-INDEX-keyed visualizations.

The end goal is to quantify KNOWLEDGE PLASTICITY, so these plots are organized by
the 4 plasticity indices themselves (NOT by raw IRT item category):

  (1) Strength    (強度)   = did the new fact take hold      -> direct items
                             (+ was the old object suppressed -> contradicted)
  (2) Stability   (安定度) = survives further edits           -> FUTURE (continuous editing)
  (3) Propagation (波及度) = ripples to logical consequences  -> logical items
  (4) Resilience  (復元力) = recovers after perturb / restore -> FUTURE (continuous editing)

Auxiliary (control, not one of the 4): Locality = unrelated facts preserved
  -> invariant / neighbor_invariant items.

Figures:
  1. plasticity_profile.png       : 4 index panels (by method); (2)(4) = future placeholders
  2. plasticity_vs_structure.png  : each MEASURABLE index vs victim degree, by topology
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "outputs/plasticity/responses_matrix.csv"
IRT_LOG = "outputs/plasticity/irt/_irt_final.log"
OUT = Path("outputs/plasticity/figs/final"); OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["ft", "ft_all", "rome", "memit", "pmet", "alphaedit", "grace", "kn", "mend"]
# index -> item category(ies) that operationalize it (single-shot measurable ones)
IDX_CAT = {"install": "direct", "suppress": "contradicted", "propagation": "logical"}


def load():
    """Single pass: method x category correct-tallies (for the by-method profile)."""
    mc = defaultdict(lambda: [0, 0])                 # (method,category) -> [correct, n]
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            method = r["respondent_id"].split("__")[2]
            mc[(method, r["category"])][0] += int(r["correct"])
            mc[(method, r["category"])][1] += 1
    return mc


def _acc(mc, method, cat):
    s, n = mc[(method, cat)]
    return (s / n if n else np.nan), n


def profile(mc):
    """2x2: one panel per plasticity index. (1)(3) measured by method; (2)(4) future."""
    meths = [m for m in METHODS if mc[(m, "direct")][1] > 0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) Strength: install (direct) + suppress-old (contradicted)
    ax = axes[0, 0]
    x = np.arange(len(meths)); w = 0.38
    inst = [_acc(mc, m, "direct")[0] for m in meths]
    supp = [_acc(mc, m, "contradicted")[0] for m in meths]
    ax.bar(x - w / 2, inst, w, label="install new (direct)", color="#1f77b4")
    ax.bar(x + w / 2, supp, w, label="suppress old (contradicted)", color="#d62728")
    ax.set_title("(1) Strength\n= new fact takes hold (+ old suppressed)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(meths, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (3) Propagation: logical
    ax = axes[1, 0]
    prop = [_acc(mc, m, "logical")[0] for m in meths]
    ax.bar(x, prop, color="#ff7f0e")
    ax.set_title("(3) Propagation\n= ripples to logical consequences (inverse/composition)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(meths, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y")

    # (2) Stability & (4) Resilience: future placeholders
    for ax, (num, name, jp, why) in zip(
            (axes[0, 1], axes[1, 1]),
            [("2", "Stability", "",
              "survives SUBSEQUENT edits\n(does an edit stay after later edits?)"),
             ("4", "Resilience", "",
              "recovers after perturbation / restore\n(re-edit back, measure recovery)")]):
        ax.text(0.5, 0.62, f"({num}) {name}", ha="center", va="center",
                fontsize=15, transform=ax.transAxes)
        ax.text(0.5, 0.42, why, ha="center", va="center", fontsize=11,
                color="#444", transform=ax.transAxes)
        ax.text(0.5, 0.18, "FUTURE — requires CONTINUOUS / sequential editing\n"
                "(index defined; single-shot pipeline ready; measurement pending)",
                ha="center", va="center", fontsize=10, color="#b7791f",
                bbox=dict(boxstyle="round", fc="#fffaf0", ec="#dd6b20"),
                transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle("Knowledge Plasticity profile by index  (n=1728 respondents; single-shot)",
                 fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "plasticity_profile.png", dpi=130); plt.close()


def _parse_irt(log):
    """Pull the regularised structural coefficients from the scalable-IRT log."""
    txt = Path(log).read_text()
    def g(pat):
        m = re.search(pat, txt); return float(m.group(1)) if m else None
    return dict(
        base=g(r"victim_degree_z\s+([+-][\d.]+)"),
        cat={c: g(rf"deg_x_cat\[{c}\]\s+([+-][\d.]+)")
             for c in ("direct", "logical", "contradicted", "invariant", "neighbor_invariant")},
        topo={t: g(rf"deg_x_topo\[{t}\]\s+([+-][\d.]+)") for t in ("ba", "er", "ring")},
        lrt_p=g(r"LRT chi2=[\d.]+ df=\d+ p=([\d.eE+-]+)"),
    )


def sensitivity(log=IRT_LOG):
    """Honest, index-keyed view of the structural effect: it is a CONDITIONAL
    (world/size/method-controlled) IRT estimate — marginally small (ceiling), but
    robust — so we plot the fitted coefficients, keyed to plasticity indices, not
    the raw marginal (which ceiling-effects wash out)."""
    c = _parse_irt(log)
    if c["base"] is None:
        print("  [sensitivity] no IRT log coefficients; skipping"); return
    # per-index degree slope = base + category interaction  (logit per +1 SD degree)
    idx = [("(1) Strength\n(install, direct)", "direct", "#1f77b4"),
           ("(1) Strength\n(suppress, contra)", "contradicted", "#d62728"),
           ("(3) Propagation\n(logical)", "logical", "#ff7f0e"),
           ("Locality*\n(invariant)", "invariant", "#2ca02c")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))

    ax = axes[0]
    labels = [l for l, _, _ in idx]
    slopes = [c["base"] + c["cat"][k] for _, k, _ in idx]
    cols = [col for _, _, col in idx]
    ax.bar(labels, slopes, color=cols)
    ax.axhline(0, color="k", lw=0.8)
    for i, v in enumerate(slopes):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="top" if v < 0 else "bottom", fontsize=10)
    ax.set_ylabel("degree sensitivity  (logit per +1 SD degree)")
    ax.set_title("Which plasticity index does entity degree hurt?\n"
                 "Strength(install) is ~10x more degree-sensitive than any other")
    ax.tick_params(axis="x", labelsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    order = ["ba", "er", "ring"]
    names = ["BA\n(scale-free)", "ER\n(uniform)", "ring\n(uniform)"]
    y = [c["topo"][t] for t in order]
    ax.bar(names, y, color=["#c53030" if v < -0.01 else "#999" for v in y])
    ax.axhline(0, color="k", lw=0.8)
    for i, v in enumerate(y):
        ax.text(i, v, f"{v:+.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=10)
    ax.set_ylabel("degree x topology  (logit)")
    ax.set_title("...and only in scale-free worlds\n"
                 "(degree penalty concentrates in BA)")
    ax.grid(alpha=0.3, axis="y")

    p = c["lrt_p"]
    fig.suptitle("Structure -> Plasticity (IRT conditional estimate; world/size/method controlled)."
                 f"  Effect is small marginally (ceiling) but robust: LRT p={p:.1e}", fontsize=12)
    fig.text(0.5, 0.005, "* Locality is a control, not one of the 4 plasticity indices.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "plasticity_vs_structure.png", dpi=130); plt.close()


if __name__ == "__main__":
    mc = load()
    profile(mc)
    sensitivity()
    print("wrote plasticity-index figures ->", OUT)
