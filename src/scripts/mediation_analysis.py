#!/usr/bin/env python3
"""Mediation check: KG degree -> representation concentration -> editing difficulty.

Answers the circularity critique (DA#4): is the victim-degree effect on
locality (invariant preservation) explained by the *learned internal
representation* (L0 FFN concentration), rather than the generating graph alone?

For each category, compare within-respondent logistic models:
  (A) correct ~ victim_degree_z              (+ respondent FE)
  (B) correct ~ victim_degree_z + gini_z     (+ respondent FE)
If the degree coefficient shrinks from (A) to (B) and gini_z is significant,
the representation concentration mediates the degree effect.
"""
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REPR_DIR = Path("outputs/plasticity/repr")
MATRIX = "outputs/plasticity/responses_matrix.csv"


def load_repr():
    """(world__size) -> entity -> gini, with gini z-scored within model."""
    table = {}
    for fp in REPR_DIR.glob("*_repr.csv"):
        key = fp.name.replace("_repr.csv", "")  # world__size
        ent_g = {}
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            ent_g[r["entity"]] = float(r["gini"])
        g = np.array(list(ent_g.values()))
        mu, sd = g.mean(), g.std() + 1e-9
        table[key] = {e: (v - mu) / sd for e, v in ent_g.items()}
    return table


def slope_ci(X, y, idx, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    base = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
    coefs = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) < 2:
            continue
        coefs.append(LogisticRegression(C=1e6, max_iter=2000).fit(X[b], y[b]).coef_[0][idx])
    return base.coef_[0][idx], np.percentile(coefs, 2.5), np.percentile(coefs, 97.5)


def main():
    repr_tab = load_repr()
    rows = []
    for r in csv.DictReader(open(MATRIX, encoding="utf-8")):
        r["correct"] = int(r["correct"])
        r["victim_degree"] = float(r["victim_degree"])
        model_key = "__".join(r["respondent_id"].split("__")[:2])  # world__size
        gini = repr_tab.get(model_key, {}).get(r["item_s"])
        if gini is None:
            continue
        r["gini_z"] = gini
        r["model_key"] = model_key
        rows.append(r)

    # z-score victim_degree within model
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_key"]].append(r["victim_degree"])
    stats = {k: (np.mean(v), np.std(v) + 1e-9) for k, v in by_model.items()}
    for r in rows:
        mu, sd = stats[r["model_key"]]
        r["deg_z"] = (r["victim_degree"] - mu) / sd

    respondents = sorted({r["respondent_id"] for r in rows})
    print("=== Mediation: degree -> gini -> correctness (within respondent) ===")
    for cat in ["invariant", "neighbor_invariant", "contradicted"]:
        sub = [r for r in rows if r["category"] == cat]
        if len({r["correct"] for r in sub}) < 2:
            continue
        rids = sorted({r["respondent_id"] for r in sub})
        fe = lambda r: [1.0 if r["respondent_id"] == x else 0.0 for x in rids[1:]]
        y = np.array([r["correct"] for r in sub])
        # (A) degree only
        XA = np.array([[r["deg_z"]] + fe(r) for r in sub])
        cA, loA, hiA = slope_ci(XA, y, 0)
        # (B) degree + gini
        XB = np.array([[r["deg_z"], r["gini_z"]] + fe(r) for r in sub])
        cB, loB, hiB = slope_ci(XB, y, 0)
        gB, glo, ghi = slope_ci(XB, y, 1)
        att = (1 - cB / cA) * 100 if cA != 0 else float("nan")
        print(f"\n[{cat}]")
        print(f"  (A) degree slope          : {cA:+.3f} [{loA:+.3f},{hiA:+.3f}]")
        print(f"  (B) degree | gini         : {cB:+.3f} [{loB:+.3f},{hiB:+.3f}]  (attenuation {att:+.0f}%)")
        print(f"  (B) gini slope            : {gB:+.3f} [{glo:+.3f},{ghi:+.3f}]"
              f"{'  *' if not (glo <= 0 <= ghi) else ''}")


if __name__ == "__main__":
    main()
