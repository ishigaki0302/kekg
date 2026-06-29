#!/usr/bin/env python3
"""Explanatory Rasch / logistic analysis of the plasticity response matrix.

Model (explanatory Rasch / LLTM-style, fit as logistic regression):
    P(correct=1) = sigmoid( respondent_ability  -  item_difficulty )
    item_difficulty = linear in structural covariates
        (category, victim_degree_z, hop, rule_type)
Respondent ability enters as respondent fixed effects (dummies), so structural
coefficients are estimated *within respondent* (controls for editor competence).

Outputs:
  - coefficient table (with bootstrap 95% CIs) for the structural predictors
  - SQ2 model comparison: 5-fold CV log-loss, full (with structure) vs
    reduced (category + respondent only) -> do structural covariates add
    predictive value beyond category and who is editing?
  - a coefficient figure for the progress deck
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def load(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["correct"] = int(r["correct"])
            for k in ("victim_degree", "hop_from_edit", "edit_subject_degree"):
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = np.nan
            r["world"] = r["respondent_id"].split("__")[0]
            rows.append(r)
    return rows


def zscore_within_world(rows, key):
    by_world = defaultdict(list)
    for r in rows:
        if not np.isnan(r[key]):
            by_world[r["world"]].append(r[key])
    stats = {w: (np.mean(v), np.std(v) + 1e-9) for w, v in by_world.items()}
    for r in rows:
        m, s = stats[r["world"]]
        r[key + "_z"] = (r[key] - m) / s if not np.isnan(r[key]) else 0.0


def build_design(rows, categories, rules, respondents, use_structure=True):
    X, y = [], []
    for r in rows:
        feat = []
        # category one-hot (drop first as reference)
        for c in categories[1:]:
            feat.append(1.0 if r["category"] == c else 0.0)
        # respondent fixed effects (drop first)
        for rid in respondents[1:]:
            feat.append(1.0 if r["respondent_id"] == rid else 0.0)
        if use_structure:
            feat.append(r["victim_degree_z"])
            hop = r["hop_from_edit"]
            feat.append(hop if not np.isnan(hop) and hop >= 0 else 0.0)
            feat.append(1.0 if (np.isnan(r["hop_from_edit"]) or r["hop_from_edit"] < 0) else 0.0)
            for rl in rules[1:]:
                feat.append(1.0 if r["rule_type"] == rl else 0.0)
        X.append(feat)
        y.append(r["correct"])
    return np.array(X), np.array(y)


def per_category_degree_slope(rows, category, n_boot=300, seed=0):
    """Logistic slope of victim_degree_z within one category, bootstrap CI."""
    sub = [r for r in rows if r["category"] == category]
    if len({r["correct"] for r in sub}) < 2:
        return None
    rids = sorted({r["respondent_id"] for r in sub})
    X = np.array([[r["victim_degree_z"]] + [1.0 if r["respondent_id"] == x else 0.0
                                            for x in rids[1:]] for r in sub])
    y = np.array([r["correct"] for r in sub])
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        m = LogisticRegression(C=1e6, max_iter=2000)
        m.fit(X[idx], y[idx])
        coefs.append(m.coef_[0][0])
    if not coefs:
        return None
    return np.mean(coefs), np.percentile(coefs, 2.5), np.percentile(coefs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/plasticity/responses_matrix.csv")
    ap.add_argument("--out-dir", default="outputs/plasticity/irt")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = load(args.csv)
    print(f"loaded {len(rows)} responses, "
          f"{len({r['respondent_id'] for r in rows})} respondents, "
          f"{len({r['world'] for r in rows})} worlds")
    zscore_within_world(rows, "victim_degree")

    categories = sorted({r["category"] for r in rows})
    rules = sorted({r["rule_type"] for r in rows})
    respondents = sorted({r["respondent_id"] for r in rows})

    # SQ2: model comparison (full vs reduced) via CV log-loss
    Xf, y = build_design(rows, categories, rules, respondents, use_structure=True)
    Xr, _ = build_design(rows, categories, rules, respondents, use_structure=False)
    clf = LogisticRegression(C=1e6, max_iter=3000)
    ll_full = -cross_val_score(clf, Xf, y, cv=5, scoring="neg_log_loss").mean()
    ll_red = -cross_val_score(clf, Xr, y, cv=5, scoring="neg_log_loss").mean()
    print("\n=== SQ2: does structure add predictive value? (5-fold CV log-loss) ===")
    print(f"  reduced (category + respondent): {ll_red:.4f}")
    print(f"  full   (+ victim_degree, hop, rule): {ll_full:.4f}")
    print(f"  improvement: {ll_red - ll_full:+.4f} (positive = structure helps)")

    # per-category victim-degree slope with bootstrap CI (the central signal)
    print("\n=== victim_degree_z slope by category (logistic, within-respondent) ===")
    print("  (negative slope = higher-degree victims are HARDER for that category)")
    slope_rows = []
    for c in categories:
        res = per_category_degree_slope(rows, c)
        if res:
            m, lo, hi = res
            sig = "" if (lo <= 0 <= hi) else "  *"
            print(f"  {c:20s} slope={m:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]{sig}")
            slope_rows.append((c, m, lo, hi))

    # figure: per-category degree slope with CI
    if slope_rows:
        cs = [r[0] for r in slope_rows]
        ms = [r[1] for r in slope_rows]
        los = [r[1] - r[2] for r in slope_rows]
        his = [r[3] - r[1] for r in slope_rows]
        plt.figure(figsize=(7, 4))
        plt.errorbar(range(len(cs)), ms, yerr=[los, his], fmt="o", capsize=5, color="#c53030")
        plt.axhline(0, color="gray", ls="--", lw=1)
        plt.xticks(range(len(cs)), cs, rotation=20, ha="right")
        plt.ylabel("victim_degree slope (logit)")
        plt.title("Structure -> plasticity: victim-degree effect by category\n"
                  "(95% bootstrap CI; <0 = high-degree victims harder)")
        plt.tight_layout(); plt.savefig(out / "degree_slope_by_category.png", dpi=130)
        plt.close()
        print(f"\nwrote figure -> {out / 'degree_slope_by_category.png'}")


if __name__ == "__main__":
    main()
