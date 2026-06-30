#!/usr/bin/env python3
"""Scalable explanatory-IRT / logistic analysis of the plasticity matrix.

Built to scale to ~1536 respondents (millions of rows): sparse design matrix +
SAGA solver + L2 regularisation. The L2 penalty on the respondent / edit-testlet
fixed effects acts as shrinkage (an empirical-Bayes approximation of random
effects), which also fixes the CV degradation seen with unregularised FE.

Model (explanatory Rasch / LLTM-style):
    logit P(correct) = a_category + respondent_FE + edit_testlet_FE + rule_FE
                       + b1*victim_degree_z + b2*hop + b3*hop_missing
                       + (victim_degree_z x topology)        # BA vs ER vs ring
                       + (victim_degree_z x category)        # which axis

Outputs:
  - SQ2: full vs reduced — 3-fold CV log-loss + likelihood-ratio test
  - victim_degree slope overall and per-topology (is the degree effect BA-specific?)
  - victim_degree slope per category (locality/propagation/suppression)
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


def load(path):
    cols = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cols["y"].append(int(r["correct"]))
            cols["category"].append(r["category"])
            cols["respondent"].append(r["respondent_id"])
            cols["rule"].append(r["rule_type"])
            w = r["respondent_id"].split("__")[0]
            cols["world"].append(w)
            cols["topology"].append(w.split("_")[0])
            cols["edit_key"].append(f"{w}/{r['edit_id']}")
            try:
                vd = float(r["victim_degree"])
            except (ValueError, TypeError):
                vd = np.nan
            cols["victim_degree"].append(vd)
            try:
                hop = float(r["hop_from_edit"])
            except (ValueError, TypeError):
                hop = np.nan
            cols["hop"].append(hop)
    return {k: np.array(v) for k, v in cols.items()}


def zscore_within(values, groups):
    out = np.zeros(len(values))
    for g in np.unique(groups):
        m = groups == g
        v = values[m]
        good = ~np.isnan(v)
        mu, sd = v[good].mean(), v[good].std() + 1e-9
        out[m] = np.where(np.isnan(v), 0.0, (v - mu) / sd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/plasticity/responses_matrix.csv")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    d = load(args.csv)
    y = d["y"]
    n_resp = len(np.unique(d["respondent"]))
    print(f"rows={len(y)} respondents={n_resp} worlds={len(np.unique(d['world']))} "
          f"topologies={sorted(set(d['topology']))}")

    deg_z = zscore_within(d["victim_degree"], d["world"])
    hop = np.where(np.isnan(d["hop"]) | (d["hop"] < 0), 0.0, d["hop"])
    hop_missing = (np.isnan(d["hop"]) | (d["hop"] < 0)).astype(float)

    # categorical one-hot blocks (sparse)
    def ohe(col):
        return OneHotEncoder(drop="first", sparse_output=True, dtype=np.float64,
                             handle_unknown="ignore").fit_transform(col.reshape(-1, 1))
    cat_b = ohe(d["category"])
    resp_b = ohe(d["respondent"])
    edit_b = ohe(d["edit_key"])
    rule_b = ohe(d["rule"])
    X_reduced = sp.hstack([cat_b, resp_b, edit_b, rule_b]).tocsr()

    # structural dense + interactions
    topo = d["topology"]
    topo_levels = sorted(set(topo))
    inter_topo = np.column_stack([deg_z * (topo == t) for t in topo_levels])
    cat_levels = sorted(set(d["category"]))
    inter_cat = np.column_stack([deg_z * (d["category"] == c) for c in cat_levels])
    struct = np.column_stack([deg_z, hop, hop_missing, inter_topo, inter_cat])
    X_full = sp.hstack([X_reduced, sp.csr_matrix(struct)]).tocsr()

    def _clf():
        return LogisticRegression(solver="saga", C=args.C, max_iter=500)

    def fit(X):
        return _clf().fit(X, y)

    def cvll(X):
        return -cross_val_score(_clf(), X, y, cv=args.cv,
                                scoring="neg_log_loss").mean()

    def loglik(m, X):
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    print("fitting (saga, sparse)...")
    ll_full, ll_red = cvll(X_full), cvll(X_reduced)
    mf, mr = fit(X_full), fit(X_reduced)
    lr = 2 * (loglik(mf, X_full) - loglik(mr, X_reduced))
    df = X_full.shape[1] - X_reduced.shape[1]
    print("\n=== SQ2: structure adds predictive value? ===")
    print(f"  reduced CV log-loss: {ll_red:.4f}")
    print(f"  full    CV log-loss: {ll_full:.4f}  (improvement {ll_red - ll_full:+.4f})")
    print(f"  LRT chi2={lr:.1f} df={df} p={chi2.sf(lr, df):.2e}")

    # coefficient map for the dense structural tail
    names = (["victim_degree_z", "hop", "hop_missing"]
             + [f"deg_x_topo[{t}]" for t in topo_levels]
             + [f"deg_x_cat[{c}]" for c in cat_levels])
    coefs = mf.coef_[0][-struct.shape[1]:]
    print("\n=== structural coefficients (regularised) ===")
    for nm, cf in zip(names, coefs):
        print(f"  {nm:22s} {cf:+.4f}")
    print("\n[deg_x_topo] 負ほど高次数victimが困難。BA<ER/ringなら次数効果はBA寄り")


if __name__ == "__main__":
    main()
