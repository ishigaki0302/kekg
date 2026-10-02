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
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

ENRICHED_NUMERIC_FIELDS = [
    "victim_betweenness",
    "victim_clustering",
    "victim_core",
    "victim_pagerank",
    "victim_frequency",
    "editor_betweenness",
    "editor_clustering",
    "editor_core",
    "editor_pagerank",
    "editor_frequency",
    "edit_fanout",
]


def load(path):
    cols = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cols["y"].append(int(r["correct"]))
            cols["category"].append(r["category"])
            cols["respondent"].append(r["respondent_id"])
            cols["rule"].append(r["rule_type"])
            parts = r["respondent_id"].split("__")
            w = parts[0]
            cols["world"].append(w)
            cols["size"].append(parts[1])
            cols["method"].append(parts[2])
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
            for field in ENRICHED_NUMERIC_FIELDS:
                try:
                    v = float(r.get(field, ""))
                except (ValueError, TypeError):
                    v = np.nan
                cols[field].append(v)
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
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--solver", choices=["saga", "lbfgs"], default="saga")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--edit-fe", type=int, default=1,
                    help="1=include edit-testlet FE (slow), 0=drop for convergence")
    ap.add_argument("--subsample", type=int, default=0,
                    help="fit on a random subsample of rows for speed (0=all)")
    ap.add_argument("--respondent-fe", type=int, default=1,
                    help="1=respondent FE (1728, slow), 0=world+size+method FE (fast)")
    ap.add_argument("--enriched-covariates", default="victim_clustering",
                    help=("comma-separated enriched covariates to add with category "
                          "interactions; use 'none' for base degree/hop only"))
    ap.add_argument("--label", default="", help="label for optional --out-csv row")
    ap.add_argument("--out-csv", default="",
                    help="optional CSV path to append one summary row for this fit")
    args = ap.parse_args()

    d = load(args.csv)
    if args.subsample and len(d["y"]) > args.subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(d["y"]), args.subsample, replace=False)
        d = {k: v[idx] for k, v in d.items()}
        print(f"subsampled to {args.subsample} rows")
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
    rule_b = ohe(d["rule"])
    if args.respondent_fe:
        ctrl = [ohe(d["respondent"])]           # 1728 dummies (slow)
    else:
        ctrl = [ohe(d["world"]), ohe(d["size"]), ohe(d["method"])]  # ~40 dummies (fast)
    blocks = [cat_b] + ctrl + [rule_b]
    if args.edit_fe:
        blocks.append(ohe(d["edit_key"]))       # edit-testlet FE (slow)
    X_reduced = sp.hstack(blocks).tocsr()

    # structural dense + interactions
    topo = d["topology"]
    topo_levels = sorted(set(topo))
    inter_topo = np.column_stack([deg_z * (topo == t) for t in topo_levels])
    cat_levels = sorted(set(d["category"]))
    inter_cat = np.column_stack([deg_z * (d["category"] == c) for c in cat_levels])
    struct = np.column_stack([deg_z, hop, hop_missing, inter_topo, inter_cat])
    struct_names = (["victim_degree_z", "hop", "hop_missing"]
                    + [f"deg_x_topo[{t}]" for t in topo_levels]
                    + [f"deg_x_cat[{c}]" for c in cat_levels])
    covariates = [
        c.strip() for c in args.enriched_covariates.split(",")
        if c.strip() and c.strip().lower() != "none"
    ]
    unknown = sorted(set(covariates) - set(ENRICHED_NUMERIC_FIELDS))
    if unknown:
        raise SystemExit(f"unknown enriched covariate(s): {', '.join(unknown)}")
    used_covariates = []
    for cov in covariates:
        if cov not in d or not np.isfinite(d[cov]).any():
            print(f"warning: skipping unavailable covariate {cov}")
            continue
        cov_z = zscore_within(d[cov], d["world"])
        inter_cov = np.column_stack([cov_z * (d["category"] == c) for c in cat_levels])
        short = cov.removeprefix("victim_")
        struct = np.column_stack([struct, cov_z, inter_cov])
        struct_names += [f"{short}_z"] + [f"{short}_x_cat[{c}]" for c in cat_levels]
        used_covariates.append(cov)
    X_full = sp.hstack([X_reduced, sp.csr_matrix(struct)]).tocsr()
    print(f"features: reduced={X_reduced.shape[1]} full={X_full.shape[1]} "
          f"struct={struct.shape[1]} solver={args.solver} C={args.C} "
          f"tol={args.tol} max_iter={args.max_iter}")
    print(f"enriched_covariates={used_covariates if used_covariates else 'none'}")

    def _clf():
        return LogisticRegression(
            solver=args.solver,
            C=args.C,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.random_state,
        )

    def fit(X, label):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = _clf().fit(X, y)
        conv_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        n_iter = int(np.max(model.n_iter_))
        status = "converged" if not conv_warnings and n_iter < args.max_iter else "not_converged"
        print(f"  fit[{label}]: {status} n_iter={n_iter}/{args.max_iter}")
        if conv_warnings:
            print(f"  fit[{label}] warning: {conv_warnings[-1].message}")
        return model

    def cvll(X):
        return -cross_val_score(_clf(), X, y, cv=args.cv,
                                scoring="neg_log_loss").mean()

    def loglik(m, X):
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    print("fitting (sparse logistic)...")
    ll_full, ll_red = cvll(X_full), cvll(X_reduced)
    mf, mr = fit(X_full, "full"), fit(X_reduced, "reduced")
    lr = 2 * (loglik(mf, X_full) - loglik(mr, X_reduced))
    df = X_full.shape[1] - X_reduced.shape[1]
    p_value = chi2.sf(lr, df)
    improvement = ll_red - ll_full
    print("\n=== SQ2: structure adds predictive value? ===")
    print(f"  reduced CV log-loss: {ll_red:.4f}")
    print(f"  full    CV log-loss: {ll_full:.4f}  (improvement {improvement:+.4f})")
    print(f"  LRT chi2={lr:.1f} df={df} p={p_value:.2e}")

    coefs = mf.coef_[0][-struct.shape[1]:]
    print("\n=== structural coefficients (regularised) ===")
    for nm, cf in zip(struct_names, coefs):
        print(f"  {nm:22s} {cf:+.4f}")
    print("\n[deg_x_topo] 負ほど高次数victimが困難。BA<ER/ringなら次数効果はBA寄り")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        coef_map = dict(zip(struct_names, coefs))
        fields = [
            "label", "csv", "rows", "respondents", "subsample", "solver", "C", "tol",
            "max_iter", "edit_fe", "respondent_fe", "cv", "enriched_covariates",
            "reduced_features", "full_features", "struct_features",
            "reduced_cv_log_loss", "full_cv_log_loss", "cv_improvement",
            "lrt_chi2", "lrt_df", "lrt_p", "full_n_iter", "reduced_n_iter",
            "victim_degree_z", "deg_x_topo[ba]", "deg_x_topo[er]", "deg_x_topo[ring]",
            "deg_x_cat[direct]",
        ]
        row = {
            "label": args.label or ("+".join(used_covariates) if used_covariates else "base"),
            "csv": args.csv,
            "rows": len(y),
            "respondents": n_resp,
            "subsample": args.subsample,
            "solver": args.solver,
            "C": args.C,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "edit_fe": args.edit_fe,
            "respondent_fe": args.respondent_fe,
            "cv": args.cv,
            "enriched_covariates": ";".join(used_covariates),
            "reduced_features": X_reduced.shape[1],
            "full_features": X_full.shape[1],
            "struct_features": struct.shape[1],
            "reduced_cv_log_loss": ll_red,
            "full_cv_log_loss": ll_full,
            "cv_improvement": improvement,
            "lrt_chi2": lr,
            "lrt_df": df,
            "lrt_p": p_value,
            "full_n_iter": int(np.max(mf.n_iter_)),
            "reduced_n_iter": int(np.max(mr.n_iter_)),
        }
        for name in fields:
            if name not in row:
                row[name] = coef_map.get(name, "")
        write_header = not out.exists() or out.stat().st_size == 0
        with out.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nwrote summary row to {out}")


if __name__ == "__main__":
    main()
