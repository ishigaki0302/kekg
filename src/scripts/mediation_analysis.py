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
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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


def slope_ci(X, y, idx, n_boot=300, seed=0, max_iter=2000):
    rng = np.random.default_rng(seed)
    base = LogisticRegression(C=1e6, max_iter=max_iter).fit(X, y)
    coefs = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) < 2:
            continue
        coefs.append(LogisticRegression(C=1e6, max_iter=max_iter).fit(X[b], y[b]).coef_[0][idx])
    return base.coef_[0][idx], np.percentile(coefs, 2.5), np.percentile(coefs, 97.5)


def respondent_fe(respondent_ids):
    try:
        enc = OneHotEncoder(drop="first", sparse_output=True)
    except TypeError:
        enc = OneHotEncoder(drop="first", sparse=True)
    return enc.fit_transform(np.array(respondent_ids).reshape(-1, 1))


SUMMARY_FIELDS = [
    "category", "n", "n_boot", "max_rows_per_category",
    "degree_only", "degree_only_lo", "degree_only_hi",
    "degree_given_gini", "degree_given_gini_lo", "degree_given_gini_hi",
    "gini", "gini_lo", "gini_hi", "attenuation_pct",
]


def write_summary_csv(path, summaries):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_plot(path, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cats = [r["category"] for r in summaries]
    y = np.arange(len(cats))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ax = axes[0]
    deg_a = np.array([r["degree_only"] for r in summaries])
    deg_b = np.array([r["degree_given_gini"] for r in summaries])
    deg_a_err = np.array([
        deg_a - np.array([r["degree_only_lo"] for r in summaries]),
        np.array([r["degree_only_hi"] for r in summaries]) - deg_a,
    ])
    deg_b_err = np.array([
        deg_b - np.array([r["degree_given_gini_lo"] for r in summaries]),
        np.array([r["degree_given_gini_hi"] for r in summaries]) - deg_b,
    ])
    ax.errorbar(deg_a, y - 0.10, xerr=deg_a_err, fmt="o", label="degree only")
    ax.errorbar(deg_b, y + 0.10, xerr=deg_b_err, fmt="s", label="degree | gini")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y, cats)
    ax.set_xlabel("degree coefficient")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    g = np.array([r["gini"] for r in summaries])
    g_err = np.array([
        g - np.array([r["gini_lo"] for r in summaries]),
        np.array([r["gini_hi"] for r in summaries]) - g,
    ])
    ax.errorbar(g, y, xerr=g_err, fmt="o", color="#2f6f4e")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("gini coefficient | degree")
    ax.set_title("Representation concentration")
    fig.suptitle("Mediation diagnostic: degree -> gini -> correctness")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows-per-category", type=int, default=0,
                    help="0 keeps all rows; positive values run a deterministic diagnostic subsample")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--categories", nargs="+",
                    default=["invariant", "neighbor_invariant", "contradicted"])
    ap.add_argument("--out-csv", default="", help="optional path for a mediation summary CSV")
    ap.add_argument("--out-fig", default="", help="optional path for a mediation summary PNG")
    args = ap.parse_args()

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
    rng = np.random.default_rng(args.seed)
    print("=== Mediation: degree -> gini -> correctness (within respondent) ===", flush=True)
    print(f"rows={len(rows)} respondents={len(respondents)} n_boot={args.n_boot} "
          f"max_rows_per_category={args.max_rows_per_category}", flush=True)
    summaries = []
    for cat_i, cat in enumerate(args.categories):
        sub = [r for r in rows if r["category"] == cat]
        if args.max_rows_per_category and len(sub) > args.max_rows_per_category:
            keep = rng.choice(len(sub), size=args.max_rows_per_category, replace=False)
            sub = [sub[i] for i in sorted(keep)]
        if len({r["correct"] for r in sub}) < 2:
            continue
        print(f"\n[{cat}] n={len(sub)}", flush=True)
        y = np.array([r["correct"] for r in sub])
        fe = respondent_fe([r["respondent_id"] for r in sub])
        deg = sp.csr_matrix(np.array([r["deg_z"] for r in sub])[:, None])
        gini = sp.csr_matrix(np.array([r["gini_z"] for r in sub])[:, None])
        # (A) degree only
        XA = sp.hstack([deg, fe]).tocsr()
        cA, loA, hiA = slope_ci(XA, y, 0, n_boot=args.n_boot,
                                seed=args.seed + cat_i * 10 + 1, max_iter=args.max_iter)
        # (B) degree + gini
        XB = sp.hstack([deg, gini, fe]).tocsr()
        cB, loB, hiB = slope_ci(XB, y, 0, n_boot=args.n_boot,
                                seed=args.seed + cat_i * 10 + 2, max_iter=args.max_iter)
        gB, glo, ghi = slope_ci(XB, y, 1, n_boot=args.n_boot,
                                seed=args.seed + cat_i * 10 + 3, max_iter=args.max_iter)
        att = (1 - cB / cA) * 100 if cA != 0 else float("nan")
        print(f"  (A) degree slope          : {cA:+.3f} [{loA:+.3f},{hiA:+.3f}]")
        print(f"  (B) degree | gini         : {cB:+.3f} [{loB:+.3f},{hiB:+.3f}]  (attenuation {att:+.0f}%)")
        print(f"  (B) gini slope            : {gB:+.3f} [{glo:+.3f},{ghi:+.3f}]"
              f"{'  *' if not (glo <= 0 <= ghi) else ''}")
        summaries.append({
            "category": cat,
            "n": len(sub),
            "n_boot": args.n_boot,
            "max_rows_per_category": args.max_rows_per_category,
            "degree_only": cA,
            "degree_only_lo": loA,
            "degree_only_hi": hiA,
            "degree_given_gini": cB,
            "degree_given_gini_lo": loB,
            "degree_given_gini_hi": hiB,
            "gini": gB,
            "gini_lo": glo,
            "gini_hi": ghi,
            "attenuation_pct": att,
        })
    if args.out_csv:
        write_summary_csv(args.out_csv, summaries)
        print(f"\nwrote {args.out_csv}", flush=True)
    if args.out_fig:
        write_summary_plot(args.out_fig, summaries)
        print(f"wrote {args.out_fig}", flush=True)


if __name__ == "__main__":
    main()
