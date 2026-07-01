#!/usr/bin/env python3
"""Run editing-plasticity evaluation -> long-format response matrix (for IRT).

Example:
  uv run python src/scripts/run_plasticity_eval.py \
      --model-dir outputs/models/symbolic_main_seed42 \
      --config configs/train_symbolic_main.yaml \
      --corpus outputs/symbolic/main/corpus.train.txt \
      --func-map outputs/symbolic/main/func_map.json \
      --layer 5 --n-per-bin 5 \
      --out outputs/plasticity/responses_rome_L5.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.edit.rome import ROME
from src.edit.ft_edit import FTEditor
from src.eval.plasticity_eval import (
    load_respondent,
    rebuild_world,
    sample_edits,
    evaluate_edit,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", required=True, help="KG corpus for ROME mom2 stats")
    ap.add_argument("--func-map", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n-per-bin", type=int, default=5)
    ap.add_argument("--max-invariant", type=int, default=20)
    ap.add_argument("--edit-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    # world params (must match generation)
    ap.add_argument("--num-entities", type=int, default=1200)
    ap.add_argument("--num-generic-relations", type=int, default=50)
    ap.add_argument("--target-generic-triples", type=int, default=24000)
    ap.add_argument("--ba-m", type=int, default=6)
    ap.add_argument("--world-seed", type=int, default=42)
    ap.add_argument("--topology", default="ba", choices=["ba", "er", "ring"])
    ap.add_argument("--method", default="rome",
                    choices=["rome", "ft", "ft_all", "memit", "alphaedit",
                             "grace", "kn", "pmet", "mend", "ke"])
    ap.add_argument("--memit-layers", default="0,1,2,3,4", help="layers for MEMIT")
    ap.add_argument("--respondent-id", default="rome_seed42_L5")
    ap.add_argument(
        "--mom2-n-samples", type=int, default=-1,
        help="ROME second-moment (C) samples. -1 = use the FULL known corpus "
             "(exact C over all facts; the controlled-world rigor upgrade).",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # Exact C over the complete known knowledge set (vs Wikipedia estimate).
    corpus_size = sum(1 for ln in open(args.corpus, encoding="utf-8") if ln.strip())
    mom2_n = corpus_size if args.mom2_n_samples < 0 else args.mom2_n_samples
    print(f"corpus facts={corpus_size} | mom2_n_samples={mom2_n} "
          f"({'EXACT full-corpus C' if mom2_n >= corpus_size else 'sampled C'})")

    model, tok = load_respondent(args.model_dir, args.config, device)
    print("respondent loaded")

    world = rebuild_world(
        dict(
            num_entities=args.num_entities,
            num_generic_relations=args.num_generic_relations,
            target_generic_triples=args.target_generic_triples,
            ba_m=args.ba_m,
            seed=args.world_seed,
            topology=args.topology,
        ),
        func_map_path=args.func_map,
    )
    print("world rebuilt (func_map matches saved)")

    # unique stats cache per respondent (world__size__method) — avoids both
    # cross-model d_mlp collision AND concurrent-write races on the C .npz when
    # multiple C-using methods of the same model run in parallel.
    stats_name = args.respondent_id
    edit_layers = None
    if args.method in ("rome", "memit", "pmet"):
        # PMET-style = multi-layer FFN edit with more v-optimisation steps
        # (adaptation; close to MEMIT in this symbolic setting).
        v_steps = 50 if args.method == "pmet" else 20
        editor = ROME(model, tok, device=device, kg_corpus_path=args.corpus,
                      mom2_n_samples=mom2_n, stats_name=stats_name,
                      v_num_grad_steps=v_steps)
        if args.method == "memit":
            edit_layers = [int(x) for x in args.memit_layers.split(",")]
        elif args.method == "pmet":
            edit_layers = [0, 1, 2]
    elif args.method == "kn":
        from src.edit.kn_edit import KNEditor
        editor = KNEditor(model, tok, device=device, default_layer=args.layer)
    elif args.method == "alphaedit":
        from src.edit.alpha_edit import AlphaEditEditor
        editor = AlphaEditEditor(model, tok, device=device, kg_corpus_path=args.corpus,
                                 default_layer=args.layer, stats_name=stats_name,
                                 mom2_n_samples=mom2_n)
    elif args.method == "grace":
        from src.edit.grace_edit import GRACEEditor
        editor = GRACEEditor(model, tok, device=device, default_layer=args.layer)
    elif args.method == "ft_all":
        editor = FTEditor(model, tok, device=device, default_layer=args.layer, scope="all")
    elif args.method in ("mend", "ke"):
        # learned editors: load the pre-trained per-model editor weights
        # editor is keyed by (world, size), NOT stats_name (which now includes method)
        import os
        model_key = "__".join(args.respondent_id.split("__")[:2])
        ed_path = f"outputs/respondents/editors/{model_key}__{args.method}.pt"
        if not os.path.exists(ed_path):
            raise SystemExit(f"editor not trained: {ed_path} (run train_editors.py first)")
        if args.method == "mend":
            from src.edit.mend_edit import MENDEditor
            editor = MENDEditor(model, tok, device=device, default_layer=args.layer)
        else:
            from src.edit.ke_edit import KEEditor
            editor = KEEditor(model, tok, device=device, default_layer=args.layer)
        editor.load(ed_path)
    else:
        editor = FTEditor(model, tok, device=device, default_layer=args.layer)
    if edit_layers is not None:  # clamp to model depth (tiny/xs have few layers)
        edit_layers = [L for L in edit_layers if L < model.config.n_layers]
    print(f"editor: {args.method} (edit_layers={edit_layers})")

    plans = sample_edits(world, n_per_bin=args.n_per_bin, seed=args.edit_seed)
    print(f"edits planned: {len(plans)} ({args.n_per_bin}/bin)")

    all_rows = []
    n_success = 0
    edit_times = []
    for i, plan in enumerate(plans):
        rows, ok, dt = evaluate_edit(
            editor, tok, world, plan, layer=args.layer, device=device,
            max_invariant=args.max_invariant, item_rng_seed=args.edit_seed + i,
            edit_layers=edit_layers,
        )
        for row in rows:
            row["respondent_id"] = args.respondent_id
            row["edit_id"] = f"e{i:04d}"
        all_rows.extend(rows)
        n_success += int(ok)
        edit_times.append(dt)
        print(f"[{i+1}/{len(plans)}] s={plan.s} bin={plan.degree_bin} "
              f"edit_success={ok} items={len(rows)} t={dt:.2f}s")

    # Efficiency summary (separate file -> does not change the matrix schema)
    import statistics
    eff_dir = Path(args.out).parent.parent / "efficiency"
    eff_dir.mkdir(parents=True, exist_ok=True)
    with (eff_dir / f"{args.respondent_id}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["respondent_id", "method", "n_edits", "mean_edit_s", "median_edit_s", "total_edit_s"])
        w.writerow([args.respondent_id, args.method, len(edit_times),
                    f"{statistics.mean(edit_times):.4f}",
                    f"{statistics.median(edit_times):.4f}",
                    f"{sum(edit_times):.2f}"])

    # write CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "respondent_id", "edit_id", "edit_s", "edit_o_new", "edit_degree_bin",
        "edit_success", "item_s", "item_r", "category", "rule_type",
        "victim_degree", "hop_from_edit", "edit_subject_degree", "correct",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # summary
    print("\n=== SUMMARY ===")
    print(f"edits: {len(plans)} | edit_success_rate: {n_success/max(1,len(plans)):.3f}")
    by_cat = defaultdict(lambda: [0, 0])
    for r in all_rows:
        by_cat[r["category"]][0] += r["correct"]
        by_cat[r["category"]][1] += 1
    print("per-category accuracy (correct/total):")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"  {cat:20s} {c/n:.3f}  ({c}/{n})")
    # direct success by degree bin
    by_bin = defaultdict(lambda: [0, 0])
    for r in all_rows:
        if r["category"] == "direct":
            by_bin[r["edit_degree_bin"]][0] += r["correct"]
            by_bin[r["edit_degree_bin"]][1] += 1
    print("direct (edit) success by edit-degree bin:")
    for b, (c, n) in sorted(by_bin.items()):
        print(f"  {b:6s} {c/n:.3f}  ({c}/{n})")
    print(f"\nwrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
