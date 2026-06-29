"""Exp 6: Cross-Layer Shattering Analysis.

Tests whether the degree-conditional shattering pattern (Exp 1b) holds
across different editing layers. If the pattern is robust, it's a property
of the model's knowledge storage, not an artifact of layer choice.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
from src.edit.rome import ROME
from src.utils.io import load_yaml

# Reuse utilities from exp1b
from src.scripts.exp1b_degree_shattering import (
    load_model_and_tokenizer, load_kg, bfs_hop,
    compute_logit_matrix, compute_shattering,
)


def run_cross_layer_shattering(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp6_cross_layer",
    num_edits=20,
    num_test_triples=2000,
    seed=42,
):
    print("=" * 60)
    print("Exp 6: Cross-Layer Shattering Analysis")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_orig, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, adjacency, degree = load_kg(kg_corpus)
    entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))
    subj_triples = defaultdict(list)
    for t in all_triples:
        subj_triples[t["s"]].append(t)

    # Test triples
    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]

    # Baseline
    print("Computing baseline...")
    D0 = compute_logit_matrix(model_orig, tokenizer, test_triples, device)

    # Select edit triples (fixed across layers)
    edit_triples_by_bin = {}
    for bin_name, (lo, hi) in [("low", (1, 15)), ("high", (35, 999))]:
        subjects = [e for e, d in degree.items() if lo <= d < hi]
        np.random.shuffle(subjects)
        edits = []
        for subj in subjects:
            if len(edits) >= num_edits:
                break
            trs = subj_triples.get(subj, [])
            if trs:
                t = trs[0]
                new_o = t["o"]
                while new_o == t["o"]:
                    new_o = entities[np.random.randint(len(entities))]
                edits.append({**t, "o_new": new_o})
        edit_triples_by_bin[bin_name] = edits
        print(f"  {bin_name}: {len(edits)} edit triples")

    # Test layers
    test_layers = [1, 3, 5, 7, 9, 11]
    results = {}

    for layer in test_layers:
        print(f"\n--- Editing Layer {layer} ---")
        layer_results = {}

        for bin_name, edits in edit_triples_by_bin.items():
            model = deepcopy(model_orig).to(device)
            model.eval()

            rome = ROME(
                model=model, tokenizer=tokenizer, device=device,
                kg_corpus_path=kg_corpus, mom2_n_samples=1000,
                use_mom2_adjustment=True, v_num_grad_steps=20,
            )

            successes = 0
            for edit in edits:
                model, result = rome.apply_edit(
                    s=edit["s"], r=edit["r"], o_target=edit["o_new"],
                    layer=layer, copy_model=False,
                )
                if result.success:
                    successes += 1

            D_star = compute_logit_matrix(model, tokenizer, test_triples, device)
            R_global = compute_shattering(D_star, D0)

            # Victim degree breakdown
            R_by_victim = {}
            for vd, (vlo, vhi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
                indices = [i for i, t in enumerate(test_triples) if vlo <= degree.get(t["s"], 0) < vhi]
                if len(indices) >= 5:
                    idx = torch.tensor(indices)
                    R_by_victim[vd] = compute_shattering(D_star[idx], D0[idx])

            layer_results[bin_name] = {
                "R_global": float(R_global),
                "R_by_victim": {k: float(v) for k, v in R_by_victim.items()},
                "edit_success_rate": successes / len(edits),
            }

            print(f"  {bin_name}: R={R_global:.4f}, success={successes}/{len(edits)}")
            for vd, rv in R_by_victim.items():
                print(f"    victim_{vd}: {rv:.4f}")

            del model, rome
            torch.cuda.empty_cache()

        results[str(layer)] = layer_results

    # Save
    with open(out_dir / "cross_layer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for bin_name, color in [("low", "blue"), ("high", "red")]:
        layers = sorted(results.keys(), key=int)
        R_globals = [results[l][bin_name]["R_global"] for l in layers]
        axes[0].plot([int(l) for l in layers], R_globals, "o-", color=color, label=f"editor={bin_name}")

    axes[0].set_xlabel("Edit Layer")
    axes[0].set_ylabel("R(D*)")
    axes[0].set_title("Global Shattering vs Edit Layer")
    axes[0].legend()

    # Victim degree ratio across layers
    for bin_name, color in [("low", "blue"), ("high", "red")]:
        layers = sorted(results.keys(), key=int)
        ratios = []
        for l in layers:
            R_low_v = results[l][bin_name]["R_by_victim"].get("low", 0)
            R_high_v = results[l][bin_name]["R_by_victim"].get("high", 0)
            ratios.append(R_low_v / (R_high_v + 1e-10))
        axes[1].plot([int(l) for l in layers], ratios, "o-", color=color, label=f"editor={bin_name}")

    axes[1].set_xlabel("Edit Layer")
    axes[1].set_ylabel("R(victim=low) / R(victim=high)")
    axes[1].set_title("Victim Vulnerability Ratio vs Edit Layer")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "cross_layer_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp6_cross_layer")
    parser.add_argument("--num-edits", type=int, default=20)
    parser.add_argument("--num-test-triples", type=int, default=2000)
    args = parser.parse_args()

    run_cross_layer_shattering(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
