"""Exp 13: Multi-Layer vs Single-Layer Editing.

Compares editing at a single layer (L5) vs editing at multiple layers
simultaneously, as done by MEMIT. Tests whether spreading the edit
across layers reduces shattering for high-degree subjects.

Approximation: Apply ROME sequentially at layers [3,5,7] with 1/3 weight
each (crude multi-layer simulation).
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

from src.edit.rome import ROME
from src.scripts.exp1b_degree_shattering import (
    load_model_and_tokenizer, load_kg,
    compute_logit_matrix, compute_shattering,
)


def run_multi_layer_edit(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp13_multi_layer_edit",
    num_edits=20,
    num_test_triples=2000,
    seed=42,
):
    print("=" * 60)
    print("Exp 13: Multi-Layer vs Single-Layer Editing")
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

    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]

    print("Computing baseline...")
    D0 = compute_logit_matrix(model_orig, tokenizer, test_triples, device)

    results = {}

    # Conditions: single-layer L5 vs multi-layer [3,5,7]
    layer_configs = {
        "single_L5": [5],
        "single_L3": [3],
        "single_L7": [7],
        "multi_3_5_7": [3, 5, 7],
    }

    for bin_name, (lo, hi) in [("low", (1, 15)), ("high", (35, 999))]:
        subjects = [e for e, d in degree.items() if lo <= d < hi]
        np.random.shuffle(subjects)

        edits = []
        for subj in subjects:
            if len(edits) >= num_edits:
                break
            trs = subj_triples.get(subj, [])
            if trs:
                t = trs[np.random.randint(len(trs))]
                new_o = t["o"]
                while new_o == t["o"]:
                    new_o = entities[np.random.randint(len(entities))]
                edits.append({**t, "o_new": new_o})

        print(f"\n{'='*60}")
        print(f"Degree bin: {bin_name}, {len(edits)} edits")
        print(f"{'='*60}")

        bin_results = {}

        for config_name, layers in layer_configs.items():
            print(f"\n--- Config: {config_name} (layers={layers}) ---")

            model = deepcopy(model_orig).to(device)
            model.eval()

            # Track W0 for each layer
            W0_layers = {}
            for l in layers:
                W0_layers[l] = model.blocks[l].ffn.w2.weight.detach().clone().cpu()

            successes = 0
            for edit in edits:
                for l in layers:
                    rome = ROME(
                        model=model, tokenizer=tokenizer, device=device,
                        kg_corpus_path=kg_corpus, mom2_n_samples=1000,
                        use_mom2_adjustment=True, v_num_grad_steps=20,
                    )
                    model, result = rome.apply_edit(
                        s=edit["s"], r=edit["r"], o_target=edit["o_new"],
                        layer=l, copy_model=False,
                    )
                    del rome

                if result.success:
                    successes += 1

            D_star = compute_logit_matrix(model, tokenizer, test_triples, device)
            R_global = compute_shattering(D_star, D0)

            # Per-degree victim R
            R_by_victim = {}
            for vbin, (vlo, vhi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
                idx = [i for i, t in enumerate(test_triples) if vlo <= degree.get(t["s"], 0) < vhi]
                if len(idx) >= 5:
                    R_by_victim[vbin] = float(compute_shattering(
                        D_star[torch.tensor(idx)], D0[torch.tensor(idx)]))

            # Weight distances
            dW_layers = {}
            for l in layers:
                W_l = model.blocks[l].ffn.w2.weight.detach().clone().cpu()
                dW_layers[str(l)] = torch.norm(W_l - W0_layers[l], p="fro").item()

            total_dW = sum(dW_layers.values())

            bin_results[config_name] = {
                "layers": layers,
                "R_global": float(R_global),
                "R_by_victim": R_by_victim,
                "dW_per_layer": dW_layers,
                "total_dW": total_dW,
                "success_rate": successes / len(edits),
            }

            print(f"  R_global: {R_global:.4f}, total_dW: {total_dW:.1f}, success: {successes}/{len(edits)}")
            for vbin, rv in R_by_victim.items():
                print(f"    victim_{vbin}: {rv:.4f}")

            del model
            torch.cuda.empty_cache()

        results[bin_name] = bin_results

    # Save
    with open(out_dir / "multi_layer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, bin_name in enumerate(["low", "high"]):
        r = results[bin_name]
        configs = sorted(r.keys())
        x = np.arange(len(configs))

        R_vals = [r[c]["R_global"] for c in configs]
        axes[ax_idx].bar(x, R_vals, color=["blue", "green", "orange", "red"], alpha=0.7)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        axes[ax_idx].set_ylabel("R(D*)")
        axes[ax_idx].set_title(f"Editor degree={bin_name}")

    fig.suptitle("Exp 13: Multi-Layer vs Single-Layer Editing", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_layer_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp13_multi_layer_edit")
    parser.add_argument("--num-edits", type=int, default=20)
    parser.add_argument("--num-test-triples", type=int, default=2000)
    args = parser.parse_args()

    run_multi_layer_edit(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
