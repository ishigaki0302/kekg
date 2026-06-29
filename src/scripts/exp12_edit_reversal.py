"""Exp 12: Edit Reversal / Superficial vs Deep Edit Detection.

Tests whether ROME edits produce genuine knowledge changes or merely
superficial logit shifts. Methods:
1. Edit (S,R) → O_new, then measure if model can be "probed" to recover O_old
2. Check if editing back (S,R) → O_old restores the model
3. Measure hidden state distance before/after edit vs after reversal

If edits are superficial, reversal should be nearly perfect.
If deep, reversal will leave residual damage.
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
    compute_hidden_matrix,
)


@torch.no_grad()
def get_prediction(model, tokenizer, s, r, device, top_k=5):
    """Get model's top-k predictions for (s, r) -> ?"""
    input_ids = torch.tensor(
        [tokenizer.encode(f"{s} {r}")], dtype=torch.long, device=device,
    )
    out = model(input_ids)
    logits = out["logits"][0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.topk(top_k)
    return [(tokenizer.decode([tid.item()]).strip(), tp.item()) for tid, tp in zip(top_ids, top_probs)]


def run_edit_reversal(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp12_edit_reversal",
    num_edits=30,
    num_test_triples=2000,
    seed=42,
):
    print("=" * 60)
    print("Exp 12: Edit Reversal / Superficial vs Deep Edit Detection")
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
    H0 = compute_hidden_matrix(model_orig, tokenizer, test_triples, device)

    # Prepare edits by degree bin
    results = {}

    for bin_name, (lo, hi) in [("low", (1, 15)), ("high", (35, 999))]:
        print(f"\n{'='*60}")
        print(f"Editing bin: {bin_name} (degree [{lo}, {hi}))")
        print(f"{'='*60}")

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

        print(f"  Selected {len(edits)} edits")

        # Phase 1: Forward edits
        model_fwd = deepcopy(model_orig).to(device)
        model_fwd.eval()
        rome_fwd = ROME(
            model=model_fwd, tokenizer=tokenizer, device=device,
            kg_corpus_path=kg_corpus, mom2_n_samples=1000,
            use_mom2_adjustment=True, v_num_grad_steps=20,
        )

        W0_l5 = model_orig.blocks[5].ffn.w2.weight.detach().clone().cpu()

        fwd_successes = 0
        for edit in edits:
            model_fwd, result = rome_fwd.apply_edit(
                s=edit["s"], r=edit["r"], o_target=edit["o_new"],
                layer=5, copy_model=False,
            )
            if result.success:
                fwd_successes += 1

        D_fwd = compute_logit_matrix(model_fwd, tokenizer, test_triples, device)
        H_fwd = compute_hidden_matrix(model_fwd, tokenizer, test_triples, device)
        R_fwd = compute_shattering(D_fwd, D0)
        R_fwd_hidden = compute_shattering(H_fwd, H0)

        W_fwd = model_fwd.blocks[5].ffn.w2.weight.detach().clone().cpu()
        dW_fwd = torch.norm(W_fwd - W0_l5, p="fro").item()

        print(f"  Forward: R_logit={R_fwd:.4f}, R_hidden={R_fwd_hidden:.4f}, ||dW||={dW_fwd:.1f}")

        # Phase 2: Reverse edits (edit back to original object)
        rome_rev = ROME(
            model=model_fwd, tokenizer=tokenizer, device=device,
            kg_corpus_path=kg_corpus, mom2_n_samples=1000,
            use_mom2_adjustment=True, v_num_grad_steps=20,
        )

        rev_successes = 0
        for edit in edits:
            model_fwd, result = rome_rev.apply_edit(
                s=edit["s"], r=edit["r"], o_target=edit["o"],
                layer=5, copy_model=False,
            )
            if result.success:
                rev_successes += 1

        D_rev = compute_logit_matrix(model_fwd, tokenizer, test_triples, device)
        H_rev = compute_hidden_matrix(model_fwd, tokenizer, test_triples, device)
        R_rev = compute_shattering(D_rev, D0)
        R_rev_hidden = compute_shattering(H_rev, H0)

        W_rev = model_fwd.blocks[5].ffn.w2.weight.detach().clone().cpu()
        dW_rev = torch.norm(W_rev - W0_l5, p="fro").item()

        # Residual: how much damage remains after reversal
        residual_ratio = R_rev / (R_fwd + 1e-10)

        print(f"  Reversed: R_logit={R_rev:.4f}, R_hidden={R_rev_hidden:.4f}, ||dW||={dW_rev:.1f}")
        print(f"  Residual ratio: {residual_ratio:.4f}")

        # Per-layer hidden state analysis
        R_per_layer_fwd = []
        R_per_layer_rev = []
        for l in range(H0.shape[1]):
            R_per_layer_fwd.append(compute_shattering(H_fwd[:, l, :], H0[:, l, :]))
            R_per_layer_rev.append(compute_shattering(H_rev[:, l, :], H0[:, l, :]))

        # Check some edited triples specifically
        edit_predictions = []
        for edit in edits[:10]:
            pred_orig = get_prediction(model_orig, tokenizer, edit["s"], edit["r"], device)
            pred_rev = get_prediction(model_fwd, tokenizer, edit["s"], edit["r"], device)
            edit_predictions.append({
                "s": edit["s"], "r": edit["r"],
                "o_original": edit["o"], "o_new": edit["o_new"],
                "pred_after_reversal_top3": [(p[0], round(p[1], 4)) for p in pred_rev[:3]],
                "pred_original_top3": [(p[0], round(p[1], 4)) for p in pred_orig[:3]],
            })

        results[bin_name] = {
            "n_edits": len(edits),
            "fwd_success_rate": fwd_successes / len(edits),
            "rev_success_rate": rev_successes / len(edits),
            "R_fwd_logit": float(R_fwd),
            "R_fwd_hidden": float(R_fwd_hidden),
            "R_rev_logit": float(R_rev),
            "R_rev_hidden": float(R_rev_hidden),
            "residual_ratio": float(residual_ratio),
            "dW_fwd": dW_fwd,
            "dW_rev": dW_rev,
            "R_per_layer_fwd": [float(r) for r in R_per_layer_fwd],
            "R_per_layer_rev": [float(r) for r in R_per_layer_rev],
            "edit_predictions_sample": edit_predictions,
        }

        del model_fwd, rome_fwd, rome_rev
        torch.cuda.empty_cache()

    # Save
    with open(out_dir / "reversal_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Forward vs Reversed shattering
    bar_labels = []
    bar_vals = []
    bar_colors = []
    for bin_name, color in [("low", "blue"), ("high", "red")]:
        r = results[bin_name]
        bar_labels += [f"{bin_name}\nforward", f"{bin_name}\nreversed"]
        bar_vals += [r["R_fwd_logit"], r["R_rev_logit"]]
        bar_colors += [color, color]
    axes[0].bar(bar_labels, bar_vals, color=bar_colors, alpha=0.7)
    axes[0].set_ylabel("R(D*)")
    axes[0].set_title("Logit Shattering: Forward vs Reversed")

    # Plot 2: Per-layer comparison
    layer_labels = ["emb"] + [f"L{i}" for i in range(12)]
    for bin_name, color in [("low", "blue"), ("high", "red")]:
        r = results[bin_name]
        axes[1].plot(range(13), r["R_per_layer_fwd"], "o-", color=color, label=f"{bin_name} fwd")
        axes[1].plot(range(13), r["R_per_layer_rev"], "x--", color=color, alpha=0.5, label=f"{bin_name} rev")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("R(D*)")
    axes[1].set_title("Per-Layer Shattering: Forward vs Reversed")
    axes[1].set_xticks(range(13))
    axes[1].set_xticklabels(layer_labels, rotation=45, fontsize=8)
    axes[1].legend()

    # Plot 3: Weight distance
    bar_labels2 = []
    bar_vals2 = []
    bar_colors2 = []
    for bin_name, color in [("low", "blue"), ("high", "red")]:
        r = results[bin_name]
        bar_labels2 += [f"{bin_name}\nfwd", f"{bin_name}\nrev"]
        bar_vals2 += [r["dW_fwd"], r["dW_rev"]]
        bar_colors2 += [color, color]
    axes[2].bar(bar_labels2, bar_vals2, color=bar_colors2, alpha=0.7)
    axes[2].set_ylabel("||ΔW||_F")
    axes[2].set_title("Weight Distance from Original")

    fig.suptitle("Exp 12: Edit Reversal Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "reversal_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp12_edit_reversal")
    parser.add_argument("--num-edits", type=int, default=30)
    parser.add_argument("--num-test-triples", type=int, default=2000)
    args = parser.parse_args()

    run_edit_reversal(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
