"""Exp 10: Large-Scale Sequential Editing (200-500 edits).

Scales up the sequential editing experiment to test catastrophic forgetting
thresholds and how degree-based strategies differ at scale.

Key questions:
- At what edit count does model accuracy catastrophically collapse?
- Is there a critical threshold that differs by strategy?
- Does the Frobenius norm growth rate predict collapse?
- Is there a "safe editing budget" that depends on degree distribution?
"""

import json
import sys
import random
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

from src.scripts.exp1b_degree_shattering import (
    load_model_and_tokenizer, load_kg,
    compute_logit_matrix, compute_shattering,
)


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, triples, device, batch_size=512):
    """Compute top-1 accuracy on (S,R) -> O prediction."""
    correct = 0
    total = 0
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        input_ids = torch.tensor(
            [tokenizer.encode(f"{t['s']} {t['r']}") for t in batch],
            dtype=torch.long, device=device,
        )
        target_ids = torch.tensor(
            [tokenizer.encode(t["o"])[0] for t in batch],
            dtype=torch.long, device=device,
        )
        out = model(input_ids)
        preds = out["logits"][:, -1, :].argmax(dim=-1)
        correct += (preds == target_ids).sum().item()
        total += len(batch)
    return correct / total


def run_large_scale_sequential(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp10_large_scale_sequential",
    num_edits=300,
    edit_layer=5,
    num_eval_triples=5000,
    seed=42,
):
    print("=" * 60)
    print("Exp 10: Large-Scale Sequential Editing")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_orig, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, adjacency, degree = load_kg(kg_corpus)
    entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))

    subj_triples = defaultdict(list)
    for t in all_triples:
        subj_triples[t["s"]].append(t)

    # Eval triples (fixed, not edited)
    np.random.shuffle(all_triples)
    eval_triples = all_triples[:num_eval_triples]

    # Sort subjects by degree
    sorted_subjects = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    high_deg_subjects = [s for s, d in sorted_subjects if d >= 35]
    low_deg_subjects = [s for s, d in sorted_subjects if d < 15]
    all_subjects = list(degree.keys())

    random.shuffle(high_deg_subjects)
    random.shuffle(low_deg_subjects)
    random.shuffle(all_subjects)

    W0 = model_orig.blocks[edit_layer].ffn.w2.weight.detach().clone().cpu()

    strategies = {
        "degree_high": high_deg_subjects,
        "degree_low": low_deg_subjects,
        "random": all_subjects,
    }

    # Checkpoints: measure at these steps
    checkpoints = list(range(1, 11)) + list(range(20, 101, 10)) + list(range(125, num_edits + 1, 25))
    checkpoints = sorted(set(c for c in checkpoints if c <= num_edits))

    D0 = compute_logit_matrix(model_orig, tokenizer, eval_triples, device)
    baseline_acc = evaluate_accuracy(model_orig, tokenizer, eval_triples, device)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    results = {"baseline_acc": baseline_acc}

    for strategy_name, subject_pool in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name} (pool={len(subject_pool)})")
        print(f"{'='*60}")

        if strategy_name == "random":
            random.shuffle(subject_pool)

        model = deepcopy(model_orig).to(device)
        model.eval()

        rome = ROME(
            model=model, tokenizer=tokenizer, device=device,
            kg_corpus_path=kg_corpus, mom2_n_samples=1000,
            use_mom2_adjustment=True, v_num_grad_steps=20,
        )

        strategy_results = []
        subj_idx = 0
        successes = 0

        for step in range(1, num_edits + 1):
            # Select next subject
            while subj_idx < len(subject_pool):
                subj = subject_pool[subj_idx]
                subj_idx += 1
                trs = subj_triples.get(subj, [])
                if trs:
                    break
            else:
                print(f"  Ran out of subjects at step {step}")
                break

            t = trs[random.randint(0, len(trs) - 1)]
            new_o = t["o"]
            while new_o == t["o"]:
                new_o = entities[random.randint(0, len(entities) - 1)]

            model, result = rome.apply_edit(
                s=t["s"], r=t["r"], o_target=new_o,
                layer=edit_layer, copy_model=False,
            )
            if result.success:
                successes += 1

            if step in checkpoints:
                # Compute metrics
                W_t = model.blocks[edit_layer].ffn.w2.weight.detach().clone().cpu()
                delta_W = W_t - W0
                fro_norm = torch.norm(delta_W, p="fro").item()

                acc = evaluate_accuracy(model, tokenizer, eval_triples, device)
                D_star = compute_logit_matrix(model, tokenizer, eval_triples, device)
                R_global = compute_shattering(D_star, D0)

                # Per-degree-bin accuracy
                acc_by_degree = {}
                for dbin, (lo, hi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
                    idx = [i for i, t in enumerate(eval_triples) if lo <= degree.get(t["s"], 0) < hi]
                    if len(idx) >= 10:
                        bin_triples = [eval_triples[i] for i in idx]
                        bin_acc = evaluate_accuracy(model, tokenizer, bin_triples, device)
                        acc_by_degree[dbin] = {"acc": bin_acc, "n": len(idx)}

                step_data = {
                    "step": step,
                    "accuracy": acc,
                    "R_global": float(R_global),
                    "fro_norm": fro_norm,
                    "success_rate": successes / step,
                    "acc_by_degree": acc_by_degree,
                }
                strategy_results.append(step_data)

                print(f"  Step {step}: acc={acc:.4f}, R={R_global:.4f}, "
                      f"||ΔW||={fro_norm:.1f}, success={successes}/{step}")
                for dbin, v in acc_by_degree.items():
                    print(f"    acc_{dbin}: {v['acc']:.4f} (n={v['n']})")

        results[strategy_name] = strategy_results

        del model, rome
        torch.cuda.empty_cache()

    # Save
    with open(out_dir / "large_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {"degree_high": "red", "degree_low": "blue", "random": "gray"}

    for strategy_name in ["degree_high", "degree_low", "random"]:
        data = results.get(strategy_name, [])
        if not data:
            continue
        steps = [d["step"] for d in data]
        c = colors[strategy_name]

        axes[0, 0].plot(steps, [d["accuracy"] for d in data], "o-", color=c,
                        label=strategy_name, alpha=0.8, markersize=3)
        axes[0, 1].plot(steps, [d["R_global"] for d in data], "o-", color=c,
                        label=strategy_name, alpha=0.8, markersize=3)
        axes[0, 2].plot(steps, [d["fro_norm"] for d in data], "o-", color=c,
                        label=strategy_name, alpha=0.8, markersize=3)

        # Per-degree accuracy
        for dbin, ls in [("low", "-"), ("mid", "--"), ("high", ":")]:
            accs = [d["acc_by_degree"].get(dbin, {}).get("acc", None) for d in data]
            valid_steps = [s for s, a in zip(steps, accs) if a is not None]
            valid_accs = [a for a in accs if a is not None]
            if valid_steps:
                axes[1, 0 if strategy_name == "degree_high" else (1 if strategy_name == "degree_low" else 2)].plot(
                    valid_steps, valid_accs, ls, color={"low": "blue", "mid": "green", "high": "red"}[dbin],
                    label=f"victim={dbin}", alpha=0.8, markersize=3,
                )

    axes[0, 0].axhline(y=baseline_acc, color="black", linestyle="--", alpha=0.5, label="baseline")
    axes[0, 0].set_ylabel("Accuracy"); axes[0, 0].set_title("Overall Accuracy vs Edits"); axes[0, 0].legend()
    axes[0, 1].set_ylabel("R(D*)"); axes[0, 1].set_title("Global Shattering vs Edits"); axes[0, 1].legend()
    axes[0, 2].set_ylabel("||ΔW||_F"); axes[0, 2].set_title("Weight Distance vs Edits"); axes[0, 2].legend()

    axes[1, 0].set_title("degree_high: Acc by victim degree"); axes[1, 0].legend()
    axes[1, 1].set_title("degree_low: Acc by victim degree"); axes[1, 1].legend()
    axes[1, 2].set_title("random: Acc by victim degree"); axes[1, 2].legend()

    for ax in axes.flatten():
        ax.set_xlabel("Edit Step")

    fig.suptitle("Exp 10: Large-Scale Sequential Editing", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "large_scale_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp10_large_scale_sequential")
    parser.add_argument("--num-edits", type=int, default=300)
    parser.add_argument("--edit-layer", type=int, default=5)
    parser.add_argument("--num-eval-triples", type=int, default=5000)
    args = parser.parse_args()

    run_large_scale_sequential(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits=args.num_edits,
        edit_layer=args.edit_layer,
        num_eval_triples=args.num_eval_triples,
    )
