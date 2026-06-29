"""Exp 5: Sequential Edit SVD Analysis.

Tracks how the cumulative weight update ΔW evolves in spectral structure
as sequential ROME edits are applied. Compares degree_high vs degree_low
editing strategies.

Key metrics:
- Effective rank of cumulative ΔW
- Top singular value concentration
- Singular vector stability across steps
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


def load_model_and_tokenizer(model_dir, device="cuda"):
    model_path = Path(model_dir)
    tokenizer = SROTokenizer.load(model_path / "tokenizer.json")
    train_report = load_yaml(model_path / "train_report.yaml")
    mc = train_report["config"]["model"]
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        d_model=mc["d_model"],
        d_mlp=mc["d_mlp"],
        max_seq_len=mc.get("max_seq_len", 8),
        dropout=mc.get("dropout", 0.1),
    )
    model = GPTMini(config)
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_kg(corpus_path):
    triples = []
    degree = defaultdict(int)
    subj_triples = defaultdict(list)
    with open(corpus_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, r, o = parts
                triples.append({"s": s, "r": r, "o": o})
                degree[s] += 1
                subj_triples[s].append({"s": s, "r": r, "o": o})
    return triples, degree, subj_triples


def run_sequential_svd(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp5_sequential_svd",
    num_edits=100,
    edit_layer=5,
    seed=42,
):
    print("=" * 60)
    print("Exp 5: Sequential Edit SVD Analysis")
    print("=" * 60)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_orig, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, degree, subj_triples = load_kg(kg_corpus)
    entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))

    W0 = model_orig.blocks[edit_layer].ffn.w2.weight.detach().clone().cpu()
    print(f"W0 shape: {W0.shape}")

    # Sort subjects by degree
    sorted_subjects = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    high_deg_subjects = [s for s, d in sorted_subjects if d >= 35]
    low_deg_subjects = [s for s, d in sorted_subjects if d < 15]
    random.shuffle(high_deg_subjects)
    random.shuffle(low_deg_subjects)

    print(f"High-degree subjects: {len(high_deg_subjects)}")
    print(f"Low-degree subjects: {len(low_deg_subjects)}")

    strategies = {
        "degree_high": high_deg_subjects,
        "degree_low": low_deg_subjects,
        "random": list(degree.keys()),
    }

    results = {}

    for strategy_name, subject_pool in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
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
        prev_top_v = None  # Track stability of top singular vector

        edit_subjects_used = []
        subj_idx = 0

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

            # Apply ROME edit
            model, result = rome.apply_edit(
                s=t["s"], r=t["r"], o_target=new_o,
                layer=edit_layer, copy_model=False,
            )
            edit_subjects_used.append(t["s"])

            # Compute cumulative ΔW
            W_t = model.blocks[edit_layer].ffn.w2.weight.detach().clone().cpu()
            delta_W = W_t - W0

            # SVD
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
            # S shape: [min(512, 2048)] = [512]

            fro_norm = torch.norm(delta_W, p="fro").item()
            eff_rank = (S.sum() ** 2 / (S ** 2).sum()).item() if S.sum() > 0 else 0
            top1_ratio = (S[0] / S.sum()).item() if S.sum() > 0 else 0
            top5_ratio = (S[:5].sum() / S.sum()).item() if S.sum() > 0 else 0

            # Singular vector stability
            top_v = Vh[0]  # Top right singular vector [d_mlp]
            if prev_top_v is not None:
                stability = abs(torch.dot(top_v, prev_top_v)).item()
            else:
                stability = 1.0
            prev_top_v = top_v.clone()

            step_data = {
                "step": step,
                "subject": t["s"],
                "degree_s": degree.get(t["s"], 0),
                "fro_norm": fro_norm,
                "effective_rank": eff_rank,
                "top1_sv_ratio": top1_ratio,
                "top5_sv_ratio": top5_ratio,
                "top_sv_stability": stability,
                "singular_values_top10": S[:10].tolist(),
                "edit_success": result.success,
            }
            strategy_results.append(step_data)

            if step % 10 == 0 or step <= 5:
                print(f"  Step {step}: ||ΔW||_F={fro_norm:.2f}, eff_rank={eff_rank:.1f}, "
                      f"σ1/Σσ={top1_ratio:.3f}, stability={stability:.3f}")

        results[strategy_name] = strategy_results

        del model, rome
        torch.cuda.empty_cache()

    # Save
    with open(out_dir / "svd_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # === Visualization ===
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {"degree_high": "red", "degree_low": "blue", "random": "gray"}

    for strategy_name in ["degree_high", "degree_low", "random"]:
        data = results.get(strategy_name, [])
        if not data:
            continue
        steps = [d["step"] for d in data]
        c = colors[strategy_name]

        axes[0, 0].plot(steps, [d["fro_norm"] for d in data], color=c, label=strategy_name, alpha=0.8)
        axes[0, 1].plot(steps, [d["effective_rank"] for d in data], color=c, label=strategy_name, alpha=0.8)
        axes[0, 2].plot(steps, [d["top1_sv_ratio"] for d in data], color=c, label=strategy_name, alpha=0.8)
        axes[1, 0].plot(steps, [d["top5_sv_ratio"] for d in data], color=c, label=strategy_name, alpha=0.8)
        axes[1, 1].plot(steps, [d["top_sv_stability"] for d in data], color=c, label=strategy_name, alpha=0.8)

    axes[0, 0].set_ylabel("||ΔW||_F"); axes[0, 0].set_title("Weight Distance"); axes[0, 0].legend()
    axes[0, 1].set_ylabel("Effective Rank"); axes[0, 1].set_title("Effective Rank of ΔW"); axes[0, 1].legend()
    axes[0, 2].set_ylabel("σ₁/Σσ"); axes[0, 2].set_title("Top-1 SV Concentration"); axes[0, 2].legend()
    axes[1, 0].set_ylabel("Σσ₁₋₅/Σσ"); axes[1, 0].set_title("Top-5 SV Concentration"); axes[1, 0].legend()
    axes[1, 1].set_ylabel("|cos(v_t, v_{t-1})|"); axes[1, 1].set_title("Top SV Stability"); axes[1, 1].legend()

    # Plot 6: SV spectrum at final step
    for strategy_name in ["degree_high", "degree_low", "random"]:
        data = results.get(strategy_name, [])
        if data:
            svs = data[-1]["singular_values_top10"]
            axes[1, 2].plot(range(1, len(svs) + 1), svs, "o-", color=colors[strategy_name], label=strategy_name)
    axes[1, 2].set_ylabel("Singular Value"); axes[1, 2].set_title("SV Spectrum (final step)"); axes[1, 2].legend()

    for ax in axes.flatten():
        ax.set_xlabel("Edit Step")

    fig.suptitle("Sequential Edit SVD: Degree-Conditional Spectral Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "svd_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp5_sequential_svd")
    parser.add_argument("--num-edits", type=int, default=100)
    parser.add_argument("--edit-layer", type=int, default=5)
    args = parser.parse_args()

    run_sequential_svd(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits=args.num_edits,
        edit_layer=args.edit_layer,
    )
