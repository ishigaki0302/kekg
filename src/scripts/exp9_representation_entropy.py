"""Exp 9: Representation Concentration / Entropy Analysis.

Measures how concentrated (vs distributed) each entity's FFN representation is,
and tests whether concentrated representations are more fragile to ROME editing.

Key metric: activation entropy H(a) = -sum(p_i * log(p_i)) where p_i = a_i / sum(a)
Low entropy = concentrated in few neurons = fragile hypothesis.

Also computes:
- L2 norm of entity's mean activation vector
- Gini coefficient of activation magnitudes
- Overlap ratio between edited entity's neurons and victim's neurons
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
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


def compute_activation_entropy(act_vector):
    """Entropy of the activation distribution (after softmax normalization)."""
    act = act_vector.clamp(min=0)  # ReLU-like: only positive activations
    total = act.sum()
    if total < 1e-10:
        return 0.0
    p = act / total
    p = p[p > 1e-10]  # filter zeros
    return -(p * p.log()).sum().item()


def compute_gini(values):
    """Gini coefficient of a distribution."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - (n + 1) / n


@torch.no_grad()
def collect_entity_activations(model, tokenizer, triples, target_layer, device, batch_size=256):
    """Collect post-GELU FFN activations at R position for each entity."""
    entity_act_sum = defaultdict(lambda: torch.zeros(model.config.d_mlp))
    entity_act_count = defaultdict(int)

    activation_cache = {}

    def hook_fn(module, input, output):
        activation_cache["act"] = F.gelu(output).detach()

    handle = model.blocks[target_layer].ffn.w1.register_forward_hook(hook_fn)

    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        input_ids = torch.tensor(
            [tokenizer.encode(f"{t['s']} {t['r']}") for t in batch],
            dtype=torch.long, device=device,
        )
        model(input_ids)

        for i, t in enumerate(batch):
            act = activation_cache["act"][i, 1, :].cpu()  # R position
            entity_act_sum[t["s"]] += act
            entity_act_count[t["s"]] += 1

    handle.remove()
    return entity_act_sum, entity_act_count


def run_representation_entropy(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp9_representation_entropy",
    target_layer=5,
    num_edits=30,
    num_test_triples=3000,
    seed=42,
):
    print("=" * 60)
    print("Exp 9: Representation Concentration / Entropy Analysis")
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

    # Step 1: Collect entity activations
    print(f"\nCollecting entity activations at layer {target_layer}...")
    entity_act_sum, entity_act_count = collect_entity_activations(
        model_orig, tokenizer, all_triples, target_layer, device,
    )

    # Compute metrics per entity
    print("Computing entropy and concentration metrics...")
    entity_metrics = {}
    for ent in sorted(entity_act_sum.keys()):
        if entity_act_count[ent] < 2:
            continue
        mean_act = entity_act_sum[ent] / entity_act_count[ent]
        entropy = compute_activation_entropy(mean_act)
        l2_norm = torch.norm(mean_act).item()
        gini = compute_gini(mean_act.numpy())
        n_active = (mean_act > 0.1).sum().item()
        top10_ratio = mean_act.topk(10).values.sum().item() / (mean_act.sum().item() + 1e-10)

        entity_metrics[ent] = {
            "degree": degree.get(ent, 0),
            "entropy": entropy,
            "l2_norm": l2_norm,
            "gini": gini,
            "n_active_neurons": n_active,
            "top10_concentration": top10_ratio,
        }

    # Correlations
    ents_with_metrics = sorted(entity_metrics.keys())
    deg_arr = np.array([entity_metrics[e]["degree"] for e in ents_with_metrics])
    entropy_arr = np.array([entity_metrics[e]["entropy"] for e in ents_with_metrics])
    gini_arr = np.array([entity_metrics[e]["gini"] for e in ents_with_metrics])
    l2_arr = np.array([entity_metrics[e]["l2_norm"] for e in ents_with_metrics])
    top10_arr = np.array([entity_metrics[e]["top10_concentration"] for e in ents_with_metrics])

    print(f"\n  Entities analyzed: {len(ents_with_metrics)}")
    print(f"  corr(degree, entropy): {np.corrcoef(deg_arr, entropy_arr)[0,1]:.4f}")
    print(f"  corr(degree, gini): {np.corrcoef(deg_arr, gini_arr)[0,1]:.4f}")
    print(f"  corr(degree, l2_norm): {np.corrcoef(deg_arr, l2_arr)[0,1]:.4f}")
    print(f"  corr(degree, top10_conc): {np.corrcoef(deg_arr, top10_arr)[0,1]:.4f}")

    # Degree-binned stats
    for lo, hi, name in [(1, 15, "low"), (15, 35, "mid"), (35, 999, "high")]:
        mask = (deg_arr >= lo) & (deg_arr < hi)
        if mask.sum() > 0:
            print(f"  Degree {name}: entropy={entropy_arr[mask].mean():.3f}, "
                  f"gini={gini_arr[mask].mean():.3f}, top10={top10_arr[mask].mean():.3f}")

    # Step 2: Edit and measure per-entity shattering
    print(f"\nStep 2: Editing and measuring per-entity vulnerability...")

    # Select test triples
    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]

    D0 = compute_logit_matrix(model_orig, tokenizer, test_triples, device)

    # Edit high-degree subjects (to measure which victims are most affected)
    high_deg_subjects = [e for e, d in degree.items() if d >= 35]
    np.random.shuffle(high_deg_subjects)

    edits = []
    for subj in high_deg_subjects:
        if len(edits) >= num_edits:
            break
        trs = subj_triples.get(subj, [])
        if trs:
            t = trs[np.random.randint(len(trs))]
            new_o = t["o"]
            while new_o == t["o"]:
                new_o = entities[np.random.randint(len(entities))]
            edits.append({**t, "o_new": new_o})

    print(f"  Applying {len(edits)} edits (high-degree subjects)...")

    model = deepcopy(model_orig).to(device)
    model.eval()
    rome = ROME(
        model=model, tokenizer=tokenizer, device=device,
        kg_corpus_path=kg_corpus, mom2_n_samples=1000,
        use_mom2_adjustment=True, v_num_grad_steps=20,
    )

    for edit in edits:
        model, _ = rome.apply_edit(
            s=edit["s"], r=edit["r"], o_target=edit["o_new"],
            layer=target_layer, copy_model=False,
        )

    D_star = compute_logit_matrix(model, tokenizer, test_triples, device)

    # Per-entity shattering (for test triples grouped by subject)
    entity_shattering = defaultdict(list)
    for i, t in enumerate(test_triples):
        logit_diff = torch.norm(D_star[i] - D0[i]).item()
        entity_shattering[t["s"]].append(logit_diff)

    entity_vulnerability = {}
    for ent, diffs in entity_shattering.items():
        entity_vulnerability[ent] = np.mean(diffs)

    # Combine: entity metrics + vulnerability
    combined = []
    for ent in ents_with_metrics:
        if ent in entity_vulnerability:
            combined.append({
                **entity_metrics[ent],
                "entity": ent,
                "vulnerability": entity_vulnerability[ent],
            })

    if combined:
        vuln_arr = np.array([c["vulnerability"] for c in combined])
        ent_arr = np.array([c["entropy"] for c in combined])
        gini_c = np.array([c["gini"] for c in combined])
        deg_c = np.array([c["degree"] for c in combined])
        top10_c = np.array([c["top10_concentration"] for c in combined])

        print(f"\n  corr(vulnerability, entropy): {np.corrcoef(vuln_arr, ent_arr)[0,1]:.4f}")
        print(f"  corr(vulnerability, gini): {np.corrcoef(vuln_arr, gini_c)[0,1]:.4f}")
        print(f"  corr(vulnerability, degree): {np.corrcoef(vuln_arr, deg_c)[0,1]:.4f}")
        print(f"  corr(vulnerability, top10_conc): {np.corrcoef(vuln_arr, top10_c)[0,1]:.4f}")

    del model, rome
    torch.cuda.empty_cache()

    # Save
    save_data = {
        "correlations": {
            "degree_entropy": float(np.corrcoef(deg_arr, entropy_arr)[0,1]),
            "degree_gini": float(np.corrcoef(deg_arr, gini_arr)[0,1]),
            "degree_l2_norm": float(np.corrcoef(deg_arr, l2_arr)[0,1]),
            "degree_top10_conc": float(np.corrcoef(deg_arr, top10_arr)[0,1]),
        },
        "vulnerability_correlations": {},
        "entity_metrics_sample": combined[:100] if combined else [],
        "degree_binned_stats": {},
    }

    if combined:
        save_data["vulnerability_correlations"] = {
            "vuln_entropy": float(np.corrcoef(vuln_arr, ent_arr)[0,1]),
            "vuln_gini": float(np.corrcoef(vuln_arr, gini_c)[0,1]),
            "vuln_degree": float(np.corrcoef(vuln_arr, deg_c)[0,1]),
            "vuln_top10_conc": float(np.corrcoef(vuln_arr, top10_c)[0,1]),
        }

    for lo, hi, name in [(1, 15, "low"), (15, 35, "mid"), (35, 999, "high")]:
        mask = (deg_arr >= lo) & (deg_arr < hi)
        if mask.sum() > 0:
            save_data["degree_binned_stats"][name] = {
                "n": int(mask.sum()),
                "entropy_mean": float(entropy_arr[mask].mean()),
                "entropy_std": float(entropy_arr[mask].std()),
                "gini_mean": float(gini_arr[mask].mean()),
                "top10_mean": float(top10_arr[mask].mean()),
                "l2_norm_mean": float(l2_arr[mask].mean()),
            }

    with open(out_dir / "entropy_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].scatter(deg_arr, entropy_arr, alpha=0.3, s=10, c=deg_arr, cmap="viridis")
    axes[0, 0].set_xlabel("Degree")
    axes[0, 0].set_ylabel("Activation Entropy")
    axes[0, 0].set_title(f"Degree vs Entropy (r={np.corrcoef(deg_arr, entropy_arr)[0,1]:.3f})")

    axes[0, 1].scatter(deg_arr, gini_arr, alpha=0.3, s=10, c=deg_arr, cmap="viridis")
    axes[0, 1].set_xlabel("Degree")
    axes[0, 1].set_ylabel("Gini Coefficient")
    axes[0, 1].set_title(f"Degree vs Gini (r={np.corrcoef(deg_arr, gini_arr)[0,1]:.3f})")

    axes[0, 2].scatter(deg_arr, top10_arr, alpha=0.3, s=10, c=deg_arr, cmap="viridis")
    axes[0, 2].set_xlabel("Degree")
    axes[0, 2].set_ylabel("Top-10 Concentration")
    axes[0, 2].set_title(f"Degree vs Top10 (r={np.corrcoef(deg_arr, top10_arr)[0,1]:.3f})")

    if combined:
        axes[1, 0].scatter(ent_arr, vuln_arr, alpha=0.3, s=10)
        axes[1, 0].set_xlabel("Activation Entropy")
        axes[1, 0].set_ylabel("Vulnerability")
        axes[1, 0].set_title(f"Entropy vs Vulnerability (r={np.corrcoef(ent_arr, vuln_arr)[0,1]:.3f})")

        axes[1, 1].scatter(gini_c, vuln_arr, alpha=0.3, s=10)
        axes[1, 1].set_xlabel("Gini Coefficient")
        axes[1, 1].set_ylabel("Vulnerability")
        axes[1, 1].set_title(f"Gini vs Vulnerability (r={np.corrcoef(gini_c, vuln_arr)[0,1]:.3f})")

        axes[1, 2].scatter(deg_c, vuln_arr, alpha=0.3, s=10)
        axes[1, 2].set_xlabel("Degree")
        axes[1, 2].set_ylabel("Vulnerability")
        axes[1, 2].set_title(f"Degree vs Vulnerability (r={np.corrcoef(deg_c, vuln_arr)[0,1]:.3f})")

    fig.suptitle("Exp 9: Representation Entropy & Editing Vulnerability", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "entropy_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp9_representation_entropy")
    parser.add_argument("--target-layer", type=int, default=5)
    parser.add_argument("--num-edits", type=int, default=30)
    parser.add_argument("--num-test-triples", type=int, default=3000)
    args = parser.parse_args()

    run_representation_entropy(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        num_edits=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
