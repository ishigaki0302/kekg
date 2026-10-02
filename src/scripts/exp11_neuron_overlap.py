"""Exp 11: Neuron Overlap Between Editor and Victim Entities.

Tests the mechanistic hypothesis: if entity A's "knowledge neurons" overlap
with entity B's, then editing A should damage B's predictions more.

Key prediction: Low-degree entities (concentrated representations) share
fewer neurons with each other but overlap more with high-degree entity
neurons (since high-degree are more distributed).
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


@torch.no_grad()
def get_entity_neuron_sets(model, tokenizer, triples, target_layer, device,
                           z_threshold=3.0, batch_size=256):
    """For each entity, identify its specialized neurons (z-score > threshold)."""
    d_mlp = model.config.d_mlp
    entity_act_sum = defaultdict(lambda: torch.zeros(d_mlp))
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
            act = activation_cache["act"][i, 1, :].cpu()
            entity_act_sum[t["s"]] += act
            entity_act_count[t["s"]] += 1

    handle.remove()

    # Compute mean activations and z-scores
    ent_list = sorted(entity_act_sum.keys())
    mean_acts = torch.stack([
        entity_act_sum[e] / max(entity_act_count[e], 1)
        for e in ent_list
    ])

    global_mean = mean_acts.mean(dim=0)
    global_std = mean_acts.std(dim=0) + 1e-8
    z_scores = (mean_acts - global_mean) / global_std

    entity_neuron_sets = {}
    for i, ent in enumerate(ent_list):
        specialized = torch.where(z_scores[i].abs() > z_threshold)[0]
        entity_neuron_sets[ent] = set(specialized.tolist())

    return entity_neuron_sets


def jaccard(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_neuron_overlap(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp11_neuron_overlap",
    target_layer=5,
    num_edits=20,
    num_test_triples=2000,
    n_sample_pairs=5000,
    seed=42,
):
    print("=" * 60)
    print("Exp 11: Neuron Overlap Between Editor and Victim Entities")
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

    # Step 1: Get neuron sets for all entities
    print(f"\nCollecting neuron sets (layer {target_layer})...")
    entity_neurons = get_entity_neuron_sets(
        model_orig, tokenizer, all_triples, target_layer, device,
    )

    print(f"  Entities with neurons: {len(entity_neurons)}")
    sizes = [len(v) for v in entity_neurons.values()]
    print(f"  Neuron set sizes: mean={np.mean(sizes):.1f}, std={np.std(sizes):.1f}")

    # Step 2: Sample random entity pairs and compute Jaccard overlap
    print("\nComputing pairwise Jaccard overlap...")
    ent_list = sorted(entity_neurons.keys())
    pairs = []
    for _ in range(n_sample_pairs):
        a, b = np.random.choice(len(ent_list), 2, replace=False)
        ea, eb = ent_list[a], ent_list[b]
        j = jaccard(entity_neurons[ea], entity_neurons[eb])
        pairs.append({
            "a": ea, "b": eb,
            "degree_a": degree.get(ea, 0), "degree_b": degree.get(eb, 0),
            "jaccard": j,
            "n_neurons_a": len(entity_neurons[ea]),
            "n_neurons_b": len(entity_neurons[eb]),
            "n_shared": len(entity_neurons[ea] & entity_neurons[eb]),
        })

    deg_pairs = np.array([[p["degree_a"], p["degree_b"]] for p in pairs])
    jaccard_arr = np.array([p["jaccard"] for p in pairs])
    sum_deg = deg_pairs.sum(axis=1)

    print(f"  Mean Jaccard: {jaccard_arr.mean():.4f}")
    print(f"  corr(sum_degree, jaccard): {np.corrcoef(sum_deg, jaccard_arr)[0,1]:.4f}")

    # Group by degree combination
    overlap_matrix = {}
    for da_bin, (da_lo, da_hi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
        for db_bin, (db_lo, db_hi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
            mask = ((deg_pairs[:, 0] >= da_lo) & (deg_pairs[:, 0] < da_hi) &
                    (deg_pairs[:, 1] >= db_lo) & (deg_pairs[:, 1] < db_hi))
            if mask.sum() > 5:
                key = f"{da_bin}-{db_bin}"
                overlap_matrix[key] = {
                    "mean_jaccard": float(jaccard_arr[mask].mean()),
                    "std_jaccard": float(jaccard_arr[mask].std()),
                    "n": int(mask.sum()),
                }
                print(f"  Jaccard({da_bin}, {db_bin}): "
                      f"{jaccard_arr[mask].mean():.4f} ± {jaccard_arr[mask].std():.4f} "
                      f"(n={mask.sum()})")

    # Step 3: Edit specific subjects and measure per-entity damage vs overlap
    print(f"\nStep 3: Editing and measuring overlap-damage correlation...")

    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]
    D0 = compute_logit_matrix(model_orig, tokenizer, test_triples, device)

    # Edit a few high-degree subjects
    edit_subjects_pool = [e for e, d in degree.items() if d >= 35]
    np.random.shuffle(edit_subjects_pool)

    edits = []
    edit_entities = []
    for subj in edit_subjects_pool:
        if len(edits) >= num_edits:
            break
        trs = subj_triples.get(subj, [])
        if trs:
            t = trs[np.random.randint(len(trs))]
            new_o = t["o"]
            while new_o == t["o"]:
                new_o = entities[np.random.randint(len(entities))]
            edits.append({**t, "o_new": new_o})
            edit_entities.append(subj)

    # Compute union of editor neurons
    editor_neurons = set()
    for e in edit_entities:
        editor_neurons |= entity_neurons.get(e, set())

    print(f"  Editor neurons (union): {len(editor_neurons)}")

    # Apply edits
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

    # Per-entity damage and overlap with editors
    entity_damage = defaultdict(list)
    for i, t in enumerate(test_triples):
        damage = torch.norm(D_star[i] - D0[i]).item()
        entity_damage[t["s"]].append(damage)

    overlap_damage_pairs = []
    for ent, damages in entity_damage.items():
        if ent in entity_neurons and ent not in edit_entities:
            victim_neurons = entity_neurons[ent]
            overlap = len(victim_neurons & editor_neurons) / (len(victim_neurons) + 1e-10)
            overlap_damage_pairs.append({
                "entity": ent,
                "degree": degree.get(ent, 0),
                "mean_damage": np.mean(damages),
                "overlap_with_editors": overlap,
                "n_victim_neurons": len(victim_neurons),
                "n_shared_with_editors": len(victim_neurons & editor_neurons),
            })

    if overlap_damage_pairs:
        overlap_arr = np.array([p["overlap_with_editors"] for p in overlap_damage_pairs])
        damage_arr = np.array([p["mean_damage"] for p in overlap_damage_pairs])
        deg_victim = np.array([p["degree"] for p in overlap_damage_pairs])

        corr_od = np.corrcoef(overlap_arr, damage_arr)[0, 1]
        corr_dd = np.corrcoef(deg_victim, damage_arr)[0, 1]
        print(f"\n  corr(overlap_with_editors, damage): {corr_od:.4f}")
        print(f"  corr(degree, damage): {corr_dd:.4f}")

    del model, rome
    torch.cuda.empty_cache()

    # Save
    save_data = {
        "overlap_matrix": overlap_matrix,
        "overlap_damage_correlation": float(corr_od) if overlap_damage_pairs else None,
        "degree_damage_correlation": float(corr_dd) if overlap_damage_pairs else None,
        "editor_neurons_count": len(editor_neurons),
        "n_edit_entities": len(edit_entities),
        "overlap_damage_sample": overlap_damage_pairs[:200],
    }

    with open(out_dir / "neuron_overlap_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Overlap matrix heatmap
    bins = ["low", "mid", "high"]
    matrix = np.zeros((3, 3))
    for i, a in enumerate(bins):
        for j, b in enumerate(bins):
            key = f"{a}-{b}"
            if key in overlap_matrix:
                matrix[i, j] = overlap_matrix[key]["mean_jaccard"]
    im = axes[0].imshow(matrix, cmap="YlOrRd", vmin=0)
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(bins)
    axes[0].set_yticklabels(bins)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center")
    axes[0].set_title("Jaccard Overlap by Degree Bin")
    axes[0].set_xlabel("Entity B degree")
    axes[0].set_ylabel("Entity A degree")
    fig.colorbar(im, ax=axes[0])

    # Plot 2: Overlap vs Damage
    if overlap_damage_pairs:
        axes[1].scatter(overlap_arr, damage_arr, alpha=0.3, s=10,
                       c=deg_victim, cmap="viridis")
        axes[1].set_xlabel("Neuron Overlap with Editors")
        axes[1].set_ylabel("Mean Logit Damage")
        axes[1].set_title(f"Overlap vs Damage (r={corr_od:.3f})")

    # Plot 3: Degree vs Damage colored by overlap
    if overlap_damage_pairs:
        sc = axes[2].scatter(deg_victim, damage_arr, alpha=0.3, s=10,
                           c=overlap_arr, cmap="coolwarm")
        axes[2].set_xlabel("Victim Degree")
        axes[2].set_ylabel("Mean Logit Damage")
        axes[2].set_title(f"Degree vs Damage (r={corr_dd:.3f})")
        fig.colorbar(sc, ax=axes[2], label="Overlap")

    fig.suptitle("Exp 11: Neuron Overlap & Editing Interference", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "neuron_overlap_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp11_neuron_overlap")
    parser.add_argument("--target-layer", type=int, default=5)
    parser.add_argument("--num-edits", type=int, default=20)
    parser.add_argument("--num-test-triples", type=int, default=2000)
    args = parser.parse_args()

    run_neuron_overlap(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        num_edits=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
