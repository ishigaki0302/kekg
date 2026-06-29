"""Exp 3: FFN Knowledge Neuron Analysis.

Identifies FFN neurons specialized for specific entities and examines
the relationship between neuron specialization and entity degree.

Key questions:
- Do high-degree entities activate more neurons or fewer specialized ones?
- Is neuron specialization concentrated in specific layers?
- How does the neuron activation pattern relate to degree?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
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


def run_knowledge_neurons(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:1",
    output_dir="outputs/exp3_knowledge_neurons",
    num_triples=10000,
    target_layers=None,
    seed=42,
):
    print("=" * 60)
    print("Exp 3: FFN Knowledge Neuron Analysis")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, degree, subj_triples = load_kg(kg_corpus)
    n_layers = model.config.n_layers
    d_mlp = model.config.d_mlp

    if target_layers is None:
        target_layers = [3, 4, 5, 6, 7, 8]  # Focus on middle layers

    entities = sorted(set(t["s"] for t in all_triples))
    print(f"Entities: {len(entities)}, Triples: {len(all_triples)}")
    print(f"Target layers: {target_layers}")

    # Sample triples
    np.random.shuffle(all_triples)
    sample_triples = all_triples[:num_triples]
    print(f"Using {len(sample_triples)} triples")

    # === Step 1: Collect FFN activations per entity per layer ===
    print("\nStep 1: Collecting FFN activations...")

    # For each target layer, accumulate entity -> activation sum and count
    entity_activation_sum = {l: defaultdict(lambda: torch.zeros(d_mlp)) for l in target_layers}
    entity_activation_count = {l: defaultdict(int) for l in target_layers}

    # Register hooks for all target layers
    activation_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # w1 output before GELU is the input; output is w1(x)
            # We want post-GELU activation = GELU(w1(x))
            activation_cache[layer_idx] = F.gelu(output).detach()
        return hook_fn

    handles = []
    for l in target_layers:
        h = model.blocks[l].ffn.w1.register_forward_hook(make_hook(l))
        handles.append(h)

    batch_size = 256
    for start in range(0, len(sample_triples), batch_size):
        batch = sample_triples[start:start + batch_size]
        input_ids = torch.tensor(
            [tokenizer.encode(f"{t['s']} {t['r']}") for t in batch],
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            model(input_ids)

        # Accumulate activations per entity
        for i, t in enumerate(batch):
            for l in target_layers:
                act = activation_cache[l][i, 1, :].cpu()  # R position
                entity_activation_sum[l][t["s"]] += act
                entity_activation_count[l][t["s"]] += 1

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {min(start + batch_size, len(sample_triples))}/{len(sample_triples)}")

    for h in handles:
        h.remove()

    # Compute mean activations per entity per layer
    print("\nStep 2: Computing mean activations and z-scores...")

    layer_results = {}

    for l in target_layers:
        # Entity mean activation matrix: [n_entities, d_mlp]
        ent_list = sorted(entity_activation_sum[l].keys())
        mean_acts = torch.stack([
            entity_activation_sum[l][e] / max(entity_activation_count[l][e], 1)
            for e in ent_list
        ])  # [n_entities, d_mlp]

        # Global statistics per neuron
        global_mean = mean_acts.mean(dim=0)  # [d_mlp]
        global_std = mean_acts.std(dim=0) + 1e-8  # [d_mlp]

        # Z-scores: how specialized each neuron is for each entity
        z_scores = (mean_acts - global_mean) / global_std  # [n_entities, d_mlp]

        # Knowledge neurons: neurons with |z| > threshold for a given entity
        threshold = 3.0  # z > 3 = "specialized"

        # Per-entity analysis
        entity_stats = []
        for i, ent in enumerate(ent_list):
            z = z_scores[i]
            n_specialized = (z.abs() > threshold).sum().item()
            max_z = z.abs().max().item()
            mean_activation = mean_acts[i].mean().item()
            activation_sparsity = (mean_acts[i] < 0.01).float().mean().item()

            entity_stats.append({
                "entity": ent,
                "degree": degree.get(ent, 0),
                "n_specialized_neurons": n_specialized,
                "max_z_score": max_z,
                "mean_activation": mean_activation,
                "activation_sparsity": activation_sparsity,
                "n_triples_seen": entity_activation_count[l][ent],
            })

        # Neuron-level analysis: how many entities is each neuron specialized for?
        neuron_specialization = (z_scores.abs() > threshold).sum(dim=0)  # [d_mlp]

        # Top shared neurons (activated by many entities)
        top_shared = neuron_specialization.topk(20)
        # Top specific neurons (activated by few entities)
        specialized_mask = neuron_specialization > 0
        if specialized_mask.sum() > 0:
            specific_counts = neuron_specialization[specialized_mask]
            specific_indices = torch.where(specialized_mask)[0]
            bottom_k = min(20, len(specific_counts))
            _, bottom_idx = specific_counts.topk(bottom_k, largest=False)
            top_specific = [(specific_indices[i].item(), specific_counts[i].item()) for i in bottom_idx]
        else:
            top_specific = []

        layer_results[l] = {
            "entity_stats": entity_stats,
            "n_active_neurons": int(specialized_mask.sum()),
            "neuron_specialization_mean": float(neuron_specialization[specialized_mask].float().mean()) if specialized_mask.any() else 0,
            "neuron_specialization_std": float(neuron_specialization[specialized_mask].float().std()) if specialized_mask.any() else 0,
            "top_shared_neurons": [
                {"neuron_idx": int(top_shared.indices[i]), "n_entities": int(top_shared.values[i])}
                for i in range(len(top_shared.indices))
            ],
            "top_specific_neurons": [
                {"neuron_idx": int(idx), "n_entities": int(cnt)}
                for idx, cnt in top_specific
            ],
        }

        # Print summary
        degrees_arr = np.array([s["degree"] for s in entity_stats])
        n_spec_arr = np.array([s["n_specialized_neurons"] for s in entity_stats])

        corr = np.corrcoef(degrees_arr, n_spec_arr)[0, 1] if len(degrees_arr) > 2 else 0
        print(f"\n  Layer {l}:")
        print(f"    Active neurons (z>{threshold} for any entity): {specialized_mask.sum()}/{d_mlp}")
        print(f"    corr(degree, n_specialized_neurons): {corr:.4f}")
        print(f"    Mean specialized neurons per entity: {n_spec_arr.mean():.1f}")

        # Degree-binned stats
        for lo, hi in [(1, 15), (15, 35), (35, 999)]:
            mask = (degrees_arr >= lo) & (degrees_arr < hi)
            if mask.sum() > 0:
                print(f"    Degree [{lo:3d}, {hi:3d}): n={mask.sum():4d}, "
                      f"mean_specialized={n_spec_arr[mask].mean():.1f}, "
                      f"sparsity={np.mean([s['activation_sparsity'] for s, m in zip(entity_stats, mask) if m]):.3f}")

    # Save results
    # Convert to serializable format
    save_data = {}
    for l, data in layer_results.items():
        save_data[str(l)] = {
            "n_active_neurons": data["n_active_neurons"],
            "neuron_specialization_mean": data["neuron_specialization_mean"],
            "entity_stats": data["entity_stats"],
            "top_shared_neurons": data["top_shared_neurons"][:10],
        }

    with open(out_dir / "knowledge_neuron_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # === Visualization ===
    print("\nGenerating plots...")
    n_plots = len(target_layers)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, l in enumerate(target_layers):
        if idx >= len(axes):
            break
        stats = layer_results[l]["entity_stats"]
        degrees_arr = [s["degree"] for s in stats]
        n_spec_arr = [s["n_specialized_neurons"] for s in stats]

        ax = axes[idx]
        scatter = ax.scatter(degrees_arr, n_spec_arr, alpha=0.3, s=10,
                           c=degrees_arr, cmap="viridis")
        ax.set_xlabel("Entity Degree")
        ax.set_ylabel("# Specialized Neurons")
        corr = np.corrcoef(degrees_arr, n_spec_arr)[0, 1]
        ax.set_title(f"Layer {l} (r={corr:.3f})")

    fig.suptitle("Knowledge Neurons: Degree vs Specialization", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "knowledge_neuron_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return layer_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", default="outputs/exp3_knowledge_neurons")
    parser.add_argument("--num-triples", type=int, default=10000)
    args = parser.parse_args()

    run_knowledge_neurons(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_triples=args.num_triples,
    )
