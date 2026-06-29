"""Exp 1: Entity Embedding Geometry Analysis.

Analyzes whether the KG's graph structure is reflected in the model's
token embedding space.

Key questions:
- Do KG neighbors have similar embeddings?
- Do high-degree entities have different embedding properties?
- Does embedding geometry correlate with graph structure?
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
import seaborn as sns

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


def load_kg_structure(corpus_path):
    """Load KG and compute graph properties."""
    triples = []
    adjacency = defaultdict(set)  # entity -> set of neighbor entities
    degree = defaultdict(int)  # entity -> count of triples where entity is subject

    with open(corpus_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, r, o = parts
                triples.append((s, r, o))
                adjacency[s].add(o)
                adjacency[o].add(s)
                degree[s] += 1

    entities = sorted(set(e for t in triples for e in [t[0], t[2]]))
    return triples, entities, adjacency, degree


def compute_hop_distances_sample(entities, adjacency, sample_size=200, max_hop=4):
    """Compute pairwise hop distances for a sample of entities using BFS."""
    from collections import deque

    sample = np.random.choice(entities, min(sample_size, len(entities)), replace=False)
    distances = {}

    for src in sample:
        visited = {src: 0}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if visited[node] >= max_hop:
                continue
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)
        for dst in sample:
            if dst in visited and src != dst:
                distances[(src, dst)] = visited[dst]

    return distances, list(sample)


def run_embedding_geometry(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp1_embedding_geometry",
    seed=42,
):
    print("=" * 60)
    print("Exp 1: Entity Embedding Geometry Analysis")
    print("=" * 60)

    np.random.seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and KG
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    triples, entities, adjacency, degree = load_kg_structure(kg_corpus)
    print(f"Entities: {len(entities)}, Triples: {len(triples)}")

    # === 1. Extract embeddings ===
    print("\n1. Extracting embeddings...")
    with torch.no_grad():
        all_embeddings = model.token_embed.weight.cpu()  # [vocab, 512]

    entity_ids = [tokenizer.get_id(e) for e in entities]
    entity_embeddings = all_embeddings[entity_ids]  # [N_entities, 512]
    print(f"  Entity embeddings shape: {entity_embeddings.shape}")

    # Also get relation embeddings
    relations = sorted(set(t[1] for t in triples))
    relation_ids = [tokenizer.get_id(r) for r in relations]
    relation_embeddings = all_embeddings[relation_ids]

    # === 2. Degree vs Embedding Norm ===
    print("\n2. Degree vs Embedding Norm...")
    degrees = np.array([degree.get(e, 0) for e in entities])
    norms = entity_embeddings.norm(dim=1).numpy()

    corr_degree_norm = np.corrcoef(degrees, norms)[0, 1]
    print(f"  Correlation(degree, embedding_norm): {corr_degree_norm:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(degrees, norms, alpha=0.3, s=10, c=degrees, cmap="viridis")
    ax.set_xlabel("Degree (as subject)")
    ax.set_ylabel("Embedding L2 Norm")
    ax.set_title(f"Degree vs Embedding Norm (r={corr_degree_norm:.3f})")
    plt.colorbar(scatter, label="Degree")
    fig.tight_layout()
    fig.savefig(out_dir / "degree_vs_norm.png", dpi=150)
    plt.close()

    # === 3. Cosine Similarity vs Hop Distance ===
    print("\n3. Cosine Similarity vs Hop Distance...")
    hop_distances, sample_entities = compute_hop_distances_sample(
        entities, adjacency, sample_size=300, max_hop=4
    )

    sample_ids = [tokenizer.get_id(e) for e in sample_entities]
    sample_embs = all_embeddings[sample_ids]  # [300, 512]

    # Pairwise cosine similarity for sample
    sample_embs_norm = F.normalize(sample_embs, dim=1)
    cos_sim_matrix = (sample_embs_norm @ sample_embs_norm.T).numpy()

    # Collect (hop_distance, cosine_similarity) pairs
    hop_cos_pairs = defaultdict(list)
    entity_to_idx = {e: i for i, e in enumerate(sample_entities)}

    for (src, dst), hop in hop_distances.items():
        if src in entity_to_idx and dst in entity_to_idx:
            i, j = entity_to_idx[src], entity_to_idx[dst]
            hop_cos_pairs[hop].append(cos_sim_matrix[i, j])

    print(f"  Hop distance distribution:")
    for hop in sorted(hop_cos_pairs.keys()):
        vals = hop_cos_pairs[hop]
        print(f"    Hop {hop}: n={len(vals)}, mean_cos={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    hops_list = sorted(hop_cos_pairs.keys())
    means = [np.mean(hop_cos_pairs[h]) for h in hops_list]
    stds = [np.std(hop_cos_pairs[h]) for h in hops_list]
    counts = [len(hop_cos_pairs[h]) for h in hops_list]

    ax.errorbar(hops_list, means, yerr=stds, fmt="o-", capsize=5)
    for h, m, c in zip(hops_list, means, counts):
        ax.annotate(f"n={c}", (h, m), textcoords="offset points", xytext=(10, 5), fontsize=8)
    ax.set_xlabel("Hop Distance in KG")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Embedding Similarity vs KG Hop Distance")
    fig.tight_layout()
    fig.savefig(out_dir / "hop_vs_cosine.png", dpi=150)
    plt.close()

    # === 4. Degree-binned embedding analysis ===
    print("\n4. Degree-binned Embedding Analysis...")
    degree_bins = [(0, 10), (10, 20), (20, 40), (40, 100), (100, 9999)]
    for lo, hi in degree_bins:
        mask = (degrees >= lo) & (degrees < hi)
        if mask.sum() == 0:
            continue
        bin_norms = norms[mask]
        bin_embs = entity_embeddings[mask]
        # Intra-bin cosine similarity
        if bin_embs.shape[0] > 1:
            bin_norm_embs = F.normalize(bin_embs, dim=1)
            intra_cos = (bin_norm_embs @ bin_norm_embs.T).numpy()
            # Exclude diagonal
            np.fill_diagonal(intra_cos, np.nan)
            mean_intra = np.nanmean(intra_cos)
        else:
            mean_intra = float("nan")
        print(f"  Degree [{lo:3d}, {hi:3d}): n={mask.sum():4d}, "
              f"mean_norm={bin_norms.mean():.4f}, "
              f"mean_intra_cos={mean_intra:.4f}")

    # === 5. E vs R embedding separation ===
    print("\n5. Entity vs Relation Embedding Space...")
    e_mean = entity_embeddings.mean(dim=0)
    r_mean = relation_embeddings.mean(dim=0)
    e_r_cos = F.cosine_similarity(e_mean.unsqueeze(0), r_mean.unsqueeze(0)).item()
    print(f"  cos(mean_E, mean_R) = {e_r_cos:.4f}")
    print(f"  Entity embedding norm: mean={entity_embeddings.norm(dim=1).mean():.4f}")
    print(f"  Relation embedding norm: mean={relation_embeddings.norm(dim=1).mean():.4f}")

    # === 6. UMAP Visualization ===
    print("\n6. UMAP Visualization...")
    try:
        import umap

        # Combine entity and relation embeddings for visualization
        all_embs = torch.cat([entity_embeddings, relation_embeddings], dim=0).numpy()
        labels = ["entity"] * len(entities) + ["relation"] * len(relations)

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.3)
        embedding_2d = reducer.fit_transform(all_embs)

        # Plot entities colored by degree
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: colored by type (entity vs relation)
        e_2d = embedding_2d[:len(entities)]
        r_2d = embedding_2d[len(entities):]
        axes[0].scatter(e_2d[:, 0], e_2d[:, 1], c="steelblue", alpha=0.3, s=5, label="Entity")
        axes[0].scatter(r_2d[:, 0], r_2d[:, 1], c="red", alpha=0.5, s=15, label="Relation")
        axes[0].legend()
        axes[0].set_title("UMAP: Entity vs Relation")

        # Right: entities colored by degree
        scatter = axes[1].scatter(
            e_2d[:, 0], e_2d[:, 1],
            c=degrees, cmap="viridis", alpha=0.5, s=10,
            norm=matplotlib.colors.LogNorm(vmin=max(1, degrees.min()), vmax=degrees.max())
        )
        plt.colorbar(scatter, ax=axes[1], label="Degree (log scale)")
        axes[1].set_title("UMAP: Entities colored by Degree")

        fig.tight_layout()
        fig.savefig(out_dir / "umap_embeddings.png", dpi=150)
        plt.close()
        print("  UMAP saved.")
    except ImportError:
        print("  umap-learn not installed, skipping UMAP.")

    # === 7. Adjacency vs Cosine Similarity (Mantel-like test) ===
    print("\n7. Graph Adjacency vs Embedding Similarity...")
    # For sample entities, compute adjacency matrix
    adj_pairs = []
    cos_pairs = []
    for i, e1 in enumerate(sample_entities):
        for j, e2 in enumerate(sample_entities):
            if i < j:
                is_adjacent = 1.0 if e2 in adjacency.get(e1, set()) else 0.0
                adj_pairs.append(is_adjacent)
                cos_pairs.append(cos_sim_matrix[i, j])

    adj_pairs = np.array(adj_pairs)
    cos_pairs = np.array(cos_pairs)

    # Point-biserial correlation
    from scipy.stats import pointbiserialr
    r_pb, p_pb = pointbiserialr(adj_pairs, cos_pairs)
    print(f"  Point-biserial r(adjacency, cos_sim) = {r_pb:.4f}, p = {p_pb:.2e}")

    adjacent_cos = cos_pairs[adj_pairs == 1.0]
    non_adjacent_cos = cos_pairs[adj_pairs == 0.0]
    print(f"  Adjacent pairs: n={len(adjacent_cos)}, mean_cos={adjacent_cos.mean():.4f}")
    print(f"  Non-adjacent:   n={len(non_adjacent_cos)}, mean_cos={non_adjacent_cos.mean():.4f}")

    # === Save summary ===
    summary = {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_triples": len(triples),
        "degree_stats": {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "std": float(degrees.std()),
        },
        "corr_degree_norm": float(corr_degree_norm),
        "hop_vs_cosine": {
            str(h): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for h, v in hop_cos_pairs.items()
        },
        "adjacency_cosine": {
            "point_biserial_r": float(r_pb),
            "p_value": float(p_pb),
            "adjacent_mean_cos": float(adjacent_cos.mean()) if len(adjacent_cos) > 0 else None,
            "non_adjacent_mean_cos": float(non_adjacent_cos.mean()),
        },
        "entity_relation_cos": float(e_r_cos),
    }

    with open(out_dir / "embedding_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {out_dir}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp1_embedding_geometry")
    args = parser.parse_args()

    run_embedding_geometry(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
    )
