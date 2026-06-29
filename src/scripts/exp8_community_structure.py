"""Exp 8: Community Structure & Editing Interference.

Tests whether ROME edits primarily disrupt within-community triples
or cross-community triples. Uses Louvain community detection on the
BA graph and measures R(D*) broken down by community membership.

Key hypothesis: Editing a node damages same-community triples more,
and this effect is amplified for high-degree (hub) nodes that bridge
communities.
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
    load_model_and_tokenizer, load_kg,
    compute_logit_matrix, compute_shattering,
)


def detect_communities(adjacency):
    """Louvain community detection using networkx."""
    import networkx as nx

    G = nx.Graph()
    for node, neighbors in adjacency.items():
        for nb in neighbors:
            G.add_edge(node, nb)

    communities = nx.community.louvain_communities(G, seed=42)
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    return node_community, len(communities)


def compute_betweenness(adjacency, sample_k=500):
    """Approximate betweenness centrality."""
    import networkx as nx

    G = nx.Graph()
    for node, neighbors in adjacency.items():
        for nb in neighbors:
            G.add_edge(node, nb)

    bc = nx.betweenness_centrality(G, k=min(sample_k, len(G)))
    return bc


def run_community_structure(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp8_community_structure",
    num_edits_per_condition=30,
    num_test_triples=3000,
    seed=42,
):
    print("=" * 60)
    print("Exp 8: Community Structure & Editing Interference")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_orig, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, adjacency, degree = load_kg(kg_corpus)
    entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))

    # Community detection
    print("\nDetecting communities...")
    node_community, n_communities = detect_communities(adjacency)
    print(f"  Found {n_communities} communities")

    # Community sizes
    comm_sizes = defaultdict(int)
    for _, c in node_community.items():
        comm_sizes[c] += 1
    sizes = sorted(comm_sizes.values(), reverse=True)
    print(f"  Top 5 community sizes: {sizes[:5]}")

    # Betweenness centrality
    print("Computing betweenness centrality...")
    betweenness = compute_betweenness(adjacency)
    print(f"  Max betweenness: {max(betweenness.values()):.4f}")
    print(f"  Mean betweenness: {np.mean(list(betweenness.values())):.4f}")

    # Classify entities: hub (high degree + high betweenness) vs peripheral
    degree_arr = np.array([degree.get(e, 0) for e in entities])
    bc_arr = np.array([betweenness.get(e, 0) for e in entities])
    degree_median = np.median(degree_arr)
    bc_median = np.median(bc_arr)

    hubs = [e for e in entities
            if degree.get(e, 0) > degree_median and betweenness.get(e, 0) > bc_median]
    peripherals = [e for e in entities
                   if degree.get(e, 0) <= degree_median and betweenness.get(e, 0) <= bc_median]

    print(f"\n  Hubs (high-degree + high-betweenness): {len(hubs)}")
    print(f"  Peripherals (low-degree + low-betweenness): {len(peripherals)}")

    # Test triples
    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]

    # Classify test triples by community relationship
    intra_community = []
    inter_community = []
    for i, t in enumerate(test_triples):
        c_s = node_community.get(t["s"], -1)
        c_o = node_community.get(t["o"], -1)
        if c_s == c_o and c_s >= 0:
            intra_community.append(i)
        elif c_s >= 0 and c_o >= 0:
            inter_community.append(i)

    print(f"\n  Test triples: intra-community={len(intra_community)}, "
          f"inter-community={len(inter_community)}")

    # Baseline
    print("\nComputing baseline...")
    D0 = compute_logit_matrix(model_orig, tokenizer, test_triples, device)

    # Subject triple index
    subj_triples = defaultdict(list)
    for t in all_triples:
        subj_triples[t["s"]].append(t)

    results = {}

    # Two editing conditions: hubs vs peripherals
    for condition_name, subject_pool in [("hub", hubs), ("peripheral", peripherals)]:
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name} (pool={len(subject_pool)})")
        print(f"{'='*60}")

        np.random.shuffle(subject_pool)

        # Select edits
        edits = []
        for subj in subject_pool:
            if len(edits) >= num_edits_per_condition:
                break
            trs = subj_triples.get(subj, [])
            if trs:
                t = trs[np.random.randint(len(trs))]
                new_o = t["o"]
                while new_o == t["o"]:
                    new_o = entities[np.random.randint(len(entities))]
                edits.append({**t, "o_new": new_o})

        print(f"  Selected {len(edits)} edits")

        model = deepcopy(model_orig).to(device)
        model.eval()

        rome = ROME(
            model=model, tokenizer=tokenizer, device=device,
            kg_corpus_path=kg_corpus, mom2_n_samples=1000,
            use_mom2_adjustment=True, v_num_grad_steps=20,
        )

        edit_communities = []
        successes = 0
        for edit in edits:
            model, result = rome.apply_edit(
                s=edit["s"], r=edit["r"], o_target=edit["o_new"],
                layer=5, copy_model=False,
            )
            if result.success:
                successes += 1
            edit_communities.append(node_community.get(edit["s"], -1))

        D_star = compute_logit_matrix(model, tokenizer, test_triples, device)

        # Global shattering
        R_global = compute_shattering(D_star, D0)

        # Intra vs inter community shattering
        R_intra = compute_shattering(
            D_star[torch.tensor(intra_community)],
            D0[torch.tensor(intra_community)],
        ) if len(intra_community) >= 5 else 0

        R_inter = compute_shattering(
            D_star[torch.tensor(inter_community)],
            D0[torch.tensor(inter_community)],
        ) if len(inter_community) >= 5 else 0

        # Same-community-as-editor vs different
        edited_comms = set(edit_communities)
        same_comm_idx = [i for i, t in enumerate(test_triples)
                         if node_community.get(t["s"], -1) in edited_comms]
        diff_comm_idx = [i for i, t in enumerate(test_triples)
                         if node_community.get(t["s"], -1) not in edited_comms
                         and node_community.get(t["s"], -1) >= 0]

        R_same_comm = compute_shattering(
            D_star[torch.tensor(same_comm_idx)],
            D0[torch.tensor(same_comm_idx)],
        ) if len(same_comm_idx) >= 5 else 0

        R_diff_comm = compute_shattering(
            D_star[torch.tensor(diff_comm_idx)],
            D0[torch.tensor(diff_comm_idx)],
        ) if len(diff_comm_idx) >= 5 else 0

        # Also break down by victim degree
        R_by_victim_degree = {}
        for dbin, (lo, hi) in [("low", (1, 15)), ("mid", (15, 35)), ("high", (35, 999))]:
            idx = [i for i, t in enumerate(test_triples) if lo <= degree.get(t["s"], 0) < hi]
            if len(idx) >= 5:
                R_by_victim_degree[dbin] = float(compute_shattering(
                    D_star[torch.tensor(idx)], D0[torch.tensor(idx)]))

        condition_result = {
            "n_edits": len(edits),
            "edit_success_rate": successes / len(edits),
            "R_global": float(R_global),
            "R_intra_community": float(R_intra),
            "R_inter_community": float(R_inter),
            "R_same_comm_as_editor": float(R_same_comm),
            "R_diff_comm_from_editor": float(R_diff_comm),
            "R_by_victim_degree": R_by_victim_degree,
            "n_intra": len(intra_community),
            "n_inter": len(inter_community),
            "n_same_comm": len(same_comm_idx),
            "n_diff_comm": len(diff_comm_idx),
            "edited_communities": [int(c) for c in edit_communities],
        }

        results[condition_name] = condition_result
        print(f"  R_global: {R_global:.4f}")
        print(f"  R_intra_community: {R_intra:.4f}")
        print(f"  R_inter_community: {R_inter:.4f}")
        print(f"  R_same_comm_as_editor: {R_same_comm:.4f}")
        print(f"  R_diff_comm_from_editor: {R_diff_comm:.4f}")
        for k, v in R_by_victim_degree.items():
            print(f"  R_victim_{k}: {v:.4f}")

        del model, rome
        torch.cuda.empty_cache()

    # Add graph metadata
    results["graph_metadata"] = {
        "n_communities": n_communities,
        "community_sizes": {str(k): v for k, v in sorted(comm_sizes.items())},
        "n_hubs": len(hubs),
        "n_peripherals": len(peripherals),
        "betweenness_stats": {
            "mean": float(np.mean(list(betweenness.values()))),
            "std": float(np.std(list(betweenness.values()))),
            "max": float(max(betweenness.values())),
        },
        "degree_betweenness_corr": float(np.corrcoef(degree_arr, bc_arr)[0, 1]),
    }

    # Save
    with open(out_dir / "community_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: R breakdown by community relationship
    categories = ["Global", "Intra-comm", "Inter-comm", "Same as editor", "Diff from editor"]
    for condition in ["hub", "peripheral"]:
        r = results[condition]
        vals = [r["R_global"], r["R_intra_community"], r["R_inter_community"],
                r["R_same_comm_as_editor"], r["R_diff_comm_from_editor"]]
        x = np.arange(len(categories))
        width = 0.35
        offset = -width/2 if condition == "hub" else width/2
        color = "red" if condition == "hub" else "blue"
        axes[0].bar(x + offset, vals, width, label=condition, color=color, alpha=0.7)

    axes[0].set_xticks(np.arange(len(categories)))
    axes[0].set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("R(D*)")
    axes[0].set_title("Shattering by Community Relationship")
    axes[0].legend()

    # Plot 2: Degree vs Betweenness scatter
    axes[1].scatter(degree_arr, bc_arr, alpha=0.3, s=10)
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Betweenness Centrality")
    corr_db = results["graph_metadata"]["degree_betweenness_corr"]
    axes[1].set_title(f"Degree vs Betweenness (r={corr_db:.3f})")

    # Plot 3: Victim degree breakdown
    for condition, color in [("hub", "red"), ("peripheral", "blue")]:
        r = results[condition]["R_by_victim_degree"]
        bins_list = ["low", "mid", "high"]
        vals = [r.get(b, 0) for b in bins_list]
        x = np.arange(len(bins_list))
        width = 0.35
        offset = -width/2 if condition == "hub" else width/2
        axes[2].bar(x + offset, vals, width, label=condition, color=color, alpha=0.7)

    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(["low", "mid", "high"])
    axes[2].set_ylabel("R(D*)")
    axes[2].set_title("Shattering by Victim Degree")
    axes[2].legend()

    fig.suptitle("Exp 8: Community Structure & Editing Interference", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "community_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp8_community_structure")
    parser.add_argument("--num-edits", type=int, default=30)
    parser.add_argument("--num-test-triples", type=int, default=3000)
    args = parser.parse_args()

    run_community_structure(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits_per_condition=args.num_edits,
        num_test_triples=args.num_test_triples,
    )
