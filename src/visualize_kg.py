"""
Knowledge Graph Visualization and Statistics
Generates comprehensive visualizations and statistics for graph.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Set
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def load_graph(graph_path: Path) -> Tuple[List[Dict], nx.DiGraph]:
    """Load graph from jsonl file and create NetworkX graph"""
    triples = []
    G = nx.DiGraph()

    with open(graph_path, 'r') as f:
        for line in f:
            triple = json.loads(line.strip())
            triples.append(triple)
            G.add_edge(triple['s'], triple['o'], relation=triple['r'])

    return triples, G


def compute_statistics(triples: List[Dict], G: nx.DiGraph) -> Dict:
    """Compute various statistics about the knowledge graph"""
    entities = set()
    relations = set()

    for triple in triples:
        entities.add(triple['s'])
        entities.add(triple['o'])
        relations.add(triple['r'])

    # Count reciprocal edges
    reciprocal_count = 0
    edge_pairs = set()

    for triple in triples:
        s, o = triple['s'], triple['o']
        if (o, s) in edge_pairs:
            reciprocal_count += 1
        edge_pairs.add((s, o))

    # Degree statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes()}

    stats = {
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_triples': len(triples),
        'num_reciprocal_pairs': reciprocal_count,
        'reciprocal_ratio': reciprocal_count / len(triples) if len(triples) > 0 else 0,
        'avg_in_degree': np.mean(list(in_degrees.values())),
        'avg_out_degree': np.mean(list(out_degrees.values())),
        'avg_total_degree': np.mean(list(total_degrees.values())),
        'max_in_degree': max(in_degrees.values()),
        'max_out_degree': max(out_degrees.values()),
        'max_total_degree': max(total_degrees.values()),
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G)
    }

    return stats


def plot_kg_overview(G: nx.DiGraph, output_path: Path):
    """Plot KG overview: skeleton + sample subgraph + reachability histogram"""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # 1. Skeleton view (simplified layout of major hubs)
    ax1 = fig.add_subplot(gs[0, 0])
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    top_node_ids = [node for node, _ in top_nodes]
    subgraph = G.subgraph(top_node_ids)

    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    node_sizes = [degree_dict[node] * 10 for node in subgraph.nodes()]
    nx.draw_networkx(subgraph, pos, ax=ax1, node_size=node_sizes,
                     node_color='lightblue', edge_color='gray', alpha=0.6,
                     with_labels=False, arrows=True, arrowsize=5)
    ax1.set_title('KG Skeleton (Top 50 Nodes by Degree)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Random sample subgraph
    ax2 = fig.add_subplot(gs[0, 1])
    all_nodes = list(G.nodes())
    if len(all_nodes) > 30:
        sample_nodes = np.random.choice(all_nodes, 30, replace=False)
    else:
        sample_nodes = all_nodes
    sample_subgraph = G.subgraph(sample_nodes)

    pos2 = nx.spring_layout(sample_subgraph, k=0.8, iterations=50)
    nx.draw_networkx(sample_subgraph, pos2, ax=ax2, node_size=200,
                     node_color='lightgreen', edge_color='gray', alpha=0.6,
                     with_labels=True, font_size=6, arrows=True, arrowsize=5)
    ax2.set_title('Random Sample (30 Nodes)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. Reachability histogram (BFS from random nodes)
    ax3 = fig.add_subplot(gs[0, 2])
    reachability_counts = []

    # Sample 20 random starting nodes
    start_nodes = np.random.choice(all_nodes, min(20, len(all_nodes)), replace=False)
    for start in start_nodes:
        reachable = len(nx.descendants(G, start)) + 1  # +1 for the node itself
        reachability_counts.append(reachable)

    ax3.hist(reachability_counts, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Reachable Nodes', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Reachability Distribution\n(from 20 random nodes)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_degree_distributions(G: nx.DiGraph, output_path: Path):
    """Plot degree distributions: in/out histograms + total degree CCDF (log-log)"""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [in_degrees[i] + out_degrees[i] for i in range(len(in_degrees))]

    # 1. In-degree histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(in_degrees, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('In-Degree', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('In-Degree Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'Mean: {np.mean(in_degrees):.2f}\nMax: {max(in_degrees)}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Out-degree histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(out_degrees, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Out-Degree', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.95, f'Mean: {np.mean(out_degrees):.2f}\nMax: {max(out_degrees)}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Total degree CCDF (log-log)
    ax3 = fig.add_subplot(gs[0, 2])
    degree_counts = Counter(total_degrees)
    degrees = sorted(degree_counts.keys())
    ccdf = []
    total_nodes = len(total_degrees)

    for d in degrees:
        ccdf_value = sum(degree_counts[k] for k in degree_counts if k >= d) / total_nodes
        ccdf.append(ccdf_value)

    ax3.loglog(degrees, ccdf, 'o-', color='purple', markersize=4, alpha=0.7)
    ax3.set_xlabel('Total Degree (k)', fontsize=10)
    ax3.set_ylabel('P(X ≥ k) [CCDF]', fontsize=10)
    ax3.set_title('Total Degree CCDF (Log-Log)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_hop_distances(G: nx.DiGraph, start_nodes: List[str]) -> Tuple[np.ndarray, List[float]]:
    """Compute hop distances from start nodes using BFS"""
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)

    hop_matrix = np.full((len(start_nodes), n), np.inf)
    reachability_curves = []

    for i, start in enumerate(start_nodes):
        if start not in node_to_idx:
            continue

        # BFS to compute shortest paths
        distances = nx.single_source_shortest_path_length(G, start)

        for node, dist in distances.items():
            if node in node_to_idx:
                hop_matrix[i, node_to_idx[node]] = dist

        # Compute reachability curve
        max_hops = int(np.max([d for d in distances.values() if d != np.inf]))
        reachability = []
        for h in range(max_hops + 1):
            reachable_count = sum(1 for d in distances.values() if d <= h)
            reachability.append(reachable_count / n * 100)  # percentage
        reachability_curves.append(reachability)

    return hop_matrix, reachability_curves


def plot_hop_distributions(G: nx.DiGraph, output_path: Path):
    """Plot hop distributions: heatmap + reachability curves"""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Select 10 random start nodes
    all_nodes = list(G.nodes())
    start_nodes = np.random.choice(all_nodes, min(10, len(all_nodes)), replace=False).tolist()

    hop_matrix, reachability_curves = compute_hop_distances(G, start_nodes)

    # 1. Hop distance heatmap
    ax1 = fig.add_subplot(gs[0, 0])

    # For visualization, limit to finite values and clip at max 10 hops
    hop_display = np.where(np.isinf(hop_matrix), 11, hop_matrix)
    hop_display = np.clip(hop_display, 0, 11)

    im = ax1.imshow(hop_display, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Target Node Index', fontsize=10)
    ax1.set_ylabel('Start Node Index', fontsize=10)
    ax1.set_title('Hop Distance Heatmap\n(10 random start nodes)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(start_nodes)))
    ax1.set_yticklabels([f'Node {i}' for i in range(len(start_nodes))])

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Hops (11 = unreachable)', fontsize=9)

    # 2. Reachability curves
    ax2 = fig.add_subplot(gs[0, 1])

    for i, curve in enumerate(reachability_curves):
        ax2.plot(range(len(curve)), curve, marker='o', markersize=3,
                label=f'Start {i}', alpha=0.7)

    ax2.set_xlabel('Number of Hops', fontsize=10)
    ax2.set_ylabel('Reachability (%)', fontsize=10)
    ax2.set_title('Reachability Curves\n(% of nodes reached within k hops)',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Knowledge Graph')
    parser.add_argument('--graph_path', type=str, required=True,
                        help='Path to graph.jsonl file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {args.graph_path}...")
    triples, G = load_graph(Path(args.graph_path))
    print(f"Loaded {len(triples)} triples, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(triples, G)

    # Save statistics
    stats_path = output_dir / 'kg_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Print key statistics
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Entities:          {stats['num_entities']:,}")
    print(f"Relations:         {stats['num_relations']:,}")
    print(f"Triples:           {stats['num_triples']:,}")
    print(f"Reciprocal pairs:  {stats['num_reciprocal_pairs']:,}")
    print(f"Reciprocal ratio:  {stats['reciprocal_ratio']:.4f}")
    print(f"Avg in-degree:     {stats['avg_in_degree']:.2f}")
    print(f"Avg out-degree:    {stats['avg_out_degree']:.2f}")
    print(f"Avg total degree:  {stats['avg_total_degree']:.2f}")
    print(f"Graph density:     {stats['density']:.6f}")
    print("="*60)

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("  - KG Overview...")
    plot_kg_overview(G, output_dir / 'kg_overview.png')

    print("  - Degree Distributions...")
    plot_degree_distributions(G, output_dir / 'degree_distributions.png')

    print("  - Hop Distributions...")
    plot_hop_distributions(G, output_dir / 'hop_distributions.png')

    print(f"\nAll outputs saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
