#!/usr/bin/env python3
"""
Analyze ripple effects from multiple editing experiments.
Plot logit changes by degree and hop distance (subject, before_object, after_object).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import networkx as nx

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_knowledge_graph(kg_file):
    """Load knowledge graph and compute degrees."""
    G = nx.DiGraph()
    triples = []

    with open(kg_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            s, r, o = data['s'], data['r'], data['o']
            G.add_edge(s, o)
            triples.append((s, r, o))

    # Compute degrees (in + out)
    degrees = {}
    for node in G.nodes():
        degrees[node] = G.in_degree(node) + G.out_degree(node)

    return G, degrees, triples

def compute_hop_distances(G, source_nodes):
    """Compute hop distances from multiple source nodes."""
    hop_distances = {}

    for node in G.nodes():
        min_dist = float('inf')
        for source in source_nodes:
            if source in G:
                try:
                    dist = nx.shortest_path_length(G.to_undirected(), source, node)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    pass
        hop_distances[node] = min_dist if min_dist != float('inf') else None

    return hop_distances

def analyze_ripple_experiments():
    """Analyze all ripple experiments."""

    # Load knowledge graph
    kg_file = 'data/kg/ba/graph.jsonl'
    print(f"Loading knowledge graph from {kg_file}...")
    G, degrees, triples = load_knowledge_graph(kg_file)
    print(f"Loaded {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Load all experiments
    exp_dirs = sorted(Path('outputs').glob('ripple_exp_*'))
    print(f"\nFound {len(exp_dirs)} experiments")

    all_data = []

    for exp_dir in exp_dirs:
        exp_id = exp_dir.name.split('_')[-1]

        # Load ripple analysis
        ripple_file = exp_dir / 'ripple_analysis.json'
        if not ripple_file.exists():
            print(f"Warning: {ripple_file} not found, skipping")
            continue

        with open(ripple_file, 'r') as f:
            ripple_data = json.load(f)

        # Load edit result
        edit_file = exp_dir / 'edit_result.json'
        with open(edit_file, 'r') as f:
            edit_result = json.load(f)

        subject = edit_result['s']
        before_obj = edit_result['o_original']
        after_obj = edit_result['o_target']

        print(f"  Exp {exp_id}: {subject} -> {before_obj} changed to {after_obj}")

        # Compute hop distances for three bases
        hop_from_subject = compute_hop_distances(G, [subject])
        hop_from_before = compute_hop_distances(G, [before_obj])
        hop_from_after = compute_hop_distances(G, [after_obj])

        # Process each affected triple
        for triple_data in ripple_data['triples']:
            s = triple_data['s']
            o = triple_data['o']
            logit_change = abs(triple_data['ripple_effect'])

            # Get degree (use subject degree as representative)
            degree = degrees.get(s, 0)

            # Get hop distances
            hop_s_subj = hop_from_subject.get(s, None)
            hop_o_subj = hop_from_subject.get(o, None)
            hop_subj = min([h for h in [hop_s_subj, hop_o_subj] if h is not None], default=None)

            hop_s_before = hop_from_before.get(s, None)
            hop_o_before = hop_from_before.get(o, None)
            hop_before = min([h for h in [hop_s_before, hop_o_before] if h is not None], default=None)

            hop_s_after = hop_from_after.get(s, None)
            hop_o_after = hop_from_after.get(o, None)
            hop_after = min([h for h in [hop_s_after, hop_o_after] if h is not None], default=None)

            all_data.append({
                'exp_id': exp_id,
                'degree': degree,
                'hop_subject': hop_subj,
                'hop_before': hop_before,
                'hop_after': hop_after,
                'logit_change': logit_change
            })

    print(f"\nTotal data points: {len(all_data)}")
    return all_data

def create_plots(all_data):
    """Create visualization plots."""

    # Convert to arrays
    degrees = np.array([d['degree'] for d in all_data])
    hop_subject = np.array([d['hop_subject'] for d in all_data])
    hop_before = np.array([d['hop_before'] for d in all_data])
    hop_after = np.array([d['hop_after'] for d in all_data])
    logit_changes = np.array([d['logit_change'] for d in all_data])

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 1. Logit change by degree
    ax1 = plt.subplot(2, 3, 1)
    degree_bins = [0, 5, 10, 15, 20, 30, 50, 100]
    degree_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-50', '50+']
    degree_binned = np.digitize(degrees, degree_bins)

    degree_means = []
    degree_stds = []
    degree_x = []
    for i in range(1, len(degree_bins)):
        mask = degree_binned == i
        if mask.sum() > 0:
            degree_means.append(logit_changes[mask].mean())
            degree_stds.append(logit_changes[mask].std())
            degree_x.append(i-1)

    ax1.bar(degree_x, degree_means, yerr=degree_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Degree Range', fontsize=12)
    ax1.set_ylabel('Mean Logit Change', fontsize=12)
    ax1.set_title('Ripple Effect by Node Degree', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(degree_labels)))
    ax1.set_xticklabels(degree_labels, rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 2-4. Logit change by hop distance (3 bases)
    hop_configs = [
        (hop_subject, 'hop_subject', 'Subject-based Hop Distance', 'coral'),
        (hop_before, 'hop_before', 'Before-Object-based Hop Distance', 'lightgreen'),
        (hop_after, 'hop_after', 'After-Object-based Hop Distance', 'plum')
    ]

    for idx, (hop_data, name, title, color) in enumerate(hop_configs, start=2):
        ax = plt.subplot(2, 3, idx)

        # Filter valid hop data
        valid_mask = ~np.isnan(hop_data) & (hop_data < 10)
        valid_hops = hop_data[valid_mask]
        valid_logits = logit_changes[valid_mask]

        if len(valid_hops) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        # Compute statistics by hop
        max_hop = int(valid_hops.max())
        hop_means = []
        hop_stds = []
        hop_counts = []
        hop_x = []

        for h in range(max_hop + 1):
            mask = valid_hops == h
            if mask.sum() > 0:
                hop_means.append(valid_logits[mask].mean())
                hop_stds.append(valid_logits[mask].std())
                hop_counts.append(mask.sum())
                hop_x.append(h)

        # Plot
        ax.bar(hop_x, hop_means, yerr=hop_stds, capsize=5, alpha=0.7, color=color)

        # Add count labels
        for i, (x, count) in enumerate(zip(hop_x, hop_counts)):
            ax.text(x, hop_means[i] + hop_stds[i] + 0.3, f'n={count}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Hop Distance', fontsize=12)
        ax.set_ylabel('Mean Logit Change', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(hop_x)
        ax.grid(axis='y', alpha=0.3)

    # 5. Scatter: Degree vs Logit Change
    ax5 = plt.subplot(2, 3, 5)

    # Sample if too many points
    if len(degrees) > 2000:
        sample_idx = np.random.choice(len(degrees), 2000, replace=False)
        degrees_plot = degrees[sample_idx]
        logits_plot = logit_changes[sample_idx]
    else:
        degrees_plot = degrees
        logits_plot = logit_changes

    ax5.scatter(degrees_plot, logits_plot, alpha=0.3, s=20, color='steelblue')

    # Add trend line
    z = np.polyfit(degrees[degrees < 100], logit_changes[degrees < 100], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, min(100, degrees.max()), 100)
    ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')

    ax5.set_xlabel('Node Degree', fontsize=12)
    ax5.set_ylabel('Logit Change', fontsize=12)
    ax5.set_title('Degree vs Ripple Effect', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Comparison of three hop bases
    ax6 = plt.subplot(2, 3, 6)

    hop_comparison_data = []
    hop_comparison_labels = []

    for hop_data, name, _, color in hop_configs:
        valid_mask = ~np.isnan(hop_data) & (hop_data < 10)
        if valid_mask.sum() > 0:
            for h in range(int(hop_data[valid_mask].max()) + 1):
                mask = (hop_data == h) & valid_mask
                if mask.sum() > 0:
                    hop_comparison_data.append({
                        'base': name.replace('hop_', ''),
                        'hop': h,
                        'mean': logit_changes[mask].mean()
                    })

    # Plot grouped bar chart
    bases = ['subject', 'before', 'after']
    colors_comp = ['coral', 'lightgreen', 'plum']
    max_hop_comp = max([d['hop'] for d in hop_comparison_data])

    x = np.arange(max_hop_comp + 1)
    width = 0.25

    for i, (base, color) in enumerate(zip(bases, colors_comp)):
        means = []
        for h in range(max_hop_comp + 1):
            matching = [d['mean'] for d in hop_comparison_data if d['base'] == base and d['hop'] == h]
            means.append(matching[0] if matching else 0)
        ax6.bar(x + i * width, means, width, label=base.capitalize(), alpha=0.7, color=color)

    ax6.set_xlabel('Hop Distance', fontsize=12)
    ax6.set_ylabel('Mean Logit Change', fontsize=12)
    ax6.set_title('Comparison of Hop Bases', fontsize=14, fontweight='bold')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(x)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = 'outputs/ripple_effects_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Save statistics
    stats = {
        'total_data_points': len(all_data),
        'degree_stats': {
            'mean': float(degrees.mean()),
            'std': float(degrees.std()),
            'min': int(degrees.min()),
            'max': int(degrees.max())
        },
        'logit_change_stats': {
            'mean': float(logit_changes.mean()),
            'std': float(logit_changes.std()),
            'min': float(logit_changes.min()),
            'max': float(logit_changes.max())
        },
        'hop_subject_stats': {
            'mean': float(np.nanmean(hop_subject)),
            'valid_count': int((~np.isnan(hop_subject)).sum())
        },
        'hop_before_stats': {
            'mean': float(np.nanmean(hop_before)),
            'valid_count': int((~np.isnan(hop_before)).sum())
        },
        'hop_after_stats': {
            'mean': float(np.nanmean(hop_after)),
            'valid_count': int((~np.isnan(hop_after)).sum())
        }
    }

    stats_file = 'outputs/ripple_effects_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")

if __name__ == '__main__':
    print("=" * 70)
    print("Ripple Effects Analysis")
    print("=" * 70)

    all_data = analyze_ripple_experiments()
    create_plots(all_data)

    print("\nAnalysis complete!")
