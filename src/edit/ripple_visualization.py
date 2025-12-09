"""Visualization for ripple effect analysis."""

from typing import List, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from matplotlib.patches import FancyBboxPatch

from .ripple_analysis import TripleLogits


def visualize_ripple_on_graph(
    results: List[TripleLogits],
    edited_s: str,
    edited_r: str,
    edited_o_original: str,
    edited_o_target: str,
    graph: nx.MultiDiGraph,
    save_path: Path,
    max_nodes: int = 100,
    figsize: tuple = (20, 16)
):
    """
    Visualize ripple effect on the knowledge graph.

    Args:
        results: List of TripleLogits
        edited_s: Subject of edited triple
        edited_r: Relation of edited triple
        edited_o_original: Original object
        edited_o_target: Target object
        graph: Knowledge graph
        save_path: Path to save visualization
        max_nodes: Maximum number of nodes to display
        figsize: Figure size
    """
    # Build subgraph centered on edited triple
    # Include nodes within certain hop distance
    nodes_to_include = set([edited_s, edited_o_original, edited_o_target])

    # Add nodes from results with significant ripple effect
    ripple_map = {}
    for result in results:
        ripple_map[result.s] = result.ripple_effect
        ripple_map[result.o] = max(ripple_map.get(result.o, 0), result.ripple_effect)

    # Sort by ripple effect and include top nodes
    sorted_entities = sorted(ripple_map.items(), key=lambda x: x[1], reverse=True)
    for entity, _ in sorted_entities[:max_nodes]:
        nodes_to_include.add(entity)

    # Create subgraph
    subgraph = graph.subgraph(nodes_to_include)

    # Map ripple effects to nodes
    node_ripple = {}
    for node in subgraph.nodes():
        node_ripple[node] = ripple_map.get(node, 0)

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    try:
        # Try spring layout with edited entities at center
        pos = nx.spring_layout(
            subgraph,
            k=2.0,
            iterations=50,
            seed=42
        )
    except:
        pos = nx.random_layout(subgraph, seed=42)

    # Normalize ripple effects for coloring
    max_ripple = max(node_ripple.values()) if node_ripple.values() else 1.0
    if max_ripple == 0:
        max_ripple = 1.0

    # Node colors based on ripple effect
    node_colors = []
    node_sizes = []
    for node in subgraph.nodes():
        ripple = node_ripple.get(node, 0)
        intensity = ripple / max_ripple

        # Special coloring for edited entities
        if node == edited_s or node == edited_o_target:
            # Red for edited triple
            node_colors.append('red')
            node_sizes.append(1500)
        elif node == edited_o_original:
            # Orange for original object
            node_colors.append('orange')
            node_sizes.append(1200)
        else:
            # Blue gradient based on ripple effect
            node_colors.append(plt.cm.Blues(0.3 + 0.7 * intensity))
            node_sizes.append(300 + 700 * intensity)

    # Draw edges
    nx.draw_networkx_edges(
        subgraph,
        pos,
        alpha=0.2,
        edge_color='gray',
        arrows=True,
        arrowsize=10,
        arrowstyle='->'
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors='black',
        linewidths=2
    )

    # Draw labels
    labels = {node: node for node in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels,
        font_size=8,
        font_weight='bold'
    )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=15, label=f'Edited: {edited_s}, {edited_o_target}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=12, label=f'Original: {edited_o_original}'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Blues(0.9),
                   markersize=12, label='High ripple effect'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Blues(0.4),
                   markersize=8, label='Low ripple effect')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title
    ax.set_title(
        f'Ripple Effect Visualization\nEdited: ({edited_s}, {edited_r}, {edited_o_original} → {edited_o_target})',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_ripple_statistics(
    results: List[TripleLogits],
    stats: Dict,
    save_path: Path,
    figsize: tuple = (16, 10)
):
    """
    Visualize ripple effect statistics.

    Args:
        results: List of TripleLogits
        stats: Statistics dictionary
        save_path: Path to save visualization
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Ripple effect by hop distance
    ax1 = fig.add_subplot(gs[0, 0])
    hop_stats = stats['by_hop_distance']
    if hop_stats:
        hops = sorted(hop_stats.keys())
        means = [hop_stats[h]['mean_ripple'] for h in hops]
        stds = [hop_stats[h]['std_ripple'] for h in hops]

        ax1.errorbar(hops, means, yerr=stds, marker='o', markersize=8,
                     capsize=5, capthick=2, linewidth=2, color='steelblue')
        ax1.set_xlabel('Hop Distance from Edited Triple', fontsize=12)
        ax1.set_ylabel('Mean Ripple Effect', fontsize=12)
        ax1.set_title('Ripple Effect vs. Graph Distance', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)

    # 2. Ripple effect by subject degree
    ax2 = fig.add_subplot(gs[0, 1])
    degree_stats = stats['by_subject_degree']
    if degree_stats:
        degrees = sorted(degree_stats.keys())
        means = [degree_stats[d]['mean_ripple'] for d in degrees]

        ax2.scatter(degrees, means, s=100, alpha=0.6, color='coral')
        ax2.set_xlabel('Subject Degree', fontsize=12)
        ax2.set_ylabel('Mean Ripple Effect', fontsize=12)
        ax2.set_title('Ripple Effect vs. Node Degree', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        # Add trend line
        if len(degrees) > 1:
            z = np.polyfit(degrees, means, 1)
            p = np.poly1d(z)
            ax2.plot(degrees, p(degrees), "r--", alpha=0.8, linewidth=2)

    # 3. Distribution of ripple effects
    ax3 = fig.add_subplot(gs[1, 0])
    ripple_effects = [r.ripple_effect for r in results]
    ax3.hist(ripple_effects, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(np.mean(ripple_effects), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(ripple_effects):.3f}')
    ax3.set_xlabel('Ripple Effect', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Ripple Effects', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')

    # 4. Scatter: hop distance vs ripple effect
    ax4 = fig.add_subplot(gs[1, 1])
    hops = [r.hop_distance for r in results if r.hop_distance >= 0]
    ripples = [r.ripple_effect for r in results if r.hop_distance >= 0]

    if hops:
        # Use hexbin for density visualization
        hexbin = ax4.hexbin(hops, ripples, gridsize=20, cmap='YlOrRd', mincnt=1)
        ax4.set_xlabel('Hop Distance', fontsize=12)
        ax4.set_ylabel('Ripple Effect', fontsize=12)
        ax4.set_title('Ripple Effect Distribution by Distance', fontsize=14, fontweight='bold')
        cb = plt.colorbar(hexbin, ax=ax4)
        cb.set_label('Count', fontsize=10)

    plt.suptitle('Ripple Effect Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_top_affected_triples(
    results: List[TripleLogits],
    edited_s: str,
    edited_r: str,
    edited_o_target: str,
    save_path: Path,
    top_k: int = 20,
    figsize: tuple = (14, 10)
):
    """
    Visualize top affected triples.

    Args:
        results: List of TripleLogits
        edited_s: Subject of edited triple
        edited_r: Relation of edited triple
        edited_o_target: Target object
        save_path: Path to save visualization
        top_k: Number of top triples to show
        figsize: Figure size
    """
    # Sort by ripple effect
    sorted_results = sorted(results, key=lambda x: x.ripple_effect, reverse=True)
    top_results = sorted_results[:top_k]

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    triple_labels = [f"{r.s[:6]}..{r.r[:4]}..{r.o[:6]}" for r in top_results]
    ripple_values = [r.ripple_effect for r in top_results]
    hop_distances = [r.hop_distance if r.hop_distance >= 0 else -1 for r in top_results]

    # Color by hop distance
    colors = []
    for hop in hop_distances:
        if hop == 0:
            colors.append('red')
        elif hop == 1:
            colors.append('orange')
        elif hop == 2:
            colors.append('yellow')
        elif hop >= 3:
            colors.append('green')
        else:
            colors.append('gray')

    y_pos = np.arange(len(triple_labels))
    bars = ax.barh(y_pos, ripple_values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(triple_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Ripple Effect', fontsize=12)
    ax.set_title(f'Top {top_k} Most Affected Triples\nEdited: ({edited_s}, {edited_r}, {edited_o_target})',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Hop 0 (edited)'),
        plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.7, label='Hop 1'),
        plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.7, label='Hop 2'),
        plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.7, label='Hop 3+'),
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.7, label='Unreachable')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add value labels
    for i, (bar, val, hop) in enumerate(zip(bars, ripple_values, hop_distances)):
        ax.text(val, i, f' {val:.3f} (h={hop})',
                va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
