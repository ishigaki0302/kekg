"""Visualization utilities for analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA

from .seqedit import CKEReport


def plot_degree_vs_impact(
    degrees: Dict[str, int],
    impacts: Dict[str, float],
    output_path: Union[str, Path],
    title: str = "Degree vs Impact"
) -> None:
    """
    Plot scatter of entity degree vs edit impact.

    Args:
        degrees: Entity -> degree
        impacts: Entity -> impact score
        output_path: Output file path
        title: Plot title
    """
    # Align entities
    entities = list(set(degrees.keys()) & set(impacts.keys()))
    degree_vals = [degrees[e] for e in entities]
    impact_vals = [impacts[e] for e in entities]

    plt.figure(figsize=(8, 6))
    plt.scatter(degree_vals, impact_vals, alpha=0.6, s=50)
    plt.xlabel("Entity Degree")
    plt.ylabel("Impact Score (I)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(degree_vals) > 1:
        z = np.polyfit(degree_vals, impact_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(degree_vals), max(degree_vals), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.7, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
        plt.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: Union[str, Path],
    title: str = "Embedding PCA",
    n_components: int = 2
) -> None:
    """
    Plot PCA of embeddings.

    Args:
        embeddings: Embedding matrix [n_samples, d_model]
        labels: List of labels for each embedding
        output_path: Output path
        title: Plot title
        n_components: Number of PCA components
    """
    # PCA
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if n_components == 2:
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=50)

        # Annotate some points
        for i, label in enumerate(labels):
            if i % max(1, len(labels) // 20) == 0:  # Annotate every Nth point
                plt.annotate(label, (proj[i, 0], proj[i, 1]), fontsize=8, alpha=0.7)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

    plt.title(title)
    plt.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ripple_heatmap(
    ripple_data: Dict[str, Dict[int, float]],
    output_path: Union[str, Path],
    metric: str = "acc_delta",
    title: str = "Ripple Effect by Hop"
) -> None:
    """
    Plot heatmap of ripple effects by hop distance.

    Args:
        ripple_data: Entity -> {hop -> metric_value}
        output_path: Output path
        metric: Metric to visualize
        title: Plot title
    """
    # Build matrix
    entities = sorted(ripple_data.keys())
    hops = sorted(set(h for data in ripple_data.values() for h in data.keys()))

    matrix = []
    for entity in entities:
        row = []
        for hop in hops:
            value = ripple_data[entity].get(hop, {}).get(metric, 0.0)
            row.append(value)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(10, max(6, len(entities) * 0.3)))
    sns.heatmap(
        matrix,
        xticklabels=[f"Hop {h}" for h in hops],
        yticklabels=entities,
        cmap="RdYlGn_r",
        center=0,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": metric}
    )
    plt.title(title)
    plt.xlabel("Hop Distance")
    plt.ylabel("Entity")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cke_rank_histogram(
    reports: List[CKEReport],
    output_path: Union[str, Path],
    title: str = "CKE Rank Distribution by Condition"
) -> None:
    """
    Plot histogram of ranks across CKE scenarios.

    Args:
        reports: List of CKE reports
        output_path: Output path
        title: Plot title
    """
    fig, axes = plt.subplots(1, len(reports), figsize=(5 * len(reports), 4), sharey=True)

    if len(reports) == 1:
        axes = [axes]

    for ax, report in zip(axes, reports):
        ranks = [step.local_rank for step in report.steps]

        ax.hist(ranks, bins=range(1, max(ranks) + 2), alpha=0.7, edgecolor='black')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
        ax.set_title(f"{report.scenario.condition}")
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cke_step_progression(
    reports: List[CKEReport],
    output_path: Union[str, Path],
    metric: str = "local_acc",
    title: str = "CKE Step Progression"
) -> None:
    """
    Plot metric progression across CKE steps.

    Args:
        reports: List of CKE reports
        output_path: Output path
        metric: Metric to plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    for report in reports:
        steps = [step.step for step in report.steps]
        values = [getattr(step, metric) for step in report.steps]

        plt.plot(steps, values, marker='o', label=report.scenario.condition, linewidth=2)

    plt.xlabel("Edit Step")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cke_heatmap(
    report: CKEReport,
    output_path: Union[str, Path],
    title: str = "CKE Edit Matrix"
) -> None:
    """
    Plot heatmap of CKE edits (step x metric).

    Args:
        report: CKE report
        output_path: Output path
        title: Plot title
    """
    # Build matrix
    metrics = ["local_acc", "retention_acc", "overwrite_correct"]
    matrix = []

    for step in report.steps:
        row = [
            step.local_acc,
            step.retention_acc,
            float(step.overwrite_correct)
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(8, max(4, len(report.steps) * 0.4)))
    sns.heatmap(
        matrix,
        xticklabels=["Local Acc", "Retention Acc", "Overwrite"],
        yticklabels=[f"Step {s.step}" for s in report.steps],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Score"}
    )
    plt.title(title)
    plt.ylabel("Edit Step")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_order_effect_boxplot(
    reports_fixed: List[CKEReport],
    reports_shuffle: List[CKEReport],
    output_path: Union[str, Path],
    metric: str = "plasticity_score",
    title: str = "Order Effect Comparison"
) -> None:
    """
    Plot boxplot comparing fixed vs shuffle order.

    Args:
        reports_fixed: Reports with fixed order
        reports_shuffle: Reports with shuffled order
        output_path: Output path
        metric: Metric to compare
        title: Plot title
    """
    data_fixed = [getattr(r, metric) for r in reports_fixed]
    data_shuffle = [getattr(r, metric) for r in reports_shuffle]

    plt.figure(figsize=(8, 6))
    positions = [1, 2]
    plt.boxplot(
        [data_fixed, data_shuffle],
        positions=positions,
        labels=["Fixed Order", "Shuffled Order"],
        widths=0.6
    )

    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
