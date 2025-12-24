"""Analysis and visualization for sequential editing results.

This module provides functions for loading and visualizing the results
of sequential editing experiments, including time series plots,
hop/degree heatmaps, and failure histograms.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_stats(path: str) -> pd.DataFrame:
    """Load stats.jsonl file into a pandas DataFrame.

    Args:
        path: Path to stats.jsonl file

    Returns:
        DataFrame with step-level metrics
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            # ベースライン行かどうか（= 1つ目だけキーが少ないケース）
            if "edit" not in record:
                # ここでは「無い情報は None」で埋めることにする
                flat_record = {
                    "step": record["step"],
                    "edit_s": None,
                    "edit_r": None,
                    "edit_o_old": None,
                    "edit_o_new": None,
                    "edit_success": None,
                    "edit_margin": None,
                    "retention_rate": None,
                    "kg_top1_acc": record.get("kg_top1_acc"),  # 後方互換性のため保持
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", record.get("kg_top1_acc")),
                    "weight_fro_norm": None,
                    "ripple_mean_abs_delta": None,
                    # ベースラインフラグが欲しければここで保持する
                    "is_baseline": record.get("is_baseline", True),
                }
            else:
                # 通常のレコード（edit や ripple があるケース）
                flat_record = {
                    "step": record["step"],
                    "edit_s": record["edit"]["s"],
                    "edit_r": record["edit"]["r"],
                    "edit_o_old": record["edit"]["o_old"],
                    "edit_o_new": record["edit"]["o_new"],
                    "edit_success": record["edit_success"]["is_success_top1"],
                    "edit_margin": record["edit_success"]["margin"],
                    "retention_rate": record["retention_rate"],
                    "kg_top1_acc": record.get("kg_top1_acc"),  # 後方互換性のため保持
                    "edited_acc": record.get("edited_acc", record.get("kg_top1_acc")),
                    "retain_acc": record.get("retain_acc", record.get("kg_top1_acc")),
                    "weight_fro_norm": record["weight_fro_norm"],
                    "ripple_mean_abs_delta": record["ripple"]["mean_abs_delta"],
                    "is_baseline": record.get("is_baseline", False),
                }
            records.append(flat_record)

    return pd.DataFrame(records)


def load_ripple(path: str) -> pd.DataFrame:
    """Load ripple_triples.jsonl file into a pandas DataFrame.

    Args:
        path: Path to ripple_triples.jsonl file

    Returns:
        DataFrame with per-triple ripple effect data
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

    return pd.DataFrame(records)


def load_acc(path: str) -> pd.DataFrame:
    """Load triple_acc.jsonl file into a pandas DataFrame.

    Args:
        path: Path to triple_acc.jsonl file

    Returns:
        DataFrame with per-triple accuracy at each step
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

    return pd.DataFrame(records)


def plot_time_series(stats: pd.DataFrame, out_path: str) -> None:
    """Plot time series of key metrics across editing steps.

    Creates a 5-panel figure showing:
    1. Edit success rate
    2. Retention rate
    3. Edited triples accuracy (cumulative)
    4. Retain triples accuracy (unedited triples)
    5. Weight distance from base model

    Args:
        stats: DataFrame from load_stats()
        out_path: Output path for the figure
    """
    # ベースライン行と通常行を分ける
    baseline = stats[stats["is_baseline"]]
    non_base = stats[~stats["is_baseline"]].copy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Sequential Editing: Time Series Metrics", fontsize=16, y=0.995)

    # 1. Edit success（ベースラインは使わない）
    ax = axes[0, 0]
    if not non_base.empty:
        ax.plot(
            non_base["step"],
            non_base["edit_success"].astype(float),
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Edit Success (0/1)")
    ax.set_title("Edit Success per Step")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # 2. Retention rate（ベースラインは使わない）
    ax = axes[0, 1]
    if not non_base.empty:
        ax.plot(
            non_base["step"],
            non_base["retention_rate"],
            marker="o",
            linewidth=2,
            color="C1",
        )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Retention Rate")
    ax.set_title("Retention of Past Edits")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Edited triples accuracy (累積)
    ax = axes[1, 0]
    if not non_base.empty:
        ax.plot(
            non_base["step"],
            non_base["edited_acc"],
            marker="o",
            linewidth=2,
            color="C2",
            label="Edited Triples Acc",
        )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on Edited Triples (Cumulative)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Retain triples accuracy（未編集トリプル）
    ax = axes[1, 1]
    # 通常ステップの Retain Acc
    if not non_base.empty:
        ax.plot(
            non_base["step"],
            non_base["retain_acc"],
            marker="o",
            linewidth=2,
            color="C3",
            label="Retain Triples Acc",
        )

    # ベースライン Retain Acc を横のハッチ付き帯で可視化
    if not baseline.empty:
        base_retain_acc = baseline["retain_acc"].iloc[0]
        # ちょっとだけ厚みを持たせた帯にハッチをかける
        eps = 0.005
        band = ax.axhspan(
            base_retain_acc - eps,
            base_retain_acc + eps,
            facecolor="none",
            edgecolor="black",
            hatch="///",
            linewidth=0,
        )
        band.set_label(f"Baseline = {base_retain_acc:.3f}")

    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy on Retain Triples (Unedited)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Weight distance（ベースラインは使わない）
    ax = axes[2, 0]
    if not non_base.empty:
        ax.plot(
            non_base["step"],
            non_base["weight_fro_norm"],
            marker="o",
            linewidth=2,
            color="C4",
        )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Weight Distance from Base Model")
    ax.grid(True, alpha=0.3)

    # 6. 空白のプロット（将来の拡張用）
    ax = axes[2, 1]
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved time series plot to: {out_path}")


def plot_hop_degree_heatmaps(ripple: pd.DataFrame, out_path_prefix: str) -> None:
    """Plot heatmaps of delta logits grouped by hop distance and degree bins.

    Creates 4 separate heatmaps:
    1. Hop from edit subject
    2. Hop from original object
    3. Hop from new object
    4. Entity degree bins

    Args:
        ripple: DataFrame from load_ripple()
        out_path_prefix: Prefix for output paths (will append _hop_subj.png, etc.)
    """
    # Filter out unreachable entities (hop > max_hop)
    max_hop = ripple["hop_subj"].max()
    if max_hop > 10:  # Assume unreachable if > 10
        max_hop = 10
    ripple_filtered = ripple[
        (ripple["hop_subj"] <= max_hop)
        & (ripple["hop_before"] <= max_hop)
        & (ripple["hop_after"] <= max_hop)
    ].copy()

    # Add absolute delta logit column
    ripple_filtered["abs_delta_logit"] = ripple_filtered["delta_logit"].abs()

    # 1. Hop from subject
    hop_subj_data = (
        ripple_filtered.groupby(["step", "hop_subj"])["abs_delta_logit"]
        .mean()
        .reset_index()
    )
    hop_subj_pivot = hop_subj_data.pivot(
        index="hop_subj", columns="step", values="abs_delta_logit"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        hop_subj_pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean |Δ Logit|"},
        ax=ax,
        annot=False,
    )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Hop Distance from Edit Subject")
    ax.set_title("Ripple Effect by Hop Distance from Edit Subject")
    plt.tight_layout()
    out_path = f"{out_path_prefix}_hop_subj.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hop-subject heatmap to: {out_path}")

    # 2. Hop from before (original object)
    hop_before_data = (
        ripple_filtered.groupby(["step", "hop_before"])["abs_delta_logit"]
        .mean()
        .reset_index()
    )
    hop_before_pivot = hop_before_data.pivot(
        index="hop_before", columns="step", values="abs_delta_logit"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        hop_before_pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean |Δ Logit|"},
        ax=ax,
        annot=False,
    )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Hop Distance from Original Object")
    ax.set_title("Ripple Effect by Hop Distance from Original Object")
    plt.tight_layout()
    out_path = f"{out_path_prefix}_hop_before.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hop-before heatmap to: {out_path}")

    # 3. Hop from after (new object)
    hop_after_data = (
        ripple_filtered.groupby(["step", "hop_after"])["abs_delta_logit"]
        .mean()
        .reset_index()
    )
    hop_after_pivot = hop_after_data.pivot(
        index="hop_after", columns="step", values="abs_delta_logit"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        hop_after_pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean |Δ Logit|"},
        ax=ax,
        annot=False,
    )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Hop Distance from New Object")
    ax.set_title("Ripple Effect by Hop Distance from New Object")
    plt.tight_layout()
    out_path = f"{out_path_prefix}_hop_after.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hop-after heatmap to: {out_path}")

    # 4. Degree bins
    # Create degree bins dynamically
    degree_bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, 999]
    ripple_filtered["degree_bin"] = pd.cut(
        ripple_filtered["degree_s"],
        bins=degree_bins,
        labels=[f"{degree_bins[i]}-{degree_bins[i+1]}" for i in range(len(degree_bins) - 1)],
        include_lowest=True,
    )

    degree_data = (
        ripple_filtered.groupby(["step", "degree_bin"])["abs_delta_logit"]
        .mean()
        .reset_index()
    )
    degree_pivot = degree_data.pivot(
        index="degree_bin", columns="step", values="abs_delta_logit"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        degree_pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean |Δ Logit|"},
        ax=ax,
        annot=False,
    )
    ax.set_xlabel("Edit Step")
    ax.set_ylabel("Entity Degree Bin")
    ax.set_title("Ripple Effect by Entity Degree")
    plt.tight_layout()
    out_path = f"{out_path_prefix}_degree.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved degree heatmap to: {out_path}")


def plot_failure_hist(acc: pd.DataFrame, out_path: str) -> None:
    """Plot histogram of when triples first fail (become incorrect).

    Args:
        acc: DataFrame from load_acc()
        out_path: Output path for the figure
    """
    # Find the first step where each triple becomes incorrect
    first_failure_steps = []

    # Group by triple ID
    for tid, group in acc.groupby("tid"):
        # Sort by step
        group_sorted = group.sort_values("step")

        # Find first step where is_correct is False
        failed_steps = group_sorted[~group_sorted["is_correct"]]
        if not failed_steps.empty:
            first_failure_step = failed_steps.iloc[0]["step"]
            first_failure_steps.append(first_failure_step)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    if first_failure_steps:
        ax.hist(
            first_failure_steps,
            bins=range(
                1, int(max(first_failure_steps)) + 2
            ),  # One bin per step
            edgecolor="black",
            alpha=0.7,
            color="C0",
        )
        ax.set_xlabel("Step of First Failure")
        ax.set_ylabel("Number of Triples")
        ax.set_title(
            f"Distribution of First Failure Steps ({len(first_failure_steps)} failures)"
        )
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No failures detected",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of Triples")
        ax.set_title("Distribution of First Failure Steps (No failures)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved failure histogram to: {out_path}")


def run_all_plots(output_dir: str = "outputs/sequential") -> None:
    """Run all visualization functions on the output directory.

    Args:
        output_dir: Directory containing stats.jsonl, ripple_triples.jsonl, etc.
    """
    base = Path(output_dir)

    print(f"Loading data from: {base}")

    # Load data
    stats = load_stats(base / "stats.jsonl")
    ripple = load_ripple(base / "ripple_triples.jsonl")
    acc = load_acc(base / "triple_acc.jsonl")

    print(f"Loaded {len(stats)} steps, {len(ripple)} ripple records, {len(acc)} accuracy records")

    # Generate plots
    print("\nGenerating plots...")
    plot_time_series(stats, str(base / "plots_time_series.png"))
    plot_hop_degree_heatmaps(ripple, str(base / "plots_hop_degree"))
    plot_failure_hist(acc, str(base / "plots_failure_hist.png"))

    print("\n✓ All plots generated successfully!")
