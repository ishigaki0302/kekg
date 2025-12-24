"""Compare results from different edit selection modes.

This script loads results from multiple selection mode experiments and
generates comparison plots.

Usage:
    python -m src.scripts.compare_selection_modes --base-dir outputs/sequential_comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)


def load_stats_for_mode(mode_dir: Path) -> pd.DataFrame:
    """Load stats.jsonl for a single mode."""
    records = []
    stats_file = mode_dir / "stats.jsonl"

    if not stats_file.exists():
        return pd.DataFrame()

    with open(stats_file, "r") as f:
        for line in f:
            record = json.loads(line)

            # Extract key metrics
            if "edit" not in record:
                # Baseline
                flat_record = {
                    "step": record["step"],
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", record.get("kg_top1_acc", 0.0)),
                    "is_baseline": True,
                }
            else:
                # Regular step
                flat_record = {
                    "step": record["step"],
                    "edit_success": record["edit_success"]["is_success_top1"],
                    "retention_rate": record["retention_rate"],
                    "edited_acc": record.get("edited_acc", record.get("kg_top1_acc")),
                    "retain_acc": record.get("retain_acc", record.get("kg_top1_acc")),
                    "weight_fro_norm": record["weight_fro_norm"],
                    "is_baseline": False,
                }
            records.append(flat_record)

    return pd.DataFrame(records)


def plot_comparison(base_dir: str, output_path: str) -> None:
    """Generate comparison plots for all selection modes.

    Args:
        base_dir: Base directory containing mode subdirectories
        output_path: Output path for the comparison figure
    """
    base = Path(base_dir)

    # Load data for all modes
    modes = ["degree_high", "degree_low", "hop_high", "hop_low"]
    mode_labels = {
        "degree_high": "Degree High",
        "degree_low": "Degree Low",
        "hop_high": "Hop High (Far)",
        "hop_low": "Hop Low (Near)",
    }

    data = {}
    for mode in modes:
        mode_dir = base / mode
        if mode_dir.exists():
            df = load_stats_for_mode(mode_dir)
            if not df.empty:
                data[mode] = df
            else:
                print(f"Warning: No data found for mode '{mode}'")
        else:
            print(f"Warning: Directory not found for mode '{mode}': {mode_dir}")

    if not data:
        print("Error: No data loaded. Cannot generate comparison plot.")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Selection Mode Comparison", fontsize=18, y=0.995)

    # Color scheme for each mode
    colors = {
        "degree_high": "C0",
        "degree_low": "C1",
        "hop_high": "C2",
        "hop_low": "C3",
    }

    # 1. Edit Success Rate
    ax = axes[0, 0]
    for mode, df in data.items():
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "edit_success" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["edit_success"].astype(float),
                marker="o",
                linewidth=2,
                label=mode_labels[mode],
                color=colors[mode],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Edit Success (0/1)", fontsize=12)
    ax.set_title("Edit Success Rate", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Retention Rate
    ax = axes[0, 1]
    for mode, df in data.items():
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "retention_rate" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["retention_rate"],
                marker="o",
                linewidth=2,
                label=mode_labels[mode],
                color=colors[mode],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.set_title("Retention of Past Edits", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Edited Triples Accuracy
    ax = axes[0, 2]
    for mode, df in data.items():
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "edited_acc" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["edited_acc"],
                marker="o",
                linewidth=2,
                label=mode_labels[mode],
                color=colors[mode],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy on Edited Triples (Cumulative)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Retain Triples Accuracy
    ax = axes[1, 0]
    for mode, df in data.items():
        # Plot baseline as horizontal line
        baseline = df[df["is_baseline"]]
        if not baseline.empty and "retain_acc" in baseline.columns:
            base_acc = baseline["retain_acc"].iloc[0]
            ax.axhline(
                y=base_acc,
                color=colors[mode],
                linestyle="--",
                linewidth=1,
                alpha=0.3,
            )

        # Plot retention accuracy
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "retain_acc" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["retain_acc"],
                marker="o",
                linewidth=2,
                label=mode_labels[mode],
                color=colors[mode],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy on Retain Triples (Unedited)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5. Weight Distance
    ax = axes[1, 1]
    for mode, df in data.items():
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "weight_fro_norm" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["weight_fro_norm"],
                marker="o",
                linewidth=2,
                label=mode_labels[mode],
                color=colors[mode],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Frobenius Norm", fontsize=12)
    ax.set_title("Weight Distance from Base Model", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 6. Final metrics summary (bar chart)
    ax = axes[1, 2]

    # Extract final step metrics
    final_metrics = []
    for mode in modes:
        if mode in data:
            df = data[mode]
            non_base = df[~df["is_baseline"]]
            if not non_base.empty:
                last_step = non_base.iloc[-1]
                final_metrics.append({
                    "mode": mode_labels[mode],
                    "edited_acc": last_step.get("edited_acc", 0),
                    "retain_acc": last_step.get("retain_acc", 0),
                })

    if final_metrics:
        final_df = pd.DataFrame(final_metrics)
        x = np.arange(len(final_df))
        width = 0.35

        ax.bar(x - width/2, final_df["edited_acc"], width, label="Edited Acc", alpha=0.8)
        ax.bar(x + width/2, final_df["retain_acc"], width, label="Retain Acc", alpha=0.8)

        ax.set_xlabel("Selection Mode", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Final Step Accuracy Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(final_df["mode"], rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No final metrics available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved comparison plot to: {output_path}")


def print_summary_stats(base_dir: str) -> None:
    """Print summary statistics for all modes.

    Args:
        base_dir: Base directory containing mode subdirectories
    """
    base = Path(base_dir)
    modes = ["degree_high", "degree_low", "hop_high", "hop_low"]
    mode_labels = {
        "degree_high": "Degree High",
        "degree_low": "Degree Low",
        "hop_high": "Hop High (Far)",
        "hop_low": "Hop Low (Near)",
    }

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for mode in modes:
        mode_dir = base / mode
        if not mode_dir.exists():
            continue

        df = load_stats_for_mode(mode_dir)
        if df.empty:
            continue

        non_base = df[~df["is_baseline"]]
        if non_base.empty:
            continue

        print(f"\n{mode_labels[mode]}:")
        print("-" * 40)

        # Edit success rate
        if "edit_success" in non_base.columns:
            success_rate = non_base["edit_success"].astype(float).mean()
            print(f"  Edit Success Rate:     {success_rate:.3f}")

        # Final retention rate
        if "retention_rate" in non_base.columns:
            final_retention = non_base["retention_rate"].iloc[-1]
            print(f"  Final Retention Rate:  {final_retention:.3f}")

        # Final accuracies
        if "edited_acc" in non_base.columns:
            final_edited_acc = non_base["edited_acc"].iloc[-1]
            print(f"  Final Edited Acc:      {final_edited_acc:.3f}")

        if "retain_acc" in non_base.columns:
            baseline = df[df["is_baseline"]]
            baseline_retain = baseline["retain_acc"].iloc[0] if not baseline.empty else 0
            final_retain_acc = non_base["retain_acc"].iloc[-1]
            retain_drop = baseline_retain - final_retain_acc
            print(f"  Baseline Retain Acc:   {baseline_retain:.3f}")
            print(f"  Final Retain Acc:      {final_retain_acc:.3f}")
            print(f"  Retain Acc Drop:       {retain_drop:.3f}")

        # Final weight distance
        if "weight_fro_norm" in non_base.columns:
            final_weight_dist = non_base["weight_fro_norm"].iloc[-1]
            print(f"  Final Weight Distance: {final_weight_dist:.3f}")

    print("\n" + "=" * 80)


def main():
    """Main entry point for comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare sequential editing results across selection modes"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/sequential_comparison",
        help="Base directory containing mode subdirectories (default: outputs/sequential_comparison)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison plot (default: <base-dir>/comparison.png)",
    )

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = str(Path(args.base_dir) / "comparison.png")

    print("=" * 80)
    print("Selection Mode Comparison Analysis")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Output plot: {args.output}")
    print("=" * 80)

    # Generate comparison plot
    plot_comparison(args.base_dir, args.output)

    # Print summary statistics
    print_summary_stats(args.base_dir)

    print("\n✓ Comparison analysis complete!")


if __name__ == "__main__":
    main()
