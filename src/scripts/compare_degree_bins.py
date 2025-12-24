"""Compare results from degree-binned editing experiments.

This script loads results from multiple degree bin experiments and
generates comparison plots showing how editing performance varies across
different degree ranges.

Usage:
    python -m src.scripts.compare_degree_bins --base-dir outputs/degree_binned --num-bins 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (18, 12)


def load_bin_stats(bin_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load stats.jsonl and config.json for a single bin.

    Returns:
        Tuple of (stats_df, config_dict)
    """
    stats_file = bin_dir / "stats.jsonl"
    config_file = bin_dir / "config.json"

    if not stats_file.exists():
        return pd.DataFrame(), {}

    # Load stats
    records = []
    with open(stats_file, "r") as f:
        for line in f:
            record = json.loads(line)

            # Extract key metrics
            if "edit" not in record:
                # Baseline
                flat_record = {
                    "step": record["step"],
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", 0.0),
                    "is_baseline": True,
                }
            else:
                # Regular step
                flat_record = {
                    "step": record["step"],
                    "edit_success": record["edit_success"]["is_success_top1"],
                    "retention_rate": record["retention_rate"],
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", 0.0),
                    "weight_fro_norm": record["weight_fro_norm"],
                    "is_baseline": False,
                }
            records.append(flat_record)

    stats_df = pd.DataFrame(records)

    # Load config
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    return stats_df, config


def get_bin_label(config: Dict, bin_id: int) -> str:
    """Generate label for a bin based on its degree range.

    Args:
        config: Bin configuration dictionary
        bin_id: Bin ID number

    Returns:
        Label string (e.g., "Bin 0: deg [50-60]")
    """
    if "bin_degree_min" in config and "bin_degree_max" in config:
        deg_min = config["bin_degree_min"]
        deg_max = config["bin_degree_max"]
        return f"Bin {bin_id}: deg [{deg_min}-{deg_max}]"
    else:
        return f"Bin {bin_id}"


def plot_degree_bin_comparison(base_dir: str, num_bins: int, output_path: str) -> None:
    """Generate comparison plots for degree bins.

    Args:
        base_dir: Base directory containing bin subdirectories
        num_bins: Number of bins to load
        output_path: Output path for the comparison figure
    """
    base = Path(base_dir)

    # Load data for all bins
    bin_data = {}
    bin_configs = {}
    bin_labels = {}

    for bin_id in range(num_bins):
        bin_dir = base / f"bin_{bin_id}"
        if bin_dir.exists():
            df, config = load_bin_stats(bin_dir)
            if not df.empty:
                bin_data[bin_id] = df
                bin_configs[bin_id] = config
                bin_labels[bin_id] = get_bin_label(config, bin_id)
            else:
                print(f"Warning: No data found for bin {bin_id}")
        else:
            print(f"Warning: Directory not found for bin {bin_id}: {bin_dir}")

    if not bin_data:
        print("Error: No data loaded. Cannot generate comparison plot.")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Degree-Binned Sequential Editing Comparison", fontsize=18, y=0.995)

    # Color scheme - use a gradient from high degree (blue) to low degree (red)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(bin_data)))

    # 1. Edit Success Rate by Bin
    ax = axes[0, 0]
    for i, (bin_id, df) in enumerate(sorted(bin_data.items())):
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "edit_success" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["edit_success"].astype(float),
                marker="o",
                linewidth=2,
                label=bin_labels[bin_id],
                color=colors[i],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Edit Success (0/1)", fontsize=12)
    ax.set_title("Edit Success Rate by Degree Bin", fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Retention Rate by Bin
    ax = axes[0, 1]
    for i, (bin_id, df) in enumerate(sorted(bin_data.items())):
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "retention_rate" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["retention_rate"],
                marker="o",
                linewidth=2,
                label=bin_labels[bin_id],
                color=colors[i],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.set_title("Retention of Past Edits by Degree Bin", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Edited Triples Accuracy by Bin
    ax = axes[0, 2]
    for i, (bin_id, df) in enumerate(sorted(bin_data.items())):
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "edited_acc" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["edited_acc"],
                marker="o",
                linewidth=2,
                label=bin_labels[bin_id],
                color=colors[i],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy on Edited Triples by Degree Bin", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 4. Retain Triples Accuracy by Bin
    ax = axes[1, 0]
    for i, (bin_id, df) in enumerate(sorted(bin_data.items())):
        # Plot baseline as horizontal line
        baseline = df[df["is_baseline"]]
        if not baseline.empty and "retain_acc" in baseline.columns:
            base_acc = baseline["retain_acc"].iloc[0]
            ax.axhline(
                y=base_acc,
                color=colors[i],
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
                label=bin_labels[bin_id],
                color=colors[i],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy on Retain Triples by Degree Bin", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 5. Weight Distance by Bin
    ax = axes[1, 1]
    for i, (bin_id, df) in enumerate(sorted(bin_data.items())):
        non_base = df[~df["is_baseline"]]
        if not non_base.empty and "weight_fro_norm" in non_base.columns:
            ax.plot(
                non_base["step"],
                non_base["weight_fro_norm"],
                marker="o",
                linewidth=2,
                label=bin_labels[bin_id],
                color=colors[i],
                alpha=0.7,
                markersize=3,
            )
    ax.set_xlabel("Edit Step", fontsize=12)
    ax.set_ylabel("Frobenius Norm", fontsize=12)
    ax.set_title("Weight Distance from Base Model by Degree Bin", fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # 6. Final metrics by degree bin (bar chart)
    ax = axes[1, 2]

    # Extract final step metrics for each bin
    final_metrics = []
    for bin_id in sorted(bin_data.keys()):
        df = bin_data[bin_id]
        config = bin_configs.get(bin_id, {})
        non_base = df[~df["is_baseline"]]

        if not non_base.empty:
            last_step = non_base.iloc[-1]

            # Get degree range
            deg_min = config.get("bin_degree_min", 0)
            deg_max = config.get("bin_degree_max", 0)
            deg_avg = config.get("bin_degree_avg", 0)

            final_metrics.append({
                "bin_id": bin_id,
                "label": f"Bin {bin_id}\n[{deg_min}-{deg_max}]",
                "degree_avg": deg_avg,
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
                "retention_rate": last_step.get("retention_rate", 0),
            })

    if final_metrics:
        final_df = pd.DataFrame(final_metrics)
        x = np.arange(len(final_df))
        width = 0.25

        ax.bar(x - width, final_df["edited_acc"], width, label="Edited Acc", alpha=0.8, color="C0")
        ax.bar(x, final_df["retain_acc"], width, label="Retain Acc", alpha=0.8, color="C1")
        ax.bar(x + width, final_df["retention_rate"], width, label="Retention Rate", alpha=0.8, color="C2")

        ax.set_xlabel("Degree Bin", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Final Step Metrics by Degree Bin", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(final_df["label"], rotation=0, ha="center", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis="y")

        # Add average degree as text on top
        for i, row in final_df.iterrows():
            ax.text(i, 1.02, f"avg={row['degree_avg']:.1f}",
                   ha='center', va='bottom', fontsize=8, rotation=0)
    else:
        ax.text(0.5, 0.5, "No final metrics available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved degree bin comparison plot to: {output_path}")


def print_summary_stats(base_dir: str, num_bins: int) -> None:
    """Print summary statistics for all degree bins.

    Args:
        base_dir: Base directory containing bin subdirectories
        num_bins: Number of bins to analyze
    """
    base = Path(base_dir)

    print("\n" + "=" * 80)
    print("Degree Bin Summary Statistics")
    print("=" * 80)

    for bin_id in range(num_bins):
        bin_dir = base / f"bin_{bin_id}"
        if not bin_dir.exists():
            continue

        df, config = load_bin_stats(bin_dir)
        if df.empty:
            continue

        non_base = df[~df["is_baseline"]]
        if non_base.empty:
            continue

        # Get degree range
        deg_min = config.get("bin_degree_min", "?")
        deg_max = config.get("bin_degree_max", "?")
        deg_avg = config.get("bin_degree_avg", 0)
        bin_start = config.get("bin_start_idx", "?")
        bin_end = config.get("bin_end_idx", "?")

        print(f"\nBin {bin_id}: Degree range [{deg_min}, {deg_max}] (avg={deg_avg:.2f})")
        print(f"  Triple indices: [{bin_start}, {bin_end})")
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
    """Main entry point for degree bin comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare degree-binned sequential editing results"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing bin subdirectories",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        required=True,
        help="Number of bins to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison plot (default: <base-dir>/degree_bins_comparison.png)",
    )

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = str(Path(args.base_dir) / "degree_bins_comparison.png")

    print("=" * 80)
    print("Degree Bin Comparison Analysis")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of bins: {args.num_bins}")
    print(f"Output plot: {args.output}")
    print("=" * 80)

    # Generate comparison plot
    plot_degree_bin_comparison(args.base_dir, args.num_bins, args.output)

    # Print summary statistics
    print_summary_stats(args.base_dir, args.num_bins)

    print("\n✓ Degree bin comparison analysis complete!")


if __name__ == "__main__":
    main()
