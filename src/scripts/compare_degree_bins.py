"""
Compare results from degree-binned editing experiments.

This script loads results from multiple degree bin experiments and
generates comparison plots showing how editing performance varies across
different degree ranges.

Usage:
    python -m src.scripts.compare_degree_bins --base-dir outputs/degree_binned --num-bins 30
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_bin_stats(bin_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load stats.jsonl and config.json for a single bin.

    Returns:
        Tuple of (stats_df, config_dict)
    """
    stats_file = bin_dir / "stats.jsonl"
    config_file = bin_dir / "config.json"

    if not stats_file.exists():
        return pd.DataFrame(), {}

    records: List[Dict[str, Any]] = []
    with open(stats_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Baseline: no "edit"
            if "edit" not in record:
                flat_record = {
                    "step": record.get("step", 0),
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", 0.0),
                    "is_baseline": True,
                }
            else:
                flat_record = {
                    "step": record.get("step", 0),
                    "edit_success": record["edit_success"]["is_success_top1"],
                    "retention_rate": record["retention_rate"],
                    "edited_acc": record.get("edited_acc", 0.0),
                    "retain_acc": record.get("retain_acc", 0.0),
                    "weight_fro_norm": record["weight_fro_norm"],
                    "is_baseline": False,
                }

            records.append(flat_record)

    stats_df = pd.DataFrame(records)
    stats_df = stats_df.sort_values(["is_baseline", "step"], ascending=[False, True])

    config: Dict[str, Any] = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    return stats_df, config


def get_bin_label(config: Dict[str, Any], bin_id: int) -> str:
    """Generate label for a bin based on its degree range."""
    if "bin_degree_min" in config and "bin_degree_max" in config:
        deg_min = config["bin_degree_min"]
        deg_max = config["bin_degree_max"]
        deg_avg = config.get("bin_degree_avg", None)
        if deg_avg is None:
            return f"Bin {bin_id}: deg [{deg_min}-{deg_max}]"
        return f"Bin {bin_id}: deg [{deg_min}-{deg_max}] (avg={deg_avg:.1f})"
    return f"Bin {bin_id}"


def _apply_xaxis(ax: plt.Axes, xscale: str) -> None:
    """Apply x-axis scaling and label."""
    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlabel("Edit Step (+1, log scale)", fontsize=12)
    else:
        ax.set_xlabel("Edit Step", fontsize=12)


def _output_with_suffix(output_path: Path, suffix: str) -> Path:
    """Insert suffix before extension."""
    return output_path.with_name(output_path.stem + suffix + output_path.suffix)


def plot_degree_bin_comparison(base_dir: str, num_bins: int, output_path: str) -> None:
    """Generate comparison plots for degree bins (linear + log)."""
    base = Path(base_dir)
    out_base = Path(output_path)

    # Load data for all bins
    bin_data: Dict[int, pd.DataFrame] = {}
    bin_configs: Dict[int, Dict[str, Any]] = {}
    bin_labels: Dict[int, str] = {}

    # NOTE: keep your original directory naming: bin_{id}
    for bin_id in range(num_bins):
        bin_dir = base / f"bin_{bin_id}"
        if not bin_dir.exists():
            print(f"Warning: Directory not found for bin {bin_id}: {bin_dir}")
            continue

        df, config = load_bin_stats(bin_dir)
        if df.empty:
            print(f"Warning: No data found for bin {bin_id}")
            continue

        bin_data[bin_id] = df
        bin_configs[bin_id] = config
        bin_labels[bin_id] = get_bin_label(config, bin_id)

    if not bin_data:
        print("Error: No data loaded. Cannot generate comparison plot.")
        return

    sorted_items = sorted(bin_data.items())  # (bin_id, df)
    sorted_bin_ids = [bid for bid, _ in sorted_items]

    # stable colors for bins
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(sorted_items)))

    def _draw_one(xscale: str, out_path: Path) -> None:
        # 2x3 plots + right legend panel
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.25, hspace=0.35)

        axes = np.empty((2, 3), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        axes[0, 2] = fig.add_subplot(gs[0, 2])
        axes[1, 0] = fig.add_subplot(gs[1, 0])
        axes[1, 1] = fig.add_subplot(gs[1, 1])
        axes[1, 2] = fig.add_subplot(gs[1, 2])

        legend_ax = fig.add_subplot(gs[:, 3])
        legend_ax.axis("off")

        fig.suptitle(f"Degree-Binned Sequential Editing Comparison ({xscale})", fontsize=18, y=0.995)

        # Helper: x values (log needs +1)
        def _xvals(non_base: pd.DataFrame) -> pd.Series:
            x = non_base["step"].astype(int)
            return x + 1 if xscale == "log" else x

        # 1. Edit Success Rate
        ax = axes[0, 0]
        for i, (bin_id, df) in enumerate(sorted_items):
            non_base = df[~df["is_baseline"]]
            if not non_base.empty and "edit_success" in non_base.columns:
                ax.plot(
                    _xvals(non_base),
                    non_base["edit_success"].astype(float),
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    alpha=0.7,
                    markersize=3,
                )
        _apply_xaxis(ax, xscale)
        ax.set_ylabel("Edit Success (0/1)", fontsize=12)
        ax.set_title("Edit Success Rate by Degree Bin", fontsize=14)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        # 2. Retention Rate
        ax = axes[0, 1]
        for i, (bin_id, df) in enumerate(sorted_items):
            non_base = df[~df["is_baseline"]]
            if not non_base.empty and "retention_rate" in non_base.columns:
                ax.plot(
                    _xvals(non_base),
                    non_base["retention_rate"],
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    alpha=0.7,
                    markersize=3,
                )
        _apply_xaxis(ax, xscale)
        ax.set_ylabel("Retention Rate", fontsize=12)
        ax.set_title("Retention of Past Edits by Degree Bin", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 3. Edited Triples Accuracy
        ax = axes[0, 2]
        for i, (bin_id, df) in enumerate(sorted_items):
            non_base = df[~df["is_baseline"]]
            if not non_base.empty and "edited_acc" in non_base.columns:
                ax.plot(
                    _xvals(non_base),
                    non_base["edited_acc"],
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    alpha=0.7,
                    markersize=3,
                )
        _apply_xaxis(ax, xscale)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy on Edited Triples by Degree Bin", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 4. Retain Triples Accuracy
        ax = axes[1, 0]
        for i, (bin_id, df) in enumerate(sorted_items):
            baseline = df[df["is_baseline"]]
            if not baseline.empty and "retain_acc" in baseline.columns:
                base_acc = baseline["retain_acc"].iloc[0]
                ax.axhline(y=base_acc, color=colors[i], linestyle="--", linewidth=1, alpha=0.3)

            non_base = df[~df["is_baseline"]]
            if not non_base.empty and "retain_acc" in non_base.columns:
                ax.plot(
                    _xvals(non_base),
                    non_base["retain_acc"],
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    alpha=0.7,
                    markersize=3,
                )
        _apply_xaxis(ax, xscale)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy on Retain Triples by Degree Bin", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 5. Weight Distance
        ax = axes[1, 1]
        for i, (bin_id, df) in enumerate(sorted_items):
            non_base = df[~df["is_baseline"]]
            if not non_base.empty and "weight_fro_norm" in non_base.columns:
                ax.plot(
                    _xvals(non_base),
                    non_base["weight_fro_norm"],
                    marker="o",
                    linewidth=2,
                    color=colors[i],
                    alpha=0.7,
                    markersize=3,
                )
        _apply_xaxis(ax, xscale)
        ax.set_ylabel("Frobenius Norm", fontsize=12)
        ax.set_title("Weight Distance from Base Model by Degree Bin", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 6. Final metrics by degree bin (bar chart) - categorical axis (no log)
        ax = axes[1, 2]
        final_metrics = []
        for bin_id in sorted_bin_ids:
            df = bin_data[bin_id]
            config = bin_configs.get(bin_id, {})
            non_base = df[~df["is_baseline"]]
            if not non_base.empty:
                last_step = non_base.iloc[-1]
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
            ax.bar(x,         final_df["retain_acc"], width, label="Retain Acc", alpha=0.8, color="C1")
            ax.bar(x + width, final_df["retention_rate"], width, label="Retention Rate", alpha=0.8, color="C2")
            ax.set_xlabel("Degree Bin", fontsize=12)
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_title("Final Step Metrics by Degree Bin", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(final_df["label"], rotation=0, ha="center", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            for irow, row in final_df.iterrows():
                ax.text(irow, 1.02, f"avg={row['degree_avg']:.1f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No final metrics available", ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.axis("off")

        # ===== Right-side legend panel =====
        bin_handles = [
            Line2D([0], [0], color=colors[i], marker="o", linewidth=2, alpha=0.7, markersize=4)
            for i in range(len(sorted_items))
        ]
        bin_label_list = [bin_labels[bid] for bid in sorted_bin_ids]

        if len(sorted_bin_ids) <= 15:
            ncol_bins = 1
        elif len(sorted_bin_ids) <= 30:
            ncol_bins = 2
        else:
            ncol_bins = 3

        leg_bins = legend_ax.legend(
            bin_handles,
            bin_label_list,
            title="Degree bins",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=9,
            title_fontsize=10,
            frameon=False,
            ncol=ncol_bins,
            handlelength=2.0,
            columnspacing=1.0,
            labelspacing=0.6,
            borderaxespad=0.0,
        )
        legend_ax.add_artist(leg_bins)

        metric_handles = [
            Patch(facecolor="C0", alpha=0.8, label="Edited Acc"),
            Patch(facecolor="C1", alpha=0.8, label="Retain Acc"),
            Patch(facecolor="C2", alpha=0.8, label="Retention Rate"),
        ]
        legend_ax.legend(
            handles=metric_handles,
            title="Bar metrics",
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            fontsize=9,
            title_fontsize=10,
            frameon=False,
            borderaxespad=0.0,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {out_path}")

    # Save two versions
    _draw_one("linear", _output_with_suffix(out_base, "_xlinear"))
    _draw_one("log",    _output_with_suffix(out_base, "_xlog"))


def print_summary_stats(base_dir: str, num_bins: int) -> None:
    """Print summary statistics for each bin."""
    base = Path(base_dir)

    print("\n" + "=" * 80)
    print("Summary Statistics by Degree Bin")
    print("=" * 80)

    for bin_id in range(num_bins):
        bin_dir = base / f"bin_{bin_id}"
        if not bin_dir.exists():
            continue

        df, config = load_bin_stats(bin_dir)
        if df.empty:
            continue

        label = get_bin_label(config, bin_id)
        non_base = df[~df["is_baseline"]]

        print(f"\n{label}")
        print("-" * len(label))

        if not non_base.empty:
            if "edit_success" in non_base.columns:
                success_rate = non_base["edit_success"].astype(float).mean()
                print(f"  Edit Success Rate:     {success_rate:.3f}")

            if "retention_rate" in non_base.columns:
                final_retention = float(non_base["retention_rate"].iloc[-1])
                print(f"  Final Retention Rate:  {final_retention:.3f}")

            if "edited_acc" in non_base.columns:
                final_edited_acc = float(non_base["edited_acc"].iloc[-1])
                print(f"  Final Edited Acc:      {final_edited_acc:.3f}")

            if "retain_acc" in non_base.columns:
                baseline = df[df["is_baseline"]]
                baseline_retain = float(baseline["retain_acc"].iloc[0]) if not baseline.empty else 0.0
                final_retain_acc = float(non_base["retain_acc"].iloc[-1])
                retain_drop = baseline_retain - final_retain_acc
                print(f"  Baseline Retain Acc:   {baseline_retain:.3f}")
                print(f"  Final Retain Acc:      {final_retain_acc:.3f}")
                print(f"  Retain Acc Drop:       {retain_drop:.3f}")

            if "weight_fro_norm" in non_base.columns:
                final_weight_dist = float(non_base["weight_fro_norm"].iloc[-1])
                print(f"  Final Weight Distance: {final_weight_dist:.3f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare degree-binned sequential editing results")

    # Accept both --base-dir (preferred) and --base_dir (legacy/accidental overwrite)
    parser.add_argument("--base-dir", dest="base_dir", type=str, required=False, help="Base directory containing bin subdirectories")
    parser.add_argument("--base_dir", dest="base_dir", type=str, required=False, help=argparse.SUPPRESS)

    parser.add_argument("--num-bins", dest="num_bins", type=int, required=False, help="Number of bins to analyze")
    parser.add_argument("--num_bins", dest="num_bins", type=int, required=False, help=argparse.SUPPRESS)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path base (default: <base-dir>/degree_bins_comparison.png). Will save *_xlinear and *_xlog.",
    )

    args = parser.parse_args()

    # Validate required args manually to keep CLI backward compatible
    if args.base_dir is None or args.num_bins is None:
        parser.error("the following arguments are required: --base-dir, --num-bins")

    if args.output is None:
        args.output = str(Path(args.base_dir) / "degree_bins_comparison.png")

    print("=" * 80)
    print("Degree Bin Comparison Analysis")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of bins: {args.num_bins}")
    print(f"Output base:    {args.output} (will create *_xlinear / *_xlog)")
    print("=" * 80)

    plot_degree_bin_comparison(args.base_dir, args.num_bins, args.output)
    print_summary_stats(args.base_dir, args.num_bins)

    print("\n✓ Degree bin comparison analysis complete!")


if __name__ == "__main__":
    main()