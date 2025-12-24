"""Compare results from multiple trials of hop high/low experiments.

This script loads results from multiple trials for both hop_high and hop_low,
and generates comparison plots showing:
  - Individual trial lines (thin, semi-transparent)
  - Mean line (thick)
  - 95% confidence interval (shaded area)

Usage:
    python -m src.scripts.compare_hop_multi_trial \
        --base-dir outputs/hop_multi_trial_no_alias \
        --num-trials 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 14)


def load_trial_stats(trial_dir: Path) -> pd.DataFrame:
    """Load stats.jsonl for a single trial."""
    stats_file = trial_dir / "stats.jsonl"

    if not stats_file.exists():
        return pd.DataFrame()

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

    return pd.DataFrame(records)


def load_all_trials(base_dir: Path, mode: str, num_trials: int) -> List[pd.DataFrame]:
    """Load stats for all trials of a given mode.

    Args:
        base_dir: Base directory containing mode subdirectories
        mode: "hop_high" or "hop_low"
        num_trials: Number of trials to load

    Returns:
        List of DataFrames, one per trial
    """
    trial_dfs = []
    mode_dir = base_dir / mode

    for trial_id in range(num_trials):
        trial_dir = mode_dir / f"trial_{trial_id}"
        if trial_dir.exists():
            df = load_trial_stats(trial_dir)
            if not df.empty:
                trial_dfs.append(df)
            else:
                print(f"Warning: No data found for {mode} trial {trial_id}")
        else:
            print(f"Warning: Directory not found for {mode} trial {trial_id}: {trial_dir}")

    return trial_dfs


def compute_statistics(trial_dfs: List[pd.DataFrame], metric: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute mean and 95% confidence interval for a metric across trials.

    Args:
        trial_dfs: List of DataFrames from different trials
        metric: Metric name to compute statistics for

    Returns:
        Tuple of (mean_df, lower_ci_df, upper_ci_df)
    """
    # Find the minimum number of steps across all trials
    min_steps = min(len(df) for df in trial_dfs)

    # Get steps from first trial, truncated to min_steps
    steps = trial_dfs[0]["step"].values[:min_steps]

    # Collect metric values for each step across trials
    all_values = []
    for df in trial_dfs:
        if metric in df.columns:
            # Truncate to min_steps to ensure all arrays have same length
            all_values.append(df[metric].values[:min_steps])
        else:
            # If metric doesn't exist, fill with zeros
            all_values.append(np.zeros(min_steps))

    # Convert to numpy array: shape (num_trials, num_steps)
    # Explicitly set dtype to float64 to avoid object array issues
    all_values = np.array(all_values, dtype=np.float64)

    # Compute mean and 95% CI using t-distribution
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0, ddof=1)  # Sample standard deviation
    n_trials = len(trial_dfs)

    # t-distribution critical value for 95% CI
    t_critical = stats.t.ppf(0.975, df=n_trials - 1) if n_trials > 1 else 1.96

    # Standard error of the mean
    sem = std_values / np.sqrt(n_trials)

    # Confidence interval
    ci_margin = t_critical * sem
    lower_ci = mean_values - ci_margin
    upper_ci = mean_values + ci_margin

    # Create DataFrames
    mean_df = pd.DataFrame({"step": steps, metric: mean_values})
    lower_ci_df = pd.DataFrame({"step": steps, metric: lower_ci})
    upper_ci_df = pd.DataFrame({"step": steps, metric: upper_ci})

    return mean_df, lower_ci_df, upper_ci_df


def plot_multi_trial_comparison(
    base_dir: str,
    num_trials: int,
    output_path: str
) -> None:
    """Generate multi-trial comparison plots.

    Args:
        base_dir: Base directory containing trial subdirectories
        num_trials: Number of trials per mode
        output_path: Output path for the comparison figure
    """
    base = Path(base_dir)

    # Load data for all trials
    print("Loading trial data...")
    hop_high_trials = load_all_trials(base, "hop_high", num_trials)
    hop_low_trials = load_all_trials(base, "hop_low", num_trials)

    if not hop_high_trials or not hop_low_trials:
        print("Error: No data loaded for one or both modes. Cannot generate comparison plot.")
        return

    print(f"Loaded {len(hop_high_trials)} hop_high trials and {len(hop_low_trials)} hop_low trials")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"Hop Multi-Trial Comparison (n={len(hop_high_trials)} trials per mode)",
        fontsize=18,
        y=0.995
    )

    # Color scheme
    color_high = "C0"  # Blue
    color_low = "C1"   # Orange

    # Metrics to plot
    metrics = [
        ("edit_success", "Edit Success (0/1)", "Edit Success Rate"),
        ("retention_rate", "Retention Rate", "Retention of Past Edits"),
        ("edited_acc", "Accuracy", "Accuracy on Edited Triples (Cumulative)"),
        ("retain_acc", "Accuracy", "Accuracy on Retain Triples (Unedited)"),
        ("weight_fro_norm", "Frobenius Norm", "Weight Distance from Base Model"),
    ]

    # Plot each metric
    for idx, (metric, ylabel, title) in enumerate(metrics):
        if idx >= 5:
            break

        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Filter to non-baseline steps
        hop_high_non_base = [df[~df["is_baseline"]] for df in hop_high_trials]
        hop_low_non_base = [df[~df["is_baseline"]] for df in hop_low_trials]

        # Plot individual trials (thin, semi-transparent)
        for trial_df in hop_high_non_base:
            if metric in trial_df.columns:
                ax.plot(
                    trial_df["step"],
                    trial_df[metric],
                    color=color_high,
                    alpha=0.15,
                    linewidth=1,
                    zorder=1
                )

        for trial_df in hop_low_non_base:
            if metric in trial_df.columns:
                ax.plot(
                    trial_df["step"],
                    trial_df[metric],
                    color=color_low,
                    alpha=0.15,
                    linewidth=1,
                    zorder=1
                )

        # Compute statistics and plot mean + CI
        if hop_high_non_base and metric in hop_high_non_base[0].columns:
            mean_high, lower_high, upper_high = compute_statistics(hop_high_non_base, metric)

            # Plot confidence interval
            ax.fill_between(
                mean_high["step"],
                lower_high[metric],
                upper_high[metric],
                color=color_high,
                alpha=0.2,
                zorder=2
            )

            # Plot mean
            ax.plot(
                mean_high["step"],
                mean_high[metric],
                color=color_high,
                linewidth=3,
                label=f"Hop High (mean, n={len(hop_high_non_base)})",
                zorder=3
            )

        if hop_low_non_base and metric in hop_low_non_base[0].columns:
            mean_low, lower_low, upper_low = compute_statistics(hop_low_non_base, metric)

            # Plot confidence interval
            ax.fill_between(
                mean_low["step"],
                lower_low[metric],
                upper_low[metric],
                color=color_low,
                alpha=0.2,
                zorder=2
            )

            # Plot mean
            ax.plot(
                mean_low["step"],
                mean_low[metric],
                color=color_low,
                linewidth=3,
                label=f"Hop Low (mean, n={len(hop_low_non_base)})",
                zorder=3
            )

        ax.set_xlabel("Edit Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Set y-limits for success/accuracy metrics
        if metric in ["edit_success", "retention_rate", "edited_acc", "retain_acc"]:
            ax.set_ylim(-0.05, 1.05)

    # Final metrics comparison (bar chart with error bars)
    ax = axes[1, 2]

    # Extract final step metrics for each trial
    final_metrics_high = []
    for trial_df in hop_high_non_base:
        if not trial_df.empty:
            last_step = trial_df.iloc[-1]
            final_metrics_high.append({
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
            })

    final_metrics_low = []
    for trial_df in hop_low_non_base:
        if not trial_df.empty:
            last_step = trial_df.iloc[-1]
            final_metrics_low.append({
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
            })

    if final_metrics_high and final_metrics_low:
        # Compute means and standard errors
        edited_acc_high = [m["edited_acc"] for m in final_metrics_high]
        retain_acc_high = [m["retain_acc"] for m in final_metrics_high]
        edited_acc_low = [m["edited_acc"] for m in final_metrics_low]
        retain_acc_low = [m["retain_acc"] for m in final_metrics_low]

        mean_edited_high = np.mean(edited_acc_high)
        mean_retain_high = np.mean(retain_acc_high)
        mean_edited_low = np.mean(edited_acc_low)
        mean_retain_low = np.mean(retain_acc_low)

        sem_edited_high = np.std(edited_acc_high, ddof=1) / np.sqrt(len(edited_acc_high))
        sem_retain_high = np.std(retain_acc_high, ddof=1) / np.sqrt(len(retain_acc_high))
        sem_edited_low = np.std(edited_acc_low, ddof=1) / np.sqrt(len(edited_acc_low))
        sem_retain_low = np.std(retain_acc_low, ddof=1) / np.sqrt(len(retain_acc_low))

        # Bar chart
        x = np.arange(2)  # Two groups: Hop High, Hop Low
        width = 0.35

        ax.bar(
            x - width/2,
            [mean_edited_high, mean_edited_low],
            width,
            yerr=[sem_edited_high * 1.96, sem_edited_low * 1.96],  # 95% CI
            label="Edited Acc",
            alpha=0.8,
            capsize=5
        )
        ax.bar(
            x + width/2,
            [mean_retain_high, mean_retain_low],
            width,
            yerr=[sem_retain_high * 1.96, sem_retain_low * 1.96],  # 95% CI
            label="Retain Acc",
            alpha=0.8,
            capsize=5
        )

        ax.set_xlabel("Selection Mode", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Final Step Accuracy (Mean ± 95% CI)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(["Hop High", "Hop Low"], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No final metrics available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved multi-trial comparison plot to: {output_path}")


def print_summary_stats(base_dir: str, num_trials: int) -> None:
    """Print summary statistics for multi-trial experiments.

    Args:
        base_dir: Base directory containing trial subdirectories
        num_trials: Number of trials per mode
    """
    base = Path(base_dir)

    print("\n" + "=" * 80)
    print("Multi-Trial Summary Statistics")
    print("=" * 80)

    for mode in ["hop_high", "hop_low"]:
        trial_dfs = load_all_trials(base, mode, num_trials)

        if not trial_dfs:
            continue

        print(f"\n{mode.replace('_', ' ').title()}:")
        print(f"  Successful trials: {len(trial_dfs)} / {num_trials}")
        print("-" * 40)

        # Compute statistics for final step metrics
        non_base_trials = [df[~df["is_baseline"]] for df in trial_dfs]

        metrics = ["edit_success", "retention_rate", "edited_acc", "retain_acc", "weight_fro_norm"]
        for metric in metrics:
            final_values = []
            for trial_df in non_base_trials:
                if not trial_df.empty and metric in trial_df.columns:
                    final_values.append(trial_df[metric].iloc[-1])

            if final_values:
                mean_val = np.mean(final_values)
                std_val = np.std(final_values, ddof=1)
                sem_val = std_val / np.sqrt(len(final_values))
                ci_95 = 1.96 * sem_val

                metric_name = metric.replace("_", " ").title()
                print(f"  {metric_name:25s}: {mean_val:.3f} ± {ci_95:.3f} (95% CI)")

    print("\n" + "=" * 80)


def main():
    """Main entry point for multi-trial comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare multi-trial hop high/low sequential editing results"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing trial subdirectories",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="Number of trials per mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison plot (default: <base-dir>/multi_trial_comparison.png)",
    )

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = str(Path(args.base_dir) / "multi_trial_comparison.png")

    print("=" * 80)
    print("Hop Multi-Trial Comparison Analysis")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of trials per mode: {args.num_trials}")
    print(f"Output plot: {args.output}")
    print("=" * 80)

    # Generate comparison plot
    plot_multi_trial_comparison(args.base_dir, args.num_trials, args.output)

    # Print summary statistics
    print_summary_stats(args.base_dir, args.num_trials)

    print("\n✓ Multi-trial comparison analysis complete!")


if __name__ == "__main__":
    main()
