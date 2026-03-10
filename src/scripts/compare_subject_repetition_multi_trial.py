"""Compare results from multiple trials of subject repetition experiments.

This script loads results from multiple trials for all repetition modes (no_rep, rep_2, rep_4),
and generates comparison plots showing:
  - Individual trial lines (thin, semi-transparent)
  - Mean line (thick)
  - 95% confidence interval (shaded area)

Usage:
    python -m src.scripts.compare_subject_repetition_multi_trial \
        --base-dir outputs/subject_rep_multi_trial_no_alias \
        --num-trials 10

Optional (cut x-axis by step):
    python -m src.scripts.compare_subject_repetition_multi_trial \
        --base-dir outputs/subject_rep_multi_trial_no_alias \
        --num-trials 10 \
        --max-step 20
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

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
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Handle malformed JSONL where multiple records might be on one line
            # Find all complete JSON objects by matching balanced braces
            json_strings = []
            i = 0
            while i < len(line):
                if line[i] == '{':
                    # Found start of JSON object, find the matching closing brace
                    brace_count = 0
                    start = i
                    for j in range(i, len(line)):
                        if line[j] == '{':
                            brace_count += 1
                        elif line[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found matching closing brace
                                json_strings.append(line[start:j+1])
                                i = j + 1
                                break
                    else:
                        # No matching closing brace found
                        i += 1
                else:
                    i += 1

            for json_str in json_strings:
                try:
                    record = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Failed to parse JSON in {stats_file}:{line_num}")
                    print(f"Error: {e}")
                    print(f"JSON string (first 200 chars): {json_str[:200]}")
                    continue  # Skip this malformed record instead of crashing

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

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Ensure step is numeric + sorted
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").reset_index(drop=True)
    return df


def load_all_trials(base_dir: Path, mode: str, num_trials: int) -> List[pd.DataFrame]:
    """Load stats for all trials of a given mode.

    Args:
        base_dir: Base directory containing mode subdirectories
        mode: "no_rep", "rep_2", or "rep_4"
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


def compute_statistics(
    trial_dfs: List[pd.DataFrame], metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute mean and 95% confidence interval for a metric across trials."""
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
    all_values = np.array(all_values, dtype=np.float64)

    # Compute mean and 95% CI using t-distribution
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0, ddof=1)
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


def _apply_max_step(dfs: List[pd.DataFrame], max_step: Optional[int]) -> List[pd.DataFrame]:
    """Apply step cutoff (inclusive) to a list of DataFrames."""
    if max_step is None:
        return dfs
    cut = []
    for df in dfs:
        df2 = df[df["step"] <= max_step].copy()
        if not df2.empty:
            cut.append(df2)
    return cut


def plot_multi_trial_comparison(
    base_dir: str,
    num_trials: int,
    output_path: str,
    xscale: str = "linear",
    max_step: Optional[int] = None,
) -> None:
    """Generate multi-trial comparison plots.

    Args:
        base_dir: Base directory containing trial subdirectories
        num_trials: Number of trials per mode
        output_path: Output path for the comparison figure
        xscale: X-axis scale ("linear" or "log")
        max_step: Plot up to this edit step (inclusive). If None, use all steps.
    """
    base = Path(base_dir)

    # Load data for all trials
    print("Loading trial data...")
    no_rep_trials = load_all_trials(base, "no_rep", num_trials)
    rep_2_trials = load_all_trials(base, "rep_2", num_trials)
    rep_4_trials = load_all_trials(base, "rep_4", num_trials)

    if not no_rep_trials or not rep_2_trials or not rep_4_trials:
        print("Error: No data loaded for one or more modes. Cannot generate comparison plot.")
        return

    print(f"Loaded {len(no_rep_trials)} no_rep trials, {len(rep_2_trials)} rep_2 trials, {len(rep_4_trials)} rep_4 trials")

    # Filter to non-baseline steps
    no_rep_non_base = [df[~df["is_baseline"]].copy() for df in no_rep_trials]
    rep_2_non_base = [df[~df["is_baseline"]].copy() for df in rep_2_trials]
    rep_4_non_base = [df[~df["is_baseline"]].copy() for df in rep_4_trials]

    # Optional: cut by max_step
    no_rep_non_base = _apply_max_step(no_rep_non_base, max_step)
    rep_2_non_base = _apply_max_step(rep_2_non_base, max_step)
    rep_4_non_base = _apply_max_step(rep_4_non_base, max_step)

    if not no_rep_non_base or not rep_2_non_base or not rep_4_non_base:
        print("Error: After applying --max-step, no non-baseline data remains for one or more modes.")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    title_suffix = "" if max_step is None else f" (up to step {max_step})"
    fig.suptitle(
        f"Subject Repetition Multi-Trial Comparison (n={len(no_rep_non_base)} trials per mode){title_suffix}",
        fontsize=18,
        y=0.995,
    )

    # Color scheme
    color_no_rep = "C0"  # Blue
    color_rep_2 = "C1"   # Orange
    color_rep_4 = "C2"   # Green

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
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Plot individual trials (thin, semi-transparent)
        for trial_df in no_rep_non_base:
            if metric in trial_df.columns:
                ax.plot(
                    trial_df["step"],
                    trial_df[metric],
                    color=color_no_rep,
                    alpha=0.15,
                    linewidth=1,
                    zorder=1,
                )

        for trial_df in rep_2_non_base:
            if metric in trial_df.columns:
                ax.plot(
                    trial_df["step"],
                    trial_df[metric],
                    color=color_rep_2,
                    alpha=0.15,
                    linewidth=1,
                    zorder=1,
                )

        for trial_df in rep_4_non_base:
            if metric in trial_df.columns:
                ax.plot(
                    trial_df["step"],
                    trial_df[metric],
                    color=color_rep_4,
                    alpha=0.15,
                    linewidth=1,
                    zorder=1,
                )

        # Compute statistics and plot mean + CI
        if metric in no_rep_non_base[0].columns:
            mean_no_rep, lower_no_rep, upper_no_rep = compute_statistics(no_rep_non_base, metric)

            ax.fill_between(
                mean_no_rep["step"],
                lower_no_rep[metric],
                upper_no_rep[metric],
                color=color_no_rep,
                alpha=0.2,
                zorder=2,
            )
            ax.plot(
                mean_no_rep["step"],
                mean_no_rep[metric],
                color=color_no_rep,
                linewidth=3,
                label=f"No Rep (rep=1, n={len(no_rep_non_base)})",
                zorder=3,
            )

        if metric in rep_2_non_base[0].columns:
            mean_rep_2, lower_rep_2, upper_rep_2 = compute_statistics(rep_2_non_base, metric)

            ax.fill_between(
                mean_rep_2["step"],
                lower_rep_2[metric],
                upper_rep_2[metric],
                color=color_rep_2,
                alpha=0.2,
                zorder=2,
            )
            ax.plot(
                mean_rep_2["step"],
                mean_rep_2[metric],
                color=color_rep_2,
                linewidth=3,
                label=f"Rep 2 (rep=2, n={len(rep_2_non_base)})",
                zorder=3,
            )

        if metric in rep_4_non_base[0].columns:
            mean_rep_4, lower_rep_4, upper_rep_4 = compute_statistics(rep_4_non_base, metric)

            ax.fill_between(
                mean_rep_4["step"],
                lower_rep_4[metric],
                upper_rep_4[metric],
                color=color_rep_4,
                alpha=0.2,
                zorder=2,
            )
            ax.plot(
                mean_rep_4["step"],
                mean_rep_4[metric],
                color=color_rep_4,
                linewidth=3,
                label=f"Rep 4 (rep=4, n={len(rep_4_non_base)})",
                zorder=3,
            )

        ax.set_xlabel("Edit Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Optional log scale for x-axis
        if xscale == "log":
            ax.set_xscale("log")

        # Set y-limits for success/accuracy metrics
        if metric in ["edit_success", "retention_rate", "edited_acc", "retain_acc"]:
            ax.set_ylim(-0.05, 1.05)

    # Final metrics comparison (bar chart with error bars)
    ax = axes[1, 2]

    # Extract final step metrics for each trial
    final_metrics_no_rep = []
    for trial_df in no_rep_non_base:
        if not trial_df.empty:
            last_step = trial_df.iloc[-1]
            final_metrics_no_rep.append({
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
            })

    final_metrics_rep_2 = []
    for trial_df in rep_2_non_base:
        if not trial_df.empty:
            last_step = trial_df.iloc[-1]
            final_metrics_rep_2.append({
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
            })

    final_metrics_rep_4 = []
    for trial_df in rep_4_non_base:
        if not trial_df.empty:
            last_step = trial_df.iloc[-1]
            final_metrics_rep_4.append({
                "edited_acc": last_step.get("edited_acc", 0),
                "retain_acc": last_step.get("retain_acc", 0),
            })

    if final_metrics_no_rep and final_metrics_rep_2 and final_metrics_rep_4:
        # Compute means and SEMs
        edited_acc_no_rep = [m["edited_acc"] for m in final_metrics_no_rep]
        retain_acc_no_rep = [m["retain_acc"] for m in final_metrics_no_rep]
        edited_acc_rep_2 = [m["edited_acc"] for m in final_metrics_rep_2]
        retain_acc_rep_2 = [m["retain_acc"] for m in final_metrics_rep_2]
        edited_acc_rep_4 = [m["edited_acc"] for m in final_metrics_rep_4]
        retain_acc_rep_4 = [m["retain_acc"] for m in final_metrics_rep_4]

        mean_edited_no_rep = np.mean(edited_acc_no_rep)
        mean_retain_no_rep = np.mean(retain_acc_no_rep)
        mean_edited_rep_2 = np.mean(edited_acc_rep_2)
        mean_retain_rep_2 = np.mean(retain_acc_rep_2)
        mean_edited_rep_4 = np.mean(edited_acc_rep_4)
        mean_retain_rep_4 = np.mean(retain_acc_rep_4)

        sem_edited_no_rep = (
            np.std(edited_acc_no_rep, ddof=1) / np.sqrt(len(edited_acc_no_rep))
            if len(edited_acc_no_rep) > 1 else 0.0
        )
        sem_retain_no_rep = (
            np.std(retain_acc_no_rep, ddof=1) / np.sqrt(len(retain_acc_no_rep))
            if len(retain_acc_no_rep) > 1 else 0.0
        )
        sem_edited_rep_2 = (
            np.std(edited_acc_rep_2, ddof=1) / np.sqrt(len(edited_acc_rep_2))
            if len(edited_acc_rep_2) > 1 else 0.0
        )
        sem_retain_rep_2 = (
            np.std(retain_acc_rep_2, ddof=1) / np.sqrt(len(retain_acc_rep_2))
            if len(retain_acc_rep_2) > 1 else 0.0
        )
        sem_edited_rep_4 = (
            np.std(edited_acc_rep_4, ddof=1) / np.sqrt(len(edited_acc_rep_4))
            if len(edited_acc_rep_4) > 1 else 0.0
        )
        sem_retain_rep_4 = (
            np.std(retain_acc_rep_4, ddof=1) / np.sqrt(len(retain_acc_rep_4))
            if len(retain_acc_rep_4) > 1 else 0.0
        )

        x = np.arange(3)  # No Rep, Rep 2, Rep 4
        width = 0.35

        ax.bar(
            x - width / 2,
            [mean_edited_no_rep, mean_edited_rep_2, mean_edited_rep_4],
            width,
            yerr=[sem_edited_no_rep * 1.96, sem_edited_rep_2 * 1.96, sem_edited_rep_4 * 1.96],
            label="Edited Acc",
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x + width / 2,
            [mean_retain_no_rep, mean_retain_rep_2, mean_retain_rep_4],
            width,
            yerr=[sem_retain_no_rep * 1.96, sem_retain_rep_2 * 1.96, sem_retain_rep_4 * 1.96],
            label="Retain Acc",
            alpha=0.8,
            capsize=5,
        )

        ax.set_xlabel("Repetition Mode", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Final Step Accuracy (Mean ± 95% CI)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(["No Rep (1)", "Rep 2 (2)", "Rep 4 (4)"], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5, 0.5,
            "No final metrics available",
            ha="center", va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved multi-trial comparison plot to: {output_path}")


def print_summary_stats(base_dir: str, num_trials: int, max_step: Optional[int] = None) -> None:
    """Print summary statistics for multi-trial experiments."""
    base = Path(base_dir)

    print("\n" + "=" * 80)
    print("Multi-Trial Summary Statistics")
    if max_step is not None:
        print(f"(up to step {max_step})")
    print("=" * 80)

    for mode in ["no_rep", "rep_2", "rep_4"]:
        trial_dfs = load_all_trials(base, mode, num_trials)
        if not trial_dfs:
            continue

        non_base_trials = [df[~df["is_baseline"]].copy() for df in trial_dfs]
        non_base_trials = _apply_max_step(non_base_trials, max_step)

        if not non_base_trials:
            continue

        print(f"\n{mode.replace('_', ' ').title()}:")
        print(f"  Successful trials: {len(non_base_trials)} / {num_trials}")
        print("-" * 40)

        metrics = ["edit_success", "retention_rate", "edited_acc", "retain_acc", "weight_fro_norm"]
        for metric in metrics:
            final_values = []
            for trial_df in non_base_trials:
                if not trial_df.empty and metric in trial_df.columns:
                    final_values.append(trial_df[metric].iloc[-1])

            if final_values:
                mean_val = np.mean(final_values)
                std_val = np.std(final_values, ddof=1) if len(final_values) > 1 else 0.0
                sem_val = std_val / np.sqrt(len(final_values)) if len(final_values) > 0 else 0.0
                ci_95 = 1.96 * sem_val

                metric_name = metric.replace("_", " ").title()
                print(f"  {metric_name:25s}: {mean_val:.3f} ± {ci_95:.3f} (95% CI)")

    print("\n" + "=" * 80)


def main():
    """Main entry point for multi-trial comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Compare multi-trial subject repetition sequential editing results"
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
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Plot up to this edit step (inclusive). If omitted, plot all steps.",
    )

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        args.output = str(Path(args.base_dir) / "multi_trial_comparison.png")

    print("=" * 80)
    print("Subject Repetition Multi-Trial Comparison Analysis")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of trials per mode: {args.num_trials}")
    print(f"Output plot: {args.output}")
    if args.max_step is None:
        print("Max step: (not set) -> plot all steps")
    else:
        print(f"Max step: {args.max_step} (inclusive)")
    print("=" * 80)

    # Generate comparison plots (linear + log)
    output_linear = args.output
    output_log = str(
        Path(output_linear).with_name(
            Path(output_linear).stem + "_log" + Path(output_linear).suffix
        )
    )

    plot_multi_trial_comparison(
        args.base_dir, args.num_trials, output_linear, xscale="linear", max_step=args.max_step
    )
    plot_multi_trial_comparison(
        args.base_dir, args.num_trials, output_log, xscale="log", max_step=args.max_step
    )

    # Print summary statistics
    print_summary_stats(args.base_dir, args.num_trials, max_step=args.max_step)

    print("\n✓ Multi-trial comparison analysis complete!")


if __name__ == "__main__":
    main()
