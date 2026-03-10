"""Analyze and visualize results by hop distance for degree high/low exclusive experiments.

This script loads results from degree_high and degree_low experiments and analyzes
how retain accuracy varies by hop distance from edited subjects.

Usage:
    # Single trial analysis
    python -m src.scripts.analyze_hop_distance_effects \
        --high-dir outputs/degree_exclusive_multi_trial_no_alias/degree_high/trial_0 \
        --low-dir outputs/degree_exclusive_multi_trial_no_alias/degree_low/trial_0 \
        --output-dir outputs/degree_exclusive_multi_trial_no_alias

    # Multi-trial analysis
    python -m src.scripts.analyze_hop_distance_effects \
        --base-dir outputs/degree_exclusive_multi_trial_no_alias \
        --num-trials 10 \
        --output-dir outputs/degree_exclusive_multi_trial_no_alias
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 16)


def load_hop_accuracy_data(trial_dir: Path) -> Dict[int, Dict[int, List[bool]]]:
    """Load accuracy data grouped by step and hop distance.

    Args:
        trial_dir: Path to trial directory

    Returns:
        Dict mapping step -> hop -> list of correctness (True/False)
    """
    # Load triple accuracy records
    acc_file = trial_dir / "triple_acc.jsonl"
    if not acc_file.exists():
        print(f"Warning: {acc_file} not found")
        return {}

    # Load ripple records to get hop distances
    ripple_file = trial_dir / "ripple_triples.jsonl"
    if not ripple_file.exists():
        print(f"Warning: {ripple_file} not found")
        return {}

    # Build hop distance map: (step, tid) -> hop_subj
    hop_map = {}
    with open(ripple_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            key = (record['step'], record['tid'])
            hop_map[key] = record['hop_subj']

    # Load accuracy records and group by step and hop
    by_step_hop = defaultdict(lambda: defaultdict(list))

    with open(acc_file, 'r') as f:
        for line in f:
            record = json.loads(line)

            # Only consider retain triples
            if record.get('triple_type') != 'retain':
                continue

            step = record['step']
            tid = record['tid']
            is_correct = record['is_correct']

            # Get hop distance
            key = (step, tid)
            if key in hop_map:
                hop = hop_map[key]
                by_step_hop[step][hop].append(is_correct)

    return by_step_hop


def load_ripple_effect_data(trial_dir: Path) -> Dict[int, Dict[str, any]]:
    """Load ripple effect magnitude data by hop.

    Returns:
        Dict mapping hop -> statistics
    """
    ripple_file = trial_dir / "ripple_triples.jsonl"
    if not ripple_file.exists():
        return {}

    by_hop = defaultdict(list)

    with open(ripple_file, 'r') as f:
        for line in f:
            r = json.loads(line)
            hop = r['hop_subj']
            delta_logit = r['delta_logit']
            by_hop[hop].append(delta_logit)

    # Compute statistics
    stats = {}
    for hop, deltas in by_hop.items():
        deltas_arr = np.array(deltas)
        stats[hop] = {
            'mean_delta': np.mean(deltas_arr),
            'mean_abs_delta': np.mean(np.abs(deltas_arr)),
            'median_delta': np.median(deltas_arr),
            'std_delta': np.std(deltas_arr),
            'count': len(deltas_arr)
        }

    return stats


def load_subject_degree_data(trial_dir: Path) -> Dict[int, List[int]]:
    """Load subject degree distribution for each hop.

    Returns:
        Dict mapping hop -> list of subject degrees
    """
    ripple_file = trial_dir / "ripple_triples.jsonl"
    if not ripple_file.exists():
        return {}

    by_hop = defaultdict(list)
    seen = defaultdict(set)  # Track unique triples per hop

    with open(ripple_file, 'r') as f:
        for line in f:
            r = json.loads(line)
            if r['step'] != 1:  # Only use step 1 to avoid duplicates
                continue

            hop = r['hop_subj']
            tid = r['tid']
            degree = r['degree_s']

            if tid not in seen[hop]:
                by_hop[hop].append(degree)
                seen[hop].add(tid)

    return dict(by_hop)


def compute_hop_accuracy_series(by_step_hop: Dict[int, Dict[int, List[bool]]]) -> Dict[int, pd.DataFrame]:
    """Convert hop accuracy data to time series DataFrames.

    Args:
        by_step_hop: Dict mapping step -> hop -> list of correctness

    Returns:
        Dict mapping hop -> DataFrame with columns [step, accuracy, count]
    """
    # Get all hops and steps
    all_hops = set()
    all_steps = set()
    for step, hop_dict in by_step_hop.items():
        all_steps.add(step)
        all_hops.update(hop_dict.keys())

    # Create DataFrame for each hop
    hop_dfs = {}
    for hop in sorted(all_hops):
        records = []
        for step in sorted(all_steps):
            if hop in by_step_hop[step]:
                correct_list = by_step_hop[step][hop]
                acc = np.mean(correct_list) if correct_list else np.nan
                count = len(correct_list)
            else:
                acc = np.nan
                count = 0

            records.append({
                'step': step,
                'accuracy': acc,
                'count': count
            })

        hop_dfs[hop] = pd.DataFrame(records)

    return hop_dfs


def load_multi_trial_hop_data(base_dir: Path, mode: str, num_trials: int) -> List[Dict[int, pd.DataFrame]]:
    """Load hop accuracy data for multiple trials.

    Args:
        base_dir: Base directory containing mode subdirectories
        mode: "degree_high" or "degree_low"
        num_trials: Number of trials

    Returns:
        List of hop_dfs dicts, one per trial
    """
    trial_hop_dfs_list = []
    mode_dir = base_dir / mode

    for trial_id in range(num_trials):
        trial_dir = mode_dir / f"trial_{trial_id}"
        if not trial_dir.exists():
            print(f"Warning: {trial_dir} not found")
            continue

        by_step_hop = load_hop_accuracy_data(trial_dir)
        if not by_step_hop:
            print(f"Warning: No hop data for {mode} trial {trial_id}")
            continue

        hop_dfs = compute_hop_accuracy_series(by_step_hop)
        trial_hop_dfs_list.append(hop_dfs)

    return trial_hop_dfs_list


def aggregate_multi_trial_hop_data(trial_hop_dfs_list: List[Dict[int, pd.DataFrame]]) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Aggregate hop accuracy data across trials.

    Args:
        trial_hop_dfs_list: List of hop_dfs dicts from different trials

    Returns:
        Dict mapping hop -> (mean_df, ci_lower_df, ci_upper_df)
    """
    if not trial_hop_dfs_list:
        return {}

    # Get all hops
    all_hops = set()
    for hop_dfs in trial_hop_dfs_list:
        all_hops.update(hop_dfs.keys())

    aggregated = {}

    for hop in sorted(all_hops):
        # Collect accuracy series for this hop across trials
        acc_series = []
        count_series = []

        for hop_dfs in trial_hop_dfs_list:
            if hop in hop_dfs:
                acc_series.append(hop_dfs[hop]['accuracy'].values)
                count_series.append(hop_dfs[hop]['count'].values)

        if not acc_series:
            continue

        # Stack into array: (n_trials, n_steps)
        acc_array = np.array(acc_series)
        count_array = np.array(count_series)

        # Compute statistics across trials
        mean_acc = np.nanmean(acc_array, axis=0)
        std_acc = np.nanstd(acc_array, axis=0)
        n_trials = np.sum(~np.isnan(acc_array), axis=0)

        # 95% confidence interval
        ci_margin = 1.96 * std_acc / np.sqrt(n_trials)
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin

        # Get steps from first trial
        steps = trial_hop_dfs_list[0][hop]['step'].values if hop in trial_hop_dfs_list[0] else np.arange(len(mean_acc))

        # Create DataFrames
        mean_df = pd.DataFrame({
            'step': steps,
            'accuracy': mean_acc,
            'count': np.mean(count_array, axis=0)
        })

        ci_lower_df = pd.DataFrame({
            'step': steps,
            'accuracy': ci_lower
        })

        ci_upper_df = pd.DataFrame({
            'step': steps,
            'accuracy': ci_upper
        })

        aggregated[hop] = (mean_df, ci_lower_df, ci_upper_df)

    return aggregated


def compute_detailed_statistics(
    high_dir: Path,
    low_dir: Path
) -> Dict[str, any]:
    """Compute detailed statistics comparing high vs low.

    Returns:
        Dictionary with detailed statistics
    """
    stats = {
        'baseline_vs_final': {},
        'ripple_magnitude': {},
        'subject_degrees': {}
    }

    for mode, trial_dir in [('high', high_dir), ('low', low_dir)]:
        # Load hop accuracy data
        by_step_hop = load_hop_accuracy_data(trial_dir)

        # Baseline (step 1) vs Final (step 100)
        baseline_by_hop = {}
        final_by_hop = {}

        for hop in by_step_hop.get(1, {}).keys():
            if hop in by_step_hop.get(1, {}):
                baseline_by_hop[hop] = np.mean(by_step_hop[1][hop])
            if hop in by_step_hop.get(100, {}):
                final_by_hop[hop] = np.mean(by_step_hop[100][hop])

        stats['baseline_vs_final'][mode] = {
            'baseline': baseline_by_hop,
            'final': final_by_hop
        }

        # Ripple effect magnitude
        stats['ripple_magnitude'][mode] = load_ripple_effect_data(trial_dir)

        # Subject degrees
        stats['subject_degrees'][mode] = load_subject_degree_data(trial_dir)

    return stats


def plot_detailed_statistics(
    stats: Dict[str, any],
    output_path: Path
):
    """Plot detailed statistical comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Detailed Statistical Analysis: Degree High vs Low', fontsize=16, fontweight='bold')

    # Get all hops
    all_hops = set()
    for mode in ['high', 'low']:
        all_hops.update(stats['baseline_vs_final'][mode]['baseline'].keys())
        all_hops.update(stats['baseline_vs_final'][mode]['final'].keys())
    all_hops = sorted(all_hops)

    # 1. Baseline accuracy by hop
    ax = axes[0, 0]
    high_baseline = [stats['baseline_vs_final']['high']['baseline'].get(h, np.nan) for h in all_hops]
    low_baseline = [stats['baseline_vs_final']['low']['baseline'].get(h, np.nan) for h in all_hops]

    x = np.arange(len(all_hops))
    width = 0.35
    ax.bar(x - width/2, high_baseline, width, label='High', color='red', alpha=0.7)
    ax.bar(x + width/2, low_baseline, width, label='Low', color='blue', alpha=0.7)
    ax.set_xlabel('Hop Distance', fontsize=11)
    ax.set_ylabel('Accuracy (Step 1)', fontsize=11)
    ax.set_title('Baseline Accuracy by Hop', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Final accuracy by hop
    ax = axes[0, 1]
    high_final = [stats['baseline_vs_final']['high']['final'].get(h, np.nan) for h in all_hops]
    low_final = [stats['baseline_vs_final']['low']['final'].get(h, np.nan) for h in all_hops]

    ax.bar(x - width/2, high_final, width, label='High', color='red', alpha=0.7)
    ax.bar(x + width/2, low_final, width, label='Low', color='blue', alpha=0.7)
    ax.set_xlabel('Hop Distance', fontsize=11)
    ax.set_ylabel('Accuracy (Step 100)', fontsize=11)
    ax.set_title('Final Accuracy by Hop', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Accuracy degradation (Delta)
    ax = axes[0, 2]
    high_delta = []
    low_delta = []
    for h in all_hops:
        hb = stats['baseline_vs_final']['high']['baseline'].get(h, np.nan)
        hf = stats['baseline_vs_final']['high']['final'].get(h, np.nan)
        lb = stats['baseline_vs_final']['low']['baseline'].get(h, np.nan)
        lf = stats['baseline_vs_final']['low']['final'].get(h, np.nan)

        if not np.isnan(hb) and not np.isnan(hf):
            high_delta.append(hf - hb)
        else:
            high_delta.append(np.nan)

        if not np.isnan(lb) and not np.isnan(lf):
            low_delta.append(lf - lb)
        else:
            low_delta.append(np.nan)

    ax.bar(x - width/2, high_delta, width, label='High', color='red', alpha=0.7)
    ax.bar(x + width/2, low_delta, width, label='Low', color='blue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Hop Distance', fontsize=11)
    ax.set_ylabel('Accuracy Change (Final - Baseline)', fontsize=11)
    ax.set_title('Accuracy Degradation (Cumulative Ripple Effect)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add delta values on bars
    for i, (hd, ld) in enumerate(zip(high_delta, low_delta)):
        if not np.isnan(hd):
            ax.text(i - width/2, hd, f'{hd:.3f}', ha='center',
                   va='bottom' if hd > 0 else 'top', fontsize=9)
        if not np.isnan(ld):
            ax.text(i + width/2, ld, f'{ld:.3f}', ha='center',
                   va='bottom' if ld > 0 else 'top', fontsize=9)

    # 4. Ripple magnitude (mean |delta_logit|)
    ax = axes[1, 0]
    high_ripple = [stats['ripple_magnitude']['high'].get(h, {}).get('mean_abs_delta', np.nan) for h in all_hops]
    low_ripple = [stats['ripple_magnitude']['low'].get(h, {}).get('mean_abs_delta', np.nan) for h in all_hops]

    ax.bar(x - width/2, high_ripple, width, label='High', color='red', alpha=0.7)
    ax.bar(x + width/2, low_ripple, width, label='Low', color='blue', alpha=0.7)
    ax.set_xlabel('Hop Distance', fontsize=11)
    ax.set_ylabel('Mean |Δlogit| per Triple', fontsize=11)
    ax.set_title('Ripple Effect Magnitude (Per-Triple Logit Change)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Subject degree distribution (boxplot)
    ax = axes[1, 1]
    high_degrees_data = []
    low_degrees_data = []
    positions = []
    labels = []

    for i, h in enumerate(all_hops):
        if h in stats['subject_degrees']['high']:
            high_degrees_data.append(stats['subject_degrees']['high'][h])
            positions.append(i * 2)
            labels.append(f'H{h}')

        if h in stats['subject_degrees']['low']:
            low_degrees_data.append(stats['subject_degrees']['low'][h])
            positions.append(i * 2 + 0.8)
            labels.append(f'L{h}')

    if high_degrees_data or low_degrees_data:
        bp_high = ax.boxplot(high_degrees_data, positions=[i*2 for i in range(len(high_degrees_data))],
                            widths=0.6, patch_artist=True, showfliers=False)
        bp_low = ax.boxplot(low_degrees_data, positions=[i*2+0.8 for i in range(len(low_degrees_data))],
                           widths=0.6, patch_artist=True, showfliers=False)

        for patch in bp_high['boxes']:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
        for patch in bp_low['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.7)

        ax.set_xlabel('Hop Distance', fontsize=11)
        ax.set_ylabel('Subject Degree', fontsize=11)
        ax.set_title('Subject Degree Distribution (Hop 1+ Triples)', fontsize=12, fontweight='bold')
        ax.set_xticks([i*2+0.4 for i in range(len(all_hops))])
        ax.set_xticklabels([f'Hop {h}' for h in all_hops])
        ax.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='High'),
                          Patch(facecolor='blue', alpha=0.7, label='Low')]
        ax.legend(handles=legend_elements)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    # Compute summary statistics for Hop 1
    hop1_idx = all_hops.index(1) if 1 in all_hops else None

    if hop1_idx is not None:
        summary_text = "Summary Statistics (Hop 1):\n"
        summary_text += "=" * 40 + "\n\n"

        # Baseline
        hb = stats['baseline_vs_final']['high']['baseline'].get(1, np.nan)
        lb = stats['baseline_vs_final']['low']['baseline'].get(1, np.nan)
        summary_text += f"Baseline Accuracy:\n"
        summary_text += f"  High: {hb:.4f}\n"
        summary_text += f"  Low:  {lb:.4f}\n"
        summary_text += f"  Diff: {hb-lb:+.4f}\n\n"

        # Final
        hf = stats['baseline_vs_final']['high']['final'].get(1, np.nan)
        lf = stats['baseline_vs_final']['low']['final'].get(1, np.nan)
        summary_text += f"Final Accuracy:\n"
        summary_text += f"  High: {hf:.4f}\n"
        summary_text += f"  Low:  {lf:.4f}\n"
        summary_text += f"  Diff: {hf-lf:+.4f}\n\n"

        # Degradation
        hd = hf - hb
        ld = lf - lb
        summary_text += f"Accuracy Degradation:\n"
        summary_text += f"  High: {hd:+.4f} ({hd/hb*100:+.1f}%)\n"
        summary_text += f"  Low:  {ld:+.4f} ({ld/lb*100:+.1f}%)\n"
        summary_text += f"  Diff: {hd-ld:+.4f}\n\n"

        summary_text += "→ "
        if abs(hd) > abs(ld):
            summary_text += "High shows LARGER cumulative\n  ripple effect"
        else:
            summary_text += "Low shows LARGER cumulative\n  ripple effect"

        summary_text += "\n\n"

        # Ripple magnitude
        hr = stats['ripple_magnitude']['high'].get(1, {}).get('mean_abs_delta', np.nan)
        lr = stats['ripple_magnitude']['low'].get(1, {}).get('mean_abs_delta', np.nan)
        summary_text += f"Mean |Δlogit| per Triple:\n"
        summary_text += f"  High: {hr:.4f}\n"
        summary_text += f"  Low:  {lr:.4f}\n"
        summary_text += f"  Diff: {hr-lr:+.4f}\n\n"

        summary_text += "→ "
        if abs(hr - lr) < 0.01:
            summary_text += "Per-triple ripple is SIMILAR\n"
            summary_text += "  Difference comes from\n  cumulative effect"
        else:
            summary_text += f"Per-triple ripple differs"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed statistics to {output_path}")


def plot_hop_specific_multi_trial(
    high_aggregated: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    low_aggregated: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    hop: int,
    output_path: Path,
    xscale: str = 'linear'
):
    """Plot detailed multi-trial comparison for a specific hop (similar to multi_trial_comparison.png).

    Args:
        high_aggregated: Aggregated data for degree_high
        low_aggregated: Aggregated data for degree_low
        hop: Hop distance to plot
        output_path: Output file path
        xscale: 'linear' or 'log'
    """
    if hop not in high_aggregated and hop not in low_aggregated:
        print(f"Warning: No data for hop {hop}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Hop {hop} Multi-Trial Comparison: Degree High vs Low',
                 fontsize=16, fontweight='bold')

    color_high = 'red'
    color_low = 'blue'

    # Get data
    high_mean_df, high_ci_lower, high_ci_upper = high_aggregated.get(hop, (None, None, None))
    low_mean_df, low_ci_lower, low_ci_upper = low_aggregated.get(hop, (None, None, None))

    # Plot 1: Retain Accuracy
    ax = axes[0, 0]

    if high_mean_df is not None:
        steps = high_mean_df['step'].values
        acc = high_mean_df['accuracy'].values
        ci_l = high_ci_lower['accuracy'].values
        ci_u = high_ci_upper['accuracy'].values

        ax.plot(steps, acc, color=color_high, linewidth=2.5, label='Degree High (Mean)', marker='o', markersize=3)
        ax.fill_between(steps, ci_l, ci_u, color=color_high, alpha=0.2, label='High 95% CI')

    if low_mean_df is not None:
        steps = low_mean_df['step'].values
        acc = low_mean_df['accuracy'].values
        ci_l = low_ci_lower['accuracy'].values
        ci_u = low_ci_upper['accuracy'].values

        ax.plot(steps, acc, color=color_low, linewidth=2.5, label='Degree Low (Mean)', marker='o', markersize=3)
        ax.fill_between(steps, ci_l, ci_u, color=color_low, alpha=0.2, label='Low 95% CI')

    ax.set_xlabel('Edit Step', fontsize=12)
    ax.set_ylabel('Retain Accuracy', fontsize=12)
    ax.set_title(f'Hop {hop} Retain Accuracy Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if xscale == 'log':
        ax.set_xscale('log')

    # Plot 2: Accuracy Difference
    ax = axes[0, 1]

    if high_mean_df is not None and low_mean_df is not None:
        merged = pd.merge(high_mean_df, low_mean_df, on='step', suffixes=('_high', '_low'))
        merged['diff'] = merged['accuracy_high'] - merged['accuracy_low']

        ax.plot(merged['step'], merged['diff'], color='purple', linewidth=2.5, marker='o', markersize=3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(merged['step'], 0, merged['diff'],
                       where=(merged['diff'] > 0), color='red', alpha=0.3, label='High > Low')
        ax.fill_between(merged['step'], 0, merged['diff'],
                       where=(merged['diff'] <= 0), color='blue', alpha=0.3, label='Low > High')

        # Add mean and final diff
        mean_diff = merged['diff'].mean()
        final_diff = merged['diff'].iloc[-1]
        ax.text(0.02, 0.98, f'Mean: {mean_diff:+.4f}\nFinal: {final_diff:+.4f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Edit Step', fontsize=12)
    ax.set_ylabel('Accuracy Difference (High - Low)', fontsize=12)
    ax.set_title(f'Hop {hop} Accuracy Difference', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if xscale == 'log':
        ax.set_xscale('log')

    # Plot 3: Sample Count
    ax = axes[1, 0]

    if high_mean_df is not None:
        ax.plot(high_mean_df['step'], high_mean_df['count'], color=color_high,
               linewidth=2, label='High', alpha=0.7)
    if low_mean_df is not None:
        ax.plot(low_mean_df['step'], low_mean_df['count'], color=color_low,
               linewidth=2, label='Low', alpha=0.7)

    ax.set_xlabel('Edit Step', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title(f'Hop {hop} Sample Count Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if xscale == 'log':
        ax.set_xscale('log')

    # Plot 4: Relative Accuracy (normalized by baseline)
    ax = axes[1, 1]

    if high_mean_df is not None:
        baseline = high_mean_df['accuracy'].iloc[0]
        if baseline > 0:
            relative = high_mean_df['accuracy'].values / baseline
            ax.plot(high_mean_df['step'], relative, color=color_high,
                   linewidth=2.5, label='High (Relative)', marker='o', markersize=3)

    if low_mean_df is not None:
        baseline = low_mean_df['accuracy'].iloc[0]
        if baseline > 0:
            relative = low_mean_df['accuracy'].values / baseline
            ax.plot(low_mean_df['step'], relative, color=color_low,
                   linewidth=2.5, label='Low (Relative)', marker='o', markersize=3)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax.set_xlabel('Edit Step', fontsize=12)
    ax.set_ylabel('Relative Accuracy (vs Baseline)', fontsize=12)
    ax.set_title(f'Hop {hop} Relative Accuracy Retention', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if xscale == 'log':
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved hop {hop} multi-trial comparison to {output_path}")


def plot_hop_comparison_single_trial(
    high_hop_dfs: Dict[int, pd.DataFrame],
    low_hop_dfs: Dict[int, pd.DataFrame],
    output_path: Path
):
    """Plot hop distance comparison for single trial.

    Args:
        high_hop_dfs: Hop DataFrames for degree_high
        low_hop_dfs: Hop DataFrames for degree_low
        output_path: Path to save plot
    """
    # Get all hops
    all_hops = sorted(set(list(high_hop_dfs.keys()) + list(low_hop_dfs.keys())))

    # Create subplots
    n_hops = len(all_hops)
    fig, axes = plt.subplots(n_hops + 1, 2, figsize=(20, 5 * (n_hops + 1)))

    # Overall accuracy (all hops combined)
    ax_overall = axes[0, 0]

    # Compute overall accuracy
    high_overall = []
    low_overall = []

    for hop_dfs, overall_list in [(high_hop_dfs, high_overall), (low_hop_dfs, low_overall)]:
        if not hop_dfs:
            continue

        # Get all steps
        steps = sorted(set(step for df in hop_dfs.values() for step in df['step']))

        for step in steps:
            all_correct = []
            for hop, df in hop_dfs.items():
                step_data = df[df['step'] == step]
                if not step_data.empty and not pd.isna(step_data['accuracy'].iloc[0]):
                    acc = step_data['accuracy'].iloc[0]
                    count = step_data['count'].iloc[0]
                    # Reconstruct number of correct/incorrect
                    n_correct = int(acc * count)
                    all_correct.extend([True] * n_correct + [False] * (int(count) - n_correct))

            overall_acc = np.mean(all_correct) if all_correct else np.nan
            overall_list.append({'step': step, 'accuracy': overall_acc})

    if high_overall:
        high_overall_df = pd.DataFrame(high_overall)
        ax_overall.plot(high_overall_df['step'], high_overall_df['accuracy'],
                       label='Degree High', color='red', linewidth=2, marker='o', markersize=3)

    if low_overall:
        low_overall_df = pd.DataFrame(low_overall)
        ax_overall.plot(low_overall_df['step'], low_overall_df['accuracy'],
                       label='Degree Low', color='blue', linewidth=2, marker='o', markersize=3)

    ax_overall.set_xlabel('Edit Step', fontsize=12)
    ax_overall.set_ylabel('Retain Accuracy', fontsize=12)
    ax_overall.set_title('Overall Retain Accuracy (All Hops)', fontsize=14, fontweight='bold')
    ax_overall.legend(fontsize=11)
    ax_overall.grid(True, alpha=0.3)

    # Sample count distribution
    ax_count = axes[0, 1]

    # Get final step counts by hop
    high_final_counts = {}
    low_final_counts = {}

    for hop in all_hops:
        if hop in high_hop_dfs:
            final_step = high_hop_dfs[hop]['step'].max()
            count = high_hop_dfs[hop][high_hop_dfs[hop]['step'] == final_step]['count'].iloc[0]
            high_final_counts[hop] = int(count)

        if hop in low_hop_dfs:
            final_step = low_hop_dfs[hop]['step'].max()
            count = low_hop_dfs[hop][low_hop_dfs[hop]['step'] == final_step]['count'].iloc[0]
            low_final_counts[hop] = int(count)

    x = np.arange(len(all_hops))
    width = 0.35

    high_counts = [high_final_counts.get(hop, 0) for hop in all_hops]
    low_counts = [low_final_counts.get(hop, 0) for hop in all_hops]

    ax_count.bar(x - width/2, high_counts, width, label='Degree High', color='red', alpha=0.7)
    ax_count.bar(x + width/2, low_counts, width, label='Degree Low', color='blue', alpha=0.7)

    ax_count.set_xlabel('Hop Distance', fontsize=12)
    ax_count.set_ylabel('Sample Count', fontsize=12)
    ax_count.set_title('Sample Distribution by Hop Distance (Final Step)', fontsize=14, fontweight='bold')
    ax_count.set_xticks(x)
    ax_count.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax_count.legend(fontsize=11)
    ax_count.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    high_total = sum(high_counts)
    low_total = sum(low_counts)

    for i, (h_count, l_count) in enumerate(zip(high_counts, low_counts)):
        if h_count > 0:
            pct = 100 * h_count / high_total
            ax_count.text(i - width/2, h_count, f'{pct:.1f}%',
                         ha='center', va='bottom', fontsize=9)
        if l_count > 0:
            pct = 100 * l_count / low_total
            ax_count.text(i + width/2, l_count, f'{pct:.1f}%',
                         ha='center', va='bottom', fontsize=9)

    # Plot each hop separately
    for idx, hop in enumerate(all_hops):
        ax = axes[idx + 1, 0]

        if hop in high_hop_dfs:
            df = high_hop_dfs[hop]
            ax.plot(df['step'], df['accuracy'], label='Degree High',
                   color='red', linewidth=2, marker='o', markersize=3)

        if hop in low_hop_dfs:
            df = low_hop_dfs[hop]
            ax.plot(df['step'], df['accuracy'], label='Degree Low',
                   color='blue', linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Edit Step', fontsize=12)
        ax.set_ylabel('Retain Accuracy', fontsize=12)
        ax.set_title(f'Hop {hop} Retain Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add sample counts in legend
        legend_text = []
        if hop in high_hop_dfs:
            final_count = high_final_counts[hop]
            legend_text.append(f'High: n={final_count}')
        if hop in low_hop_dfs:
            final_count = low_final_counts[hop]
            legend_text.append(f'Low: n={final_count}')

        ax.text(0.02, 0.02, '\n'.join(legend_text),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))

        # Accuracy difference plot
        ax_diff = axes[idx + 1, 1]

        if hop in high_hop_dfs and hop in low_hop_dfs:
            high_df = high_hop_dfs[hop]
            low_df = low_hop_dfs[hop]

            # Merge on step
            merged = pd.merge(high_df, low_df, on='step', suffixes=('_high', '_low'))
            merged['diff'] = merged['accuracy_high'] - merged['accuracy_low']

            ax_diff.plot(merged['step'], merged['diff'],
                        color='purple', linewidth=2, marker='o', markersize=3)
            ax_diff.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_diff.fill_between(merged['step'], 0, merged['diff'],
                                where=(merged['diff'] > 0), color='red', alpha=0.3,
                                label='High > Low')
            ax_diff.fill_between(merged['step'], 0, merged['diff'],
                                where=(merged['diff'] <= 0), color='blue', alpha=0.3,
                                label='Low > High')

            ax_diff.set_xlabel('Edit Step', fontsize=12)
            ax_diff.set_ylabel('Accuracy Difference (High - Low)', fontsize=12)
            ax_diff.set_title(f'Hop {hop} Accuracy Difference', fontsize=14, fontweight='bold')
            ax_diff.legend(fontsize=11)
            ax_diff.grid(True, alpha=0.3)

            # Add mean difference
            mean_diff = merged['diff'].mean()
            ax_diff.text(0.02, 0.98, f'Mean diff: {mean_diff:+.4f}',
                        transform=ax_diff.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.5))
        else:
            ax_diff.text(0.5, 0.5, 'N/A\n(data missing for one mode)',
                        ha='center', va='center', fontsize=12, transform=ax_diff.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved hop comparison plot to {output_path}")


def plot_hop_comparison_multi_trial(
    high_aggregated: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    low_aggregated: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    output_path: Path
):
    """Plot hop distance comparison for multi-trial aggregated data.

    Args:
        high_aggregated: Aggregated hop data for degree_high
        low_aggregated: Aggregated hop data for degree_low
        output_path: Path to save plot
    """
    all_hops = sorted(set(list(high_aggregated.keys()) + list(low_aggregated.keys())))

    # Create subplots
    n_hops = len(all_hops)
    fig, axes = plt.subplots(n_hops + 1, 2, figsize=(20, 5 * (n_hops + 1)))

    # Overall accuracy plot
    ax_overall = axes[0, 0]

    # Compute overall accuracy across all hops
    for label, aggregated, color in [
        ('Degree High', high_aggregated, 'red'),
        ('Degree Low', low_aggregated, 'blue')
    ]:
        if not aggregated:
            continue

        # Get steps from first hop
        first_hop = list(aggregated.keys())[0]
        mean_df, _, _ = aggregated[first_hop]
        steps = mean_df['step'].values

        overall_accs = []
        overall_counts = []

        for step_idx, step in enumerate(steps):
            step_accs = []
            step_weights = []

            for hop, (mean_df, _, _) in aggregated.items():
                if step_idx < len(mean_df):
                    acc = mean_df.iloc[step_idx]['accuracy']
                    count = mean_df.iloc[step_idx]['count']

                    if not np.isnan(acc) and count > 0:
                        step_accs.append(acc)
                        step_weights.append(count)

            if step_accs:
                # Weighted average
                weighted_acc = np.average(step_accs, weights=step_weights)
                overall_accs.append(weighted_acc)
            else:
                overall_accs.append(np.nan)

        ax_overall.plot(steps, overall_accs, label=label, color=color,
                       linewidth=2.5, marker='o', markersize=3)

    ax_overall.set_xlabel('Edit Step', fontsize=12)
    ax_overall.set_ylabel('Retain Accuracy', fontsize=12)
    ax_overall.set_title('Overall Retain Accuracy (All Hops, Multi-Trial Mean)',
                        fontsize=14, fontweight='bold')
    ax_overall.legend(fontsize=11)
    ax_overall.grid(True, alpha=0.3)

    # Sample count distribution
    ax_count = axes[0, 1]

    # Get final step counts
    high_final_counts = {}
    low_final_counts = {}

    for hop in all_hops:
        if hop in high_aggregated:
            mean_df, _, _ = high_aggregated[hop]
            count = mean_df.iloc[-1]['count']
            high_final_counts[hop] = count

        if hop in low_aggregated:
            mean_df, _, _ = low_aggregated[hop]
            count = mean_df.iloc[-1]['count']
            low_final_counts[hop] = count

    x = np.arange(len(all_hops))
    width = 0.35

    high_counts = [high_final_counts.get(hop, 0) for hop in all_hops]
    low_counts = [low_final_counts.get(hop, 0) for hop in all_hops]

    ax_count.bar(x - width/2, high_counts, width, label='Degree High', color='red', alpha=0.7)
    ax_count.bar(x + width/2, low_counts, width, label='Degree Low', color='blue', alpha=0.7)

    ax_count.set_xlabel('Hop Distance', fontsize=12)
    ax_count.set_ylabel('Avg Sample Count', fontsize=12)
    ax_count.set_title('Sample Distribution by Hop Distance (Multi-Trial Average)',
                      fontsize=14, fontweight='bold')
    ax_count.set_xticks(x)
    ax_count.set_xticklabels([f'Hop {h}' for h in all_hops])
    ax_count.legend(fontsize=11)
    ax_count.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    high_total = sum(high_counts)
    low_total = sum(low_counts)

    for i, (h_count, l_count) in enumerate(zip(high_counts, low_counts)):
        if h_count > 0:
            pct = 100 * h_count / high_total
            ax_count.text(i - width/2, h_count, f'{pct:.1f}%',
                         ha='center', va='bottom', fontsize=9)
        if l_count > 0:
            pct = 100 * l_count / low_total
            ax_count.text(i + width/2, l_count, f'{pct:.1f}%',
                         ha='center', va='bottom', fontsize=9)

    # Plot each hop
    for idx, hop in enumerate(all_hops):
        ax = axes[idx + 1, 0]

        if hop in high_aggregated:
            mean_df, ci_lower_df, ci_upper_df = high_aggregated[hop]
            ax.plot(mean_df['step'], mean_df['accuracy'], label='Degree High',
                   color='red', linewidth=2.5, marker='o', markersize=3)
            ax.fill_between(mean_df['step'], ci_lower_df['accuracy'], ci_upper_df['accuracy'],
                           color='red', alpha=0.2)

        if hop in low_aggregated:
            mean_df, ci_lower_df, ci_upper_df = low_aggregated[hop]
            ax.plot(mean_df['step'], mean_df['accuracy'], label='Degree Low',
                   color='blue', linewidth=2.5, marker='o', markersize=3)
            ax.fill_between(mean_df['step'], ci_lower_df['accuracy'], ci_upper_df['accuracy'],
                           color='blue', alpha=0.2)

        ax.set_xlabel('Edit Step', fontsize=12)
        ax.set_ylabel('Retain Accuracy', fontsize=12)
        ax.set_title(f'Hop {hop} Retain Accuracy (Mean ± 95% CI)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add sample counts
        legend_text = []
        if hop in high_aggregated:
            count = high_final_counts[hop]
            legend_text.append(f'High: n≈{count:.0f}')
        if hop in low_aggregated:
            count = low_final_counts[hop]
            legend_text.append(f'Low: n≈{count:.0f}')

        ax.text(0.02, 0.02, '\n'.join(legend_text),
               transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))

        # Difference plot
        ax_diff = axes[idx + 1, 1]

        if hop in high_aggregated and hop in low_aggregated:
            high_mean_df, _, _ = high_aggregated[hop]
            low_mean_df, _, _ = low_aggregated[hop]

            merged = pd.merge(high_mean_df, low_mean_df, on='step', suffixes=('_high', '_low'))
            merged['diff'] = merged['accuracy_high'] - merged['accuracy_low']

            ax_diff.plot(merged['step'], merged['diff'],
                        color='purple', linewidth=2.5, marker='o', markersize=3)
            ax_diff.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_diff.fill_between(merged['step'], 0, merged['diff'],
                                where=(merged['diff'] > 0), color='red', alpha=0.3,
                                label='High > Low')
            ax_diff.fill_between(merged['step'], 0, merged['diff'],
                                where=(merged['diff'] <= 0), color='blue', alpha=0.3,
                                label='Low > High')

            ax_diff.set_xlabel('Edit Step', fontsize=12)
            ax_diff.set_ylabel('Accuracy Difference (High - Low)', fontsize=12)
            ax_diff.set_title(f'Hop {hop} Accuracy Difference (Multi-Trial Mean)',
                            fontsize=14, fontweight='bold')
            ax_diff.legend(fontsize=11)
            ax_diff.grid(True, alpha=0.3)

            # Add statistics
            mean_diff = merged['diff'].mean()
            final_diff = merged['diff'].iloc[-1]
            ax_diff.text(0.02, 0.98,
                        f'Mean diff: {mean_diff:+.4f}\nFinal diff: {final_diff:+.4f}',
                        transform=ax_diff.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.5))
        else:
            ax_diff.text(0.5, 0.5, 'N/A\n(data missing)',
                        ha='center', va='center', fontsize=12, transform=ax_diff.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-trial hop comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retain accuracy by hop distance"
    )

    # Mode 1: Single trial
    parser.add_argument(
        "--high-dir",
        type=str,
        help="Path to degree_high trial directory"
    )
    parser.add_argument(
        "--low-dir",
        type=str,
        help="Path to degree_low trial directory"
    )

    # Mode 2: Multi-trial
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Base directory containing degree_high and degree_low subdirectories"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to analyze (for multi-trial mode)"
    )

    # Common
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single trial mode
    if args.high_dir and args.low_dir:
        print("=" * 80)
        print("Single Trial Hop Distance Analysis")
        print("=" * 80)

        high_dir = Path(args.high_dir)
        low_dir = Path(args.low_dir)

        print(f"\nLoading degree_high data from {high_dir}")
        high_by_step_hop = load_hop_accuracy_data(high_dir)
        high_hop_dfs = compute_hop_accuracy_series(high_by_step_hop)

        print(f"Loading degree_low data from {low_dir}")
        low_by_step_hop = load_hop_accuracy_data(low_dir)
        low_hop_dfs = compute_hop_accuracy_series(low_by_step_hop)

        print("\nComputing detailed statistics...")
        detailed_stats = compute_detailed_statistics(high_dir, low_dir)

        print("\nGenerating plots...")

        # Main hop comparison
        output_path = output_dir / "hop_distance_analysis_single_trial.png"
        plot_hop_comparison_single_trial(high_hop_dfs, low_hop_dfs, output_path)

        # Detailed statistics
        stats_path = output_dir / "hop_distance_detailed_stats_single_trial.png"
        plot_detailed_statistics(detailed_stats, stats_path)

        print("\n✓ Single trial analysis complete!")

    # Multi-trial mode
    elif args.base_dir:
        print("=" * 80)
        print("Multi-Trial Hop Distance Analysis")
        print("=" * 80)

        base_dir = Path(args.base_dir)

        print(f"\nLoading degree_high data ({args.num_trials} trials)")
        high_trial_hop_dfs = load_multi_trial_hop_data(base_dir, "degree_high", args.num_trials)
        print(f"  Loaded {len(high_trial_hop_dfs)} trials")

        print(f"\nLoading degree_low data ({args.num_trials} trials)")
        low_trial_hop_dfs = load_multi_trial_hop_data(base_dir, "degree_low", args.num_trials)
        print(f"  Loaded {len(low_trial_hop_dfs)} trials")

        print("\nAggregating data across trials...")
        high_aggregated = aggregate_multi_trial_hop_data(high_trial_hop_dfs)
        low_aggregated = aggregate_multi_trial_hop_data(low_trial_hop_dfs)

        print("\nGenerating plots...")

        # Main overview
        output_path = output_dir / "hop_distance_analysis_multi_trial.png"
        plot_hop_comparison_multi_trial(high_aggregated, low_aggregated, output_path)

        # Detailed statistics (using trial 0 as representative)
        high_dir = base_dir / "degree_high" / "trial_0"
        low_dir = base_dir / "degree_low" / "trial_0"
        if high_dir.exists() and low_dir.exists():
            detailed_stats = compute_detailed_statistics(high_dir, low_dir)
            stats_path = output_dir / "hop_distance_detailed_stats.png"
            plot_detailed_statistics(detailed_stats, stats_path)

        # Generate hop-specific detailed plots
        all_hops = sorted(set(list(high_aggregated.keys()) + list(low_aggregated.keys())))
        print(f"\nGenerating hop-specific multi-trial comparisons for {len(all_hops)} hops...")

        for hop in all_hops:
            # Linear scale
            hop_path = output_dir / f"hop_{hop}_multi_trial_comparison.png"
            plot_hop_specific_multi_trial(high_aggregated, low_aggregated, hop, hop_path, xscale='linear')

            # Log scale
            hop_path_log = output_dir / f"hop_{hop}_multi_trial_comparison_log.png"
            plot_hop_specific_multi_trial(high_aggregated, low_aggregated, hop, hop_path_log, xscale='log')

        print("\n✓ Multi-trial analysis complete!")

    else:
        print("Error: Must specify either --high-dir and --low-dir, or --base-dir")
        return 1

    print(f"\nOutput saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
