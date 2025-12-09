#!/usr/bin/env python3
"""
Plot summary statistics from ROME locating results
Visualizes average best layer and 95% confidence intervals
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_results(json_path):
    """Load locating results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_statistics(data):
    """Compute mean and 95% confidence interval for best layers and effects"""
    best_layers = [item['best_layer'] for item in data]
    best_effects = [item['best_effect'] for item in data]

    # Compute statistics for layers
    layer_mean = np.mean(best_layers)
    layer_sem = stats.sem(best_layers)
    layer_ci = stats.t.interval(0.95, len(best_layers)-1, loc=layer_mean, scale=layer_sem)

    # Compute statistics for effects
    effect_mean = np.mean(best_effects)
    effect_sem = stats.sem(best_effects)
    effect_ci = stats.t.interval(0.95, len(best_effects)-1, loc=effect_mean, scale=effect_sem)

    return {
        'layers': {
            'mean': layer_mean,
            'ci': layer_ci,
            'std': np.std(best_layers, ddof=1),
            'values': best_layers
        },
        'effects': {
            'mean': effect_mean,
            'ci': effect_ci,
            'std': np.std(best_effects, ddof=1),
            'values': best_effects
        }
    }

def plot_summary(stats_dict, output_path='outputs/locating_results/summary_plot.png'):
    """Create summary plots with mean and 95% CI"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Best Layer Distribution with Mean and CI
    layers = stats_dict['layers']['values']
    layer_mean = stats_dict['layers']['mean']
    layer_ci = stats_dict['layers']['ci']

    ax1.hist(layers, bins=12, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(layer_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {layer_mean:.2f}')
    ax1.axvspan(layer_ci[0], layer_ci[1], alpha=0.2, color='red', label=f'95% CI: [{layer_ci[0]:.2f}, {layer_ci[1]:.2f}]')
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Best Layers\n(Knowledge Localization)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Best Effect Distribution with Mean and CI
    effects = stats_dict['effects']['values']
    effect_mean = stats_dict['effects']['mean']
    effect_ci = stats_dict['effects']['ci']

    ax2.hist(effects, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(effect_mean, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {effect_mean:.4f}')
    ax2.axvspan(effect_ci[0], effect_ci[1], alpha=0.2, color='darkred', label=f'95% CI: [{effect_ci[0]:.4f}, {effect_ci[1]:.4f}]')
    ax2.set_xlabel('Causal Effect', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Best Effects\n(Indirect Effect Magnitude)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to: {output_path}")

    return fig

def plot_layer_vs_effect(stats_dict, output_path='outputs/locating_results/layer_effect_scatter.png'):
    """Create scatter plot of layer vs effect"""
    layers = stats_dict['layers']['values']
    effects = stats_dict['effects']['values']

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(layers, effects, alpha=0.6, s=100, c=layers, cmap='viridis', edgecolor='black', linewidth=0.5)

    # Add mean lines
    layer_mean = stats_dict['layers']['mean']
    effect_mean = stats_dict['effects']['mean']
    ax.axvline(layer_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean Layer: {layer_mean:.2f}')
    ax.axhline(effect_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean Effect: {effect_mean:.4f}')

    ax.set_xlabel('Best Layer', fontsize=12)
    ax.set_ylabel('Best Effect (Causal Impact)', fontsize=12)
    ax.set_title('Knowledge Localization: Layer vs. Causal Effect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Layer Number', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")

    return fig

def print_statistics(stats_dict, n_samples):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("ROME LOCATING - SUMMARY STATISTICS")
    print("="*70)
    print(f"Number of samples: {n_samples}")
    print()

    print("Best Layer Statistics:")
    print(f"  Mean: {stats_dict['layers']['mean']:.2f}")
    print(f"  Standard Deviation: {stats_dict['layers']['std']:.2f}")
    print(f"  95% Confidence Interval: [{stats_dict['layers']['ci'][0]:.2f}, {stats_dict['layers']['ci'][1]:.2f}]")
    print()

    print("Best Effect Statistics:")
    print(f"  Mean: {stats_dict['effects']['mean']:.6f}")
    print(f"  Standard Deviation: {stats_dict['effects']['std']:.6f}")
    print(f"  95% Confidence Interval: [{stats_dict['effects']['ci'][0]:.6f}, {stats_dict['effects']['ci'][1]:.6f}]")
    print(f"  Min: {min(stats_dict['effects']['values']):.6f}")
    print(f"  Max: {max(stats_dict['effects']['values']):.6f}")
    print("="*70)
    print()

def main():
    # Load results
    json_path = 'outputs/locating_results/locating_summary.json'
    print(f"Loading results from: {json_path}")
    data = load_results(json_path)

    # Compute statistics
    stats_dict = compute_statistics(data)

    # Print statistics
    print_statistics(stats_dict, len(data))

    # Create plots
    print("Generating plots...")
    plot_summary(stats_dict)
    plot_layer_vs_effect(stats_dict)

    print("\nDone!")

if __name__ == '__main__':
    main()
