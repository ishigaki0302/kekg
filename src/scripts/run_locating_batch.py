"""Run causal tracing (Locating) on random samples from training data."""

import sys
import json
import random
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.edit.causal_tracing import CausalTracer
from src.utils import load_yaml, Logger


def plot_aggregate_statistics(results_summary, output_dir):
    """Plot aggregate statistics similar to ROME paper."""
    successful = [r for r in results_summary if 'error' not in r]
    if not successful:
        return

    # Prepare data
    best_layers = [r['best_layer'] for r in successful]
    best_effects = [r['best_effect'] for r in successful]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Layer distribution histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(best_layers, bins=range(min(best_layers), max(best_layers)+2),
             color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Best Layers', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Effect magnitude by layer
    ax2 = plt.subplot(2, 3, 2)
    layers_array = np.array(best_layers)
    effects_array = np.array(best_effects)
    scatter = ax2.scatter(layers_array, effects_array, alpha=0.6,
                         c=effects_array, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Effect Magnitude', fontsize=12)
    ax2.set_title('Effect Magnitude vs Layer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Effect')

    # 3. Average effect per layer
    ax3 = plt.subplot(2, 3, 3)
    layer_to_effects = {}
    for layer, effect in zip(best_layers, best_effects):
        if layer not in layer_to_effects:
            layer_to_effects[layer] = []
        layer_to_effects[layer].append(effect)

    avg_layers = sorted(layer_to_effects.keys())
    avg_effects = [np.mean(layer_to_effects[l]) for l in avg_layers]
    std_effects = [np.std(layer_to_effects[l]) for l in avg_layers]

    ax3.errorbar(avg_layers, avg_effects, yerr=std_effects,
                marker='o', markersize=8, linewidth=2, capsize=5,
                color='darkgreen', ecolor='lightgreen', alpha=0.8)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Average Effect', fontsize=12)
    ax3.set_title('Average Effect per Layer', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. High score vs low score scatter
    ax4 = plt.subplot(2, 3, 4)
    high_scores = [r['high_score'] for r in successful]
    low_scores = [r['low_score'] for r in successful]
    ax4.scatter(low_scores, high_scores, alpha=0.6, s=100,
               edgecolors='black', linewidth=0.5, color='coral')
    ax4.plot([min(low_scores), max(high_scores)],
            [min(low_scores), max(high_scores)],
            'k--', alpha=0.3, label='y=x')
    ax4.set_xlabel('Low Score (corrupted)', fontsize=12)
    ax4.set_ylabel('High Score (clean)', fontsize=12)
    ax4.set_title('Clean vs Corrupted Scores', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Effect cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_effects = np.sort(best_effects)
    cumulative = np.arange(1, len(sorted_effects) + 1) / len(sorted_effects)
    ax5.plot(sorted_effects, cumulative, linewidth=2.5, color='purple')
    ax5.set_xlabel('Effect Magnitude', fontsize=12)
    ax5.set_ylabel('Cumulative Probability', fontsize=12)
    ax5.set_title('Cumulative Distribution of Effects', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax5.axvline(np.median(sorted_effects), color='red', linestyle='--', alpha=0.5)
    ax5.legend()

    # 6. Statistics summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Total Samples: {len(results_summary)}
    Successful: {len(successful)}

    Best Layer:
      Mean: {np.mean(best_layers):.2f}
      Std:  {np.std(best_layers):.2f}
      Mode: {max(set(best_layers), key=best_layers.count)}

    Effect Magnitude:
      Mean:   {np.mean(best_effects):.4f}
      Median: {np.median(best_effects):.4f}
      Std:    {np.std(best_effects):.4f}
      Max:    {np.max(best_effects):.4f}
      Min:    {np.min(best_effects):.4f}

    Score Statistics:
      Avg High Score: {np.mean(high_scores):.6f}
      Avg Low Score:  {np.mean(low_scores):.6f}
    """

    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Causal Tracing Aggregate Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Create a second figure for heatmap of all traces
    if len(successful) > 0:
        create_average_heatmap(successful, output_dir)


def create_average_heatmap(successful_results, output_dir):
    """Create average heatmap across all successful traces."""
    # This would require storing full score matrices, which we'll add as an enhancement
    # For now, create a layer importance visualization

    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect best layer counts
    layer_counts = {}
    for r in successful_results:
        layer = r['best_layer']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    layers = sorted(layer_counts.keys())
    counts = [layer_counts[l] for l in layers]

    # Create bar chart
    bars = ax.bar(layers, counts, color='steelblue', edgecolor='black', alpha=0.7)

    # Color bars by importance
    max_count = max(counts)
    for bar, count in zip(bars, counts):
        bar.set_color(plt.cm.Purples(count / max_count))

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Layer Importance Distribution (Best Layer Frequency)',
                fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_importance.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_t0_aie_by_layer(all_results_by_kind, output_dir):
    """Plot T0 (subject entity) AIE across layers for all kinds.

    Args:
        all_results_by_kind: Dictionary with keys None, 'mlp', 'attn' containing lists of TracingResults
        output_dir: Output directory path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors matching the heatmap colors
    color_map = {
        None: '#9467bd',      # Purple
        'mlp': '#2ca02c',     # Green
        'attn': '#d62728'     # Red
    }

    kind_labels = {
        None: 'All States',
        'mlp': 'MLP',
        'attn': 'Attention'
    }

    # Plot 1: All three on same plot
    ax1 = axes[0, 0]
    for kind in [None, 'mlp', 'attn']:
        results = all_results_by_kind.get(kind, [])
        if not results:
            continue

        # Extract T0 scores across all samples
        t0_scores_per_sample = []
        for result in results:
            # T0 is the first token (index 0)
            t0_score = result.scores[0, :].numpy()  # [num_layers]
            t0_scores_per_sample.append(t0_score)

        # Average across samples
        avg_t0_scores = np.mean(t0_scores_per_sample, axis=0)  # [num_layers]
        std_t0_scores = np.std(t0_scores_per_sample, axis=0)

        layers = np.arange(len(avg_t0_scores))

        # Plot line with error band
        ax1.plot(layers, avg_t0_scores,
                linewidth=2.5, marker='o', markersize=6,
                color=color_map[kind], label=kind_labels[kind], alpha=0.8)
        ax1.fill_between(layers,
                         avg_t0_scores - std_t0_scores,
                         avg_t0_scores + std_t0_scores,
                         color=color_map[kind], alpha=0.2)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Indirect Effect (AIE)', fontsize=12, fontweight='bold')
    ax1.set_title('T0 (Subject Entity) AIE by Layer - All Components', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    # Plot 2: All States only
    ax2 = axes[0, 1]
    results = all_results_by_kind.get(None, [])
    if results:
        t0_scores_per_sample = []
        for result in results:
            t0_score = result.scores[0, :].numpy()
            t0_scores_per_sample.append(t0_score)

        avg_t0_scores = np.mean(t0_scores_per_sample, axis=0)
        std_t0_scores = np.std(t0_scores_per_sample, axis=0)
        layers = np.arange(len(avg_t0_scores))

        ax2.plot(layers, avg_t0_scores,
                linewidth=3, marker='o', markersize=8,
                color=color_map[None], alpha=0.8)
        ax2.fill_between(layers,
                         avg_t0_scores - std_t0_scores,
                         avg_t0_scores + std_t0_scores,
                         color=color_map[None], alpha=0.3)

        ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Indirect Effect', fontsize=12, fontweight='bold')
        ax2.set_title('All States', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # Plot 3: MLP only
    ax3 = axes[1, 0]
    results = all_results_by_kind.get('mlp', [])
    if results:
        t0_scores_per_sample = []
        for result in results:
            t0_score = result.scores[0, :].numpy()
            t0_scores_per_sample.append(t0_score)

        avg_t0_scores = np.mean(t0_scores_per_sample, axis=0)
        std_t0_scores = np.std(t0_scores_per_sample, axis=0)
        layers = np.arange(len(avg_t0_scores))

        ax3.plot(layers, avg_t0_scores,
                linewidth=3, marker='s', markersize=8,
                color=color_map['mlp'], alpha=0.8)
        ax3.fill_between(layers,
                         avg_t0_scores - std_t0_scores,
                         avg_t0_scores + std_t0_scores,
                         color=color_map['mlp'], alpha=0.3)

        ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Indirect Effect', fontsize=12, fontweight='bold')
        ax3.set_title('MLP Only', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Attention only
    ax4 = axes[1, 1]
    results = all_results_by_kind.get('attn', [])
    if results:
        t0_scores_per_sample = []
        for result in results:
            t0_score = result.scores[0, :].numpy()
            t0_scores_per_sample.append(t0_score)

        avg_t0_scores = np.mean(t0_scores_per_sample, axis=0)
        std_t0_scores = np.std(t0_scores_per_sample, axis=0)
        layers = np.arange(len(avg_t0_scores))

        ax4.plot(layers, avg_t0_scores,
                linewidth=3, marker='^', markersize=8,
                color=color_map['attn'], alpha=0.8)
        ax4.fill_between(layers,
                         avg_t0_scores - std_t0_scores,
                         avg_t0_scores + std_t0_scores,
                         color=color_map['attn'], alpha=0.3)

        ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Indirect Effect', fontsize=12, fontweight='bold')
        ax4.set_title('Attention Only', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # Overall title
    n_samples = len(all_results_by_kind.get(None, []))
    fig.suptitle(f'T0 (Subject Entity Token) Average Indirect Effect by Layer (n={n_samples})',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()
    save_path = output_dir / 't0_aie_by_layer.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return save_path


def plot_average_heatmap(all_results, output_dir, kind_suffix=""):
    """Plot average heatmap across all samples."""
    if not all_results:
        return

    # Collect all score matrices
    all_scores = []
    for result in all_results:
        all_scores.append(result.scores.numpy())

    # Convert to numpy array and compute mean
    scores_array = np.array(all_scores)  # [num_samples, num_tokens, num_layers]
    avg_scores = np.mean(scores_array, axis=0)  # [num_tokens, num_layers]

    # Get average low/high scores
    avg_low = np.mean([r.low_score for r in all_results])
    avg_high = np.mean([r.high_score for r in all_results])

    # Use token labels from first result (they should all be similar length)
    # Pad with generic labels if needed
    max_tokens = avg_scores.shape[0]
    token_strs = [f"T{i}" for i in range(max_tokens)]

    # Get kind from first result
    kind = all_results[0].kind if hasattr(all_results[0], 'kind') else None

    # Use ROME style formatting with different colors for different kinds
    cmap_map = {
        None: 'Purples',
        'mlp': 'Greens',
        'attn': 'Reds'
    }
    cmap = cmap_map.get(kind, 'Purples')

    with plt.rc_context(rc={'font.family': 'sans-serif'}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Plot average heatmap
        h = ax.pcolor(
            avg_scores,
            cmap=cmap,
            vmin=avg_low
        )

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(token_strs))])
        ax.set_xticks([0.5 + i for i in range(0, avg_scores.shape[1])])
        ax.set_xticklabels(list(range(0, avg_scores.shape[1])))
        ax.set_yticklabels(token_strs, fontsize=9)

        # ROME style labels
        kind_name = {'mlp': 'MLP', 'attn': 'Attention'}.get(kind, 'State')
        ax.set_title(f'Average Causal Effect: {kind_name} (n={len(all_results)})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Token Position', fontsize=10)

        # Colorbar
        cb = plt.colorbar(h, ax=ax)
        cb.set_label('Average Effect', rotation=270, labelpad=20)

        plt.tight_layout()
        save_path = output_dir / f'average_heatmap{kind_suffix}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    return save_path


def plot_heatmap(result, save_path, title=None):
    """Plot token x layer heatmap in ROME style."""
    scores = result.scores.numpy()
    token_strs = result.input_tokens.copy()
    low_score = result.low_score
    high_score = result.high_score
    kind = result.kind if hasattr(result, 'kind') else None

    # Mark subject tokens
    for i in range(*result.subject_range):
        token_strs[i] = token_strs[i] + "*"

    # Use ROME style formatting with different colors for different kinds
    cmap_map = {
        None: 'Purples',
        'mlp': 'Greens',
        'attn': 'Reds'
    }
    cmap = cmap_map.get(kind, 'Purples')

    with plt.rc_context(rc={'font.family': 'sans-serif'}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Plot heatmap with appropriate colormap (ROME style)
        h = ax.pcolor(
            scores,
            cmap=cmap,
            vmin=low_score
        )

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(token_strs))])
        ax.set_xticks([0.5 + i for i in range(0, scores.shape[1])])
        ax.set_xticklabels(list(range(0, scores.shape[1])))
        ax.set_yticklabels(token_strs, fontsize=9)

        # ROME style labels with component-specific text
        if kind == 'mlp':
            ax.set_title('Impact of restoring MLP after corrupted input', fontsize=11)
            ax.set_xlabel('Single restored MLP layer', fontsize=10)
        elif kind == 'attn':
            ax.set_title('Impact of restoring Attention after corrupted input', fontsize=11)
            ax.set_xlabel('Single restored Attention layer', fontsize=10)
        else:
            ax.set_title('Impact of restoring state after corrupted input', fontsize=11)
            ax.set_xlabel('Single restored layer', fontsize=10)

        # Colorbar
        cb = plt.colorbar(h, ax=ax)

        # Show prediction info
        answer = result.answer.strip() if hasattr(result, 'answer') else ''
        if answer:
            cb.ax.set_title(f'p({answer})', y=-0.16, fontsize=9)

        # Add custom title if provided
        if title:
            fig.suptitle(title, fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Locating on random samples")
    parser.add_argument("--model-dir", type=str, default="outputs/models/gpt_small",
                        help="Path to trained model directory")
    parser.add_argument("--kg-file", type=str, default="data/kg/ba/graph.jsonl",
                        help="Path to knowledge graph file")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of random samples to trace")
    parser.add_argument("--output-dir", type=str, default="outputs/locating_results",
                        help="Output directory for results")
    parser.add_argument("--noise-level", type=float, default=3.0,
                        help="Noise level for causal tracing")
    parser.add_argument("--num-noise-samples", type=int, default=10,
                        help="Number of noise samples for tracing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = Logger(verbose=True)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ROME Causal Tracing - Batch Locating")
    logger.info("=" * 70)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"KG file: {args.kg_file}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Output directory: {output_dir}")

    # Load model
    logger.info("\nLoading model...")
    tokenizer = SROTokenizer.load(model_dir / "tokenizer.json")
    train_report = load_yaml(model_dir / "train_report.yaml")
    model_cfg = train_report["config"]["model"]

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        d_mlp=model_cfg["d_mlp"],
        max_seq_len=model_cfg.get("max_seq_len", 8),
        dropout=0.0
    )

    model = GPTMini(gpt_config)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")

    # Initialize causal tracer
    tracer = CausalTracer(model, tokenizer, device)

    # Load knowledge graph
    logger.info(f"\nLoading knowledge graph from {args.kg_file}...")
    with open(args.kg_file, 'r') as f:
        kg_data = [json.loads(line) for line in f]

    # Sample random triples
    if len(kg_data) > args.num_samples:
        sampled_data = random.sample(kg_data, args.num_samples)
    else:
        sampled_data = kg_data
        logger.info(f"Warning: Only {len(kg_data)} samples available")

    logger.info(f"Sampled {len(sampled_data)} triples")

    # Run causal tracing on each sample
    results_summary = []

    # Store all results for averaging (organized by kind)
    all_results_by_kind = {
        None: [],  # All states
        'mlp': [],
        'attn': []
    }

    for idx, triple in enumerate(sampled_data):
        s, r, o = triple['s'], triple['r'], triple['o']

        logger.info(f"\n[{idx+1}/{len(sampled_data)}] Processing: {s} {r} -> {o}")

        try:
            # Run causal tracing for each kind (all, mlp, attn)
            for kind in [None, 'mlp', 'attn']:
                kind_str = kind if kind else "all"
                logger.info(f"  Tracing {kind_str}...")

                result = tracer.trace_important_states(
                    s=s,
                    r=r,
                    o_target=o,
                    noise_level=args.noise_level,
                    num_samples=args.num_noise_samples,
                    kind=kind
                )

                # Store for averaging
                all_results_by_kind[kind].append(result)

                # Find best layer
                subject_last_idx = result.subject_range[1] - 1
                layer_scores = result.scores[subject_last_idx, :]
                best_layer = torch.argmax(layer_scores).item()
                best_effect = layer_scores[best_layer].item()

                # Save visualization
                kind_suffix = f"_{kind}" if kind else ""
                plot_path = output_dir / f"trace_{idx:03d}_{s}_{r}{kind_suffix}.png"
                plot_heatmap(
                    result,
                    plot_path,
                    title=f"Sample {idx+1} ({kind_str}): {s} {r} → {o}"
                )

                # Store summary (only for 'all' kind to avoid duplication)
                if kind is None:
                    logger.info(f"  Best layer: {best_layer} (effect: {best_effect:.4f})")
                    logger.info(f"  Predicted: {result.answer.strip()} | Target: {o}")
                    logger.info(f"  Correct: {result.correct_prediction}")

                    results_summary.append({
                        'index': idx,
                        's': s,
                        'r': r,
                        'o_target': o,
                        'predicted': result.answer.strip(),
                        'correct': result.correct_prediction,
                        'best_layer': best_layer,
                        'best_effect': best_effect,
                        'high_score': result.high_score,
                        'low_score': result.low_score
                    })

        except Exception as e:
            logger.info(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'index': idx,
                's': s,
                'r': r,
                'o_target': o,
                'error': str(e)
            })

    # Save summary
    summary_path = output_dir / "locating_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("Summary Statistics:")
    logger.info(f"{'=' * 70}")

    successful = [r for r in results_summary if 'error' not in r]
    correct = [r for r in successful if r['correct']]

    logger.info(f"Total samples: {len(results_summary)}")
    logger.info(f"Successful: {len(successful)}")

    if successful:
        logger.info(f"Correct predictions: {len(correct)} ({len(correct)/len(successful)*100:.1f}%)")
    else:
        logger.info("No successful traces - all samples encountered errors")

    if successful:
        best_layers = [r['best_layer'] for r in successful]
        logger.info(f"Average best layer: {np.mean(best_layers):.1f} ± {np.std(best_layers):.1f}")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_path}")

    # Generate aggregate visualizations
    if successful:
        logger.info("\nGenerating aggregate analysis plots...")
        plot_aggregate_statistics(results_summary, output_dir)
        logger.info(f"Aggregate plots saved to: {output_dir}/aggregate_analysis.png")
        logger.info(f"Layer importance plot saved to: {output_dir}/layer_importance.png")

        # Generate average heatmaps for each kind
        logger.info("\nGenerating average heatmaps...")
        for kind in [None, 'mlp', 'attn']:
            if all_results_by_kind[kind]:
                kind_str = kind if kind else "all"
                kind_suffix = f"_{kind}" if kind else ""
                avg_path = plot_average_heatmap(
                    all_results_by_kind[kind],
                    output_dir,
                    kind_suffix=kind_suffix
                )
                logger.info(f"Average heatmap ({kind_str}) saved to: {avg_path}")

        # Generate T0 AIE by layer plot
        logger.info("\nGenerating T0 AIE by layer plot...")
        t0_path = plot_t0_aie_by_layer(all_results_by_kind, output_dir)
        logger.info(f"T0 AIE plot saved to: {t0_path}")


if __name__ == "__main__":
    main()
