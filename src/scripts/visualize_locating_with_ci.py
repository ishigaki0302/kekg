"""Visualize ROME causal tracing results with 95% confidence intervals from 500 random samples."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.edit.rome import ROME, EditSpec
from src.utils import load_yaml, Logger


def main():
    logger = Logger(verbose=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    logger.info("=" * 70)
    logger.info("ROME Causal Tracing Visualization with 95% CI (500 samples)")
    logger.info("=" * 70)

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

    # Initialize ROME
    kg_corpus_path = "data/kg/ba/corpus.train.txt"

    rome = ROME(
        model,
        tokenizer,
        device=device,
        kg_corpus_path=kg_corpus_path,
        mom2_n_samples=3000,
        use_mom2_adjustment=True
    )

    # Load training data
    logger.info(f"\nLoading training triples from: {kg_corpus_path}")
    triples = []
    with open(kg_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append(tuple(parts))

    logger.info(f"Total triples: {len(triples)}")

    # Random sample 500 triples
    n_samples = min(500, len(triples))
    sampled_triples = random.sample(triples, n_samples)
    logger.info(f"Sampled {n_samples} triples for causal tracing\n")

    # Collect causal tracing results for all samples
    n_layers = model_cfg["n_layers"]
    all_token_layer_grids = []  # List of grids, each with shape [n_tokens, n_layers]
    max_tokens = 0

    logger.info("Running causal tracing on 500 samples...")
    for i, (subject, relation, obj) in enumerate(tqdm(sampled_triples, desc="Causal tracing")):
        try:
            # Run causal tracing
            token_layer_grid = rome.compute_full_causal_tracing(
                s=subject,
                r=relation,
                o_target=obj,
                noise_level=3.0
            )
            all_token_layer_grids.append(token_layer_grid)
            max_tokens = max(max_tokens, token_layer_grid.shape[0])
        except Exception as e:
            # Skip failed samples
            continue

    logger.info(f"\nSuccessfully traced {len(all_token_layer_grids)} samples")
    logger.info(f"Max token length: {max_tokens}")

    # Pad all grids to same size
    padded_grids = []
    for grid in all_token_layer_grids:
        n_tokens = grid.shape[0]
        if n_tokens < max_tokens:
            # Pad with zeros
            padded = np.zeros((max_tokens, n_layers))
            padded[:n_tokens, :] = grid
            padded_grids.append(padded)
        else:
            padded_grids.append(grid)

    # Stack all grids: [n_samples, n_tokens, n_layers]
    all_grids = np.stack(padded_grids, axis=0)
    logger.info(f"Stacked grids shape: {all_grids.shape} (samples × tokens × layers)")

    # Compute statistics across samples
    mean_grid = np.mean(all_grids, axis=0)  # [n_tokens, n_layers]
    std_grid = np.std(all_grids, axis=0)    # [n_tokens, n_layers]

    # 95% confidence interval (±1.96 * std)
    ci_95 = 1.96 * std_grid / np.sqrt(len(all_grids))

    logger.info(f"\nMean grid shape: {mean_grid.shape}")
    logger.info(f"Std grid shape: {std_grid.shape}")
    logger.info(f"95% CI grid shape: {ci_95.shape}")

    # Create visualization
    logger.info("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean Indirect Effect (Token×Layer Heatmap)
    ax1 = axes[0, 0]
    vmax = np.abs(mean_grid).max()
    im1 = ax1.imshow(mean_grid, cmap='RdYlGn', aspect='auto',
                     vmin=-vmax, vmax=vmax)
    ax1.set_yticks(range(max_tokens))
    ax1.set_yticklabels([f'Token {i}' for i in range(max_tokens)], fontsize=9)
    ax1.set_xticks(range(0, n_layers, 2))
    ax1.set_xticklabels(range(0, n_layers, 2))
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Token Position', fontsize=12)
    ax1.set_title(f'Mean Indirect Effect (n={len(all_grids)})', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Mean Indirect Effect', rotation=270, labelpad=20)

    # Plot 2: 95% Confidence Interval Width (Token×Layer Heatmap)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(ci_95, cmap='YlOrRd', aspect='auto', vmin=0)
    ax2.set_yticks(range(max_tokens))
    ax2.set_yticklabels([f'Token {i}' for i in range(max_tokens)], fontsize=9)
    ax2.set_xticks(range(0, n_layers, 2))
    ax2.set_xticklabels(range(0, n_layers, 2))
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Token Position', fontsize=12)
    ax2.set_title('95% Confidence Interval Width', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('±95% CI', rotation=270, labelpad=20)

    # Plot 3: Layer-wise aggregation with error bars (averaging over all token positions)
    ax3 = axes[1, 0]
    # Average over token dimension
    layer_means = np.mean(mean_grid, axis=0)  # [n_layers]
    layer_ci = np.mean(ci_95, axis=0)         # [n_layers]

    layers = np.arange(n_layers)
    ax3.bar(layers, layer_means, color='steelblue', alpha=0.7)
    ax3.errorbar(layers, layer_means, yerr=layer_ci, fmt='none',
                 ecolor='black', capsize=5, capthick=2, linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('Mean Indirect Effect', fontsize=12)
    ax3.set_title('Layer-wise Average (with 95% CI)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_xticks(layers)

    # Highlight best layer
    best_layer = np.argmax(layer_means)
    ax3.bar(best_layer, layer_means[best_layer], color='red', alpha=0.7)

    # Plot 4: Token position-wise aggregation with error bars (averaging over all layers)
    ax4 = axes[1, 1]
    # Average over layer dimension
    token_means = np.mean(mean_grid, axis=1)  # [n_tokens]
    token_ci = np.mean(ci_95, axis=1)         # [n_tokens]

    token_positions = np.arange(max_tokens)
    ax4.bar(token_positions, token_means, color='forestgreen', alpha=0.7)
    ax4.errorbar(token_positions, token_means, yerr=token_ci, fmt='none',
                 ecolor='black', capsize=5, capthick=2, linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Token Position', fontsize=12)
    ax4.set_ylabel('Mean Indirect Effect', fontsize=12)
    ax4.set_title('Token-wise Average (with 95% CI)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticks(token_positions)

    plt.tight_layout()

    output_path = "outputs/causal_tracing_with_ci.png"
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nVisualization saved to: {output_path}")

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("Summary Statistics")
    logger.info("=" * 70)
    logger.info(f"Number of samples: {len(all_grids)}")
    logger.info(f"Max token length: {max_tokens}")
    logger.info(f"Number of layers: {n_layers}")
    logger.info(f"\nBest layer (highest mean effect): {best_layer}")
    logger.info(f"  Mean effect: {layer_means[best_layer]:.6f}")
    logger.info(f"  95% CI: ±{layer_ci[best_layer]:.6f}")
    logger.info(f"\nTop 5 layers by mean effect:")
    sorted_layers = np.argsort(layer_means)[::-1]
    for i, layer_idx in enumerate(sorted_layers[:5]):
        logger.info(f"  {i+1}. Layer {layer_idx:2d}: {layer_means[layer_idx]:+.6f} ± {layer_ci[layer_idx]:.6f}")
    logger.info("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
