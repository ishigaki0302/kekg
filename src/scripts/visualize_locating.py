"""Visualize ROME causal tracing results."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.edit.rome import ROME, EditSpec
from src.utils import load_yaml, Logger


def main():
    logger = Logger(verbose=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    logger.info("=" * 70)
    logger.info("ROME Causal Tracing Visualization")
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

    # Test edit
    subject = "E_000__a0"
    relation = "R_04__a3"
    new_object = "E_002"

    logger.info(f"\nInput: {subject} {relation}")
    logger.info(f"Target: {new_object}\n")

    # Get causal tracing results for visualization
    logger.info("Running causal tracing...")

    target_layer, layer_effects, token_layer_grid = rome.locate_important_layer_with_scores(
        s=subject,
        r=relation,
        o_target=new_object,
        noise_level=3.0
    )

    logger.info(f"\nSelected layer: {target_layer}")
    logger.info(f"\nTop 5 layers:")
    sorted_effects = sorted(layer_effects, key=lambda x: x['effect'], reverse=True)
    for i, le in enumerate(sorted_effects[:5]):
        logger.info(f"  {i+1}. Layer {le['layer']:2d}: effect={le['effect']:+.4f}")

    # Create visualization
    logger.info("\nCreating visualization...")

    # Get token strings
    input_text = f"{subject} {relation}"
    tokens = tokenizer.encode(input_text)
    token_strs = [tokenizer.get_token(t) for t in tokens]

    # Mark subject tokens
    s_tokens = tokenizer.encode(subject)
    for i in range(len(s_tokens)):
        token_strs[i] = token_strs[i] + "*"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full Token×Layer Heatmap (like ROME paper)
    im1 = ax1.imshow(token_layer_grid, cmap='RdYlGn', aspect='auto',
                     vmin=-np.abs(token_layer_grid).max(),
                     vmax=np.abs(token_layer_grid).max())
    ax1.set_yticks(range(len(token_strs)))
    ax1.set_yticklabels(token_strs, fontsize=10)
    ax1.set_xticks(range(0, token_layer_grid.shape[1], 2))
    ax1.set_xticklabels(range(0, token_layer_grid.shape[1], 2))
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Token Position (* = subject)', fontsize=12)
    ax1.set_title('Causal Tracing: Impact of Restoring Each State', fontsize=14)
    ax1.invert_yaxis()

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Indirect Effect', rotation=270, labelpad=20)

    # Highlight selected layer with a vertical line
    ax1.axvline(x=target_layer, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Plot 2: Layer effects for subject_last token (aggregated view)
    layers = [le['layer'] for le in layer_effects]
    effects = [le['effect'] for le in layer_effects]

    colors = ['red' if l == target_layer else 'steelblue' for l in layers]
    ax2.bar(layers, effects, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Causal Effect', fontsize=12)
    ax2.set_title(f'Subject Last Token Effects\n(Red = Selected Layer {target_layer})', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = "outputs/causal_tracing_visualization.png"
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nVisualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
