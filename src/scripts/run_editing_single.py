"""Run ROME editing on a single (s,r,o) triple."""

import sys
import json
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.edit.rome import ROME
from src.edit.ripple_analysis import RippleAnalyzer
from src.edit.ripple_visualization import (
    visualize_ripple_on_graph,
    visualize_ripple_statistics,
    visualize_top_affected_triples
)
from src.utils import load_yaml, Logger


def visualize_editing_result(
    rome: ROME,
    s: str,
    r: str,
    o_original: str,
    o_target: str,
    result,
    save_dir: Path
):
    """Create visualizations for the editing result."""

    # 1. Locating heatmap
    logger = Logger(verbose=True)
    logger.info("\nGenerating locating visualization...")

    target_layer, layer_effects, token_layer_grid = rome.locate_important_layer_with_scores(
        s=s, r=r, o_target=o_target
    )

    # Get token strings
    input_text = f"{s} {r}"
    # Use original tokenizer (rome.tokenizer is wrapped)
    tokens = rome.original_tokenizer.encode(input_text)
    token_strs = [rome.original_tokenizer.get_token(t) for t in tokens]

    # Mark subject tokens
    s_tokens = rome.original_tokenizer.encode(s)
    for i in range(len(s_tokens)):
        token_strs[i] = token_strs[i] + "*"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Token×Layer Heatmap
    token_layer_grid_np = token_layer_grid.cpu().numpy() if torch.is_tensor(token_layer_grid) else token_layer_grid
    im1 = ax1.imshow(token_layer_grid_np, cmap='RdYlGn', aspect='auto',
                     vmin=-np.abs(token_layer_grid_np).max(),
                     vmax=np.abs(token_layer_grid_np).max())
    ax1.set_yticks(range(len(token_strs)))
    ax1.set_yticklabels(token_strs, fontsize=10)
    ax1.set_xticks(range(0, token_layer_grid_np.shape[1], 2))
    ax1.set_xticklabels(range(0, token_layer_grid_np.shape[1], 2))
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Token Position (* = subject)', fontsize=12)
    ax1.set_title('Causal Tracing: Impact of Restoring Each State', fontsize=14)
    ax1.invert_yaxis()

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Indirect Effect', rotation=270, labelpad=20)

    # Highlight selected layer
    ax1.axvline(x=target_layer, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Plot 2: Layer effects
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
    plt.savefig(save_dir / "locating_result.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Locating visualization saved to: {save_dir / 'locating_result.png'}")

    # 2. Editing result visualization
    logger.info("Generating editing result visualization...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart comparing before/after
    categories = ['Before Edit', 'After Edit']
    original_vals = [1 if result.original_prediction == o_original else 0,
                     0]
    target_vals = [0,
                   1 if result.new_prediction == o_target else 0]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_vals, width, label=f'Original: {o_original}',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, target_vals, width, label=f'Target: {o_target}',
                   color='green', alpha=0.8)

    ax.set_ylabel('Prediction Match', fontsize=12)
    ax.set_title(f'ROME Editing Result\n{s} {r} → ?', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0, 1.2])

    # Add text annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       '✓', ha='center', va='bottom', fontsize=20, color='darkgreen')

    # Add info text
    info_text = f"Edited Layer: {result.layer}\n"
    info_text += f"Success: {'Yes' if result.success else 'No'}\n"
    info_text += f"Before: {result.original_prediction}\n"
    info_text += f"After: {result.new_prediction}"

    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / "editing_result.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Editing visualization saved to: {save_dir / 'editing_result.png'}")


def main():
    parser = argparse.ArgumentParser(description="Run ROME editing on a single triple")
    parser.add_argument("--model-dir", type=str, default="outputs/models/gpt_small",
                        help="Path to trained model directory")
    parser.add_argument("--kg-corpus", type=str, default="data/kg/ba/corpus.train.txt",
                        help="Path to KG corpus for computing statistics (use trained knowledge graph)")
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject entity")
    parser.add_argument("--relation", type=str, required=True,
                        help="Relation")
    parser.add_argument("--target", type=str, required=True,
                        help="Target object (new fact)")
    parser.add_argument("--original", type=str, default=None,
                        help="Original object (for comparison)")
    parser.add_argument("--output-dir", type=str, default="outputs/editing_results",
                        help="Output directory for results")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to edit (if None, auto-locate)")
    parser.add_argument("--analyze-ripple", action="store_true",
                        help="Analyze ripple effects on entire knowledge graph")
    parser.add_argument("--max-ripple-triples", type=int, default=None,
                        help="Maximum number of triples to analyze for ripple effect (None for all)")

    args = parser.parse_args()

    logger = Logger(verbose=True)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ROME Knowledge Editing - Single Edit")
    logger.info("=" * 70)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Relation: {args.relation}")
    logger.info(f"Target: {args.target}")
    if args.original:
        logger.info(f"Original: {args.original}")
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

    # Initialize ROME
    logger.info("\nInitializing ROME...")
    rome = ROME(
        model,
        tokenizer,
        device=device,
        kg_corpus_path=args.kg_corpus,
        mom2_n_samples=1000,
        use_mom2_adjustment=False,  # Disable for now due to compatibility issues
        v_num_grad_steps=100  # Increase optimization steps
    )

    # Check original prediction if not provided
    if args.original is None:
        logger.info("\nChecking original prediction...")
        input_text = f"{args.subject} {args.relation}"
        input_ids = torch.tensor([tokenizer.encode(input_text)]).to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :]
            pred_id = torch.argmax(logits).item()
            args.original = tokenizer.get_token(pred_id).strip()

        logger.info(f"Original prediction: {args.original}")

    # Apply edit
    logger.info("\nApplying ROME edit...")
    edited_model, result = rome.apply_edit(
        s=args.subject,
        r=args.relation,
        o_target=args.target,
        layer=args.layer,
        copy_model=True
    )

    logger.info("\n" + "=" * 70)
    logger.info("Edit Result:")
    logger.info("=" * 70)
    logger.info(f"Success: {result.success}")
    logger.info(f"Edited Layer: {result.layer}")
    logger.info(f"Original Prediction: {result.original_prediction}")
    logger.info(f"New Prediction: {result.new_prediction}")
    logger.info(f"Target: {args.target}")

    # Save result
    result_data = {
        's': args.subject,
        'r': args.relation,
        'o_original': args.original,
        'o_target': args.target,
        'success': result.success,
        'layer': result.layer,
        'original_prediction': result.original_prediction,
        'new_prediction': result.new_prediction
    }

    result_path = output_dir / "edit_result.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"\nResult saved to: {result_path}")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualize_editing_result(
        rome=rome,
        s=args.subject,
        r=args.relation,
        o_original=args.original,
        o_target=args.target,
        result=result,
        save_dir=output_dir
    )

    # Analyze ripple effects if requested
    if args.analyze_ripple:
        logger.info("\n" + "=" * 70)
        logger.info("Analyzing Ripple Effects")
        logger.info("=" * 70)

        # Initialize ripple analyzer
        analyzer = RippleAnalyzer(
            model_before=model,
            model_after=edited_model,
            tokenizer=tokenizer,
            device=device
        )

        # Analyze all triples
        logger.info(f"\nAnalyzing ripple effects on knowledge graph...")
        logger.info(f"KG corpus: {args.kg_corpus}")
        if args.max_ripple_triples:
            logger.info(f"Analyzing up to {args.max_ripple_triples} triples")
        else:
            logger.info("Analyzing all triples in corpus")

        ripple_results = analyzer.analyze_all_triples(
            kg_path=args.kg_corpus,
            edited_s=args.subject,
            edited_r=args.relation,
            edited_o_original=args.original,
            edited_o_target=args.target,
            max_triples=args.max_ripple_triples
        )

        logger.info(f"Analyzed {len(ripple_results)} triples")

        # Compute statistics
        logger.info("\nComputing ripple effect statistics...")
        stats = analyzer.compute_statistics(ripple_results)

        logger.info(f"\nRipple Effect Statistics:")
        logger.info(f"  Total triples analyzed: {stats['total_triples']}")
        logger.info(f"\n  By hop distance:")
        for hop, hop_stats in sorted(stats['by_hop_distance'].items()):
            logger.info(f"    Hop {hop}: {hop_stats['count']} triples, "
                       f"mean ripple = {hop_stats['mean_ripple']:.4f} ± {hop_stats['std_ripple']:.4f}")

        # Save detailed results
        ripple_data = {
            'edit': {
                's': args.subject,
                'r': args.relation,
                'o_original': args.original,
                'o_target': args.target,
                'layer': result.layer
            },
            'statistics': stats,
            'triples': [r.to_dict() for r in ripple_results]
        }

        ripple_path = output_dir / "ripple_analysis.json"
        with open(ripple_path, 'w') as f:
            json.dump(ripple_data, f, indent=2)

        logger.info(f"\nRipple analysis saved to: {ripple_path}")

        # Create ripple visualizations
        logger.info("\nCreating ripple effect visualizations...")

        # 1. Ripple on knowledge graph
        logger.info("  - Knowledge graph with ripple effects...")
        visualize_ripple_on_graph(
            results=ripple_results,
            edited_s=args.subject,
            edited_r=args.relation,
            edited_o_original=args.original,
            edited_o_target=args.target,
            graph=analyzer.graph,
            save_path=output_dir / "ripple_graph.png"
        )

        # 2. Statistical plots
        logger.info("  - Ripple effect statistics...")
        visualize_ripple_statistics(
            results=ripple_results,
            stats=stats,
            save_path=output_dir / "ripple_stats.png"
        )

        # 3. Top affected triples
        logger.info("  - Top affected triples...")
        visualize_top_affected_triples(
            results=ripple_results,
            edited_s=args.subject,
            edited_r=args.relation,
            edited_o_target=args.target,
            save_path=output_dir / "ripple_top_affected.png"
        )

        logger.info(f"\nRipple visualizations saved to: {output_dir}")

    logger.info(f"\n{'=' * 70}")
    logger.info("Done!")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
