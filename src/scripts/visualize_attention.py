"""Visualize attention patterns and layer activations."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.utils import load_yaml, Logger

sns.set_style("whitegrid")


def visualize_attention_pattern(model, tokenizer, s, r, o, output_dir: Path):
    """Visualize attention patterns for a single triple."""
    logger = Logger(verbose=True)
    logger.info(f"Visualizing attention for: {s} {r} -> {o}")

    # Encode
    input_text = f"{s} {r}"
    input_ids = torch.tensor(
        tokenizer.encode(input_text),
        dtype=torch.long,
        device=model.token_embed.weight.device
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, return_attention=True)
        attentions = outputs["attentions"]

    # Plot attention for each layer
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    tokens = [s.split('__a')[0], r.split('__a')[0]]

    for layer_idx in range(min(n_layers, 12)):
        attn = attentions[layer_idx][0].cpu().numpy()  # [n_heads, seq_len, seq_len]

        # Average over heads
        attn_avg = attn.mean(axis=0)  # [seq_len, seq_len]

        ax = axes[layer_idx]
        sns.heatmap(attn_avg, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=tokens, yticklabels=tokens,
                    cbar=True, ax=ax, vmin=0, vmax=1)
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    plt.suptitle(f'Attention Patterns: {s} {r} -> {o}', fontsize=14, y=1.00)
    plt.tight_layout()

    output_path = output_dir / "attention_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved attention patterns to {output_path}")


def visualize_layer_activations(model, tokenizer, triples, output_dir: Path):
    """Visualize activation norms across layers."""
    logger = Logger(verbose=True)
    logger.info("Visualizing layer activations...")

    model.eval()

    # Sample triples
    sample_triples = triples[:20]

    # Collect activations
    layer_norms = [[] for _ in range(model.config.n_layers + 1)]

    with torch.no_grad():
        for s, r, o in sample_triples:
            input_text = f"{s} {r}"
            input_ids = torch.tensor(
                tokenizer.encode(input_text),
                dtype=torch.long,
                device=model.token_embed.weight.device
            ).unsqueeze(0)

            outputs = model(input_ids, return_hidden_states=True)
            hidden_states = outputs["hidden_states"]

            for layer_idx, h in enumerate(hidden_states):
                # Get norm of last token
                norm = h[0, -1, :].norm().item()
                layer_norms[layer_idx].append(norm)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    layer_indices = list(range(len(layer_norms)))
    means = [np.mean(norms) for norms in layer_norms]
    stds = [np.std(norms) for norms in layer_norms]

    ax.errorbar(layer_indices, means, yerr=stds, marker='o', capsize=5, capthick=2)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Activation Norm')
    ax.set_title('Hidden State Norms Across Layers (Mean ± Std)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layer_indices)

    plt.tight_layout()
    output_path = output_dir / "layer_activations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved layer activations to {output_path}")


def visualize_error_analysis(model, tokenizer, inference, triples, output_dir: Path):
    """Analyze and visualize prediction errors."""
    logger = Logger(verbose=True)
    logger.info("Analyzing prediction errors...")

    from src.modeling.inference_fixed import SROInferenceFixed

    # Sample and evaluate
    sample_triples = triples[:200]
    results = inference.batch_evaluate(sample_triples)

    errors = [r for r in results['details'] if not r['is_correct']]
    correct = [r for r in results['details'] if r['is_correct']]

    logger.info(f"Errors: {len(errors)}, Correct: {len(correct)}")

    # Analyze error ranks
    error_ranks = [e['rank'] for e in errors]
    correct_ranks = [c['rank'] for c in correct]

    # Plot rank distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    axes[0].hist(error_ranks, bins=20, alpha=0.7, label='Errors', color='red', edgecolor='black')
    axes[0].hist(correct_ranks, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].set_xlabel('Rank of Ground Truth')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Rank Distribution: Errors vs Correct')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Box plot
    data_to_plot = [correct_ranks, error_ranks]
    axes[1].boxplot(data_to_plot, labels=['Correct', 'Errors'])
    axes[1].set_ylabel('Rank of Ground Truth')
    axes[1].set_title('Rank Box Plot')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "error_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved error analysis to {output_path}")

    # Print statistics
    logger.info(f"\nError Statistics:")
    logger.info(f"  Mean rank (errors): {np.mean(error_ranks):.2f}")
    logger.info(f"  Median rank (errors): {np.median(error_ranks):.1f}")
    logger.info(f"  Mean rank (correct): {np.mean(correct_ranks):.2f}")


def visualize_logit_distribution(model, tokenizer, inference, triples, output_dir: Path):
    """Visualize logit distribution for predictions."""
    logger = Logger(verbose=True)
    logger.info("Visualizing logit distribution...")

    sample_triples = triples[:50]

    model.eval()
    all_logits = []

    with torch.no_grad():
        for s, r, o in sample_triples:
            input_text = f"{s} {r}"
            input_ids = torch.tensor(
                tokenizer.encode(input_text),
                dtype=torch.long,
                device=model.token_embed.weight.device
            ).unsqueeze(0)

            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :].cpu().numpy()

            # Get only entity logits
            entity_logits = []
            for token, token_id in tokenizer.vocab.items():
                if token.startswith('E_') and '__a' not in token:
                    entity_logits.append(logits[token_id])

            all_logits.append(entity_logits)

    # Plot
    all_logits = np.array(all_logits)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of max logit
    max_logits = all_logits.max(axis=1)
    axes[0].hist(max_logits, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Max Logit Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Maximum Logits')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Logit spread (max - min)
    logit_spread = all_logits.max(axis=1) - all_logits.min(axis=1)
    axes[1].hist(logit_spread, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Logit Spread (Max - Min)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Logit Spread Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "logit_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved logit distribution to {output_path}")


def main():
    logger = Logger(verbose=True)
    logger.info("Starting advanced visualizations...")

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    logger.info("Loading model...")
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

    # Create inference
    from src.modeling.inference_fixed import SROInferenceFixed
    inference = SROInferenceFixed(model, tokenizer, device)

    # Load triples
    train_triples = inference.load_corpus_triples("data/kg/ba/corpus.train.txt")

    # 1. Attention patterns
    sample_triple = train_triples[0]
    visualize_attention_pattern(model, tokenizer, *sample_triple, output_dir)

    # 2. Layer activations
    visualize_layer_activations(model, tokenizer, train_triples, output_dir)

    # 3. Error analysis
    visualize_error_analysis(model, tokenizer, inference, train_triples, output_dir)

    # 4. Logit distribution
    visualize_logit_distribution(model, tokenizer, inference, train_triples, output_dir)

    logger.info(f"\n✓ All advanced visualizations saved to {output_dir}/")
    logger.info("Generated plots:")
    logger.info("  - attention_patterns.png")
    logger.info("  - layer_activations.png")
    logger.info("  - error_analysis.png")
    logger.info("  - logit_distribution.png")


if __name__ == "__main__":
    main()
