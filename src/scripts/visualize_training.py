"""Visualize training progress and model analysis."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import SROTokenizer, GPTMini, GPTConfig
from src.modeling.inference_fixed import SROInferenceFixed
from src.utils import load_yaml, load_json, Logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_training_curves(metrics_path: Path, output_dir: Path):
    """Plot training curves from metrics."""
    logger = Logger(verbose=True)
    logger.info("Plotting training curves...")

    metrics = load_json(metrics_path)

    # Extract data
    steps = []
    train_loss = []
    train_acc = []
    eval_acc = []

    for metric_name, values in metrics.items():
        if metric_name == "loss":
            for entry in values:
                if entry['step'] is not None:
                    steps.append(entry['step'])
                    train_loss.append(entry['value'])
        elif metric_name == "acc":
            for entry in values:
                if entry['step'] is not None:
                    train_acc.append(entry['value'])
        elif metric_name == "eval_acc":
            eval_steps = [e['step'] for e in values if e['step'] is not None]
            eval_values = [e['value'] for e in values if e['step'] is not None]

    # Plot loss
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(steps, train_loss, label='Train Loss', alpha=0.7)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(steps, train_acc, label='Train Accuracy', alpha=0.7)
    if eval_steps:
        axes[1].plot(eval_steps, eval_values, label='Eval Accuracy', marker='o', alpha=0.7)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Evaluation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training curves to {output_path}")


def plot_entity_embeddings(model, tokenizer, output_dir: Path, method='pca'):
    """Plot entity embeddings with dimensionality reduction."""
    logger = Logger(verbose=True)
    logger.info(f"Plotting entity embeddings ({method})...")

    # Get all canonical entity tokens
    entity_tokens = [t for t in tokenizer.vocab.keys() if t.startswith('E_') and '__a' not in t]
    entity_tokens = sorted(entity_tokens)[:50]  # Limit to 50 for clarity

    # Get embeddings
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for token in entity_tokens:
            token_id = tokenizer.get_id(token)
            emb = model.token_embed.weight[token_id].cpu().numpy()
            embeddings.append(emb)
            labels.append(token)

    embeddings = np.array(embeddings)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        title = f'Entity Embeddings (PCA) - Explained Variance: {reducer.explained_variance_ratio_.sum():.2%}'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced = reducer.fit_transform(embeddings)
        title = 'Entity Embeddings (t-SNE)'

    # Plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=100)

    # Annotate some points
    for i, label in enumerate(labels):
        if i % max(1, len(labels) // 20) == 0:  # Annotate every Nth
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]),
                        fontsize=8, alpha=0.7)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    output_path = output_dir / f"entity_embeddings_{method}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved embeddings plot to {output_path}")


def plot_alias_clustering(model, tokenizer, kg_aliases, output_dir: Path):
    """Plot alias clustering for specific entities."""
    logger = Logger(verbose=True)
    logger.info("Plotting alias clustering...")

    # Select a few entities with aliases
    entities_to_plot = ['E_000', 'E_001', 'E_010', 'E_020', 'E_050']

    all_embeddings = []
    all_labels = []
    colors = []
    color_map = plt.cm.get_cmap('tab10')

    model.eval()
    with torch.no_grad():
        for i, entity in enumerate(entities_to_plot):
            if entity in kg_aliases['entities']:
                aliases = kg_aliases['entities'][entity]

                for alias in aliases:
                    if alias in tokenizer.vocab:
                        token_id = tokenizer.get_id(alias)
                        emb = model.token_embed.weight[token_id].cpu().numpy()
                        all_embeddings.append(emb)
                        all_labels.append(f"{entity} ({alias.split('__a')[1]})")
                        colors.append(color_map(i))

    if not all_embeddings:
        logger.warning("No alias embeddings found")
        return

    embeddings = np.array(all_embeddings)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(14, 10))

    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, c=[colors[i]], s=150, alpha=0.7, edgecolors='black', linewidths=0.5)
        plt.annotate(all_labels[i], (x, y), fontsize=8, alpha=0.8)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Alias Clustering (Same Entity Aliases Should Cluster Together)')
    plt.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color_map(i), markersize=10,
                                   label=entity) for i, entity in enumerate(entities_to_plot)]
    plt.legend(handles=legend_elements, loc='best')

    output_path = output_dir / "alias_clustering.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved alias clustering to {output_path}")


def plot_prediction_heatmap(inference: SROInferenceFixed, triples: list, output_dir: Path):
    """Plot prediction confidence heatmap."""
    logger = Logger(verbose=True)
    logger.info("Plotting prediction heatmap...")

    # Sample triples
    sample_triples = triples[:20]

    predictions = []
    ground_truths = []
    confidences = []

    for s, r, o in sample_triples:
        result = inference.predict_next(s, r, top_k=5)
        o_canonical = inference._get_canonical_entity(o)

        predictions.append(result['top_entity'])
        ground_truths.append(o_canonical)

        # Get confidence for ground truth
        entity_probs = dict(result['top_k_entities'])
        confidence = entity_probs.get(o_canonical, 0.0)
        confidences.append(confidence)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create matrix
    correct = [1 if p == g else 0 for p, g in zip(predictions, ground_truths)]

    # Bar plot
    x = np.arange(len(sample_triples))
    bars = ax.bar(x, confidences, color=['green' if c else 'red' for c in correct], alpha=0.6)

    ax.set_xlabel('Triple Index')
    ax.set_ylabel('Confidence (Probability)')
    ax.set_title('Prediction Confidence for Sample Triples (Green=Correct, Red=Wrong)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add accuracy text
    accuracy = sum(correct) / len(correct)
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / "prediction_confidence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved prediction heatmap to {output_path}")


def main():
    logger = Logger(verbose=True)
    logger.info("Starting visualization...")

    # Output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_dir = Path("outputs/models/gpt_small")

    if not (model_dir / "model.pt").exists():
        logger.error(f"Model not found at {model_dir}")
        return

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
    model.eval()

    logger.info("Model loaded")

    # 1. Training curves
    if (model_dir / "metrics.json").exists():
        plot_training_curves(model_dir / "metrics.json", output_dir)

    # 2. Entity embeddings (PCA)
    plot_entity_embeddings(model, tokenizer, output_dir, method='pca')

    # 3. Entity embeddings (t-SNE)
    plot_entity_embeddings(model, tokenizer, output_dir, method='tsne')

    # 4. Alias clustering
    kg_aliases = load_yaml("data/kg/ba/aliases.json")
    plot_alias_clustering(model, tokenizer, kg_aliases, output_dir)

    # 5. Prediction analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inference = SROInferenceFixed(model, tokenizer, device)

    train_triples = inference.load_corpus_triples("data/kg/ba/corpus.train.txt")
    plot_prediction_heatmap(inference, train_triples, output_dir)

    logger.info(f"\n✓ All visualizations saved to {output_dir}/")
    logger.info("Generated plots:")
    logger.info("  - training_curves.png")
    logger.info("  - entity_embeddings_pca.png")
    logger.info("  - entity_embeddings_tsne.png")
    logger.info("  - alias_clustering.png")
    logger.info("  - prediction_confidence.png")


if __name__ == "__main__":
    main()
