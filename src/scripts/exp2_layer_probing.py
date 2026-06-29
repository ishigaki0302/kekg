"""Exp 2: Layer-wise Probing Analysis.

Trains linear probes at each layer to decode S, R, O information
from hidden states. Reveals where knowledge is extracted.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
from src.utils.io import load_yaml


def load_model_and_tokenizer(model_dir, device="cuda"):
    model_path = Path(model_dir)
    tokenizer = SROTokenizer.load(model_path / "tokenizer.json")
    train_report = load_yaml(model_path / "train_report.yaml")
    mc = train_report["config"]["model"]
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        d_model=mc["d_model"],
        d_mlp=mc["d_mlp"],
        max_seq_len=mc.get("max_seq_len", 8),
        dropout=mc.get("dropout", 0.1),
    )
    model = GPTMini(config)
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_triples(corpus_path):
    triples = []
    with open(corpus_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append({"s": parts[0], "r": parts[1], "o": parts[2]})
    return triples


def extract_hidden_states(model, tokenizer, triples, device, batch_size=256):
    """Extract hidden states at all layers for all triples."""
    n_layers = model.config.n_layers
    n_triples = len(triples)

    # Pre-allocate: [n_triples, n_layers+1, 2, d_model]
    # layers+1 because hidden_states[0] = embedding, hidden_states[l] = after block l-1
    all_hidden = []

    for start in range(0, n_triples, batch_size):
        batch = triples[start : start + batch_size]
        input_texts = [f"{t['s']} {t['r']}" for t in batch]
        input_ids = torch.tensor(
            [tokenizer.encode(text) for text in input_texts],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(input_ids, return_hidden_states=True)
            # hidden_states: list of [batch, 2, 512] × (n_layers + 1)
            hidden = torch.stack(outputs["hidden_states"], dim=1)  # [batch, n_layers+1, 2, 512]
            all_hidden.append(hidden.cpu())

        if (start // batch_size) % 10 == 0:
            print(f"  Extracted {min(start + batch_size, n_triples)}/{n_triples}")

    return torch.cat(all_hidden, dim=0)  # [n_triples, n_layers+1, 2, 512]


def run_probing(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:1",
    output_dir="outputs/exp2_layer_probing",
    num_triples=10000,
    seed=42,
):
    print("=" * 60)
    print("Exp 2: Layer-wise Probing Analysis")
    print("=" * 60)

    np.random.seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples = load_triples(kg_corpus)
    n_layers = model.config.n_layers

    # Sample triples
    np.random.shuffle(all_triples)
    triples = all_triples[:num_triples]
    print(f"Using {len(triples)} triples")

    # Get labels
    s_labels = np.array([tokenizer.get_id(t["s"]) for t in triples])
    r_labels = np.array([tokenizer.get_id(t["r"]) for t in triples])
    o_labels = np.array([tokenizer.get_id(t["o"]) for t in triples])

    # Compute degree for each entity
    degree_map = defaultdict(int)
    for t in all_triples:
        degree_map[t["s"]] += 1
    degree_labels = np.array([degree_map.get(t["s"], 0) for t in triples])

    # Extract hidden states
    print("\nExtracting hidden states...")
    hidden_states = extract_hidden_states(model, tokenizer, triples, device)
    # Shape: [N, n_layers+1, 2, 512]
    print(f"Hidden states shape: {hidden_states.shape}")

    # Train/test split
    n_train = int(len(triples) * 0.8)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, len(triples))

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score

    results = {"object_probe": {}, "subject_probe": {}, "degree_probe": {}}

    # === Probe 1: Object Probe (h_R → O) ===
    print("\n--- Object Probe (h_R → O) ---")
    for layer in range(n_layers + 1):
        X = hidden_states[:, layer, 1, :].numpy()  # R position
        y = o_labels

        clf = LogisticRegression(max_iter=500, solver="saga", n_jobs=-1, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))
        results["object_probe"][layer] = float(acc)
        layer_name = "emb" if layer == 0 else f"L{layer-1}"
        print(f"  {layer_name:>4s}: acc={acc:.4f}")

    # === Probe 2: Subject Probe (h_R → S) ===
    print("\n--- Subject Probe (h_R → S) ---")
    for layer in range(n_layers + 1):
        X = hidden_states[:, layer, 1, :].numpy()  # R position
        y = s_labels

        clf = LogisticRegression(max_iter=500, solver="saga", n_jobs=-1, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))
        results["subject_probe"][layer] = float(acc)
        layer_name = "emb" if layer == 0 else f"L{layer-1}"
        print(f"  {layer_name:>4s}: acc={acc:.4f}")

    # === Probe 3: Degree Probe (h_S → degree, regression) ===
    print("\n--- Degree Probe (h_S → degree, regression R²) ---")
    for layer in range(n_layers + 1):
        X = hidden_states[:, layer, 0, :].numpy()  # S position
        y = degree_labels.astype(float)

        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y[train_idx])
        r2 = reg.score(X[test_idx], y[test_idx])
        results["degree_probe"][layer] = float(r2)
        layer_name = "emb" if layer == 0 else f"L{layer-1}"
        print(f"  {layer_name:>4s}: R²={r2:.4f}")

    # === Visualization ===
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]

    # Object probe
    obj_accs = [results["object_probe"][l] for l in range(n_layers + 1)]
    axes[0].plot(range(n_layers + 1), obj_accs, "o-", color="tab:blue")
    axes[0].set_xticks(range(n_layers + 1))
    axes[0].set_xticklabels(layers_labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Object Probe (h_R → O)")
    axes[0].set_ylim(0, 1)

    # Subject probe
    subj_accs = [results["subject_probe"][l] for l in range(n_layers + 1)]
    axes[1].plot(range(n_layers + 1), subj_accs, "o-", color="tab:orange")
    axes[1].set_xticks(range(n_layers + 1))
    axes[1].set_xticklabels(layers_labels, rotation=45, fontsize=8)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Subject Probe (h_R → S)")
    axes[1].set_ylim(0, 1)

    # Degree probe
    deg_r2s = [results["degree_probe"][l] for l in range(n_layers + 1)]
    axes[2].plot(range(n_layers + 1), deg_r2s, "o-", color="tab:green")
    axes[2].set_xticks(range(n_layers + 1))
    axes[2].set_xticklabels(layers_labels, rotation=45, fontsize=8)
    axes[2].set_ylabel("R²")
    axes[2].set_title("Degree Probe (h_S → degree)")

    fig.suptitle("Layer-wise Probing: Where is S, O, Degree information?", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "probing_results.png", dpi=150)
    plt.close()

    # Save
    with open(out_dir / "probing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", default="outputs/exp2_layer_probing")
    parser.add_argument("--num-triples", type=int, default=10000)
    args = parser.parse_args()

    run_probing(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_triples=args.num_triples,
    )
