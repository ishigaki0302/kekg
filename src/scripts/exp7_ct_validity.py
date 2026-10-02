"""Exp 7: Causal Tracing Validity Check for 2-token inputs.

Tests whether causal tracing produces meaningful layer localization
when the input is only 2 tokens [S, R].

Key questions:
- Does best_layer concentrate on specific layers or spread uniformly?
- Is the AIE score distribution peaked or flat across layers?
- Does the CT-selected layer yield better edit success than random layers?
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
from src.edit.causal_tracing import CausalTracer
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


def run_ct_validity(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    num_samples=500,
    noise_level=3.0,
    num_noise_samples=10,
    device="cuda:0",
    output_dir="outputs/exp7_ct_validity",
    seed=42,
):
    print("=" * 60)
    print("Exp 7: Causal Tracing Validity Check")
    print("=" * 60)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    n_layers = model.config.n_layers
    print(f"Model: {n_layers} layers, d_model={model.config.d_model}")

    # Load triples
    all_triples = load_triples(kg_corpus)
    print(f"Total triples: {len(all_triples)}")

    # Sample triples that the model predicts correctly
    np.random.seed(seed)
    np.random.shuffle(all_triples)

    tracer = CausalTracer(model, tokenizer, device)

    results = []
    correct_count = 0
    tested = 0

    for triple in all_triples:
        if len(results) >= num_samples:
            break

        s, r, o = triple["s"], triple["r"], triple["o"]

        # Check if model predicts correctly
        input_text = f"{s} {r}"
        input_ids = torch.tensor([tokenizer.encode(input_text)]).to(device)
        with torch.no_grad():
            out = model(input_ids)
            pred_id = out["logits"][0, -1, :].argmax().item()
            pred = tokenizer.get_token(pred_id)

        tested += 1
        if pred.strip() != o.strip():
            continue
        correct_count += 1

        # Run causal tracing
        best_layer, effects, grid = tracer.locate_important_layer_with_scores(
            s, r, o, noise_level=noise_level, num_samples=num_noise_samples
        )

        # grid shape: [num_tokens, num_layers]
        # For 2-token input: token 0 = S, token 1 = R
        scores_S = grid[0, :].numpy()  # AIE scores for subject token
        scores_R = grid[1, :].numpy() if grid.shape[0] > 1 else np.zeros(n_layers)

        # Compute statistics
        score_range = scores_S.max() - scores_S.min()
        score_std = scores_S.std()
        score_entropy = -np.sum(
            np.where(scores_S > 0, scores_S / scores_S.sum() * np.log(scores_S / scores_S.sum() + 1e-10), 0)
        )

        # Get degree from KG (approximate: count occurrences as subject)
        result = {
            "s": s,
            "r": r,
            "o": o,
            "best_layer": int(best_layer),
            "scores_S": scores_S.tolist(),
            "scores_R": scores_R.tolist(),
            "score_range": float(score_range),
            "score_std": float(score_std),
            "score_entropy": float(score_entropy),
        }
        results.append(result)

        if len(results) % 50 == 0:
            print(f"  Progress: {len(results)}/{num_samples} (tested {tested}, correct {correct_count})")

    print(f"\nCompleted: {len(results)} triples (tested {tested}, correct {correct_count})")

    # Save raw results
    with open(out_dir / "ct_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # === Analysis ===
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    best_layers = [r["best_layer"] for r in results]

    # 1. Layer distribution
    layer_counts = defaultdict(int)
    for bl in best_layers:
        layer_counts[bl] += 1

    print("\n1. Best Layer Distribution:")
    for layer in range(n_layers):
        count = layer_counts.get(layer, 0)
        bar = "#" * (count // 2)
        print(f"  Layer {layer:2d}: {count:4d} ({count/len(results)*100:5.1f}%) {bar}")

    # Uniformity test: chi-squared against uniform
    from scipy.stats import chisquare
    observed = [layer_counts.get(i, 0) for i in range(n_layers)]
    chi2, p_value = chisquare(observed)
    print(f"\n  Chi-squared test vs uniform: chi2={chi2:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  -> Distribution is NOT uniform (layer localization exists)")
    else:
        print("  -> Distribution is uniform (NO meaningful localization)")

    # 2. AIE score statistics
    all_ranges = [r["score_range"] for r in results]
    all_stds = [r["score_std"] for r in results]
    print(f"\n2. AIE Score Spread:")
    print(f"  Score range: mean={np.mean(all_ranges):.4f}, std={np.std(all_ranges):.4f}")
    print(f"  Score std:   mean={np.mean(all_stds):.4f}, std={np.std(all_stds):.4f}")

    # 3. Average AIE score per layer
    avg_scores = np.mean([r["scores_S"] for r in results], axis=0)
    print(f"\n3. Average AIE Score by Layer:")
    for layer in range(n_layers):
        bar = "#" * int(avg_scores[layer] * 200)
        print(f"  Layer {layer:2d}: {avg_scores[layer]:.4f} {bar}")

    # 4. Consistency: same entity → same best layer?
    entity_layers = defaultdict(list)
    for r in results:
        entity_layers[r["s"]].append(r["best_layer"])
    multi_entity = {e: layers for e, layers in entity_layers.items() if len(layers) >= 3}
    if multi_entity:
        consistencies = []
        for e, layers in multi_entity.items():
            most_common = max(set(layers), key=layers.count)
            consistency = layers.count(most_common) / len(layers)
            consistencies.append(consistency)
        print(f"\n4. Entity Consistency (entities with 3+ triples):")
        print(f"  N entities: {len(multi_entity)}")
        print(f"  Mean consistency: {np.mean(consistencies):.3f}")
        print(f"  (1.0 = always same layer, 1/12 = random)")

    # Save summary
    summary = {
        "num_triples": len(results),
        "layer_distribution": dict(layer_counts),
        "chi2_uniform": {"chi2": float(chi2), "p_value": float(p_value)},
        "avg_aie_by_layer": avg_scores.tolist(),
        "score_range_mean": float(np.mean(all_ranges)),
        "score_std_mean": float(np.mean(all_stds)),
    }
    with open(out_dir / "ct_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp7_ct_validity")
    args = parser.parse_args()

    run_ct_validity(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        num_samples=args.num_samples,
        device=args.device,
        output_dir=args.output_dir,
    )
