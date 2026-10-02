"""Exp 1b: Degree-Conditional Representation Shattering.

Measures how ROME edits distort the model's representations for unedited
triples, conditioned on the degree of the edited subject.

Key metric: R(D*) = ||D* - D_0||_F / ||D_0||_F  (Nishi et al., ICML 2025)
where D is the matrix of output logits for a fixed set of test triples.

Our extension: decompose R(D*) by:
  - Subject degree bin (low / mid / high)
  - Hop distance from edited subjects
  - Per-layer hidden state shattering
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
from src.edit.rome import ROME
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


def load_kg(corpus_path):
    """Load KG and compute degree/adjacency."""
    triples = []
    adjacency = defaultdict(set)
    degree = defaultdict(int)

    with open(corpus_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, r, o = parts
                triples.append({"s": s, "r": r, "o": o})
                adjacency[s].add(o)
                adjacency[o].add(s)
                degree[s] += 1

    return triples, adjacency, degree


def bfs_hop(entity, adjacency, max_hop=4):
    """BFS from entity, return {entity: hop_distance}."""
    from collections import deque
    visited = {entity: 0}
    queue = deque([entity])
    while queue:
        node = queue.popleft()
        if visited[node] >= max_hop:
            continue
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
    return visited


@torch.no_grad()
def compute_logit_matrix(model, tokenizer, triples, device, batch_size=512):
    """Compute logit vectors for all triples. Returns [N, vocab_size]."""
    all_logits = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        input_ids = torch.tensor(
            [tokenizer.encode(f"{t['s']} {t['r']}") for t in batch],
            dtype=torch.long, device=device,
        )
        out = model(input_ids)
        logits = out["logits"][:, -1, :]  # [batch, vocab]
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def compute_hidden_matrix(model, tokenizer, triples, device, batch_size=512):
    """Compute hidden states at all layers for all triples.
    Returns [N, n_layers+1, d_model] (R-position hidden states).
    """
    all_hidden = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        input_ids = torch.tensor(
            [tokenizer.encode(f"{t['s']} {t['r']}") for t in batch],
            dtype=torch.long, device=device,
        )
        out = model(input_ids, return_hidden_states=True)
        # hidden_states: list of [batch, seq_len, d_model] x (n_layers+1)
        # Take R position (index 1)
        hs = torch.stack([h[:, 1, :] for h in out["hidden_states"]], dim=1)  # [batch, n_layers+1, d_model]
        all_hidden.append(hs.cpu())
    return torch.cat(all_hidden, dim=0)


def compute_shattering(D_star, D_0):
    """R(D*) = ||D* - D_0||_F / ||D_0||_F"""
    diff = D_star - D_0
    return torch.norm(diff, p="fro").item() / (torch.norm(D_0, p="fro").item() + 1e-10)


def run_degree_shattering(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp1b_degree_shattering",
    num_edits_per_bin=50,
    num_test_triples=3000,
    seed=42,
):
    print("=" * 60)
    print("Exp 1b: Degree-Conditional Representation Shattering")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    model_orig, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, adjacency, degree = load_kg(kg_corpus)
    print(f"Triples: {len(all_triples)}, Entities with degree: {len(degree)}")

    # Degree bins
    degree_bins = {
        "low": (1, 15),
        "mid": (15, 35),
        "high": (35, 999),
    }

    # Group subjects by degree
    subjects_by_bin = {}
    for bin_name, (lo, hi) in degree_bins.items():
        subjects = [e for e, d in degree.items() if lo <= d < hi]
        subjects_by_bin[bin_name] = subjects
        print(f"  {bin_name} [{lo}, {hi}): {len(subjects)} subjects")

    # Build subject -> triples index
    subj_triples = defaultdict(list)
    for t in all_triples:
        subj_triples[t["s"]].append(t)

    # Select test triples (fixed across all conditions)
    np.random.shuffle(all_triples)
    test_triples = all_triples[:num_test_triples]
    print(f"\nTest triples: {len(test_triples)}")

    # Compute baseline representations
    print("Computing baseline logits and hidden states...")
    D0_logits = compute_logit_matrix(model_orig, tokenizer, test_triples, device)
    D0_hidden = compute_hidden_matrix(model_orig, tokenizer, test_triples, device)
    print(f"  D0_logits: {D0_logits.shape}, D0_hidden: {D0_hidden.shape}")

    # Checkpoints: measure shattering at these edit counts
    checkpoints = [1, 2, 5, 10, 20, 30, 50]
    checkpoints = [c for c in checkpoints if c <= num_edits_per_bin]

    results = {}

    for bin_name, (lo, hi) in degree_bins.items():
        print(f"\n{'='*60}")
        print(f"Editing subjects from degree bin: {bin_name} [{lo}, {hi})")
        print(f"{'='*60}")

        subjects = subjects_by_bin[bin_name]
        if len(subjects) == 0:
            print("  No subjects in this bin, skipping.")
            continue

        np.random.shuffle(subjects)

        # Select edit triples: 1 per subject
        edit_triples = []
        all_entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))
        for subj in subjects:
            if len(edit_triples) >= num_edits_per_bin:
                break
            trs = subj_triples.get(subj, [])
            if trs:
                t = trs[np.random.randint(len(trs))]
                # Pick a random new object different from original
                new_o = t["o"]
                while new_o == t["o"]:
                    new_o = all_entities[np.random.randint(len(all_entities))]
                edit_triples.append({**t, "o_new": new_o})

        print(f"  Edit triples selected: {len(edit_triples)}")

        # Fresh copy of model for this bin
        model = deepcopy(model_orig).to(device)
        model.eval()

        # Initialize ROME editor
        rome = ROME(
            model=model,
            tokenizer=tokenizer,
            device=device,
            kg_corpus_path=kg_corpus,
            mom2_n_samples=1000,
            use_mom2_adjustment=True,
            v_num_grad_steps=20,
        )

        bin_results = {
            "degree_bin": bin_name,
            "degree_range": [lo, hi],
            "num_edits": len(edit_triples),
            "checkpoints": [],
        }

        # Collect edited subjects for hop-distance analysis
        edited_subjects = []

        for i, edit in enumerate(edit_triples):
            step = i + 1
            print(f"  Edit {step}/{len(edit_triples)}: ({edit['s']}, {edit['r']}, {edit['o']}) -> {edit['o_new']}")

            # Apply ROME edit
            model, result = rome.apply_edit(
                s=edit["s"],
                r=edit["r"],
                o_target=edit["o_new"],
                layer=5,  # Use middle layer (consistent across bins)
                copy_model=False,
            )

            edited_subjects.append(edit["s"])

            if step not in checkpoints:
                continue

            print(f"\n  --- Checkpoint at {step} edits ---")

            # Compute post-edit representations
            D_star_logits = compute_logit_matrix(model, tokenizer, test_triples, device)
            D_star_hidden = compute_hidden_matrix(model, tokenizer, test_triples, device)

            # Global shattering
            R_logit = compute_shattering(D_star_logits, D0_logits)
            print(f"    R(D*) logit: {R_logit:.6f}")

            # Per-layer hidden shattering
            n_layers_plus1 = D0_hidden.shape[1]
            R_per_layer = []
            for layer in range(n_layers_plus1):
                R_l = compute_shattering(D_star_hidden[:, layer, :], D0_hidden[:, layer, :])
                R_per_layer.append(R_l)
                layer_name = "emb" if layer == 0 else f"L{layer-1}"
                print(f"    R(D*) {layer_name}: {R_l:.6f}")

            # Shattering by hop distance from edited subjects
            # For each test triple, find min hop from any edited subject
            hop_map = {}
            for subj in edited_subjects:
                hops = bfs_hop(subj, adjacency, max_hop=4)
                for entity, dist in hops.items():
                    if entity not in hop_map or dist < hop_map[entity]:
                        hop_map[entity] = dist

            # Group test triples by min hop to edited subjects
            test_by_hop = defaultdict(list)
            for idx, t in enumerate(test_triples):
                hop_s = hop_map.get(t["s"], 99)
                hop_o = hop_map.get(t["o"], 99)
                min_hop = min(hop_s, hop_o)
                if min_hop <= 4:
                    test_by_hop[min_hop].append(idx)
                else:
                    test_by_hop["far"].append(idx)

            R_by_hop = {}
            for hop_key in sorted(test_by_hop.keys(), key=lambda x: x if isinstance(x, int) else 99):
                indices = test_by_hop[hop_key]
                if len(indices) < 5:
                    continue
                idx_tensor = torch.tensor(indices)
                R_hop = compute_shattering(D_star_logits[idx_tensor], D0_logits[idx_tensor])
                R_by_hop[str(hop_key)] = {"R": float(R_hop), "n": len(indices)}
                print(f"    R(D*) hop={hop_key}: {R_hop:.6f} (n={len(indices)})")

            # Shattering by degree of test triple's subject
            test_by_degree = defaultdict(list)
            for idx, t in enumerate(test_triples):
                d = degree.get(t["s"], 0)
                if d < 15:
                    test_by_degree["low"].append(idx)
                elif d < 35:
                    test_by_degree["mid"].append(idx)
                else:
                    test_by_degree["high"].append(idx)

            R_by_victim_degree = {}
            for dbin, indices in test_by_degree.items():
                if len(indices) < 5:
                    continue
                idx_tensor = torch.tensor(indices)
                R_d = compute_shattering(D_star_logits[idx_tensor], D0_logits[idx_tensor])
                R_by_victim_degree[dbin] = {"R": float(R_d), "n": len(indices)}
                print(f"    R(D*) victim_degree={dbin}: {R_d:.6f} (n={len(indices)})")

            checkpoint_data = {
                "num_edits": step,
                "R_logit_global": float(R_logit),
                "R_per_layer": [float(r) for r in R_per_layer],
                "R_by_hop": R_by_hop,
                "R_by_victim_degree": R_by_victim_degree,
                "edited_subjects": list(edited_subjects),
            }
            bin_results["checkpoints"].append(checkpoint_data)

        results[bin_name] = bin_results

        # Free GPU memory
        del model, rome
        torch.cuda.empty_cache()

    # Save results
    with open(out_dir / "shattering_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # === Visualization ===
    print("\nGenerating plots...")

    # Plot 1: R(D*) vs num_edits for each degree bin
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Global logit shattering
    for bin_name in ["low", "mid", "high"]:
        if bin_name not in results:
            continue
        cps = results[bin_name]["checkpoints"]
        xs = [c["num_edits"] for c in cps]
        ys = [c["R_logit_global"] for c in cps]
        axes[0].plot(xs, ys, "o-", label=f"degree={bin_name}")
    axes[0].set_xlabel("Number of edits")
    axes[0].set_ylabel("R(D*)")
    axes[0].set_title("Global Representation Shattering")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Middle: Per-layer shattering at max edits
    layer_labels = ["emb"] + [f"L{i}" for i in range(12)]
    for bin_name in ["low", "mid", "high"]:
        if bin_name not in results:
            continue
        cps = results[bin_name]["checkpoints"]
        if cps:
            last = cps[-1]
            axes[1].plot(range(len(last["R_per_layer"])), last["R_per_layer"], "o-", label=f"degree={bin_name}")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("R(D*)")
    axes[1].set_title(f"Per-Layer Shattering (after {num_edits_per_bin} edits)")
    axes[1].set_xticks(range(13))
    axes[1].set_xticklabels(layer_labels, rotation=45, fontsize=8)
    axes[1].legend()

    # Right: Shattering by hop distance at max edits
    for bin_name in ["low", "mid", "high"]:
        if bin_name not in results:
            continue
        cps = results[bin_name]["checkpoints"]
        if cps:
            last = cps[-1]
            hops = []
            rs = []
            for h_str, v in sorted(last["R_by_hop"].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99):
                if h_str.isdigit():
                    hops.append(int(h_str))
                    rs.append(v["R"])
            if hops:
                axes[2].plot(hops, rs, "o-", label=f"degree={bin_name}")
    axes[2].set_xlabel("Hop distance from edited subjects")
    axes[2].set_ylabel("R(D*)")
    axes[2].set_title(f"Shattering by Hop Distance (after {num_edits_per_bin} edits)")
    axes[2].legend()

    fig.suptitle("Degree-Conditional Representation Shattering (Nishi R(D*) metric)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "shattering_plots.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {out_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp1b_degree_shattering")
    parser.add_argument("--num-edits-per-bin", type=int, default=50)
    parser.add_argument("--num-test-triples", type=int, default=3000)
    args = parser.parse_args()

    run_degree_shattering(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits_per_bin=args.num_edits_per_bin,
        num_test_triples=args.num_test_triples,
    )
