"""Exp 4: ROME Update Geometry Analysis.

Analyzes how ROME's rank-1 updates (ΔW = v ⊗ k^T) differ between
high-degree and low-degree subjects.

Key questions:
- Does k_star overlap with other entities' keys more for high-degree subjects?
- Does v_star align with the target embedding direction?
- Is the update magnitude (||ΔW||_F) degree-dependent?
- How does the key vector concentration explain victim-degree asymmetry?
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
    triples = []
    degree = defaultdict(int)
    subj_triples = defaultdict(list)
    with open(corpus_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, r, o = parts
                triples.append({"s": s, "r": r, "o": o})
                degree[s] += 1
                subj_triples[s].append({"s": s, "r": r, "o": o})
    return triples, degree, subj_triples


@torch.no_grad()
def get_ffn_input(model, tokenizer, s, r, layer, device):
    """Get the FFN input (after LN2) at the R position for a given (S, R) pair."""
    input_ids = torch.tensor([tokenizer.encode(f"{s} {r}")], dtype=torch.long, device=device)

    # Forward pass with hook to capture FFN input
    activations = {}

    def hook_fn(module, input, output):
        activations["ln2_output"] = input[0].detach()  # input to FFN is output of LN2

    # Actually we need the input to the FFN block, which is the output of ln2
    handle = model.blocks[layer].ffn.register_forward_hook(hook_fn)
    model(input_ids)
    handle.remove()

    # R position = index 1
    ffn_input = activations["ln2_output"][0, 1, :]  # [d_model=512]
    return ffn_input


@torch.no_grad()
def get_ffn_mid_activation(model, tokenizer, s, r, layer, device):
    """Get FFN mid-layer activation (after GELU) at R position."""
    input_ids = torch.tensor([tokenizer.encode(f"{s} {r}")], dtype=torch.long, device=device)

    activations = {}

    def hook_fn(module, input, output):
        activations["gelu_out"] = output.detach()

    # Hook after w1 + GELU
    handle = model.blocks[layer].ffn.w1.register_forward_hook(
        lambda m, i, o: activations.update({"w1_out": F.gelu(o).detach()})
    )
    model(input_ids)
    handle.remove()

    return activations["w1_out"][0, 1, :]  # [d_mlp=2048]


def extract_rome_vectors(model, tokenizer, s, r, o_new, layer, device, kg_corpus):
    """Apply a ROME edit and extract k_star, v_star, delta_W."""
    model_copy = deepcopy(model).to(device)
    model_copy.eval()

    # Get W_before
    W_before = model_copy.blocks[layer].ffn.w2.weight.detach().clone()

    rome = ROME(
        model=model_copy,
        tokenizer=tokenizer,
        device=device,
        kg_corpus_path=kg_corpus,
        mom2_n_samples=1000,
        use_mom2_adjustment=True,
        v_num_grad_steps=20,
    )

    model_copy, result = rome.apply_edit(
        s=s, r=r, o_target=o_new,
        layer=layer, copy_model=False,
    )

    # Get W_after
    W_after = model_copy.blocks[layer].ffn.w2.weight.detach().clone()

    # ΔW = W_after - W_before  (shape: [d_model, d_mlp] or [d_mlp, d_model])
    delta_W = W_after - W_before

    # SVD of rank-1 update to extract k and v directions
    # w2.weight shape: [d_model, d_mlp] = [512, 2048]
    # delta_W ≈ sigma * u @ v^T where:
    #   u (left SV) is in d_model space [512] = VALUE direction
    #   v (right SV) is in d_mlp space [2048] = KEY direction
    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
    # U: [512, 512], S: [512], Vh: [512, 2048]

    v_direction = U[:, 0]   # [d_model=512] - value direction (output space)
    k_direction = Vh[0]     # [d_mlp=2048] - key direction (FFN mid-layer space)
    sigma = S[0].item()

    # Effective rank
    if S.sum() > 0:
        eff_rank = (S.sum() ** 2) / (S ** 2).sum()
    else:
        eff_rank = torch.tensor(0.0)

    del model_copy, rome
    torch.cuda.empty_cache()

    return {
        "delta_W": delta_W.cpu(),
        "k_direction": k_direction.cpu(),
        "v_direction": v_direction.cpu(),
        "sigma": sigma,
        "singular_values": S[:10].cpu().tolist(),
        "effective_rank": eff_rank.item(),
        "delta_W_fro": torch.norm(delta_W, p="fro").item(),
        "edit_success": result.success,
    }


def run_rome_geometry(
    model_dir="outputs/models/gpt_small",
    kg_corpus="data/kg/ba/corpus.train.txt",
    device="cuda:0",
    output_dir="outputs/exp4_rome_geometry",
    num_edits_per_bin=30,
    edit_layer=5,
    seed=42,
):
    print("=" * 60)
    print("Exp 4: ROME Update Geometry Analysis")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    all_triples, degree, subj_triples = load_kg(kg_corpus)

    entities = sorted(set(e for t in all_triples for e in [t["s"], t["o"]]))
    print(f"Entities: {len(entities)}, Triples: {len(all_triples)}")

    # Degree bins
    degree_bins = {
        "low": (1, 15),
        "mid": (15, 35),
        "high": (35, 999),
    }

    # === Part A: Collect FFN mid-layer activations (keys in d_mlp space) ===
    print("\nPart A: Computing FFN mid-layer activations for all entities...")

    # For each entity, get its average post-GELU activation across its triples
    # This is in d_mlp space (2048), matching ROME's key direction from SVD
    entity_keys = {}  # entity -> mean post-GELU activation at R position [d_mlp]
    entity_ffn_inputs = {}  # entity -> mean FFN input at R position [d_model]

    for i, ent in enumerate(entities):
        trs = subj_triples.get(ent, [])
        if not trs:
            continue
        sample_trs = trs[:5] if len(trs) > 5 else trs
        mid_acts = []
        ffn_inputs = []
        for t in sample_trs:
            mid = get_ffn_mid_activation(model, tokenizer, t["s"], t["r"], edit_layer, device)
            mid_acts.append(mid)
            fi = get_ffn_input(model, tokenizer, t["s"], t["r"], edit_layer, device)
            ffn_inputs.append(fi)
        entity_keys[ent] = torch.stack(mid_acts).mean(dim=0).cpu()  # [d_mlp=2048]
        entity_ffn_inputs[ent] = torch.stack(ffn_inputs).mean(dim=0).cpu()  # [d_model=512]

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(entities)} entities processed")

    print(f"  Computed activations for {len(entity_keys)} entities")

    # Build key matrix (d_mlp space for key overlap analysis)
    ent_list = sorted(entity_keys.keys())
    key_matrix = torch.stack([entity_keys[e] for e in ent_list])  # [N, d_mlp=2048]
    key_matrix_norm = F.normalize(key_matrix, dim=1)

    # Also build FFN input matrix (d_model space for value alignment analysis)
    ffn_input_matrix = torch.stack([entity_ffn_inputs[e] for e in ent_list])  # [N, d_model=512]
    ffn_input_matrix_norm = F.normalize(ffn_input_matrix, dim=1)

    # === Part B: ROME updates for different degree bins ===
    print("\nPart B: Extracting ROME update vectors...")

    all_results = {}

    for bin_name, (lo, hi) in degree_bins.items():
        print(f"\n--- Degree bin: {bin_name} [{lo}, {hi}) ---")

        subjects = [e for e, d in degree.items() if lo <= d < hi]
        np.random.shuffle(subjects)

        bin_results = []

        for subj in subjects[:num_edits_per_bin]:
            trs = subj_triples.get(subj, [])
            if not trs:
                continue

            t = trs[np.random.randint(len(trs))]
            # Random new object
            new_o = t["o"]
            while new_o == t["o"]:
                new_o = entities[np.random.randint(len(entities))]

            print(f"  Edit: ({t['s']}, {t['r']}, {t['o']}) -> {new_o}")

            # Extract ROME vectors
            rome_data = extract_rome_vectors(
                model, tokenizer, t["s"], t["r"], new_o,
                edit_layer, device, kg_corpus,
            )

            # Key overlap: k_direction (d_mlp=2048) vs entity mid-layer activations (d_mlp=2048)
            k_dir = rome_data["k_direction"]  # [d_mlp=2048]
            k_dir_norm = F.normalize(k_dir.unsqueeze(0), dim=1)  # [1, d_mlp]
            overlaps = (key_matrix_norm @ k_dir_norm.T).squeeze()  # [N]

            # Value alignment: v_direction (d_model=512) vs entity embeddings
            v_dir = rome_data["v_direction"]  # [d_model=512]
            with torch.no_grad():
                e_old = model.token_embed.weight[tokenizer.get_id(t["o"])].cpu()
                e_new = model.token_embed.weight[tokenizer.get_id(new_o)].cpu()
                e_subj = model.token_embed.weight[tokenizer.get_id(t["s"])].cpu()

            v_dir_norm = F.normalize(v_dir.unsqueeze(0), dim=1)
            cos_v_e_new = F.cosine_similarity(v_dir_norm, F.normalize(e_new.unsqueeze(0), dim=1)).item()
            cos_v_e_old = F.cosine_similarity(v_dir_norm, F.normalize(e_old.unsqueeze(0), dim=1)).item()
            cos_v_diff = F.cosine_similarity(v_dir_norm, F.normalize((e_new - e_old).unsqueeze(0), dim=1)).item()

            # Key overlap statistics by degree
            overlap_by_degree = {"low": [], "mid": [], "high": []}
            for ent, ov in zip(ent_list, overlaps.tolist()):
                d = degree.get(ent, 0)
                if d < 15:
                    overlap_by_degree["low"].append(abs(ov))
                elif d < 35:
                    overlap_by_degree["mid"].append(abs(ov))
                else:
                    overlap_by_degree["high"].append(abs(ov))

            result = {
                "s": t["s"],
                "r": t["r"],
                "o_old": t["o"],
                "o_new": new_o,
                "degree_s": degree.get(t["s"], 0),
                "edit_success": rome_data["edit_success"],
                "delta_W_fro": rome_data["delta_W_fro"],
                "sigma_top1": rome_data["sigma"],
                "effective_rank": rome_data["effective_rank"],
                "singular_values_top5": rome_data["singular_values"][:5],
                # Key overlap stats (d_mlp space)
                "key_overlap_mean": float(overlaps.abs().mean()),
                "key_overlap_max": float(overlaps.abs().max()),
                "key_overlap_std": float(overlaps.abs().std()),
                "key_overlap_by_victim_degree": {
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                    for k, v in overlap_by_degree.items() if v
                },
                # Value alignment (d_model space)
                "cos_v_e_new": cos_v_e_new,
                "cos_v_e_old": cos_v_e_old,
                "cos_v_diff": cos_v_diff,
                # Top-N most overlapping entities
                "top_overlap_entities": [
                    {"entity": ent_list[i], "overlap": float(overlaps[i]), "degree": degree.get(ent_list[i], 0)}
                    for i in overlaps.abs().topk(10).indices.tolist()
                ],
            }
            bin_results.append(result)

        all_results[bin_name] = bin_results
        print(f"  Completed {len(bin_results)} edits for {bin_name}")

    # Save raw results
    with open(out_dir / "rome_geometry_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # === Analysis ===
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    for bin_name in ["low", "mid", "high"]:
        results = all_results.get(bin_name, [])
        if not results:
            continue
        print(f"\n--- {bin_name} (n={len(results)}) ---")

        fros = [r["delta_W_fro"] for r in results]
        sigmas = [r["sigma_top1"] for r in results]
        eff_ranks = [r["effective_rank"] for r in results]
        overlaps = [r["key_overlap_mean"] for r in results]
        max_overlaps = [r["key_overlap_max"] for r in results]

        print(f"  ||ΔW||_F: mean={np.mean(fros):.2f}, std={np.std(fros):.2f}")
        print(f"  σ_1: mean={np.mean(sigmas):.2f}, std={np.std(sigmas):.2f}")
        print(f"  Effective rank: mean={np.mean(eff_ranks):.2f}")
        print(f"  Key overlap (mean |cos|): mean={np.mean(overlaps):.4f}")
        print(f"  Key overlap (max |cos|): mean={np.mean(max_overlaps):.4f}")

        # Key overlap by victim degree
        for vd in ["low", "mid", "high"]:
            vals = [r["key_overlap_by_victim_degree"].get(vd, {}).get("mean", 0) for r in results]
            if vals:
                print(f"  Key overlap with {vd}-degree entities: {np.mean(vals):.4f}")

    # === Visualization ===
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:red"}

    # Plot 1: ||ΔW||_F distribution
    for bin_name in ["low", "mid", "high"]:
        results = all_results.get(bin_name, [])
        fros = [r["delta_W_fro"] for r in results]
        axes[0, 0].hist(fros, alpha=0.5, label=bin_name, color=colors[bin_name], bins=15)
    axes[0, 0].set_xlabel("||ΔW||_F")
    axes[0, 0].set_title("Update Magnitude")
    axes[0, 0].legend()

    # Plot 2: Effective rank
    for bin_name in ["low", "mid", "high"]:
        results = all_results.get(bin_name, [])
        ranks = [r["effective_rank"] for r in results]
        axes[0, 1].hist(ranks, alpha=0.5, label=bin_name, color=colors[bin_name], bins=15)
    axes[0, 1].set_xlabel("Effective Rank")
    axes[0, 1].set_title("Update Effective Rank")
    axes[0, 1].legend()

    # Plot 3: Mean key overlap
    for bin_name in ["low", "mid", "high"]:
        results = all_results.get(bin_name, [])
        overlaps = [r["key_overlap_mean"] for r in results]
        degrees = [r["degree_s"] for r in results]
        axes[0, 2].scatter(degrees, overlaps, alpha=0.5, label=bin_name, color=colors[bin_name])
    axes[0, 2].set_xlabel("Subject Degree")
    axes[0, 2].set_ylabel("Mean |cos(k_star, entity_key)|")
    axes[0, 2].set_title("Key Overlap vs Subject Degree")
    axes[0, 2].legend()

    # Plot 4: Key overlap by victim degree
    x_pos = [0, 1, 2]
    width = 0.25
    for i, bin_name in enumerate(["low", "mid", "high"]):
        results = all_results.get(bin_name, [])
        means = []
        for vd in ["low", "mid", "high"]:
            vals = [r["key_overlap_by_victim_degree"].get(vd, {}).get("mean", 0) for r in results]
            means.append(np.mean(vals))
        axes[1, 0].bar([x + (i - 1) * width for x in x_pos], means, width, label=f"editor={bin_name}", color=colors[bin_name])
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(["victim=low", "victim=mid", "victim=high"])
    axes[1, 0].set_ylabel("Mean |cos(k_star, victim_key)|")
    axes[1, 0].set_title("Key Overlap: Editor × Victim Degree")
    axes[1, 0].legend()

    # Plot 5: Degree vs ||ΔW||_F scatter
    all_degrees = []
    all_fros = []
    for bin_name in ["low", "mid", "high"]:
        for r in all_results.get(bin_name, []):
            all_degrees.append(r["degree_s"])
            all_fros.append(r["delta_W_fro"])
    axes[1, 1].scatter(all_degrees, all_fros, alpha=0.5, s=20)
    axes[1, 1].set_xlabel("Subject Degree")
    axes[1, 1].set_ylabel("||ΔW||_F")
    axes[1, 1].set_title("Degree vs Update Magnitude")

    # Plot 6: Singular value spectrum for each bin
    for bin_name in ["low", "mid", "high"]:
        results = all_results.get(bin_name, [])
        svs = np.array([r["singular_values_top5"] for r in results])
        mean_svs = svs.mean(axis=0)
        axes[1, 2].plot(range(1, len(mean_svs) + 1), mean_svs, "o-", label=bin_name, color=colors[bin_name])
    axes[1, 2].set_xlabel("Singular Value Index")
    axes[1, 2].set_ylabel("Singular Value")
    axes[1, 2].set_title("Mean SV Spectrum of ΔW")
    axes[1, 2].legend()

    fig.suptitle("ROME Update Geometry: Degree-Conditional Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "rome_geometry_plots.png", dpi=150)
    plt.close()

    print(f"\nAll results saved to {out_dir}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/models/gpt_small")
    parser.add_argument("--kg-corpus", default="data/kg/ba/corpus.train.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="outputs/exp4_rome_geometry")
    parser.add_argument("--num-edits-per-bin", type=int, default=30)
    parser.add_argument("--edit-layer", type=int, default=5)
    args = parser.parse_args()

    run_rome_geometry(
        model_dir=args.model_dir,
        kg_corpus=args.kg_corpus,
        device=args.device,
        output_dir=args.output_dir,
        num_edits_per_bin=args.num_edits_per_bin,
        edit_layer=args.edit_layer,
    )
