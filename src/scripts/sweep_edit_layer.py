#!/usr/bin/env python3
"""Find a ROME layer / #grad-steps with usable edit success on this model.

L5 gave 0% edit success (target prob stays ~1e-4). Sweep layers and v grad
steps on a few edits to locate a setting where ROME actually flips argmax.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.edit.rome import ROME
from src.eval.plasticity_eval import load_respondent, rebuild_world, sample_edits, R_F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/models/symbolic_main_seed42")
    ap.add_argument("--config", default="configs/train_symbolic_main.yaml")
    ap.add_argument("--corpus", default="outputs/symbolic/main/corpus.train.txt")
    ap.add_argument("--func-map", default="outputs/symbolic/main/func_map.json")
    ap.add_argument("--layers", default="0,1,2,3,5")
    ap.add_argument("--grad-steps", default="25,100")
    ap.add_argument("--n-edits", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_respondent(args.model_dir, args.config, device)
    world = rebuild_world(
        dict(num_entities=1200, num_generic_relations=50,
             target_generic_triples=24000, ba_m=6, seed=42),
        func_map_path=args.func_map,
    )
    plans = sample_edits(world, n_per_bin=max(1, args.n_edits // 3), seed=0)[: args.n_edits]
    print(f"edits: {[p.s for p in plans]}")

    layers = [int(x) for x in args.layers.split(",")]
    grad_steps = [int(x) for x in args.grad_steps.split(",")]

    print(f"{'gsteps':>6} {'layer':>5} {'succ':>5}  detail")
    for gs in grad_steps:
        rome = ROME(model, tok, device=device, kg_corpus_path=args.corpus,
                    v_num_grad_steps=gs)
        for L in layers:
            n_ok = 0
            details = []
            for p in plans:
                _, res = rome.apply_edit(p.s, R_F, p.o_new, layer=L, copy_model=True)
                n_ok += int(res.success)
                details.append(f"{p.degree_bin}:{'Y' if res.success else 'n'}"
                               f"({res.new_prediction}->want {p.o_new})")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            print(f"{gs:>6} {L:>5} {n_ok}/{len(plans)}  " + " ".join(details))


if __name__ == "__main__":
    main()
