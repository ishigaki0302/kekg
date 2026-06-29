#!/usr/bin/env python3
"""Per-entity internal-representation concentration at the edit layer (L0).

For each respondent model, run "E_i R_F" through the model and capture the
post-GELU FFN activation (dim d_mlp) at the subject position. Compute
concentration features per entity:
  - gini : Gini coefficient of |activation| (high = concentrated representation)
  - l2   : L2 norm of the activation

These are the candidate MEDIATORS for the structure -> plasticity path
(KG degree -> representation concentration -> editing difficulty), used to
answer the circularity critique (difficulty explained by learned internal
representation, not merely by the generating graph).
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.eval.plasticity_eval import load_respondent, rebuild_world
from src.kg.symbolic_world import R_F

WORLDS = {"ba_s1": ("ba", 1), "ba_s2": ("ba", 2), "ba_s42": ("ba", 42), "er_s42": ("er", 42)}
SIZES = ["base", "small"]
WORLD_KW = dict(num_entities=1200, num_generic_relations=50,
                target_generic_triples=24000, ba_m=6)


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    cum = np.cumsum(x)
    if cum[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


@torch.no_grad()
def entity_features(model, tok, entities, layer, device, batch=256):
    cap = {}
    h = model.blocks[layer].ffn.w1.register_forward_hook(
        lambda m, i, o: cap.__setitem__("a", o)
    )
    ginis, l2s = [], []
    try:
        for s in range(0, len(entities), batch):
            chunk = entities[s:s + batch]
            ids = torch.tensor([tok.encode(f"{e} {R_F}") for e in chunk], device=device)
            model(ids)
            act = F.gelu(cap["a"])[:, 0, :].float().cpu().numpy()  # subject pos
            for row in act:
                ginis.append(gini(row))
                l2s.append(float(np.linalg.norm(row)))
    finally:
        h.remove()
    return ginis, l2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/plasticity/repr")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for wid, (topo, seed) in WORLDS.items():
        world = rebuild_world(dict(seed=seed, topology=topo, **WORLD_KW))
        for size in SIZES:
            mdir = f"outputs/respondents/models/{wid}__{size}"
            cfg = f"outputs/respondents/configs/{wid}__{size}.yaml"
            model, tok = load_respondent(mdir, cfg, device)
            ginis, l2s = entity_features(model, tok, world.entities, args.layer, device)
            fp = out / f"{wid}__{size}_repr.csv"
            with fp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["entity", "degree", "gini", "l2"])
                for e, g, l in zip(world.entities, ginis, l2s):
                    w.writerow([e, world.degree(e), f"{g:.5f}", f"{l:.4f}"])
            cg = np.corrcoef([world.degree(e) for e in world.entities], ginis)[0, 1]
            print(f"{wid}__{size}: corr(degree,gini)={cg:+.3f} -> {fp}")


if __name__ == "__main__":
    main()
