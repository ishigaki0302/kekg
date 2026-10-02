#!/usr/bin/env python3
"""全 (world x size) の L0 表現特徴(gini/l2)を生成する resume 付きドライバ。

既存の compute_repr_features.py は WORLDS/SIZES がハードコード(8通り)で resume なし。
本スクリプトはそれを触らず、24 worlds x 8 sizes = 192 respondent を対象に、
- 既に出力済みの repr csv はスキップ(resume)
- --shard/--num-shards で複数GPUに分割(CUDA_VISIBLE_DEVICES と併用)
して媒介分析(mediation_analysis.py)の入力を揃える。
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.eval.plasticity_eval import load_respondent, rebuild_world
from src.scripts.compute_repr_features import entity_features

SIZES = ["tiny", "xs", "small", "small-wide", "base", "base-wide", "large", "xl"]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 42]
TOPOS = ["ba", "er", "ring"]
WORLD_KW = dict(num_entities=1200, num_generic_relations=50,
                target_generic_triples=24000, ba_m=6)


def all_combos():
    combos = []
    for topo in TOPOS:
        for seed in SEEDS:
            wid = f"{topo}_s{seed}"
            for size in SIZES:
                combos.append((wid, topo, seed, size))
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/plasticity/repr")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    combos = all_combos()
    todo = [c for i, c in enumerate(combos) if i % args.num_shards == args.shard]
    done = skipped = failed = 0

    for wid, topo, seed, size in todo:
        fp = out / f"{wid}__{size}_repr.csv"
        if fp.exists() and not args.overwrite:
            skipped += 1
            continue
        mdir = f"outputs/respondents/models/{wid}__{size}"
        cfg = f"outputs/respondents/configs/{wid}__{size}.yaml"
        if not Path(mdir).exists() or not Path(cfg).exists():
            print(f"[skip-missing] {wid}__{size}", flush=True)
            failed += 1
            continue
        try:
            world = rebuild_world(dict(seed=seed, topology=topo, **WORLD_KW))
            model, tok = load_respondent(mdir, cfg, device)
            ginis, l2s = entity_features(model, tok, world.entities, args.layer, device)
            with fp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["entity", "degree", "gini", "l2"])
                for e, g, l in zip(world.entities, ginis, l2s):
                    w.writerow([e, world.degree(e), f"{g:.5f}", f"{l:.4f}"])
            cg = np.corrcoef([world.degree(e) for e in world.entities], ginis)[0, 1]
            print(f"[ok] {wid}__{size}: corr(deg,gini)={cg:+.3f} -> {fp}", flush=True)
            done += 1
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {wid}__{size}: {e}", flush=True)
            failed += 1

    print(f"[shard {args.shard}/{args.num_shards}] done={done} skipped={skipped} failed={failed}",
          flush=True)


if __name__ == "__main__":
    main()
