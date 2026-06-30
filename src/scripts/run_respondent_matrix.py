#!/usr/bin/env python3
"""Build the respondent matrix for explanatory IRT, in parallel across GPUs.

Factorial: worlds (topology x seed) x model sizes x editing methods.
Phases:
  worlds : generate symbolic worlds (CPU, in-process)
  train  : train one respondent model per (world, size)   [GPU pool]
  eval   : run plasticity eval per (world, size, method)   [GPU pool]
  concat : merge all response CSVs into one matrix

Each (world, size) is trained once; methods reuse the same base model.
For IRT: respondents = (world, size, method); items are shared within a world.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.kg.symbolic_world import build_symbolic_world, world_to_triples
from src.kg.generator import write_txt, write_jsonl
from src.utils import save_yaml
import json

ROOT = Path(__file__).parent.parent.parent
PY = sys.executable

# 24 worlds = 3 topologies x 8 seeds. (ba_s1/ba_s2/ba_s42/er_s42 are kept so the
# already-computed respondents are reused via the resume/skip logic.)
_SEEDS = [1, 2, 3, 4, 5, 6, 42, 7]
WORLDS = [
    (f"{topo}_s{seed}", topo, seed)
    for topo in ("ba", "er", "ring")
    for seed in _SEEDS
]
SIZES = {  # 8 sizes (depth x width); small/base identical to existing for reuse
    "tiny": dict(n_layers=4, n_heads=4, d_model=128, d_mlp=512, max_seq_len=8, dropout=0.1),
    "xs": dict(n_layers=4, n_heads=8, d_model=256, d_mlp=1024, max_seq_len=8, dropout=0.1),
    "small": dict(n_layers=6, n_heads=8, d_model=256, d_mlp=1024, max_seq_len=8, dropout=0.1),
    "small-wide": dict(n_layers=6, n_heads=8, d_model=512, d_mlp=2048, max_seq_len=8, dropout=0.1),
    "base": dict(n_layers=12, n_heads=8, d_model=512, d_mlp=2048, max_seq_len=8, dropout=0.1),
    "base-wide": dict(n_layers=12, n_heads=12, d_model=768, d_mlp=3072, max_seq_len=8, dropout=0.1),
    "large": dict(n_layers=18, n_heads=10, d_model=640, d_mlp=2560, max_seq_len=8, dropout=0.1),
    "xl": dict(n_layers=24, n_heads=12, d_model=768, d_mlp=3072, max_seq_len=8, dropout=0.1),
}
METHODS = ["rome", "ft", "ft_all", "memit", "alphaedit", "grace", "kn", "pmet"]

WORLD_DIR = ROOT / "outputs/symbolic"
CFG_DIR = ROOT / "outputs/respondents/configs"
MODEL_DIR = ROOT / "outputs/respondents/models"
RESP_DIR = ROOT / "outputs/plasticity/matrix"
LOG_DIR = ROOT / "outputs/respondents/logs"
for d in (CFG_DIR, MODEL_DIR, RESP_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

WORLD_KW = dict(num_entities=1200, num_generic_relations=50,
                target_generic_triples=24000, ba_m=6)


def gen_worlds():
    for wid, topo, seed in WORLDS:
        out = WORLD_DIR / wid
        out.mkdir(parents=True, exist_ok=True)
        world = build_symbolic_world(seed=seed, topology=topo, **WORLD_KW)
        write_txt(world_to_triples(world), out / "corpus.train.txt")
        json.dump(world.func_map, open(out / "func_map.json", "w"), ensure_ascii=False)
        import numpy as np
        degs = np.array([world.degree(e) for e in world.entities])
        print(f"[world] {wid} ({topo}, seed={seed}) facts={len(world_to_triples(world))} "
              f"deg mean={degs.mean():.1f} std={degs.std():.1f}")


def write_configs():
    for wid, topo, seed in WORLDS:
        for size, mcfg in SIZES.items():
            cfg = {
                "model": mcfg,
                "train": dict(batch_size=256, lr=3.0e-4, weight_decay=0.01,
                              epochs=100, warmup_steps=1000, grad_clip=1.0,
                              eval_interval=500, save_interval=2000),
                "data": {
                    "train_path": str(WORLD_DIR / wid / "corpus.train.txt"),
                    "eval_all_path": str(WORLD_DIR / wid / "corpus.train.txt"),
                },
                "output_dir": str(MODEL_DIR / f"{wid}__{size}"),
                "seed": seed,
            }
            save_yaml(cfg, CFG_DIR / f"{wid}__{size}.yaml")


def gpu_pool(jobs, gpus):
    """jobs: list of (name, argv). Run with <=len(gpus) concurrent, pinned."""
    free = list(gpus)
    running = {}  # gpu -> (proc, name, logfile)
    pending = list(jobs)
    failed = []
    while pending or running:
        while free and pending:
            gpu = free.pop()
            name, argv = pending.pop(0)
            logf = open(LOG_DIR / f"{name}.log", "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            print(f"[launch gpu{gpu}] {name}")
            running[gpu] = (subprocess.Popen(argv, env=env, stdout=logf,
                                             stderr=subprocess.STDOUT, cwd=str(ROOT)),
                            name, logf)
        time.sleep(3)
        for gpu, (p, name, logf) in list(running.items()):
            if p.poll() is not None:
                logf.close()
                rc = p.returncode
                print(f"[done gpu{gpu}] {name} rc={rc}")
                if rc != 0:
                    failed.append(name)
                del running[gpu]
                free.append(gpu)
    if failed:
        print(f"[WARN] failed jobs: {failed}")
    return failed


def train_jobs():
    jobs = []
    for wid, topo, seed in WORLDS:
        for size in SIZES:
            if (MODEL_DIR / f"{wid}__{size}" / "model.pt").exists():
                continue  # skip already-trained models (resume)
            cfg = CFG_DIR / f"{wid}__{size}.yaml"
            argv = [PY, "src/cli/train_lm.py", "--config", str(cfg)]
            jobs.append((f"train__{wid}__{size}", argv))
    return jobs


def eval_jobs():
    jobs = []
    for wid, topo, seed in WORLDS:
        for size in SIZES:
            mdir = MODEL_DIR / f"{wid}__{size}"
            cfg = CFG_DIR / f"{wid}__{size}.yaml"
            corpus = WORLD_DIR / wid / "corpus.train.txt"
            fmap = WORLD_DIR / wid / "func_map.json"
            for method in METHODS:
                rid = f"{wid}__{size}__{method}"
                if (RESP_DIR / f"{rid}.csv").exists():
                    continue  # skip already-computed respondents (resume)
                argv = [
                    PY, "src/scripts/run_plasticity_eval.py",
                    "--model-dir", str(mdir), "--config", str(cfg),
                    "--corpus", str(corpus), "--func-map", str(fmap),
                    "--layer", "0", "--n-per-bin", "20", "--max-invariant", "20",
                    "--topology", topo, "--world-seed", str(seed),
                    "--method", method, "--respondent-id", rid,
                    "--out", str(RESP_DIR / f"{rid}.csv"),
                ]
                jobs.append((f"eval__{rid}", argv))
    return jobs


def concat():
    files = sorted(glob.glob(str(RESP_DIR / "*.csv")))
    out = ROOT / "outputs/plasticity/responses_matrix.csv"
    rows = []
    header = None
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            r = csv.reader(f)
            h = next(r)
            header = h
            for line in r:
                rows.append(line)
    if header:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print(f"[concat] {len(files)} files, {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "worlds", "train", "eval", "concat"])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=1,
                    help="concurrent jobs per GPU (models are tiny; pack several)")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")] * args.slots_per_gpu

    if args.phase in ("all", "worlds"):
        print("=== PHASE: worlds ==="); gen_worlds(); write_configs()
    if args.phase in ("all", "train"):
        print("=== PHASE: train ==="); gpu_pool(train_jobs(), gpus)
    if args.phase in ("all", "eval"):
        print("=== PHASE: eval ==="); gpu_pool(eval_jobs(), gpus)
    if args.phase in ("all", "concat"):
        print("=== PHASE: concat ==="); concat()
    print("=== orchestrator done ===")


if __name__ == "__main__":
    main()
