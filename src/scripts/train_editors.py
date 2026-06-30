#!/usr/bin/env python3
"""Train learned editors (MEND / KnowledgeEditor) per base model, in parallel.

Learned editors are model-specific, so one editor is trained per (world, size).
Outputs: outputs/respondents/editors/<world>__<size>__<method>.pt
Resume-aware (skips existing). Designed to be launched as a GPU pool like the
respondent matrix orchestrator.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ROOT = Path(__file__).parent.parent.parent
PY = sys.executable
ED_DIR = ROOT / "outputs/respondents/editors"
LOG_DIR = ROOT / "outputs/respondents/logs"
ED_DIR.mkdir(parents=True, exist_ok=True)

# import the world list / sizes from the matrix orchestrator (single source)
import importlib.util
spec = importlib.util.spec_from_file_location("rm", ROOT / "src/scripts/run_respondent_matrix.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
WORLDS, SIZES = rm.WORLDS, list(rm.SIZES)
WORLD_KW = rm.WORLD_KW


def train_one(wid, topo, seed, size, method, steps):
    """Worker (separate process): train one editor and save it."""
    import torch
    from src.eval.plasticity_eval import load_respondent, rebuild_world
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = ROOT / f"outputs/respondents/models/{wid}__{size}"
    cfg = ROOT / f"outputs/respondents/configs/{wid}__{size}.yaml"
    model, tok = load_respondent(str(mdir), str(cfg), dev)
    world = rebuild_world(dict(seed=seed, topology=topo, **WORLD_KW))
    if method == "mend":
        from src.edit.mend_edit import MENDEditor
        ed = MENDEditor(model, tok, device=dev, default_layer=0)
    elif method == "ke":
        from src.edit.ke_edit import KEEditor
        ed = KEEditor(model, tok, device=dev, default_layer=0)
    else:
        raise ValueError(method)
    ed.train_editor(world, steps=steps)
    ed.save(str(ED_DIR / f"{wid}__{size}__{method}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="mend")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--worker", default="")  # internal: "wid,topo,seed,size,method"
    args = ap.parse_args()

    if args.worker:  # run a single training (invoked as subprocess)
        wid, topo, seed, size, method = args.worker.split(",")
        train_one(wid, topo, int(seed), size, method, args.steps)
        return

    methods = args.methods.split(",")
    jobs = []
    for wid, topo, seed in WORLDS:
        for size in SIZES:
            for method in methods:
                if (ED_DIR / f"{wid}__{size}__{method}.pt").exists():
                    continue
                if not (ROOT / f"outputs/respondents/models/{wid}__{size}/model.pt").exists():
                    continue  # base model not trained yet
                name = f"editor__{wid}__{size}__{method}"
                argv = [PY, "src/scripts/train_editors.py", "--steps", str(args.steps),
                        "--worker", f"{wid},{topo},{seed},{size},{method}"]
                jobs.append((name, argv))
    print(f"editor jobs: {len(jobs)}")

    gpus = [int(x) for x in args.gpus.split(",")] * args.slots_per_gpu
    free = list(gpus)
    running = {}
    pending = list(jobs)
    while pending or running:
        while free and pending:
            g = free.pop()
            name, argv = pending.pop(0)
            lf = open(LOG_DIR / f"{name}.log", "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            print(f"[launch gpu{g}] {name}")
            running[g] = (subprocess.Popen(argv, env=env, stdout=lf,
                                           stderr=subprocess.STDOUT, cwd=str(ROOT)), name, lf)
        time.sleep(3)
        for g, (p, name, lf) in list(running.items()):
            if p.poll() is not None:
                lf.close()
                print(f"[done gpu{g}] {name} rc={p.returncode}")
                del running[g]
                free.append(g)
    print("editor training done")


if __name__ == "__main__":
    main()
