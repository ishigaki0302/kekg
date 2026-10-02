#!/usr/bin/env python3
"""Generality(言い換え) アーム: 既存の 24 worlds x 8 sizes x 9 methods を、
R_F の alias(言い換え surface form)付き world で再学習・評価する。

- モデルは alias 事実も学習するので Generality(言い換え編集) を評価できる。
- 出力は全て `*_alias` ツリーに書き、**完了済みの base matrix を一切上書きしない**。
- 既存 run_respondent_matrix.py の部品(gen_worlds/write_configs/train_jobs/
  eval_jobs/gpu_pool)をモジュール globals 差し替えで再利用(コア資産は無改変)。
resume: 既存 model.pt / respondent csv はスキップ。
"""
import argparse
import csv
import glob

import src.scripts.run_respondent_matrix as M


def concat_alias():
    files = sorted(glob.glob(str(M.RESP_DIR / "*.csv")))
    out = M.ROOT / "outputs/plasticity/responses_matrix_alias.csv"
    rows, header = [], None
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            rows.extend(list(r))
    if header:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print(f"[concat-alias] {len(files)} files, {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "worlds", "train", "eval", "concat"])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--n-aliases", type=int, default=3)
    args = ap.parse_args()

    root = M.ROOT
    # 出力ツリーを alias 用に差し替え(base を汚さない)
    M.WORLD_DIR = root / "outputs/symbolic_alias"
    M.CFG_DIR = root / "outputs/respondents_alias/configs"
    M.MODEL_DIR = root / "outputs/respondents_alias/models"
    M.RESP_DIR = root / "outputs/plasticity/matrix_alias"
    M.LOG_DIR = root / "outputs/respondents_alias/logs"
    M.EDITOR_DIR = root / "outputs/respondents_alias/editors"
    for d in (M.WORLD_DIR, M.CFG_DIR, M.MODEL_DIR, M.RESP_DIR, M.LOG_DIR, M.EDITOR_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # world 生成に alias を注入
    M.WORLD_KW = dict(M.WORLD_KW, num_rf_aliases=args.n_aliases)

    gpus = [int(x) for x in args.gpus.split(",")] * args.slots_per_gpu
    print(f"[alias] n_aliases={args.n_aliases} gpus={gpus} out={M.MODEL_DIR}", flush=True)

    if args.phase in ("all", "worlds"):
        print("=== alias PHASE: worlds ===", flush=True)
        M.gen_worlds(); M.write_configs()
    if args.phase in ("all", "train"):
        print("=== alias PHASE: train ===", flush=True)
        M.gpu_pool(M.train_jobs(), gpus)
    if args.phase in ("all", "eval"):
        print("=== alias PHASE: eval ===", flush=True)
        M.gpu_pool(M.eval_jobs(), gpus)
    if args.phase in ("all", "concat"):
        print("=== alias PHASE: concat ===", flush=True)
        concat_alias()
    print("=== alias orchestrator done ===", flush=True)


if __name__ == "__main__":
    main()
