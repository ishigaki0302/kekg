#!/usr/bin/env python3
"""
BA（Barabási–Albert）構造の Knowledge Graph を生成するスクリプト（エイリアス無し版）．

- 目的：トリプル数を「約3万件（デフォルト30000）」に揃えつつ，
        エンティティ名・リレーション名は “言い換え（__a0 等）無し” にする．
- 重要：有向グラフ（directed）として出力する．
        ※以前のように subject 側だけ別語彙（alias）になると，in-only / out-only に分裂して hop が 1 で止まりがち．
        ここでは同一語彙（E_***, R_**）のみを使うため，中継ノードが自然に生まれて multi-hop になりやすい．

出力形式：
- txt: 1行1トリプル "S R O"（training用 corpus と同形式）
- jsonl: {"s":..., "r":..., "o":...}

依存：
  pip install networkx numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set
from pathlib import Path
import argparse
import json
import random

import numpy as np
import networkx as nx


@dataclass(frozen=True)
class Triple:
    s: str
    r: str
    o: str

    def to_dict(self) -> dict:
        return {"s": self.s, "r": self.r, "o": self.o}


def _solve_m_for_target(n: int, target_directed_edges: int) -> int:
    """
    nx.barabasi_albert_graph(n, m) の undirected edge 数は
      E_und = m*(m-1)/2 + (n-m)*m
    これを bidirectional にすると directed edge 数は
      E_dir = 2*E_und = 2*n*m - m*m - m
    target に近い m を二次方程式で近似して返す．
    """
    # m^2 - (2n-1)m + target = 0  を解く（近似）
    # ただし上の式は E_dir = 2*n*m - m^2 - m
    # => m^2 - (2n-1)m + target = 0
    a = 1
    b = -(2 * n - 1)
    c = target_directed_edges

    disc = b * b - 4 * a * c
    if disc <= 0:
        return max(1, min(n - 1, int(target_directed_edges / max(1, 2 * n))))

    m1 = (-(b) - (disc ** 0.5)) / (2 * a)
    m = int(round(m1))
    return max(1, min(n - 1, m))


def generate_ba_kg_no_alias(
    num_entities: int,
    num_relations: int,
    target_triples: int = 30000,
    seed: int = 42,
    directed_mode: str = "one_way",  # "one_way" or "bidirectional"
    ensure_exact: bool = True,
) -> List[Triple]:
    """
    BAグラフからエイリアス無しのトリプル集合を生成する．

    directed_mode:
      - "one_way": undirected edge ごとに方向をランダムに1つだけ付与（= 1 triple / edge）
      - "bidirectional": 両方向を付与（= 2 triples / edge）

    ensure_exact:
      - True: target_triples に一致するまで間引き/追加を行う
      - False: BA由来の本数のまま出力（target は m 推定のみに利用）
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    entities = [f"E_{i:04d}" for i in range(num_entities)]
    relations = [f"R_{i:03d}" for i in range(num_relations)]

    # target を満たしやすい m を推定（bidirectional 前提の近似）
    # one_way の場合は directed edge が半分になるので target*2 を入れて m を少し大きめに推定
    if directed_mode == "one_way":
        m_attach = _solve_m_for_target(num_entities, target_triples * 2)
    elif directed_mode == "bidirectional":
        m_attach = _solve_m_for_target(num_entities, target_triples)
    else:
        raise ValueError(f"Unknown directed_mode: {directed_mode}")

    # BA（undirected）
    G = nx.barabasi_albert_graph(num_entities, m_attach, seed=seed)
    und_edges = list(G.edges())

    # 有向化（同一語彙のまま）
    directed_edges: List[Tuple[int, int]] = []
    if directed_mode == "one_way":
        for u, v in und_edges:
            if py_rng.random() < 0.5:
                directed_edges.append((u, v))
            else:
                directed_edges.append((v, u))
    else:  # bidirectional
        for u, v in und_edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))

    # Triple 化（relation はランダム割当）
    triples: List[Triple] = []
    for u, v in directed_edges:
        r = relations[int(rng.integers(0, num_relations))]
        triples.append(Triple(s=entities[u], r=r, o=entities[v]))

    if not ensure_exact:
        return triples

    # --- target_triples に揃える（間引き or 追加）---
    if len(triples) > target_triples:
        py_rng.shuffle(triples)
        triples = triples[:target_triples]
        return triples

    # 足りない場合：preferential に追加（degree proportional sampling）
    # 既存のエッジ集合を管理して重複を避ける
    edge_set: Set[Tuple[str, str, str]] = set((t.s, t.r, t.o) for t in triples)

    # degree を（現在の有向エッジ）から作る：preferential sampling に使う
    out_deg = np.zeros(num_entities, dtype=np.int64)
    in_deg = np.zeros(num_entities, dtype=np.int64)
    for t in triples:
        u = int(t.s.split("_")[1])
        v = int(t.o.split("_")[1])
        out_deg[u] += 1
        in_deg[v] += 1
    deg = out_deg + in_deg
    deg = deg.astype(np.float64) + 1.0  # ゼロ割防止のスムージング
    prob = deg / deg.sum()

    def sample_node() -> int:
        return int(rng.choice(num_entities, p=prob))

    # 追加ループ
    max_trials = 5_000_000
    trials = 0
    while len(triples) < target_triples and trials < max_trials:
        trials += 1
        u = sample_node()
        v = sample_node()
        if u == v:
            continue

        r = relations[int(rng.integers(0, num_relations))]
        s = entities[u]
        o = entities[v]
        key = (s, r, o)
        if key in edge_set:
            continue

        triples.append(Triple(s=s, r=r, o=o))
        edge_set.add(key)

        # degree 更新（prob も更新したいがコストが高いので，ある程度の近似で固定）
        out_deg[u] += 1
        in_deg[v] += 1

    if len(triples) < target_triples:
        raise RuntimeError(
            f"Could not reach target_triples={target_triples}. "
            f"got={len(triples)} after trials={trials}. "
            "Try increasing num_entities or num_relations, or use bidirectional mode."
        )

    return triples


def write_txt(triples: List[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(f"{t.s} {t.r} {t.o}\n")


def write_jsonl(triples: List[Triple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-entities", type=int, default=1200)
    ap.add_argument("--num-relations", type=int, default=250)
    ap.add_argument("--target-triples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--directed-mode", type=str, default="one_way", choices=["one_way", "bidirectional"])
    ap.add_argument("--out-txt", type=str, default="data/kg/ba_no_alias/corpus.train.txt")
    ap.add_argument("--out-jsonl", type=str, default="data/kg/ba_no_alias/graph.jsonl")
    ap.add_argument("--no-exact", action="store_true", help="If set, do not force exact target_triples.")
    args = ap.parse_args()

    triples = generate_ba_kg_no_alias(
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        target_triples=args.target_triples,
        seed=args.seed,
        directed_mode=args.directed_mode,
        ensure_exact=(not args.no_exact),
    )

    out_txt = Path(args.out_txt)
    out_jsonl = Path(args.out_jsonl)
    write_txt(triples, out_txt)
    write_jsonl(triples, out_jsonl)

    # 簡易サマリ
    ents = {t.s for t in triples} | {t.o for t in triples}
    rels = {t.r for t in triples}
    print("=== BA KG (no alias) generated ===")
    print(f"triples: {len(triples)}")
    print(f"entities(observed): {len(ents)} (requested: {args.num_entities})")
    print(f"relations(observed): {len(rels)} (requested: {args.num_relations})")
    print(f"directed_mode: {args.directed_mode}")
    print(f"out_txt: {out_txt}")
    print(f"out_jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()