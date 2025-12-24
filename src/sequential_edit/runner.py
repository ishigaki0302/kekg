"""Main runner for sequential editing experiments.

This module implements the core logic for performing sequential knowledge
edits using ROME and logging the resulting metrics and ripple effects.
"""

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np

from .config import SeqEditConfig
from .kg_utils import KG, Triple
from .ripple_logging import (
    compute_ripple_records,
    write_ripple_jsonl,
    compute_triple_accuracy_records,
)

# Import existing modules
from src.modeling.gpt_mini import GPTMini, GPTConfig
from src.modeling.tokenizer import SROTokenizer
from src.edit.rome import ROME
from src.utils.io import load_yaml

def select_eval_triples(
    kg: KG,
    config: SeqEditConfig,
    edit_cases: List[Dict[str, Any]],
) -> Tuple[List[Triple], List[Triple]]:
    """Eval用のtripleを選ぶ。

    Returns:
        Tuple of (edited_triples, retain_triples):
            - edited_triples: 編集対象トリプル（edit_casesと同一順序）
            - retain_triples: 編集されないトリプル（保持評価用にサンプル）
    """
    all_triples = list(kg.triples)

    # (s, r, o) -> Triple のインデックス（KG内で一意である前提）
    triple_index: Dict[Tuple[str, str, str], Triple] = {
        (t.s, t.r, t.o): t for t in all_triples
    }

    # edit_cases順に edited_triples を構成（順序ズレを防ぐ）
    edited_triples: List[Triple] = []
    edit_set = set()
    for c in edit_cases:
        key = (c["s"], c["r"], c["o_old"])
        edit_set.add(key)
        t = triple_index.get(key)
        if t is not None:
            edited_triples.append(t)

    # 編集されないトリプル（保持評価用）
    unedited_triples = [t for t in all_triples if (t.s, t.r, t.o) not in edit_set]

    # 保持評価用トリプルをサンプリング（重複なし）
    if len(unedited_triples) > config.num_retain_triples:
        random.seed(config.seed + 1)  # 異なるシードを使用
        retain_triples = random.sample(unedited_triples, config.num_retain_triples)
    else:
        retain_triples = unedited_triples

    return edited_triples, retain_triples
# def select_eval_triples(
#     kg: KG,
#     config: SeqEditConfig,
#     edit_cases: List[Dict[str, Any]],
# ) -> Tuple[List[Triple], List[Triple]]:
#     """Eval用のtripleをモードに応じて選ぶ。

#     Returns:
#         Tuple of (edited_triples, retain_triples):
#             - edited_triples: 編集されるトリプルのリスト（評価用）
#             - retain_triples: 編集されないトリプルのリスト（保持評価用）
#     """
#     # KG 側に全トリプルが格納されている前提
#     all_triples = list(kg.triples)

#     # 編集されるトリプルを抽出
#     edit_set = {(c["s"], c["r"], c["o_old"]) for c in edit_cases}
#     edited_triples = [
#         t for t in all_triples
#         if (t.s, t.r, t.o) in edit_set
#     ]

#     # 編集されないトリプルを抽出
#     unedited_triples = [
#         t for t in all_triples
#         if (t.s, t.r, t.o) not in edit_set
#     ]

#     # 保持評価用トリプルをサンプリング（重複なし）
#     if len(unedited_triples) > config.num_retain_triples:
#         random.seed(config.seed + 1)  # 異なるシードを使用
#         retain_triples = random.sample(unedited_triples, config.num_retain_triples)
#     else:
#         retain_triples = unedited_triples

#     return edited_triples, retain_triples


def load_model_and_tokenizer(
    model_dir: str, device: str = "cuda"
) -> Tuple[GPTMini, SROTokenizer]:
    """Load trained model and tokenizer from directory.

    Args:
        model_dir: Directory containing model.pt and tokenizer.json
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_dir)

    # Load tokenizer
    tokenizer = SROTokenizer.load(model_path / "tokenizer.json")

    # Load model config and weights
    train_report = load_yaml(model_path / "train_report.yaml")
    model_config = train_report["config"]["model"]

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,  # Get vocab_size from tokenizer
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        d_model=model_config["d_model"],
        d_mlp=model_config["d_mlp"],
        max_seq_len=model_config.get("max_seq_len", 8),
        dropout=model_config.get("dropout", 0.1),
    )

    model = GPTMini(gpt_config)
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def extract_rome_layer_weights(model: GPTMini) -> Dict[str, torch.Tensor]:
    """Extract weights from ROME-editable layers (FFN w2 layers).

    Args:
        model: GPTMini model

    Returns:
        Dictionary mapping layer name -> weight tensor (detached copy)
    """
    weights = {}
    for name, param in model.named_parameters():
        if "ffn.w2" in name:
            weights[name] = param.detach().clone()
    return weights


def compute_weight_distance(
    current_weights: Dict[str, torch.Tensor],
    base_weights: Dict[str, torch.Tensor],
) -> float:
    """Compute Frobenius norm distance between current and base weights.

    Args:
        current_weights: Current model weights
        base_weights: Base (original) model weights

    Returns:
        Frobenius norm of weight difference
    """
    total_norm = 0.0
    for name in base_weights:
        if name in current_weights:
            diff = current_weights[name] - base_weights[name]
            total_norm += torch.norm(diff, p="fro").item() ** 2
    return np.sqrt(total_norm)


def compute_pairwise_avg_hop(kg: KG, subjects: List[str], max_hop: int = 4) -> float:
    """Compute average pairwise hop distance among a set of subjects.

    Args:
        kg: Knowledge graph
        subjects: List of subject entities
        max_hop: Maximum hop distance to explore

    Returns:
        Average pairwise hop distance among subjects
    """
    if len(subjects) < 2:
        return 0.0

    total_hop = 0.0
    count = 0

    for i, subj_i in enumerate(subjects):
        # Compute BFS from this subject once
        hop_distances = kg.bfs_hop(subj_i, max_hop)

        # Sum distances to all other subjects in the list
        for j, subj_j in enumerate(subjects):
            if i < j:  # Count each pair only once
                distance = hop_distances.get(subj_j, max_hop + 1)  # Unreachable = max_hop + 1
                total_hop += distance
                count += 1

    return total_hop / count if count > 0 else 0.0

def sample_edit_cases(
    kg: KG,
    num_cases: int,
    seed: int = 42,
    selection_mode: str = "random",
    max_hop: int = 4,
) -> List[Dict[str, Any]]:
    """Sample edit cases from knowledge graph with various selection strategies.

    Each edit case consists of:
    - Original triple (s, r, o_old)
    - New object (o_new) that is different from o_old

    ※ alias（__a1等）は使わない前提：すべて抽象エンティティ（例: E_184）
    """
    random.seed(seed)
    np.random.seed(seed)

    all_triples = list(kg.triples)
    entities = sorted(set(kg.entities))  # すべて抽象エンティティのみの想定

    # Select triples based on selection mode
    if selection_mode == "random":
        sampled_triples = random.sample(all_triples, min(num_cases, len(all_triples)))

    elif selection_mode == "degree_high":
        sorted_triples = sorted(all_triples, key=lambda t: t.degree_s, reverse=True)
        sampled_triples = sorted_triples[:num_cases]

    elif selection_mode == "degree_low":
        sorted_triples = sorted(all_triples, key=lambda t: t.degree_s)
        sampled_triples = sorted_triples[:num_cases]

    elif selection_mode == "hop_high":
        print(f"Selecting {num_cases} triples to maximize pairwise hop distance (greedy)...")
        selected = [random.choice(all_triples)]
        remaining = [t for t in all_triples if t.tid != selected[0].tid]

        # 各ノードから選択済みsubjectへの最小距離を記録
        # min_dist_to_selected[entity] = 選択済みsubjectへの最小距離
        min_dist_to_selected = kg.bfs_hop(selected[0].s, max_hop)

        for i in range(1, num_cases):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_cases}")

            best_triple = None
            best_min_distance = -1

            candidates = remaining if len(remaining) <= 500 else random.sample(remaining, 500)

            for candidate in candidates:
                # 事前計算した距離マップから最小距離を取得（BFS不要）
                min_distance = min_dist_to_selected.get(candidate.s, max_hop + 1)
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_triple = candidate

            if best_triple:
                selected.append(best_triple)
                remaining = [t for t in remaining if t.tid != best_triple.tid]

                # 新しく選択されたsubjectからBFSして距離マップを更新
                new_hop_dist = kg.bfs_hop(best_triple.s, max_hop)
                for entity, dist in new_hop_dist.items():
                    if entity not in min_dist_to_selected:
                        min_dist_to_selected[entity] = dist
                    else:
                        min_dist_to_selected[entity] = min(min_dist_to_selected[entity], dist)

        sampled_triples = selected
        avg_pairwise_hop = compute_pairwise_avg_hop(kg, [t.s for t in sampled_triples], max_hop)
        print(f"Selected {len(sampled_triples)} triples with avg pairwise hop: {avg_pairwise_hop:.2f}")

    elif selection_mode == "hop_low":
        print(f"Selecting {num_cases} triples to minimize pairwise hop distance (greedy)...")
        selected = [random.choice(all_triples)]
        remaining = [t for t in all_triples if t.tid != selected[0].tid]

        # 各ノードから選択済みsubjectへの最小距離を記録
        # min_dist_to_selected[entity] = 選択済みsubjectへの最小距離
        min_dist_to_selected = kg.bfs_hop(selected[0].s, max_hop)

        for i in range(1, num_cases):
            if i % 10 == 0:
                print(f"  Progress: {i}/{num_cases}")

            best_triple = None
            best_min_distance = max_hop + 2

            candidates = remaining if len(remaining) <= 500 else random.sample(remaining, 500)

            for candidate in candidates:
                # 事前計算した距離マップから最小距離を取得（BFS不要）
                min_distance = min_dist_to_selected.get(candidate.s, max_hop + 1)
                if min_distance < best_min_distance:
                    best_min_distance = min_distance
                    best_triple = candidate

            if best_triple:
                selected.append(best_triple)
                remaining = [t for t in remaining if t.tid != best_triple.tid]

                # 新しく選択されたsubjectからBFSして距離マップを更新
                new_hop_dist = kg.bfs_hop(best_triple.s, max_hop)
                for entity, dist in new_hop_dist.items():
                    if entity not in min_dist_to_selected:
                        min_dist_to_selected[entity] = dist
                    else:
                        min_dist_to_selected[entity] = min(min_dist_to_selected[entity], dist)

        sampled_triples = selected
        avg_pairwise_hop = compute_pairwise_avg_hop(kg, [t.s for t in sampled_triples], max_hop)
        print(f"Selected {len(sampled_triples)} triples with avg pairwise hop: {avg_pairwise_hop:.2f}")

    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    # Build edit cases
    edit_cases: List[Dict[str, Any]] = []
    for triple in sampled_triples:
        o_old = triple.o

        possible_new_objects = [e for e in entities if e != o_old]
        if not possible_new_objects:
            continue

        o_new = random.choice(possible_new_objects)

        edit_cases.append(
            {
                "s": triple.s,
                "r": triple.r,
                "o_old": o_old,          # aliasなし前提（KGに入ってる文字列そのまま）
                "o_new": o_new,          # aliasなし前提
                "degree_s": triple.degree_s,
            }
        )

    return edit_cases
# def sample_edit_cases(
#     kg: KG, num_cases: int, seed: int = 42, selection_mode: str = "random", max_hop: int = 4
# ) -> List[Dict[str, Any]]:
#     """Sample edit cases from knowledge graph with various selection strategies.

#     Each edit case consists of:
#     - Original triple (s, r, o)
#     - New object (o_new) that is different from o (at abstract-entity level)

#     o_new は alias 付きではなく，抽象エンティティ（例: E_184）のみを使う

#     Args:
#         kg: Knowledge graph
#         num_cases: Number of edit cases to sample
#         seed: Random seed
#         selection_mode: How to select edit cases:
#             - "random": Random sampling
#             - "degree_high": Subjects with high average degree
#             - "degree_low": Subjects with low average degree
#             - "hop_high": Maximize average pairwise hop distance among edit subjects
#             - "hop_low": Minimize average pairwise hop distance among edit subjects
#         max_hop: Maximum hop distance for computing pairwise hop (used in hop_high/hop_low modes)
#     """
#     random.seed(seed)
#     np.random.seed(seed)

#     def to_abstract(ent: str) -> str:
#         # "E_184__a1" -> "E_184"
#         return ent.split("__")[0] if "__" in ent else ent

#     # Get all triples as candidates
#     all_triples = list(kg.triples)

#     # KG 内の全エンティティから「抽象エンティティ集合」を作る
#     # 例: {"E_001__a4", "E_001__a5", "E_184__a1"} -> {"E_001", "E_184"}
#     abstract_entities = sorted({to_abstract(e) for e in kg.entities})

#     # Select triples based on selection mode
#     if selection_mode == "random":
#         # Random sampling
#         sampled_triples = random.sample(all_triples, min(num_cases, len(all_triples)))

#     elif selection_mode == "degree_high":
#         # Sort by subject degree (descending) and take top N
#         sorted_triples = sorted(all_triples, key=lambda t: t.degree_s, reverse=True)
#         sampled_triples = sorted_triples[:num_cases]

#     elif selection_mode == "degree_low":
#         # Sort by subject degree (ascending) and take top N
#         sorted_triples = sorted(all_triples, key=lambda t: t.degree_s)
#         sampled_triples = sorted_triples[:num_cases]

#     elif selection_mode == "hop_high":
#         # Greedy selection to maximize pairwise hop distance among edit subjects
#         print(f"Selecting {num_cases} triples to maximize pairwise hop distance (greedy)...")

#         # Start with a random triple
#         selected = [random.choice(all_triples)]
#         remaining = [t for t in all_triples if t.tid != selected[0].tid]

#         for i in range(1, num_cases):
#             if i % 10 == 0:
#                 print(f"  Progress: {i}/{num_cases}")

#             # Find the triple whose subject is farthest from already selected subjects
#             selected_subjects = [t.s for t in selected]
#             best_triple = None
#             best_min_distance = -1

#             # Sample candidates for efficiency (if too many remaining)
#             candidates = remaining if len(remaining) <= 500 else random.sample(remaining, 500)

#             for candidate in candidates:
#                 # Compute minimum distance from this candidate to any selected subject
#                 min_distance = max_hop + 1
#                 for selected_subj in selected_subjects:
#                     hop_dist = kg.bfs_hop(candidate.s, max_hop)
#                     distance = hop_dist.get(selected_subj, max_hop + 1)
#                     min_distance = min(min_distance, distance)

#                 # Choose candidate with maximum min_distance (farthest from all selected)
#                 if min_distance > best_min_distance:
#                     best_min_distance = min_distance
#                     best_triple = candidate

#             if best_triple:
#                 selected.append(best_triple)
#                 remaining = [t for t in remaining if t.tid != best_triple.tid]

#         sampled_triples = selected
#         avg_pairwise_hop = compute_pairwise_avg_hop(kg, [t.s for t in sampled_triples], max_hop)
#         print(f"Selected {len(sampled_triples)} triples with avg pairwise hop: {avg_pairwise_hop:.2f}")

#     elif selection_mode == "hop_low":
#         # Greedy selection to minimize pairwise hop distance among edit subjects
#         print(f"Selecting {num_cases} triples to minimize pairwise hop distance (greedy)...")

#         # Start with a random triple
#         selected = [random.choice(all_triples)]
#         remaining = [t for t in all_triples if t.tid != selected[0].tid]

#         for i in range(1, num_cases):
#             if i % 10 == 0:
#                 print(f"  Progress: {i}/{num_cases}")

#             # Find the triple whose subject is closest to already selected subjects
#             selected_subjects = [t.s for t in selected]
#             best_triple = None
#             best_min_distance = max_hop + 2  # Start with large value

#             # Sample candidates for efficiency (if too many remaining)
#             candidates = remaining if len(remaining) <= 500 else random.sample(remaining, 500)

#             for candidate in candidates:
#                 # Compute minimum distance from this candidate to any selected subject
#                 min_distance = max_hop + 1
#                 for selected_subj in selected_subjects:
#                     hop_dist = kg.bfs_hop(candidate.s, max_hop)
#                     distance = hop_dist.get(selected_subj, max_hop + 1)
#                     min_distance = min(min_distance, distance)

#                 # Choose candidate with minimum min_distance (closest to any selected)
#                 if min_distance < best_min_distance:
#                     best_min_distance = min_distance
#                     best_triple = candidate

#             if best_triple:
#                 selected.append(best_triple)
#                 remaining = [t for t in remaining if t.tid != best_triple.tid]

#         sampled_triples = selected
#         avg_pairwise_hop = compute_pairwise_avg_hop(kg, [t.s for t in sampled_triples], max_hop)
#         print(f"Selected {len(sampled_triples)} triples with avg pairwise hop: {avg_pairwise_hop:.2f}")

#     else:
#         raise ValueError(f"Unknown selection mode: {selection_mode}")

#     edit_cases = []
#     for triple in sampled_triples:
#         o_old_abs = to_abstract(triple.o)

#         # 元の o と「抽象レベル」で異なるものだけ候補にする
#         possible_new_objects = [
#             e_abs for e_abs in abstract_entities if e_abs != o_old_abs
#         ]

#         # 念のためガード
#         if not possible_new_objects:
#             # 全部同じ抽象エンティティだった，などの変な状況
#             # とりあえずスキップするか，例外を投げる
#             # ここではスキップにしておく
#             continue

#         # new_object は alias なしの抽象エンティティ
#         new_object = random.choice(possible_new_objects)

#         edit_case = {
#             "s": triple.s,
#             "r": triple.r,
#             "o_old": triple.o,      # ここは今まで通り alias 付き
#             "o_new": new_object,    # ★ alias なし抽象エンティティに変更
#             "degree_s": triple.degree_s,
#         }
#         edit_cases.append(edit_case)

#     return edit_cases


def compute_step_metrics(
    step: int,
    model_before: GPTMini,
    model_after: GPTMini,
    tokenizer: SROTokenizer,
    kg: KG,
    base_weights: Dict[str, torch.Tensor],
    past_edits: List[Dict],
    current_case: Dict,
    edited_triples: List[Triple],
    retain_triples: List[Triple],
    config: SeqEditConfig,
    ripple_fp: Any,
    acc_fp: Any,
) -> Dict[str, Any]:
    """Compute comprehensive metrics for a single editing step.

    Args:
        step: Current step number
        model_before: Model before current edit
        model_after: Model after current edit
        tokenizer: Tokenizer
        kg: Knowledge graph
        base_weights: Original model weights (before any edits)
        past_edits: List of all previous edits
        current_case: Current edit case
        eval_triples: Triples for evaluation
        config: Configuration
        ripple_fp: File pointer for ripple logging
        acc_fp: File pointer for accuracy logging

    Returns:
        Dictionary of step metrics
    """
    device = config.device

    # 1. Edit success: Check if the current edit was successful
    input_text = f"{current_case['s']} {current_case['r']}"
    input_ids = torch.tensor(
        tokenizer.encode(input_text), dtype=torch.long, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model_after(input_ids)
        logits = outputs["logits"][0, -1, :]
        pred_id = logits.argmax().item()
        pred_token = tokenizer.get_token(pred_id)

        # Also get logit for target
        target_id = tokenizer.get_id(current_case["o_new"])
        target_logit = logits[target_id].item()

        # Get logit for old object
        old_id = tokenizer.get_id(current_case["o_old"])
        old_logit = logits[old_id].item()

    is_edit_success = pred_token == current_case["o_new"]
    edit_margin = target_logit - old_logit

    # 2. Retention: Evaluate all past edits
    retention_successes = []
    for past_edit in past_edits:
        past_input = f"{past_edit['s']} {past_edit['r']}"
        past_input_ids = torch.tensor(
            tokenizer.encode(past_input), dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            past_outputs = model_after(past_input_ids)
            past_logits = past_outputs["logits"][0, -1, :]
            past_pred_id = past_logits.argmax().item()
            past_pred = tokenizer.get_token(past_pred_id)

        retention_successes.append(past_pred == past_edit["o_new"])

    retention_rate = (
        np.mean(retention_successes) if retention_successes else 1.0
    )

    # 3. KG accuracy: Evaluate on edited triples (cumulative - only triples edited so far)
    # これまでに編集されたトリプルのうち、現在のステップまでに編集されたものだけを評価
    edited_so_far = edited_triples[:step]  # ステップ1〜stepまでに編集されたトリプル
    correct_edited = 0
    for triple in edited_so_far:
        eval_input = f"{triple.s} {triple.r}"
        eval_input_ids = torch.tensor(
            tokenizer.encode(eval_input), dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            eval_outputs = model_after(eval_input_ids)
            eval_logits = eval_outputs["logits"][0, -1, :]
            eval_pred_id = eval_logits.argmax().item()
            eval_pred = tokenizer.get_token(eval_pred_id)

        # 編集後のターゲット（o_new）と一致するか確認
        # past_editsから該当する編集を見つける
        matching_edit = None
        for edit in past_edits:
            if edit["s"] == triple.s and edit["r"] == triple.r and edit["o_old"] == triple.o:
                matching_edit = edit
                break

        if matching_edit and eval_pred == matching_edit["o_new"]:
            correct_edited += 1

    edited_acc = correct_edited / len(edited_so_far) if edited_so_far else 0.0

    # 4. Retention accuracy: Evaluate on unedited (retain) triples
    correct_retain = 0
    for triple in retain_triples:
        eval_input = f"{triple.s} {triple.r}"
        eval_input_ids = torch.tensor(
            tokenizer.encode(eval_input), dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            eval_outputs = model_after(eval_input_ids)
            eval_logits = eval_outputs["logits"][0, -1, :]
            eval_pred_id = eval_logits.argmax().item()
            eval_pred = tokenizer.get_token(eval_pred_id)

        if eval_pred == triple.o:
            correct_retain += 1

    retain_acc = correct_retain / len(retain_triples) if retain_triples else 0.0

    # 5. Weight distance from base model
    current_weights = extract_rome_layer_weights(model_after)
    weight_fro_norm = compute_weight_distance(current_weights, base_weights)

    # 6. Ripple effects: Compute delta logits for retain triples (unedited triples only)
    ripple_records = compute_ripple_records(
        step=step,
        kg=kg,
        triples=retain_triples,
        model_before=model_before,
        model_after=model_after,
        tokenizer=tokenizer,
        edit_subject=current_case["s"],
        edit_orig_object=current_case["o_old"],
        edit_new_object=current_case["o_new"],
        max_hop=config.max_hop,
        device=device,
    )

    # Write ripple records to file
    write_ripple_jsonl(ripple_records, ripple_fp)

    # Compute ripple statistics
    delta_logits = [abs(r["delta_logit"]) for r in ripple_records]
    mean_abs_delta = np.mean(delta_logits) if delta_logits else 0.0

    # Group by hop distance (subject-based)
    by_hop_subj = defaultdict(list)
    for r in ripple_records:
        hop = r["hop_subj"]
        if hop <= config.max_hop:
            by_hop_subj[hop].append(abs(r["delta_logit"]))

    hop_stats = {
        str(hop): np.mean(deltas) if deltas else 0.0
        for hop, deltas in by_hop_subj.items()
    }

    # Group by degree bins
    by_degree_bin = defaultdict(list)
    for r in ripple_records:
        degree = r["degree_s"]
        for bin_min, bin_max in config.degree_bins:
            if bin_min <= degree < bin_max:
                by_degree_bin[f"{bin_min}-{bin_max}"].append(
                    abs(r["delta_logit"])
                )
                break

    degree_stats = {
        bin_name: np.mean(deltas) if deltas else 0.0
        for bin_name, deltas in by_degree_bin.items()
    }

    # 7. Triple accuracy records (for both edited and retain triples)
    # Edited triples
    edited_acc_records = compute_triple_accuracy_records(
        step=step,
        triples=edited_so_far,
        model=model_after,
        tokenizer=tokenizer,
        device=device,
    )
    for record in edited_acc_records:
        record["triple_type"] = "edited"
        acc_fp.write(json.dumps(record) + "\n")

    # Retain triples
    retain_acc_records = compute_triple_accuracy_records(
        step=step,
        triples=retain_triples,
        model=model_after,
        tokenizer=tokenizer,
        device=device,
    )
    for record in retain_acc_records:
        record["triple_type"] = "retain"
        acc_fp.write(json.dumps(record) + "\n")

    acc_fp.flush()

    # 8. Compile step metrics
    step_metrics = {
        "step": step,
        "edit": {
            "s": current_case["s"],
            "r": current_case["r"],
            "o_old": current_case["o_old"],
            "o_new": current_case["o_new"],
        },
        "edit_success": {
            "is_success_top1": is_edit_success,
            "margin": edit_margin,
        },
        "retention_rate": retention_rate,
        "edited_acc": edited_acc,  # 編集済みトリプルの精度
        "retain_acc": retain_acc,  # 未編集トリプルの精度
        "weight_fro_norm": weight_fro_norm,
        "ripple": {
            "mean_abs_delta": mean_abs_delta,
            "by_hop_subj": hop_stats,
            "by_degree_bin": degree_stats,
        },
    }

    return step_metrics


def run_sequential_edits(config: SeqEditConfig) -> None:
    """Execute sequential editing experiment and log results.

    Args:
        config: Sequential editing configuration
    """
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create output directory
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), indent=2, fp=f)

    print(f"Output directory: {out_dir}")

    # Load knowledge graph from training corpus
    print("Loading knowledge graph from training corpus...")
    kg_path = Path(config.kg_dir) / "corpus.train.txt"
    kg = KG(str(kg_path))
    print(f"Loaded KG: {kg}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_dir, config.device)
    print(f"Model: {model.config.n_layers} layers, {model.get_num_params()} params")

    # Extract base weights
    base_weights = extract_rome_layer_weights(model)

    # edit_cases を先に作ってから eval_triples を選ぶ
    # Sample edit cases based on selection mode
    print(f"Sampling edit cases using mode: {config.edit_selection_mode}")
    edit_cases = sample_edit_cases(
        kg,
        config.num_steps,
        seed=config.seed,
        selection_mode=config.edit_selection_mode,
        max_hop=config.max_hop
    )
    print(f"Edit cases: {len(edit_cases)}")

    # Sample / select evaluation triples (edited and retain)
    edited_triples, retain_triples = select_eval_triples(kg, config, edit_cases)
    print(f"Edited triples (for evaluation): {len(edited_triples)}")
    print(f"Retain triples (unedited, for retention evaluation): {len(retain_triples)}")


    # Open output files
    stats_fp = open(out_dir / "stats.jsonl", "w")
    ripple_fp = open(out_dir / "ripple_triples.jsonl", "w")
    acc_fp = open(out_dir / "triple_acc.jsonl", "w")

    # Initialize ROME editor
    print("Initializing ROME editor...")
    kg_corpus_path = Path(config.kg_dir) / "corpus.train.txt"
    rome_editor = ROME(
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        kg_corpus_path=str(kg_corpus_path),
        mom2_n_samples=1000,
        use_mom2_adjustment=True,  # Enable to use KG corpus statistics
        v_num_grad_steps=config.v_num_grad_steps,
    )

    # Track past edits
    past_edits = []

    # Compute baseline accuracy before any edits (on retain triples only)
    print("\nComputing baseline accuracy before edits...")
    baseline_retain_correct = 0
    for triple in retain_triples:
        eval_input = f"{triple.s} {triple.r}"
        eval_input_ids = torch.tensor(
            tokenizer.encode(eval_input), dtype=torch.long, device=config.device
        ).unsqueeze(0)

        with torch.no_grad():
            eval_outputs = model(eval_input_ids)
            eval_logits = eval_outputs["logits"][0, -1, :]
            eval_pred_id = eval_logits.argmax().item()
            eval_pred = tokenizer.get_token(eval_pred_id)

        if eval_pred == triple.o:
            baseline_retain_correct += 1

    baseline_retain_acc = baseline_retain_correct / len(retain_triples) if retain_triples else 0.0
    print(f"Baseline Retain Accuracy (unedited triples): {baseline_retain_acc:.4f}")

    # Save baseline metrics
    baseline_metrics = {
        "step": 0,
        "edited_acc": 0.0,  # No edits yet
        "retain_acc": baseline_retain_acc,
        "is_baseline": True,
    }
    stats_fp.write(json.dumps(baseline_metrics) + "\n")
    stats_fp.flush()

    # Sequential editing loop
    print(f"\nStarting sequential editing ({config.num_steps} steps)...")
    for step, case in enumerate(edit_cases, start=1):
        print(f"\n[Step {step}/{config.num_steps}]")
        print(f"  Edit: ({case['s']}, {case['r']}, {case['o_old']}) -> {case['o_new']}")

        # 1. Copy model for delta logit computation
        model_before = deepcopy(model)

        # 2. Apply ROME edit (in-place on main model)
        edited_model, edit_result = rome_editor.apply_edit(
            s=case["s"],
            r=case["r"],
            o_target=case["o_new"],
            layer=config.edit_layer,  # Use configured layer or auto-locate
            copy_model=False,  # Edit in-place
        )

        # Update model reference (edited_model should be same as model)
        model = edited_model

        print(f"  Success: {edit_result.success}")
        print(f"  Layer: {edit_result.layer}")

        # 3. Compute step metrics
        step_metrics = compute_step_metrics(
            step=step,
            model_before=model_before,
            model_after=model,
            tokenizer=tokenizer,
            kg=kg,
            base_weights=base_weights,
            past_edits=past_edits,
            current_case=case,
            edited_triples=edited_triples,
            retain_triples=retain_triples,
            config=config,
            ripple_fp=ripple_fp,
            acc_fp=acc_fp,
        )

        print(f"  Retention: {step_metrics['retention_rate']:.3f}")
        print(f"  Edited Acc: {step_metrics['edited_acc']:.3f}")
        print(f"  Retain Acc: {step_metrics['retain_acc']:.3f}")
        print(f"  Weight Δ: {step_metrics['weight_fro_norm']:.3f}")

        # Write step metrics
        stats_fp.write(json.dumps(step_metrics) + "\n")
        stats_fp.flush()

        # Add to past edits
        past_edits.append(
            {
                "step": step,
                "s": case["s"],
                "r": case["r"],
                "o_old": case["o_old"],
                "o_new": case["o_new"],
            }
        )

    # Close files
    stats_fp.close()
    ripple_fp.close()
    acc_fp.close()

    print(f"\n✓ Sequential editing complete!")
    print(f"Results saved to: {out_dir}")
