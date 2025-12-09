"""Evaluation metrics for knowledge editing."""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass

from ..modeling import GPTMini, SROInference, SROTokenizer
from ..kg import KnowledgeGraph


@dataclass
class LocalMetrics:
    """Local edit success metrics."""
    accuracy: float  # Top-1 accuracy
    mean_rank: float
    median_rank: float
    mean_log_prob: float
    mrr: float  # Mean reciprocal rank


@dataclass
class GlobalMetrics:
    """Global side-effect metrics."""
    accuracy: float  # Overall accuracy on all triples
    accuracy_delta: float  # Change from pre-edit
    mean_log_prob: float
    log_prob_delta: float
    num_degraded: int  # Number of triples that got worse


def compute_local_metrics(
    inference: SROInference,
    edited_triples: List[Tuple[str, str, str]]
) -> LocalMetrics:
    """
    Compute local edit success metrics.

    Args:
        inference: Inference engine
        edited_triples: List of edited (s, r, o_new) triples

    Returns:
        Local metrics
    """
    results = inference.batch_evaluate(edited_triples)

    # Mean reciprocal rank
    ranks = [r["rank"] for r in results["details"]]
    mrr = np.mean([1.0 / r for r in ranks])

    return LocalMetrics(
        accuracy=results["accuracy"],
        mean_rank=results["mean_rank"],
        median_rank=results["median_rank"],
        mean_log_prob=results["mean_log_prob"],
        mrr=mrr
    )


def compute_global_metrics(
    inference_pre: SROInference,
    inference_post: SROInference,
    all_triples: List[Tuple[str, str, str]]
) -> GlobalMetrics:
    """
    Compute global side-effect metrics.

    Args:
        inference_pre: Pre-edit inference engine
        inference_post: Post-edit inference engine
        all_triples: All triples in knowledge graph

    Returns:
        Global metrics
    """
    # Evaluate on both models
    results_pre = inference_pre.batch_evaluate(all_triples)
    results_post = inference_post.batch_evaluate(all_triples)

    # Compute deltas
    acc_delta = results_post["accuracy"] - results_pre["accuracy"]
    logp_delta = results_post["mean_log_prob"] - results_pre["mean_log_prob"]

    # Count degraded triples
    num_degraded = 0
    for r_pre, r_post in zip(results_pre["details"], results_post["details"]):
        if r_post["rank"] > r_pre["rank"]:
            num_degraded += 1

    return GlobalMetrics(
        accuracy=results_post["accuracy"],
        accuracy_delta=acc_delta,
        mean_log_prob=results_post["mean_log_prob"],
        log_prob_delta=logp_delta,
        num_degraded=num_degraded
    )


def compute_impact_I(
    inference_pre: SROInference,
    inference_post: SROInference,
    all_triples: List[Tuple[str, str, str]]
) -> float:
    """
    Compute impact metric I: accuracy degradation.

    I = acc_pre - acc_post

    Args:
        inference_pre: Pre-edit inference
        inference_post: Post-edit inference
        all_triples: All triples

    Returns:
        Impact score (higher = more negative impact)
    """
    acc_pre = inference_pre.batch_evaluate(all_triples)["accuracy"]
    acc_post = inference_post.batch_evaluate(all_triples)["accuracy"]

    return acc_pre - acc_post


def compute_degree_correlation(
    impacts: Dict[str, float],
    degrees: Dict[str, int],
    method: str = "pearson"
) -> Dict[str, float]:
    """
    Compute correlation between entity degree and edit impact.

    Args:
        impacts: Dictionary mapping entity -> impact score
        degrees: Dictionary mapping entity -> degree
        method: "pearson" or "spearman"

    Returns:
        Dictionary with correlation coefficient and p-value
    """
    # Align entities
    entities = list(set(impacts.keys()) & set(degrees.keys()))

    impact_values = [impacts[e] for e in entities]
    degree_values = [degrees[e] for e in entities]

    # Compute correlation
    if method == "pearson":
        corr, pval = stats.pearsonr(degree_values, impact_values)
    elif method == "spearman":
        corr, pval = stats.spearmanr(degree_values, impact_values)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return {
        "correlation": corr,
        "p_value": pval,
        "method": method,
        "n_samples": len(entities)
    }


def compute_ripple_by_hop(
    inference_pre: SROInference,
    inference_post: SROInference,
    kg: KnowledgeGraph,
    edited_entity: str,
    max_hop: int = 2
) -> Dict[int, Dict[str, float]]:
    """
    Compute ripple effect by hop distance from edited entity.

    Args:
        inference_pre: Pre-edit inference
        inference_post: Post-edit inference
        kg: Knowledge graph
        edited_entity: Entity that was edited
        max_hop: Maximum hop distance to analyze

    Returns:
        Dictionary mapping hop -> metrics dict
    """
    # Get neighbors at each hop
    neighbors_by_hop = kg.get_neighbors(edited_entity, hop=max_hop)

    results = {}

    for hop, entities in neighbors_by_hop.items():
        if hop == 0:
            continue  # Skip the edited entity itself

        # Get triples involving these entities
        hop_triples = []
        for triple in kg.triples:
            if triple.s in entities or triple.o in entities:
                # Use first alias
                s_alias = kg.entity_aliases[triple.s][0]
                r_alias = kg.relation_aliases[triple.r][0]
                o_alias = kg.entity_aliases[triple.o][0]
                hop_triples.append((s_alias, r_alias, o_alias))

        if not hop_triples:
            continue

        # Evaluate
        eval_pre = inference_pre.batch_evaluate(hop_triples)
        eval_post = inference_post.batch_evaluate(hop_triples)

        results[hop] = {
            "num_triples": len(hop_triples),
            "acc_delta": eval_post["accuracy"] - eval_pre["accuracy"],
            "logp_delta": eval_post["mean_log_prob"] - eval_pre["mean_log_prob"],
            "rank_delta": eval_post["mean_rank"] - eval_pre["mean_rank"]
        }

    return results
