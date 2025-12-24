"""Ripple effect logging for sequential editing.

This module provides utilities for computing and logging delta logits
and metadata (hop distances, degrees) for analyzing ripple effects
during sequential knowledge editing.
"""

from typing import Iterable, TextIO, List, Dict, Any
import json
import torch
import torch.nn.functional as F

from .kg_utils import Triple, KG


def compute_ripple_records(
    step: int,
    kg: KG,
    triples: Iterable[Triple],
    model_before,
    model_after,
    tokenizer,
    edit_subject: str,
    edit_orig_object: str,
    edit_new_object: str,
    max_hop: int,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Compute ripple effect records for a set of triples.

    For each triple, computes:
    - delta_logit: Change in logit for the correct object
    - hop_subj: Hop distance from edit subject to triple subject
    - hop_before: Hop distance from original edit object to triple subject
    - hop_after: Hop distance from new edit object to triple subject
    - degree_s: Degree of triple subject

    Args:
        step: Current editing step number
        kg: Knowledge graph
        triples: Triples to evaluate
        model_before: Model before the edit
        model_after: Model after the edit
        tokenizer: Tokenizer for encoding triples
        edit_subject: Subject of the current edit
        edit_orig_object: Original object of the current edit
        edit_new_object: New object of the current edit
        max_hop: Maximum hop distance to compute
        device: Device to run on

    Returns:
        List of dictionaries containing ripple effect metrics
    """
    model_before.eval()
    model_after.eval()

    # Compute hop distances from edit entities
    hop_from_subj = kg.bfs_hop(edit_subject, max_hop)
    hop_from_orig = kg.bfs_hop(edit_orig_object, max_hop)
    hop_from_new = kg.bfs_hop(edit_new_object, max_hop)

    records = []

    with torch.no_grad():
        for triple in triples:
            # Encode input (subject + relation)
            input_text = f"{triple.s} {triple.r}"
            input_ids = torch.tensor(
                tokenizer.encode(input_text), dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, 2]

            # Get target object token ID
            target_id = tokenizer.get_id(triple.o)

            # Compute logits before edit
            outputs_before = model_before(input_ids)
            logits_before = outputs_before["logits"][0, -1, :]  # [vocab_size]
            logit_before = float(logits_before[target_id].cpu())

            # Compute logits after edit
            outputs_after = model_after(input_ids)
            logits_after = outputs_after["logits"][0, -1, :]  # [vocab_size]
            logit_after = float(logits_after[target_id].cpu())

            # Compute delta logit
            delta_logit = logit_after - logit_before

            # Compute hop distances (use max_hop + 1 if unreachable)
            hop_subj = hop_from_subj.get(triple.s, max_hop + 1)
            hop_before = hop_from_orig.get(triple.s, max_hop + 1)
            hop_after = hop_from_new.get(triple.s, max_hop + 1)

            # Create record
            record = {
                "step": step,
                "tid": triple.tid,
                "s": triple.s,
                "r": triple.r,
                "o": triple.o,
                "delta_logit": delta_logit,
                "logit_before": logit_before,
                "logit_after": logit_after,
                "hop_subj": hop_subj,
                "hop_before": hop_before,
                "hop_after": hop_after,
                "degree_s": triple.degree_s,
                "edit_s": edit_subject,
                "edit_o_old": edit_orig_object,
                "edit_o_new": edit_new_object,
            }
            records.append(record)

    return records


def write_ripple_jsonl(records: List[Dict[str, Any]], fp: TextIO) -> None:
    """Write ripple records to JSONL file.

    Args:
        records: List of ripple effect records
        fp: File pointer to write to (must be opened in write/append mode)
    """
    for record in records:
        fp.write(json.dumps(record) + "\n")
    fp.flush()  # Ensure data is written immediately


def compute_triple_accuracy_records(
    step: int,
    triples: Iterable[Triple],
    model,
    tokenizer,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Compute accuracy records for triples at a given step.

    Args:
        step: Current editing step number
        triples: Triples to evaluate
        model: Model to evaluate
        tokenizer: Tokenizer for encoding triples
        device: Device to run on

    Returns:
        List of dictionaries with accuracy metrics per triple
    """
    model.eval()
    records = []

    with torch.no_grad():
        for triple in triples:
            # Encode input (subject + relation)
            input_text = f"{triple.s} {triple.r}"
            input_ids = torch.tensor(
                tokenizer.encode(input_text), dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, 2]

            # Get target object token ID
            target_id = tokenizer.get_id(triple.o)

            # Forward pass
            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :]  # [vocab_size]

            # Check if prediction is correct
            pred_id = logits.argmax().item()
            is_correct = pred_id == target_id

            # Get predicted token
            pred_token = tokenizer.get_token(pred_id)

            # Compute probability and rank
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            target_prob = float(probs[target_id])
            rank = int((probs > target_prob).sum() + 1)

            # Create record
            record = {
                "step": step,
                "tid": triple.tid,
                "s": triple.s,
                "r": triple.r,
                "o": triple.o,
                "is_correct": is_correct,
                "predicted": pred_token,
                "probability": target_prob,
                "rank": rank,
            }
            records.append(record)

    return records
