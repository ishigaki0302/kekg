"""Editing-plasticity evaluation driver.

Applies a knowledge edit to a trained respondent model and scores the
logical-closure item battery (see ``src/kg/symbolic_world.py``), producing a
long-format response matrix for explanatory IRT.

Response semantics (one row per item):
  - positive items (direct / logical / invariant / neighbor_invariant):
        correct = 1 iff argmax prediction == gold_o
  - suppression items (contradicted):
        correct = 1 iff argmax prediction != suppress_o (old object is gone)

Each row also carries the structural covariates (victim_degree, hop_from_edit,
edit_subject_degree), the logical category, and the rule type — the explanatory
predictors for item difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import random

import torch

from src.modeling import GPTMini, GPTConfig, SROTokenizer
from src.kg.symbolic_world import (
    SymbolicWorld,
    build_symbolic_world,
    degree_bins,
    Item,
    R_F,
    CAT_CONTRADICTED,
)
from src.utils import load_yaml


# --------------------------------------------------------------------- loading
def load_respondent(model_dir: str, config_path: str, device: str):
    """Load a trained GPTMini respondent + its tokenizer."""
    model_dir = Path(model_dir)
    tok = SROTokenizer.load(model_dir / "tokenizer.json")
    mcfg = load_yaml(config_path)["model"]
    gcfg = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_model=mcfg["d_model"],
        d_mlp=mcfg["d_mlp"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg.get("dropout", 0.1),
    )
    model = GPTMini(gcfg)
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok


def rebuild_world(world_kwargs: Dict, func_map_path: Optional[str] = None) -> SymbolicWorld:
    """Deterministically rebuild the world used for training.

    Generation is seeded, so ``build_symbolic_world`` reproduces the identical
    ``func_map`` and BA substrate. If ``func_map_path`` is given, assert the
    rebuilt map matches the persisted one (guards against param drift).
    """
    world = build_symbolic_world(**world_kwargs)
    if func_map_path:
        import json

        saved = json.load(open(func_map_path, encoding="utf-8"))
        assert world.func_map == saved, (
            "Rebuilt func_map != saved func_map; world_kwargs/seed drifted."
        )
    return world


# ------------------------------------------------------------------- scoring
@torch.no_grad()
def predict_object(model, tok: SROTokenizer, s: str, r: str, device: str) -> str:
    ids = torch.tensor([tok.encode(f"{s} {r}")], device=device)
    logits = model(ids)["logits"][0, -1, :]
    return tok.get_token(int(torch.argmax(logits)))


def score_item(pred: str, item: Item) -> int:
    if item.category == CAT_CONTRADICTED:
        return int(pred.strip() != item.suppress_o.strip())
    return int(pred.strip() == item.gold_o.strip())


# ------------------------------------------------------------------- edits
@dataclass
class EditPlan:
    s: str
    o_new: str
    degree_bin: str


def sample_edits(
    world: SymbolicWorld,
    n_per_bin: int,
    seed: int = 0,
) -> List[EditPlan]:
    """Sample edits stratified by the edit-subject's degree bin."""
    rng = random.Random(seed)
    bins = degree_bins(world)
    by_bin: Dict[str, List[str]] = {}
    for e, b in bins.items():
        by_bin.setdefault(b, []).append(e)
    plans: List[EditPlan] = []
    for b, ents in by_bin.items():
        rng.shuffle(ents)
        for s in ents[:n_per_bin]:
            o_old = world.func_map[s]
            o_new = s
            while o_new == s or o_new == o_old:
                o_new = world.entities[rng.randrange(len(world.entities))]
            plans.append(EditPlan(s=s, o_new=o_new, degree_bin=b))
    return plans


def evaluate_edit(
    rome,
    tok: SROTokenizer,
    world: SymbolicWorld,
    plan: EditPlan,
    layer: int,
    device: str,
    max_invariant: int = 20,
    item_rng_seed: int = 0,
    edit_layers: Optional[List[int]] = None,
) -> (List[Dict], bool, float):
    """Apply one edit (on a fresh copy) and score its item battery.

    edit_layers (list) -> MEMIT-style multi-layer edit via ROME; else single layer.
    Returns (rows, edit_success, edit_time_s) — edit_time for Efficiency.
    """
    import time
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    if edit_layers is not None:
        edited_model, res = rome.apply_edit(
            plan.s, R_F, plan.o_new, layers=edit_layers, copy_model=True
        )
    else:
        edited_model, res = rome.apply_edit(
            plan.s, R_F, plan.o_new, layer=layer, copy_model=True
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    edit_time_s = time.perf_counter() - _t0
    items = world.build_item_battery(
        plan.s, plan.o_new, max_invariant=max_invariant, rng=random.Random(item_rng_seed)
    )
    rows: List[Dict] = []
    for it in items:
        pred = predict_object(edited_model, tok, it.s, it.r, device)
        rows.append(
            {
                "edit_s": plan.s,
                "edit_o_new": plan.o_new,
                "edit_degree_bin": plan.degree_bin,
                "edit_success": int(res.success),
                "item_s": it.s,
                "item_r": it.r,
                "category": it.category,
                "rule_type": it.rule_type,
                "victim_degree": it.covariates.get("victim_degree"),
                "hop_from_edit": it.covariates.get("hop_from_edit"),
                "edit_subject_degree": it.covariates.get("edit_subject_degree"),
                "correct": score_item(pred, it),
            }
        )
    # free the edited copy
    del edited_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, res.success, edit_time_s
