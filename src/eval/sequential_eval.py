"""Sequential (continual) editing evaluation on the symbolic world.

Applies a SEQUENCE of edits to the SAME model (cumulatively) and tracks the
Plasticity-Stability axis:
  - retention : of all edits applied so far, fraction whose direct fact
                (s, R_F) -> o_new still holds  (do earlier edits survive?)
  - collapse  : accuracy on a fixed sample of UNEDITED facts
                (does the model's general knowledge degrade?)

Chainable editors (ROME / MEMIT via edit_layers / FT) edit in place with
copy_model=False, so a single editor bound to a working-model copy accumulates.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from src.kg.symbolic_world import R_F


@torch.no_grad()
def _predict(model, tok, s, r, device):
    ids = torch.tensor([tok.encode(f"{s} {r}")], device=device)
    return tok.get_token(int(torch.argmax(model(ids)["logits"][0, -1, :])))


def run_sequence(rome, tok, world, plans, layer, device,
                 edit_layers: Optional[List[int]] = None, n_collapse=200, seed=0):
    """rome must be bound to a working-model copy (edits accumulate in place).

    Returns list of per-step dicts: step, retention, collapse, edit_success.
    """
    import random
    rng = random.Random(seed)
    work = rome.original_model  # the working model (edited in place)

    # fixed unedited sample for collapse (subjects not in the edit sequence)
    edit_subjects = {p.s for p in plans}
    pool = [e for e in world.entities if e not in edit_subjects]
    rng.shuffle(pool)
    collapse_set = pool[:n_collapse]

    edited = []  # (s, o_new)
    out = []
    for i, plan in enumerate(plans):
        if edit_layers is not None:
            rome.apply_edit(plan.s, R_F, plan.o_new, layers=edit_layers, copy_model=False)
        else:
            rome.apply_edit(plan.s, R_F, plan.o_new, layer=layer, copy_model=False)
        edited.append((plan.s, plan.o_new))

        # retention of all edits so far
        ret = sum(1 for (s, o) in edited
                  if _predict(work, tok, s, R_F, device).strip() == o.strip()) / len(edited)
        # collapse: unedited fact accuracy
        col = sum(1 for s in collapse_set
                  if _predict(work, tok, s, R_F, device).strip() == world.func_map[s].strip()) / len(collapse_set)
        edit_ok = _predict(work, tok, plan.s, R_F, device).strip() == plan.o_new.strip()
        out.append({"step": i + 1, "retention": ret, "collapse": col,
                    "edit_success": int(edit_ok), "n_edited": len(edited)})
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out
