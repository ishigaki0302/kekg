"""AlphaEdit-style editor for GPTMini (adaptation, not the original codebase).

AlphaEdit (Fang et al., ICLR 2025) constrains a locate-then-edit update to the
*null space* of preserved knowledge, so the update does not disturb facts the
model already holds. Here we adapt that idea on top of our ROME update:

  1. compute the ROME rank-1 weight delta dW on blocks[L].ffn.w2,
  2. build a projector P onto the null space of the preserved-key covariance
     C = E[k k^T]  (k = post-GELU activation = input to w2), by removing the
     top eigen-directions that carry most of C's energy,
  3. apply the projected update  W <- W0 + dW @ P.

Because dW @ P has (almost) no component along the dominant preserved-key
directions, edits to preserved facts are suppressed (better locality).
Exposes the same apply_edit signature as ROME / FTEditor.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .rome import ROME, EditResult, EditSpec


class AlphaEditEditor:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        kg_corpus_path: Optional[str] = None,
        default_layer: int = 0,
        energy: float = 0.95,
        stats_name: str = "gpt_mini",
        mom2_n_samples: int = 27600,
    ):
        self.original_model = model
        self.tok = tokenizer
        self.device = device
        self.default_layer = default_layer
        self.rome = ROME(model, tokenizer, device=device, kg_corpus_path=kg_corpus_path,
                         mom2_n_samples=mom2_n_samples, stats_name=stats_name)
        # null-space projector on the key (d_mlp) axis at the edit layer
        self.P = self._null_space_projector(kg_corpus_path, default_layer, energy)

    @torch.no_grad()
    def _null_space_projector(self, corpus_path, layer, energy):
        model = self.original_model
        cap = {}
        h = model.blocks[layer].ffn.w1.register_forward_hook(
            lambda m, i, o: cap.__setitem__("a", o)
        )
        d_mlp = model.blocks[layer].ffn.w1.out_features
        C = torch.zeros(d_mlp, d_mlp, device=self.device, dtype=torch.float64)
        n = 0
        lines = [ln.strip() for ln in open(corpus_path, encoding="utf-8") if ln.strip()]
        try:
            for s in range(0, len(lines), 256):
                batch = lines[s:s + 256]
                ids = torch.tensor([self.tok.encode(t) for t in batch], device=self.device)
                model(ids)
                k = F.gelu(cap["a"]).reshape(-1, d_mlp).double()  # all positions
                C += k.t() @ k
                n += k.shape[0]
        finally:
            h.remove()
        C /= max(1, n)
        evals, evecs = torch.linalg.eigh(C)  # ascending
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        total = evals.sum()
        csum = torch.cumsum(evals, 0)
        r = int((csum < energy * total).sum().item()) + 1  # dirs covering `energy`
        Ur = evecs[:, :r]
        P = torch.eye(d_mlp, device=self.device, dtype=torch.float64) - Ur @ Ur.t()
        return P.float()

    @torch.no_grad()
    def _predict(self, model, s, r):
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        return self.tok.get_token(int(torch.argmax(model(ids)["logits"][0, -1, :])))

    def apply_edit(self, s, r, o_target, layer=None, copy_model=True):
        L = self.default_layer if layer is None else layer
        orig_pred = self._predict(self.original_model, s, r)
        # ROME delta on a throwaway copy
        edited, _ = self.rome.apply_edit(s, r, o_target, layer=L, copy_model=True)
        W0 = self.original_model.blocks[L].ffn.w2.weight.data
        dW = (edited.blocks[L].ffn.w2.weight.data - W0)  # [d_model, d_mlp]
        dW_proj = dW @ self.P.to(dW.dtype)               # project key axis
        m = deepcopy(self.original_model)
        m.blocks[L].ffn.w2.weight.data = (W0 + dW_proj).clone()
        m.eval()
        new_pred = self._predict(m, s, r)
        del edited
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return m, EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=L, original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target),
        )
