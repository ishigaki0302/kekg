"""GRACE-style editor for GPTMini (adaptation, not the original codebase).

GRACE (Hartvigsen et al., NeurIPS 2023) keeps weights frozen and stores edits in
a discrete key-value codebook at one layer: at inference, if a hidden state is
close (within epsilon) to a stored key, it is replaced by the stored value.

Single-edit adaptation:
  1. key   = block[L] output at the last token for input "s r",
  2. value = a vector optimised (frozen model) so that substituting the
     last-token hidden with it makes the model output o_target,
  3. an inference hook substitutes the value whenever the last-token hidden is
     within epsilon of the key (else passthrough).

Because only states near the edit key are touched, GRACE preserves locality
perfectly but does not generalise/propagate (single codebook entry).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .rome import EditResult, EditSpec


class GRACEEditor:
    def __init__(self, model, tokenizer, device="cuda", default_layer=0,
                 lr=0.5, steps=100, eps_frac=0.25):
        self.original_model = model
        self.tok = tokenizer
        self.device = device
        self.default_layer = default_layer
        self.lr = lr
        self.steps = steps
        self.eps_frac = eps_frac  # epsilon = eps_frac * ||key||

    @staticmethod
    def _block_out(out):
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def _predict(self, model, s, r):
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        return self.tok.get_token(int(torch.argmax(model(ids)["logits"][0, -1, :])))

    def apply_edit(self, s, r, o_target, layer=None, copy_model=True):
        L = self.default_layer if layer is None else layer
        m = deepcopy(self.original_model)
        m.eval()
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        tgt = torch.tensor([self.tok.get_id(o_target.strip())], device=self.device)
        orig_pred = self._predict(self.original_model, s, r)

        # 1) key = block[L] last-token output
        cap = {}
        h = m.blocks[L].register_forward_hook(
            lambda mod, i, o: cap.__setitem__("h", self._block_out(o).detach())
        )
        with torch.no_grad():
            m(ids)
        h.remove()
        key = cap["h"][0, -1, :].clone()
        eps = self.eps_frac * key.norm().item()

        # 2) optimise value so output == o_target (substitute last-token hidden)
        value = key.clone().requires_grad_(True)
        opt = torch.optim.Adam([value], lr=self.lr)

        def opt_hook(mod, inp, out):
            x = self._block_out(out)
            x[:, -1, :] = value
            return out

        hh = m.blocks[L].register_forward_hook(opt_hook)
        for _ in range(self.steps):
            opt.zero_grad()
            logits = m(ids)["logits"][0, -1, :].unsqueeze(0)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            opt.step()
            if int(logits.argmax()) == int(tgt):
                break
        hh.remove()
        value = value.detach()

        # 3) install persistent inference hook (codebook of one entry)
        def infer_hook(mod, inp, out, _key=key, _val=value, _eps=eps):
            x = self._block_out(out)
            d = (x[:, -1, :] - _key).norm(dim=-1)
            mask = d < _eps
            if mask.any():
                x[mask, -1, :] = _val
            return out

        handle = m.blocks[L].register_forward_hook(infer_hook)
        # keep the handle alive on the model so it isn't garbage-collected
        m._grace_handle = handle

        new_pred = self._predict(m, s, r)
        return m, EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=L, original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target),
        )
