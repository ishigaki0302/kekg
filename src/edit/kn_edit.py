"""Knowledge-Neurons-style editor for GPTMini (adaptation of Dai et al. 2022).

Knowledge Neurons attributes a fact to a small set of FFN intermediate neurons
(post-GELU, dim d_mlp) and edits the fact by manipulating those neurons' value
vectors (columns of w2). Here:

  1. attribution = |a * d logit[o_old] / d a| at the edit layer's post-GELU
     activation `a` (grad x activation, a standard cheap proxy for integrated
     gradients) -> top-k knowledge neurons,
  2. edit = shift those neurons' value vectors along (E[o_new] - E[o_old]) in the
     unembedding direction (lm_head rows), with a scale searched until the
     prediction flips.

Distinct mechanism from ROME/MEMIT (attribution-driven, neuron-local), so it is
a genuinely different editing method, not a ROME re-parametrisation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch

from .rome import EditResult, EditSpec


class KNEditor:
    def __init__(self, model, tokenizer, device="cuda", default_layer=0,
                 topk=20, scales=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0)):
        self.original_model = model
        self.tok = tokenizer
        self.device = device
        self.default_layer = default_layer
        self.topk = topk
        self.scales = scales

    @torch.no_grad()
    def _predict(self, model, s, r):
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        return self.tok.get_token(int(torch.argmax(model(ids)["logits"][0, -1, :])))

    def _knowledge_neurons(self, model, ids, o_old_id, L):
        cap = {}

        def pre(mod, inp):
            a = inp[0]
            a.retain_grad()
            cap["a"] = a
            return None

        h = model.blocks[L].ffn.w2.register_forward_pre_hook(pre)
        logits = model(ids)["logits"][0, -1, :]
        model.zero_grad(set_to_none=True)
        logits[o_old_id].backward()
        h.remove()
        a = cap["a"][0, -1, :].detach()
        g = cap["a"].grad[0, -1, :].detach()
        attr = (a * g).abs()
        topk = torch.topk(attr, min(self.topk, attr.numel())).indices
        return topk, a

    def apply_edit(self, s, r, o_target, layer=None, copy_model=True):
        L = self.default_layer if layer is None else layer
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        orig_pred = self._predict(self.original_model, s, r)
        o_old_id = self.tok.get_id(orig_pred.strip())
        o_new_id = self.tok.get_id(o_target.strip())

        base = deepcopy(self.original_model)
        base.eval()
        topk, act = self._knowledge_neurons(base, ids, o_old_id, L)

        with torch.no_grad():
            E = base.lm_head.weight  # [vocab, d_model]
            direction = (E[o_new_id] - E[o_old_id])
            direction = direction / (direction.norm() + 1e-9)
            w2 = base.blocks[L].ffn.w2.weight  # [d_model, d_mlp]
            w2_orig = w2.data.clone()
            a_sel = act[topk]  # activations of the knowledge neurons
            best = None
            for scale in self.scales:
                w2.data = w2_orig.clone()
                # add scale*direction to each selected neuron's value vector,
                # weighted by the neuron activation magnitude (normalised)
                wsum = a_sel.abs().sum() + 1e-9
                for j, i in enumerate(topk):
                    w2.data[:, i] += scale * (a_sel[j].abs() / wsum) * direction
                pred = self._predict(base, s, r)
                if pred.strip() == o_target.strip():
                    best = scale
                    break
            if best is None:
                # keep the strongest attempt (last scale already applied)
                pred = self._predict(base, s, r)

        new_pred = self._predict(base, s, r)
        return base, EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=L, original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target),
        )
