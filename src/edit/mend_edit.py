"""MEND-style learned editor for GPTMini (adaptation of Mitchell et al. 2022).

MEND learns to transform the raw edit-loss gradient of a layer into a refined
low-rank weight update that is reliable AND local. We adapt the core idea to a
single layer (blocks[L].ffn.w2):

  raw gradient of the edit CE loss w.r.t. w2 is  g = delta (x) a
      a     = input to w2 (post-GELU activation, d_mlp) at the edit token
      delta = output-side gradient (d_model), recovered as g @ a / (a.a)
  MEND maps (a, delta) -> (a', delta') via small residual MLPs, and the applied
  update is  dW = -edit_lr * scale * (delta' (x) a').

The MLPs + scale (the hypernetwork) are TRAINED per base model on a distribution
of synthetic edits with an edit-success loss + a locality (KL) loss. The model
gradient is treated as a (detached) input feature, so training is first-order.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .rome import EditResult, EditSpec
from src.kg.symbolic_world import R_F


class _ResMLP(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(64, dim // 2)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class MENDEditor(nn.Module):
    def __init__(self, model, tokenizer, device="cuda", default_layer=0, edit_lr=1e-2):
        super().__init__()
        self.original_model = model
        self.tok = tokenizer
        self.device = device
        self.default_layer = default_layer
        self.edit_lr = edit_lr
        self.wname = f"blocks.{default_layer}.ffn.w2.weight"
        d_model = model.config.d_model
        d_mlp = model.config.d_mlp
        self.mlp_a = _ResMLP(d_mlp).to(device)
        self.mlp_d = _ResMLP(d_model).to(device)
        self.scale = nn.Parameter(torch.tensor(1.0, device=device))

    def _ids(self, s, r=R_F):
        return torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)

    def _factors(self, s, o_new_id):
        """Detached (a, delta) of the edit-loss gradient at the edit layer.

        Base model is frozen for MEND training; we temporarily enable grad on
        just w2.weight to recover the raw gradient (used only as input feature).
        """
        L = self.default_layer
        m = self.original_model
        w2 = m.blocks[L].ffn.w2.weight
        prev = w2.requires_grad
        w2.requires_grad_(True)
        w2.grad = None
        cap = {}
        h = m.blocks[L].ffn.w2.register_forward_pre_hook(
            lambda mod, i: cap.__setitem__("a", i[0]))
        logits = m(self._ids(s))["logits"][0, -1, :]
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([o_new_id], device=self.device))
        loss.backward()
        h.remove()
        a = cap["a"][0, -1, :].detach()
        gW = w2.grad.detach()                          # [d_model, d_mlp]
        delta = gW @ a / (a @ a + 1e-9)               # [d_model]
        w2.grad = None
        w2.requires_grad_(prev)
        return a.detach(), delta.detach()

    def _delta_w(self, a, delta):
        a2 = self.mlp_a(a)
        d2 = self.mlp_d(delta)
        return -self.edit_lr * self.scale * torch.outer(d2, a2)  # [d_model, d_mlp]

    # ---------------------------------------------------------------- training
    def train_editor(self, world, steps=1500, lr=1e-4, lam=1.0, loc_batch=24, seed=0):
        import random
        rng = random.Random(seed)
        ents = world.entities
        base = self.original_model
        for p in base.parameters():
            p.requires_grad_(False)
        W0 = base.blocks[self.default_layer].ffn.w2.weight.detach()
        opt = torch.optim.Adam(
            list(self.mlp_a.parameters()) + list(self.mlp_d.parameters()) + [self.scale], lr=lr)
        for step in range(steps):
            s = ents[rng.randrange(len(ents))]
            o_old = world.func_map[s]
            o_new = s
            while o_new == s or o_new == o_old:
                o_new = ents[rng.randrange(len(ents))]
            o_new_id = self.tok.get_id(o_new)
            a, delta = self._factors(s, o_new_id)
            dW = self._delta_w(a, delta)
            override = {self.wname: W0 + dW}
            # edit-success loss
            el = functional_call(base, override, (self._ids(s),))["logits"][0, -1, :]
            edit_loss = F.cross_entropy(el.unsqueeze(0),
                                        torch.tensor([o_new_id], device=self.device))
            # locality loss: KL on other random facts (base vs edited)
            loc = [e for e in (ents[rng.randrange(len(ents))] for _ in range(loc_batch)) if e != s]
            loc_ids = torch.tensor([self.tok.encode(f"{e} {R_F}") for e in loc], device=self.device)
            with torch.no_grad():
                base_lp = F.log_softmax(base(loc_ids)["logits"][:, -1, :], dim=-1)
            ed_lp = F.log_softmax(
                functional_call(base, override, (loc_ids,))["logits"][:, -1, :], dim=-1)
            loc_loss = F.kl_div(ed_lp, base_lp, log_target=True, reduction="batchmean")
            loss = edit_loss + lam * loc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def _predict(self, model, s, r):
        return self.tok.get_token(int(torch.argmax(model(self._ids(s, r))["logits"][0, -1, :])))

    def apply_edit(self, s, r, o_target, layer=None, copy_model=True):
        L = self.default_layer
        orig_pred = self._predict(self.original_model, s, r)
        o_new_id = self.tok.get_id(o_target.strip())
        a, delta = self._factors(s, o_new_id)
        with torch.no_grad():
            dW = self._delta_w(a, delta)
        m = deepcopy(self.original_model)
        m.blocks[L].ffn.w2.weight.data = (m.blocks[L].ffn.w2.weight.data + dW).clone()
        m.eval()
        new_pred = self._predict(m, s, r)
        return m, EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=L, original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target),
        )

    def save(self, path):
        torch.save({"mlp_a": self.mlp_a.state_dict(), "mlp_d": self.mlp_d.state_dict(),
                    "scale": self.scale.detach().cpu()}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.mlp_a.load_state_dict(sd["mlp_a"])
        self.mlp_d.load_state_dict(sd["mlp_d"])
        with torch.no_grad():
            self.scale.copy_(sd["scale"].to(self.device))
        return self
