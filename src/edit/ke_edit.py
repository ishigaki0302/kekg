"""KnowledgeEditor-style learned editor for GPTMini (adaptation of De Cao 2021).

Unlike MEND (which transforms the edit *gradient*), KnowledgeEditor conditions a
hypernetwork on the *edit itself* and predicts the weight update directly. Here
the condition is [subject hidden at the edit layer ; unembedding of o_new], and a
hypernetwork predicts low-rank factors (u, v) so that
    dW = -edit_lr * scale * (u (x) v)   on blocks[L].ffn.w2.
The hypernetwork is trained per base model with an edit-success loss + a
locality (KL) loss (functional forward), like MEND.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .rome import EditResult, EditSpec
from src.kg.symbolic_world import R_F


class KEEditor(nn.Module):
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
        hid = max(128, d_model)
        self.trunk = nn.Sequential(nn.Linear(2 * d_model, hid), nn.ReLU()).to(device)
        self.head_u = nn.Linear(hid, d_model).to(device)   # output-side factor
        self.head_v = nn.Linear(hid, d_mlp).to(device)     # key-side factor
        # NOTE: do NOT zero-init both heads — a rank-1 product outer(u, v) with
        # u=v=0 has zero gradient w.r.t. both factors (dead start). Default init
        # gives a small non-zero update so the hypernetwork can learn.
        self.scale = nn.Parameter(torch.tensor(1.0, device=device))

    def _ids(self, s, r=R_F):
        return torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)

    @staticmethod
    def _blk(o):
        return o[0] if isinstance(o, tuple) else o

    def _condition(self, s, o_new_id):
        """[subject hidden at edit layer ; unembedding(o_new)] (detached)."""
        L = self.default_layer
        m = self.original_model
        cap = {}
        h = m.blocks[L].register_forward_hook(
            lambda mod, i, o: cap.__setitem__("h", self._blk(o).detach()))
        with torch.no_grad():
            m(self._ids(s))
        h.remove()
        hs = cap["h"][0, -1, :]
        e = m.lm_head.weight[o_new_id].detach()
        return torch.cat([hs, e])

    def _delta_w(self, cond):
        z = self.trunk(cond)
        u = self.head_u(z)
        v = self.head_v(z)
        return -self.edit_lr * self.scale * torch.outer(u, v)  # [d_model, d_mlp]

    def train_editor(self, world, steps=1500, lr=1e-4, lam=1.0, loc_batch=24, seed=0):
        import random
        rng = random.Random(seed)
        ents = world.entities
        base = self.original_model
        for p in base.parameters():
            p.requires_grad_(False)
        W0 = base.blocks[self.default_layer].ffn.w2.weight.detach()
        params = (list(self.trunk.parameters()) + list(self.head_u.parameters())
                  + list(self.head_v.parameters()) + [self.scale])
        opt = torch.optim.Adam(params, lr=lr)
        for step in range(steps):
            s = ents[rng.randrange(len(ents))]
            o_old = world.func_map[s]
            o_new = s
            while o_new == s or o_new == o_old:
                o_new = ents[rng.randrange(len(ents))]
            o_new_id = self.tok.get_id(o_new)
            cond = self._condition(s, o_new_id)
            dW = self._delta_w(cond)
            override = {self.wname: W0 + dW}
            el = functional_call(base, override, (self._ids(s),))["logits"][0, -1, :]
            edit_loss = F.cross_entropy(el.unsqueeze(0),
                                        torch.tensor([o_new_id], device=self.device))
            loc = [ents[rng.randrange(len(ents))] for _ in range(loc_batch)]
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
        cond = self._condition(s, o_new_id)
        with torch.no_grad():
            dW = self._delta_w(cond)
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
        torch.save({"trunk": self.trunk.state_dict(), "head_u": self.head_u.state_dict(),
                    "head_v": self.head_v.state_dict(), "scale": self.scale.detach().cpu()}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.trunk.load_state_dict(sd["trunk"])
        self.head_u.load_state_dict(sd["head_u"])
        self.head_v.load_state_dict(sd["head_v"])
        with torch.no_grad():
            self.scale.copy_(sd["scale"].to(self.device))
        return self
