"""Lightweight fine-tuning (FT) editor for GPTMini.

A simple, well-understood editing baseline that gradient-descends the edit
fact ``(s, r) -> o_target`` for a few steps. To be directly comparable to ROME,
it optimises the *same* parameters ROME edits (``blocks.{L}.ffn.w2``); all other
weights are frozen. Exposes the same ``apply_edit`` signature as ``ROME`` so the
plasticity-eval driver can use either interchangeably.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rome import EditResult, EditSpec


class FTEditor:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        lr: float = 5e-3,
        steps: int = 100,
        default_layer: int = 0,
        weight_decay: float = 0.0,
    ):
        self.original_model = model
        self.tok = tokenizer
        self.device = device
        self.lr = lr
        self.steps = steps
        self.default_layer = default_layer
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _predict(self, model, s: str, r: str) -> str:
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        logits = model(ids)["logits"][0, -1, :]
        return self.tok.get_token(int(torch.argmax(logits)))

    def apply_edit(
        self,
        s: str,
        r: str,
        o_target: str,
        layer: Optional[int] = None,
        copy_model: bool = True,
    ) -> Tuple[nn.Module, EditResult]:
        L = self.default_layer if layer is None else layer
        model = deepcopy(self.original_model) if copy_model else self.original_model

        orig_pred = self._predict(model, s, r)

        # train only blocks.{L}.ffn.w2 (same target as ROME)
        target_key = f"blocks.{L}.ffn.w2"
        train_params = []
        for n, p in model.named_parameters():
            train = target_key in n
            p.requires_grad_(train)
            if train:
                train_params.append(p)

        model.train()
        opt = torch.optim.Adam(train_params, lr=self.lr, weight_decay=self.weight_decay)
        ids = torch.tensor([self.tok.encode(f"{s} {r}")], device=self.device)
        tgt = torch.tensor([self.tok.get_id(o_target.strip())], device=self.device)

        for _ in range(self.steps):
            opt.zero_grad()
            logits = model(ids)["logits"][0, -1, :].unsqueeze(0)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            opt.step()
            if int(logits.argmax()) == int(tgt):
                break

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        new_pred = self._predict(model, s, r)

        return model, EditResult(
            success=(new_pred.strip() == o_target.strip()),
            layer=L,
            original_prediction=orig_pred.strip(),
            new_prediction=new_pred.strip(),
            edit_spec=EditSpec(s=s, r=r, o_target=o_target),
        )
