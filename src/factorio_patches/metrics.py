"""Masked-patch metrics + trivial baselines.

EMPTY dominates the grid, so raw accuracy is misleading. We therefore report
non-empty precision/recall/F1 and entity-token accuracy, and compare against
trivial baselines (always-EMPTY, always-majority-entity).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .vocab import EMPTY_ID, UNK_ID


def masked_ce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy over masked cells only. logits [B,V,H,W], y/mask [B,H,W].

    ``weight`` is an optional per-class weight (e.g. down-weight EMPTY to fight
    the heavy class imbalance and prevent collapse-to-empty).
    """
    loss = F.cross_entropy(logits, y, weight=weight, reduction="none")  # [B,H,W]
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def most_common_entity_id(grids: list[np.ndarray]) -> int:
    """Most frequent non-EMPTY token id across grids (the majority-entity prior)."""
    counts: dict[int, int] = {}
    for g in grids:
        ids, c = np.unique(g, return_counts=True)
        for i, n in zip(ids.tolist(), c.tolist()):
            if i != EMPTY_ID:
                counts[i] = counts.get(i, 0) + n
    if not counts:
        return UNK_ID
    return max(counts, key=counts.get)


class MaskedCounts:
    """Accumulates masked-cell counts for one predictor across batches."""

    def __init__(self):
        self.n = 0
        self.correct = 0
        self.n_y_nonempty = 0
        self.n_pred_nonempty = 0
        self.tp_detect = 0      # both non-empty (entity vs no-entity detection)
        self.tp_exact = 0       # pred == target and target non-empty
        self.n_topk = 0

    @torch.no_grad()
    def update(self, preds, y, mask, topk_idx=None):
        m = mask
        p = preds[m]
        t = y[m]
        self.n += t.numel()
        eq = p == t
        self.correct += int(eq.sum())
        ynz = t != EMPTY_ID
        pnz = p != EMPTY_ID
        self.n_y_nonempty += int(ynz.sum())
        self.n_pred_nonempty += int(pnz.sum())
        self.tp_detect += int((ynz & pnz).sum())
        self.tp_exact += int((eq & ynz).sum())
        if topk_idx is not None:
            tk = topk_idx.permute(0, 2, 3, 1)[m]      # [Nmasked, k]
            self.n_topk += int((tk == t.unsqueeze(1)).any(dim=1).sum())

    def metrics(self, with_topk: bool = False) -> dict:
        def safe(a, b):
            return (a / b) if b else 0.0
        prec = safe(self.tp_detect, self.n_pred_nonempty)
        rec = safe(self.tp_detect, self.n_y_nonempty)
        precx = safe(self.tp_exact, self.n_pred_nonempty)
        recx = safe(self.tp_exact, self.n_y_nonempty)
        out = {
            "masked_acc": safe(self.correct, self.n),
            "non_empty_precision": prec,
            "non_empty_recall": rec,
            "non_empty_f1": safe(2 * prec * rec, prec + rec),
            "non_empty_exact_f1": safe(2 * precx * recx, precx + recx),
            "entity_token_acc": safe(self.tp_exact, self.n_y_nonempty),
            "masked_cells": self.n,
            "target_nonempty_frac": safe(self.n_y_nonempty, self.n),
        }
        if with_topk:
            out["top5_acc"] = safe(self.n_topk, self.n)
        return out


@torch.no_grad()
def evaluate(model, loader, device, prior_id: int, topk: int = 5) -> dict:
    """Run model over loader and return model + baseline metrics on identical patches."""
    model.eval()
    acc_model = MaskedCounts()
    acc_empty = MaskedCounts()
    acc_prior = MaskedCounts()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        k = min(topk, logits.shape[1])
        topk_idx = logits.topk(k=k, dim=1).indices
        acc_model.update(preds, y, mask, topk_idx=topk_idx)
        acc_empty.update(torch.full_like(y, EMPTY_ID), y, mask)
        acc_prior.update(torch.full_like(y, prior_id), y, mask)
    return {
        "model": acc_model.metrics(with_topk=True),
        "baseline_empty": acc_empty.metrics(),
        "baseline_majority_entity": acc_prior.metrics(),
    }


def format_metrics(split: str, m: dict) -> str:
    md = m["model"]
    be = m["baseline_empty"]
    bp = m["baseline_majority_entity"]
    lines = [
        f"[{split}] model        : masked_acc={md['masked_acc']:.3f} "
        f"nonempty_f1={md['non_empty_f1']:.3f} exact_f1={md['non_empty_exact_f1']:.3f} "
        f"entity_acc={md['entity_token_acc']:.3f} top5={md.get('top5_acc', 0):.3f} "
        f"(P={md['non_empty_precision']:.3f} R={md['non_empty_recall']:.3f})",
        f"[{split}] base(empty)  : masked_acc={be['masked_acc']:.3f} nonempty_f1={be['non_empty_f1']:.3f} "
        f"entity_acc={be['entity_token_acc']:.3f}",
        f"[{split}] base(majority): masked_acc={bp['masked_acc']:.3f} nonempty_f1={bp['non_empty_f1']:.3f} "
        f"entity_acc={bp['entity_token_acc']:.3f}",
    ]
    return "\n".join(lines)
