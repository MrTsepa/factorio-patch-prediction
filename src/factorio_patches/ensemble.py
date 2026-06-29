"""Ensemble + test-time-augmentation evaluation (shared by the local harness and Modal).

Averages probabilities over a model ensemble and (optionally) the 8 D4 views, then scores
exact entity-token accuracy over masked, non-empty-target cells.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .dataset import FactorioPatchDataset
from .metrics import maskgit_decode
from .symmetry import build_perms, tta_logits
from .vocab import EMPTY_ID


@torch.no_grad()
def predict_probs(models, x, mask, vocab, device, tta=False, maskgit=0, perms_t=None):
    """Averaged probabilities [B,V,H,W] over the ensemble (and D4 views if tta)."""
    accs = []
    for m in models:
        if maskgit:
            pred = maskgit_decode(m, x, mask, steps=maskgit)
            accs.append(torch.nn.functional.one_hot(pred, len(vocab)).permute(0, 3, 1, 2).float())
        elif tta:
            accs.append(tta_logits(m, x, vocab, device, perms_t))
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                accs.append(m(x).float().softmax(dim=1))
    return torch.stack(accs).mean(0)


@torch.no_grad()
def entity_acc(models, grids, vocab, device, tta=False, maskgit=0, n=1500, bs=128, seed=0):
    """Exact entity-token accuracy over masked, non-empty-target cells (matches metrics.py)."""
    ds = FactorioPatchDataset(grids, train=False, length=n, seed=seed)
    perms_t = [torch.as_tensor(p, device=device) for p in build_perms(vocab)] if tta else None
    tp = tot = 0
    for b in DataLoader(ds, batch_size=bs):
        x, y, mask = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
        pred = predict_probs(models, x, mask, vocab, device, tta, maskgit, perms_t).argmax(1)
        m = mask & (y != EMPTY_ID)
        tp += int((pred[m] == y[m]).sum()); tot += int(m.sum())
    return tp / max(tot, 1)
